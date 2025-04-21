"""
hybrid_chunker.py
-----------------

A high‑level semantic chunking module that breaks long text into
coherent pieces no larger than a model’s context window, then
refines those pieces via an iterative, hill‑climbing optimizer.

Key components:
  - SectionChunker    : Coarse splits on paragraphs or headings
  - SentenceChunker   : Finer splits on sentence boundaries
  - FixedCharacterChunker : Fallback fixed‑stride splitter
  - RandomInitializer : Produces initial (start,end) groupings
  - HybridChunker     : Combines the above, then optimizes chunk
                        boundaries for:
                        * high intra‑chunk cohesion
                        * low inter‑chunk similarity
                        * full coverage, minimal overlap
                        * no overflow beyond token limit

Configurable parameters (via `config` dict):
  - chunk_size     (int)   : max tokens per chunk (≤ encoder.ctx_length)
  - max_iteration  (int)   : hill‑climb iterations
  - min_coverage   (float) : [0.0,1.0] minimum coverage
  - update_rate    (float) : (0.0,1.0] fraction of chunks tweaked per step
  - randomness    (float) : [0.0,1.0] random‑restart chance
  - delta          (float) : ≥0.0 early‑stop improvement threshold
  - patient        (int)   : non‑improving rounds before stopping
"""

import logging
import random
from functools import lru_cache
from typing import Optional

# External packages
from pydantic import BaseModel, field_validator

# Custom packages
from .._encoder import Encoder
from .._chunkers import ChunkerMetrics, RandomInitializer
from .basic import SectionChunker, SentenceChunker, FixedCharacterChunker, FixedCharacterChunkerConfig
from .utility import estimate_token_count

logger = logging.getLogger(__name__)


class HybridChunkerConfig(BaseModel):
    """
    Configuration for the HybridChunker.

    Attributes:
        chunk_size (int): Maximum number of tokens per chunk.
        max_iteration (int): Maximum number of optimization iterations.
        min_coverage (float): Minimum required coverage of the input text.
        update_rate (float): Fraction of chunks updated in each optimization iteration.
        randomness (float): Randomness factor for chunk updates.
        delta (float): Minimum improvement in evaluation score to reset early stopping counter.
        patient (int): Number of non-improving iterations before early stopping.
    """
    chunk_size: int = 512
    max_iteration: int = 20
    min_coverage: float = 0.9
    update_rate: float = 0.5
    randomness: float = 0.25
    delta: float = 0.0001
    patient: int = 5

    @field_validator("n_chunk")
    def validate_n_chunk(cls, v: int) -> int:
        """
        Validates the number of chunks to create.

        Args:
            v (int): Number of chunks.

        Returns:
            int: Validated number of chunks.
        """
        if v < 1:
            raise ValueError("Number of chunks must be at least 1.")
        return v

    @field_validator("chunk_size")
    def validate_chunk_size(cls, v: int) -> int:
        """
        Validates the maximum number of tokens per chunk.

        Args:
            v (int): Maximum number of tokens.

        Returns:
            int: Validated maximum number of tokens.
        """
        if v <= 0:
            raise ValueError("Chunk size must be greater than 0.")
        return v

    @field_validator("max_iteration")
    def validate_max_iteration(cls, v: int) -> int:
        """
        Validates the maximum number of optimization iterations.

        Args:
            v (int): Maximum number of iterations.

        Returns:
            int: Validated maximum number of iterations.
        """
        if v < 1:
            raise ValueError("Maximum iterations must be at least 1.")
        return v

    @field_validator("min_coverage")
    def validate_min_coverage(cls, v: float) -> float:
        """
        Validates the minimum required coverage of the input text.

        Args:
            v (float): Minimum coverage.

        Returns:
            float: Validated minimum coverage.
        """
        if not (0.0 < v <= 1.0):
            raise ValueError("Coverage must be between 0.0 and 1.0.")
        return v

    @field_validator("update_rate")
    def validate_update_rate(cls, v: float) -> float:
        """
        Validates the fraction of chunks updated in each optimization iteration.

        Args:
            v (float): Update rate.

        Returns:
            float: Validated update rate.
        """
        if not (0.0 < v <= 1.0):
            raise ValueError("Update rate must be between 0.0 and 1.0.")
        return v

    @field_validator("randomness")
    def validate_randomness(cls, v: float) -> float:
        """
        Validates the randomness factor for chunk updates.

        Args:
            v (float): Randomness.

        Returns:
            float: Validated randomness.
        """
        if not (0.0 <= v <= 1.0):
            raise ValueError("Randomness must be between 0.0 and 1.0.")
        return v

    @field_validator("delta")
    def validate_delta(cls, v: float) -> float:
        """
        Validates the minimum improvement in evaluation score to reset early stopping counter.

        Args:
            v (float): Delta.

        Returns:
            float: Validated delta.
        """
        if v < 0.0:
            raise ValueError("Delta must be greater than or equal to 0.0.")
        return v

    @field_validator("patient")
    def validate_patient(cls, v: int) -> int:
        """
        Validates the number of non-improving iterations before early stopping.

        Args:
            v (int): Patient.

        Returns:
            int: Validated patient.
        """
        if v < 0:
            raise ValueError("Patient must be greater than or equal to 0.")
        return v

    def step_forward(
        self, current_grouping: list[tuple[int, int]], N_LINE: int
    ) -> list[tuple[int, int]]:
        """
        Performs one step of the optimization process by randomly adjusting the boundaries of a subset of chunks.

        This function selects a fraction of the current chunks (determined by `self.update_rate`)
        and randomly either increases or decreases their start or end indices by one,
        within the bounds of the total number of text units (`N_LINE`). It also ensures
        that no generated chunk has a length of less than 2. Duplicate chunks resulting
        from these adjustments are removed, and if the number of chunks decreases due
        to duplication, new random chunks are added to maintain the original number
        of chunks.

        **Infinite Loop Prevention**: 
            On every update round, try to find a new valid chunk up to `N_COMBS` times.
            It's expected that on certain update rounds, no point is updated.
            
            * If no valid chunk is found, break the loop when `N_COMBS` is reached.
            * If a valid chunk is found, break the loop early. Update the point.

        **Criterias for Valid Chunk**:
            1. Not duplicated with other chunks
            2. Up to HybridChunker.C elements in a chunk
            
        Args:
            current_grouping (List[Tuple[int, int]]): The current list of chunk tuples,
                where each tuple represents a chunk's (start_index, end_index).
            N_LINE (int): The total number of text units (e.g., sentences) in the input.
                This defines the upper bound for the chunk end indices.

        Returns:
            new_grouping (List[Tuple[int, int]]): 
            A new list of chunk tuples representing the grouping
            after one step of random adjustment.
        """
        new_grouping: list[tuple[int, int]] = current_grouping[:]
        G = len(current_grouping)
        F = max(1, int(G * self.update_rate))
        N_COMBS = self.calculate_number_of_combinations(N_LINE, HybridChunker.C)
        # Update a random chunk `F` times
        for _ in range(F):
            # Make sure the chunk is not too small!
            left, right = 0, 0
            point = -1
            found: bool = False
            loop_counter: int = 0
            while loop_counter < N_COMBS:
                # Randomly select a chunk
                point = random.randint(0, G - 1)
                reference_tuple = new_grouping[point]
                # 0: decrement, 1: increment
                increment = random.randint(0, 1) == 0

                if increment:
                    left = reference_tuple[0]
                    right = min(N_LINE, reference_tuple[1] + 1)
                else:
                    left = max(0, reference_tuple[0] - 1)
                    right = reference_tuple[1]

                for gs, ge in new_grouping:
                    found = gs == left and ge == right
                    if found:
                        break

                if right - left >= HybridChunker.C and not found:
                    break

                loop_counter += 1

            if found:
                new_grouping[point] = (left, right)

        return new_grouping

    @staticmethod
    def calculate_cosine_similarity(
        vec1: list[float], vec2: list[float]
    ) -> float:
        """
        Calculates the cosine similarity between two vectors.

        Args:
            vec1 (List[float]): The first vector.
            vec2 (List[float]): The second vector.

        Returns:
            similarity (float): The cosine similarity between the two vectors.
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1)**0.5
        norm2 = sum(b * b for b in vec2)**0.5
        if norm1 != 0 and norm2 != 0:
            similarity = dot_product / (norm1 * norm2)
        else:
            similarity = 0
        return similarity

    @lru_cache(maxsize=512)
    def str_to_embedding(self, text: str) -> list[float]:
        """
        Encodes a text string into its embedding vector using the encoder.

        This method uses memoization (lru_cache) to avoid redundant encoding of the
        same text.

        Args:
            text (str): The text string to encode.

        Returns:
            out (List[float]): The embedding vector of the text.
        """
        return self.__encoder.encode(text)

    @staticmethod
    def isascii(text: str) -> bool:
        """
        Determines if a given text string consists entirely of ASCII characters.

        This method uses the `charade` library to detect the encoding of the input
        text. If the detected encoding is ASCII, the function returns `True`;
        otherwise, it returns `False`.

        Args:
            text (str): The text string to be checked.

        Returns:
            out (bool): `True` if the text is entirely ASCII, `False` otherwise.
        """
        byte_sentence = text.encode("utf-8")
        result = charade.detect(byte_sentence)
        return result["encoding"] == "ascii"

    def eval(self, *args) -> float:
        """
        Evaluates the current chunking based on semantic coherence and other metrics.

        The evaluation considers intra-group cohesion (similarity within chunks),
        inter-group cohesion (similarity between chunks), coverage (proportion of
        text included in chunks), overlap (redundancy between chunks) and
        overflow (proportion of text beyond the context_length).

        Args:
            *args: Variable length argument list.  Must contain:
                - lines (List[str]): A list of text units (e.g., sentences).
                - grouping (List[Tuple[int, int]]): A list of chunk tuples.
                - capacity (int): The total number of text units.
                - verbose (bool): Whether to log metrics.

        Returns:
            score (float): A score representing the quality of the chunking. Higher scores indicate better chunking.
            ```python
            C: float = 0.25
            postive_metrics = [intra-group cohesion, coverage]
            negative_metrics = [inter-group cohesion, overlapped, overflow]
            score = sum(positive_metrics) - sum(negative_metrics) * C
            ```
        """
        assert len(args) >= 4, "Expect lines, grouping, capacity, verbose."
        lines, grouping, capacity, verbose, *_ = args
        intra_group_cohesion: float = self.__calculate_intra_group_cohesion(
            lines, grouping, verbose
        )
        inter_group_cohesion: float = self.__calculate_inter_group_cohesion(
            lines, grouping, verbose
        )
        overflow: float = self.__calculate_overflow(lines, grouping, verbose)
        overlapped = ChunkerMetrics.calculate_overlapped(capacity, grouping)
        coverage = ChunkerMetrics.calculate_coverage(capacity, grouping)

        C: float = 0.25
        positive_metrics = [intra_group_cohesion, coverage]
        negative_metrics = [inter_group_cohesion, overlapped, overflow]
        score: float = sum(positive_metrics) - sum(negative_metrics) * C

        if verbose:
            logger.warning("metrics/overlapped: %.4f", overlapped)
            logger.warning("metrics/coverage: %.4f", coverage)
            logger.warning("metrics/score: %.4f", score)
        return score

    def __calculate_intra_group_cohesion(
        self,
        lines: list[str],
        grouping: list[tuple[int, int]],
        verbose: bool = True
    ) -> float:
        """
        Calculates the average semantic cohesion within each chunk.

        For each chunk containing more than one text unit, this method computes the
        cosine similarity between the embeddings of all possible contiguous sub-parts
        of the chunk and averages these similarities. The overall intra-group cohesion
        is then the average of these per-chunk cohesion scores.

        Args:
            lines (List[str]): A list of the original text units.
            grouping (List[Tuple[int, int]]): A list of tuples, where each tuple
                (start, end) defines a chunk's boundaries.
            verbose (bool, optional): If `True`, logs the calculated intra-group
                cohesion. Defaults to `True`.

        Returns:
            intra_group_cohesion (float): The average intra-group semantic cohesion across all chunks.

        Example:
            >>> lines = [L1, L2, L3, L4, L5]
            >>> grouping = [(0, 3), (3, 5)]
            >>> # cs = method to compute cosine similarity between two vectors
            >>> group1_cohesion = [cs(L1, L2+L3), cs(L1+L2, L3)] / 2
            >>> group2_cohesion = [cs(L4, L5)] / 1
            >>> intra_group_cohesion = (group1_cohesion + group2_cohesion) / 2
        """
        intra_group_cohesion: float = 0
        for g_start, g_end in grouping:
            if g_end - g_start < 2:
                # Only one line
                continue

            group_cohesion: float = 0.0
            for vi in range(g_start, g_end - 1):
                a_text = self.reconstruct_custom_chunk(
                    lines[g_start:vi + 1], "sentence"
                )
                b_text = self.reconstruct_custom_chunk(
                    lines[vi + 1:g_end], "sentence"
                )
                part_a_embedding = self.str_to_embedding(a_text)
                part_b_embedding = self.str_to_embedding(b_text)

                cosine_similarity = self.calculate_cosine_similarity(
                    part_a_embedding, part_b_embedding
                )

                group_cohesion += cosine_similarity
            group_cohesion /= g_end - g_start - 1

            intra_group_cohesion += group_cohesion
        intra_group_cohesion /= len(grouping)

        if verbose:
            logger.warning(
                "metrics/intra_group_cohesion: %.4f", intra_group_cohesion
            )
        return intra_group_cohesion

    def __calculate_inter_group_cohesion(
        self,
        lines: list[str],
        grouping: list[tuple[int, int]],
        verbose: bool = True
    ) -> float:
        """
        Calculates the average semantic cohesion between different chunks.

        This method computes the cosine similarity between the embeddings of each
        pair of distinct chunks and averages these similarities to provide a measure
        of how semantically related the different chunks are to each other. Lower
        values are generally preferred, indicating that the chunks are more distinct.

        The cosine similarity is computed for each pair of distinct chunks.
        
        Args:
            lines (List[str]): A list of the original text units.
            grouping (List[Tuple[int, int]]): A list of tuples, where each tuple
                (start, end) defines a chunk's boundaries.
            verbose (bool, optional): If `True`, logs the calculated inter-group
                cohesion. Defaults to `True`.

        Returns:
            inter_group_cohesion (float): 
            The average inter-group semantic cohesion between all pairs of distinct chunks.

        Example:
            >>> lines = [L1, L2, L3, L4, L5, L6, L7, L8, L9]
            >>> grouping = [(0, 3), (3, 5), (5, 9)]
            >>> group_line = [L1+L2+L3, L4+L5, L6+L7+L8+L9]
            >>> # cs = method to compute cosine similarity between two vectors
            >>> inter_group_cohesion = [
                cs(group_line[0], group_line[1]), 
                cs(group_line[0], group_line[2]), 
                cs(group_line[1], group_line[2])
                ] / len(grouping)
        """
        n_groups = len(grouping)
        if n_groups < 2:
            return 0

        n_compare = n_groups * (n_groups - 1) / 2
        inter_group_cohesion: float = 0
        for i in range(n_groups - 1):
            i_start, i_end = grouping[i]
            i_line = self.reconstruct_custom_chunk(
                lines[i_start:i_end], "sentence"
            )
            i_embedding = self.str_to_embedding(i_line)
            for j in range(i + 1, n_groups):
                j_start, j_end = grouping[j]
                j_line = self.reconstruct_custom_chunk(
                    lines[j_start:j_end], "sentence"
                )
                j_embedding = self.str_to_embedding(j_line)
                cosine_similarity = self.calculate_cosine_similarity(
                    i_embedding, j_embedding
                )
                inter_group_cohesion += cosine_similarity
        inter_group_cohesion /= n_compare

        if verbose:
            logger.warning(
                "metrics/inter_group_cohesion: %.4f", inter_group_cohesion
            )
        return inter_group_cohesion

    def __calculate_overflow(
        self,
        lines: list[str],
        grouping: list[tuple[int, int]],
        verbose: bool = True
    ) -> float:
        """
        Calculates a penalty based on the estimated token count of each chunk exceeding the maximum allowed `chunk_size`.

        This method estimates the number of tokens in each chunk based on its length
        and whether it primarily contains ASCII characters (assuming a lower byte-to-token
        ratio for ASCII). The overflow for each chunk is the proportion by which its
        estimated token count exceeds `chunk_size`. The overall overflow is the average
        overflow across all chunks.

        Args:
            lines (List[str]): A list of the original text units.
            grouping (List[Tuple[int, int]]): A list of tuples, where each tuple
                (start, end) defines a chunk's boundaries.
            verbose (bool, optional): If `True`, logs the calculated overflow.
                Defaults to `True`.

        Returns:
            overflow (float): 
            The average overflow penalty across all chunks. 
            A value of 0 indicates no chunks exceed the token limit. 
            Positive values indicate the degree of exceedance.

        Notes:
        * **Token Estimation**: The token scale factor is 0.5 for ASCII and 0.75 for non-ASCII.

        Example:
            >>> lines = [L1, L2, L3, L4, L5, L6, L7, L8, L9]
            >>> grouping = [(0, 3), (3, 5), (5, 9)]
            >>> # est_token_count = method to estimate token count
            >>> overflow = [
                    max(0, est_token_count(L1+L2+L3) / chunk_size), 
                    max(0, est_token_count(L4+L5) / chunk_size), 
                    max(0, est_token_count(L6+L7+L8+L9) / chunk_size)
                ] / len(grouping)
        """
        overflow: float = 0.0
        for g_start, g_end in grouping:
            g_line = self.reconstruct_custom_chunk(
                lines[g_start:g_end], "sentence"
            )
            coef = 0.5 if self.isascii(g_line) else 0.75
            est_token_count = len(g_line) * coef
            g_overflow = max(0.0, est_token_count / self.chunk_size - 1.0)
            overflow += g_overflow
        overflow /= len(grouping)

        if verbose:
            logger.warning("metrics/overflow: %.4f", overflow)
        return overflow

    @staticmethod
    def calculate_number_of_combinations(N: int, C: int) -> int:
        if N == C:
            return 1

        if N < C or N < 2 * C:
            return 0

        return N - 2 * C + 1

    def resolve_tight_search_space(self,
                                   lines: list[str]) -> Optional[list[str]]:
        parts = [
            self.reconstruct_custom_chunk(lines[:HybridChunker.C], "sentence"),
            self.reconstruct_custom_chunk(lines[HybridChunker.C:], "sentence")
        ]
        counter: int = 0
        for part in parts:
            part_token_count = len(part) * 0.5 if self.isascii(part) else 0.75
            if part_token_count > self.chunk_size:
                counter += 1
        if counter == 0:
            return parts

        return None

    def split(self, long_text: str):
        """
        Splits a long text into a list of semantically coherent chunks.

        This is the main entry point for the `HybridChunker`. It first performs an
        initial coarse-grained splitting of the text by sections. For sections that
        are estimated to exceed the `chunk_size`, it further splits them into sentences
        and then uses the `find_best_grouping` method to optimize the sentence-level
        chunking based on semantic coherence. Smaller sections are either directly
        added to the output or merged with subsequent sections if they don't exceed
        the `chunk_size` when combined.

        Args:
            long_text (str): The input text to be split into chunks.

        Returns:
            output (List[str]): 
            A list of strings, where each string represents a semantically
            coherent chunk of the original text.
        """
        logger.warning("[BEG] split")
        logger.warning("CONFIG: %s", self.config)
        logger.warning("Encoder: %s", self.encoder)

        if not isinstance(long_text, str):
            raise TypeError(
                f"Expected 'long_text' to be str, got {type(long_text).__name__}."
            )

        # Sanitize argument `long_text`
        text = long_text.strip()
        if len(text) == 0:
            raise ValueError("Expect long_text to be non-empty string.")

        coef = 0.5 if self.isascii(text) else 0.75
        if len(text) * coef < self.chunk_size:
            return [text]

        section_chunker = SectionChunker({})
        sentence_chunker = SentenceChunker({})

        output: list[str] = []
        temp: str = ""
        for section in section_chunker.split(text):
            section_len = len(section)
            coef = 0.5 if self.isascii(section) else 0.75
            est_tc = section_len * coef
            if est_tc > self.chunk_size:
                logger.warning("Content:\n%s", section)
                sentences = sentence_chunker.split(section)
                L = len(sentences)
                n_combs = HybridChunker.calculate_number_of_combinations(
                    L, HybridChunker.C
                )
                if n_combs == 1:
                    parts = self.resolve_tight_search_space(sentences)
                    if parts:
                        output.extend(parts)
                        continue

                    splitter = FixedCharacterChunker(
                        {
                            "chunk_size": 128,
                            "stride_rate": 0.9
                        }
                    )
                    sentences = splitter.split(section)
                    L = len(sentences)

                # When distributed evenly, every chunks are about chunk_size.
                # 20 sentences per group
                # At least 2 groups
                G: int = max(
                    2,
                    int(est_tc // self.chunk_size),
                    L // 20,
                )
                grouping = self.find_best_grouping(sentences, G)
                for g_start, g_end in grouping:
                    g_chunk = self.reconstruct_custom_chunk(
                        sentences[g_start:g_end], "sentence"
                    )
                    output.append(g_chunk)
            elif est_tc > self.chunk_size * 0.75:
                if temp:
                    output.append(temp)
                    temp = ""
                output.append(section)
            elif est_tc + len(temp) * coef > self.chunk_size:
                output.append(temp)
                temp = section
            else:
                if temp:
                    temp = self.reconstruct_custom_chunk(
                        [temp, section], "section"
                    )
                else:
                    temp = section
        if temp:
            output.append(temp)

        logger.warning("[END] split")
        return output

    def optimize(self, grouping: list[tuple[int, int]],
                 n_line: int) -> list[tuple[int, int]]:
        """
        Performs a single optimization step on the current chunk grouping.

        The optimization step either initializes a new random grouping (based on the
        `temperature` parameter) or performs a random adjustment of the existing
        chunk boundaries using the `step_forward` method.

        Args:
            grouping (List[Tuple[int, int]]): The current list of chunk tuples.
            n_line (int): The total number of text units.

        Returns:
            grouping (List[Tuple[int, int]]): 
            The new list of chunk tuples after the optimization step.
        """
        if random.random() < self.temperature:
            logger.warning("Initializing new random grouping")
            initializer = RandomInitializer(n_line, len(grouping))
            return initializer.init()
        return self.step_forward(grouping, n_line)

    def find_best_grouping(self, lines: list[str],
                           G: int) -> list[tuple[int, int]]:
        """
        Iteratively optimizes the grouping of text units into semantically coherent chunks.

        This method starts with an initial grouping and iteratively refines it by
        proposing new groupings using the `optimize` method and evaluating their
        quality using the `eval` method. It keeps track of the best grouping found
        so far based on the evaluation score and the coverage of the text. The
        optimization process continues for a maximum number of iterations (`max_iteration`)
        or until no significant improvement is observed for a certain number of
        consecutive iterations (`patient`), implementing an early stopping mechanism.

        Args:
            lines (List[str]): A list of text units (e.g., sentences) to be grouped.
            G (int): Expected number of groups.
            C (int): Minimum number of text units per group.

        Returns:
            best_grouping (List[Tuple[int, int]]): 
            The best grouping of text units found during the
            optimization process, sorted by the start index of each chunk.
        """
        logger.warning("[BEG] find_best_grouping")
        L = len(lines)
        initializer = RandomInitializer(L, G)
        grouping = initializer.init()

        # Variables
        iteration: int = 0
        best_score: float = 0.0
        best_grouping: list[tuple[int, int]] = grouping[:]
        non_improving_counter: int = 0

        logger.warning("Lines: %d, Groups: %d", L, G)
        while iteration < self.max_iteration:
            logger.warning("======= [%d] =======", iteration)
            iteration += 1
            score: float = self.eval(lines, grouping, L)
            if score - best_score < self.delta:
                non_improving_counter += 1
                if non_improving_counter >= self.patient:
                    logger.warning("Early Stopping!")
                    logger.warning("Grouping: %s", grouping)
                    break
            else:
                non_improving_counter = 0

            coverage: float = ChunkerMetrics.calculate_coverage(L, grouping)
            if score > best_score and coverage >= self.min_coverage:
                best_score = score
                best_grouping = grouping[:]
                logger.warning("Improved! Score: %.4f.", score)
                logger.warning("Grouping: %s", grouping)

            grouping = self.optimize(grouping, L)

        score: float = self.eval(lines, grouping, L)
        coverage: float = ChunkerMetrics.calculate_coverage(L, grouping)
        if score > best_score and coverage >= self.min_coverage:
            best_score = score
            best_grouping = grouping[:]
            logger.warning("Improved! Score: %.4f.", score)
            logger.warning("Grouping: %s", grouping)

        # Sort the best grouping
        best_grouping.sort(key=lambda x: x[0])
        logger.warning("Best Grouping: %s", best_grouping)
        logger.warning("[END] find_best_grouping")
        return best_grouping

    @staticmethod
    def reconstruct_custom_chunk(partial_chunk: list[str], level: str) -> str:
        """
        Reconstructs a text chunk from a list of text units.

        The reconstruction method depends on the specified level (e.g., "section"
        or "sentence").

        Args:
            partial_chunk (List[str]): A list of text units.
            level (str): The level of text unit ("section" or "sentence").

        Returns:
            output (str): 
            The reconstructed text chunk.
        """
        if level == "section":
            return "\n\n".join([chunk.strip() for chunk in partial_chunk])
        return " ".join([chunk.strip() for chunk in partial_chunk])
