import random
import logging
from math import ceil
from pydantic import BaseModel, field_validator

from .._chunkers import ChunkerMetrics, RandomInitializer
from .._encoder import Encoder
from .basic import SentenceChunker, FixedGroupChunkerConfig, FixedGroupChunker
from .utility import reconstruct_chunk, estimate_token_count, all_within_chunk_size

logger = logging.getLogger(__name__)


class SemanticChunkerConfig(BaseModel):
    """
    Attributes:
        chunk_size (int): The size of each chunk in terms of tokens.
            Default is 128.
        update_rate (float): The rate at which the grouping is updated during optimization.
            Default is 0.3.
        min_coverage (float): The minimum coverage required for a grouping to be considered valid.
            Default is 0.8.
        max_iteration (int): The maximum number of iterations for the optimization process.
            Default is 20.
        randomness (float): The randomness factor for the optimization process.
            Default is 0.2. Higher values indicate more randomness.
    """
    n_chunk: int = 1
    chunk_size: int = 128
    update_rate: float = 0.3
    min_coverage: float = 0.8
    max_iteration: int = 20
    randomness: float = 0.2

    @field_validator("n_chunk")
    @classmethod
    def validate_n_chunk(cls, value: int) -> int:    # pylint: disable=no-self-argument
        if value <= 0:
            raise ValueError("n_chunk must be greater than 0")
        return value

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, value: int) -> int:    # pylint: disable=no-self-argument
        if value <= 0:
            raise ValueError("K must be greater than 0")
        return value

    @field_validator("update_rate")
    @classmethod
    def validate_update_rate(cls, value: float) -> float:    # pylint: disable=no-self-argument
        if value < 0 or value > 1:
            raise ValueError("update_rate must be between 0 and 1")
        return value

    @field_validator("min_coverage")
    @classmethod
    def validate_min_coverage(cls, value: float) -> float:    # pylint: disable=no-self-argument
        if value < 0 or value > 1:
            raise ValueError("min_coverage must be between 0 and 1")
        return value

    @field_validator("max_iteration")
    @classmethod
    def validate_max_iteration(cls, value: int) -> int:    # pylint: disable=no-self-argument
        if value < 1:
            raise ValueError("max_iteration must be at least 1")
        return value

    @field_validator("randomness")
    @classmethod
    def validate_randomness(cls, value: float) -> float:    # pylint: disable=no-self-argument
        if value < 0 or value > 1:
            raise ValueError("randomness must be between 0 and 1")
        return value


class SemanticChunker:
    """
    """
    C: int = 2

    def __init__(
        self,
        encoder: Encoder,
        config: SemanticChunkerConfig,
    ):
        self.__encoder = encoder
        self.__config = config

        # Cache Variables
        self.__pws_cache: dict[tuple[int, int], float] = {}
        self.__e_cache: dict[tuple[int, int], tuple[list[float], int]] = {}

    @property
    def config(self) -> SemanticChunkerConfig:
        return self.__config

    @property
    def encoder(self) -> Encoder:
        return self.__encoder

    @staticmethod
    def drop_duplicates(
        grouping: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        unique_set = set()
        for group in grouping:
            if group not in unique_set:
                unique_set.add(group)
        return [*unique_set]

    @staticmethod
    def calculate_number_of_combinations(N: int, C: int) -> int:
        if N == C:
            return 1

        if N < C or N < 2 * C:
            return 0

        return N - 2 * C + 1

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
        F = max(1, int(G * self.config.update_rate))
        N_COMBS = self.calculate_number_of_combinations(
            N_LINE, SemanticChunker.C
        )
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

                if right - left >= SemanticChunker.C and not found:
                    break

                loop_counter += 1

            if not found:
                new_grouping[point] = (left, right)

        return new_grouping

    @staticmethod
    def calculate_cosine_similarity(
        vec1: list[float], vec2: list[float]
    ) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1)**0.5
        norm2 = sum(b * b for b in vec2)**0.5
        similarity = dot_product / (
            norm1 * norm2
        ) if norm1 != 0 and norm2 != 0 else 0
        return similarity

    def calculate_pairwise_similarity(
        self,
        embeddings: list[list[float]],
        start: int,
        end: int,
    ) -> float:
        """
        Computes the average pairwise cosine similarity between lines within a group.

        Args:
            embeddings (List[List[float]]): Embeddings of all lines.
            start (int): Start index of the group.
            end (int): End index of the group.

        Returns:
            float: Average pairwise similarity score.

        range(start, end)
        x = sum(cosine_similarity(vec_i, vec_j)) where i != j
        y = end - start
        result = x / y
        """
        if end - start <= 1:
            return 0

        pairwise_similarities = []
        for vi in range(start, end):
            for vj in range(vi + 1, end):
                key = (vi, vj)
                if key not in self.__pws_cache:
                    self.__pws_cache[key] = self.calculate_cosine_similarity(
                        embeddings[vi], embeddings[vj]
                    )
                similarity = self.__pws_cache[key]
                # logger.info("%d vs %d => %.4f", vi, vj, similarity)
                pairwise_similarities.append(similarity)
        return (
            sum(pairwise_similarities) /
            len(pairwise_similarities) if pairwise_similarities else 0
        )

    def _encode(self, lines: list[str], start: int,
                end: int) -> tuple[list[float], int]:
        """Cached encoding.

        Args:
            lines (List[str]): A list of all lines.
            start (int): The start index
            end (int): The end index

        Returns:
            Tuple[List[float], int]: The corresponding embedding and token counts

        Notes:
        * Only embed the line when it's not found in the cache.
        """
        key = (start, end)
        if key not in self.__e_cache:
            self.__e_cache[key] = self.__encoder.encode_v2(
                reconstruct_chunk(lines[start:end]) if end -
                start > 1 else lines[start]
            )
        return self.__e_cache[key]

    def eval(self, *args) -> float:
        """
        Evaluates the current chunking based on semantic coherence and other metrics.

        Args:
            args (list[Any]): Variable length argument list. Must contain:
                - embeddings (List[List[float]]): Embeddings of all lines.
                - grouping (List[Tuple[int, int]]): The current grouping of lines.
                - capacity (int): The total number of lines.
                - verbose (bool): Whether to log metrics.

        Returns:
            score (float): A score representing the quality of the chunking. 
            Higher score indicate better chunking.
            ```python
            score = cohesion - overlapped + coverage
            ```
        """
        assert len(args) >= 4, "Expect embeddings, grouping, capacity, verbose."
        embeddings, grouping, capacity, verbose, *_ = args
        cohesion: float = 0
        for g_start, g_end in grouping:
            cohesion += self.calculate_pairwise_similarity(
                embeddings, g_start, g_end
            )
        cohesion /= len(grouping) if grouping else 1

        overlapped = ChunkerMetrics.calculate_overlapped(capacity, grouping)
        coverage = ChunkerMetrics.calculate_coverage(capacity, grouping)

        score: float = cohesion - overlapped + coverage
        if verbose:
            logger.info("metrics/cohesion: %.4f", cohesion)
            logger.info("metrics/overlapped: %.4f", overlapped)
            logger.info("metrics/coverage: %.4f", coverage)
            logger.info("metrics/score: %.4f", score)
        return score

    def split(self, long_text: str):
        """
        Splits the input `long_text` into semantically coherent chunks.
        This is the main entry point for `SemanticChunker`.

        ## High-level Steps:
        1. Run input validation.
        2. Split the text into sentences using `SentenceChunker`.
        3. For every sentence that is longer than `max_part_size`, 
            run `FixedGroupChunker` to split it into smaller parts.
        4. If the number of parts is equal to `n_chunk`, return the parts.
        5. Otherwise, run the optimization algorithm to find the best grouping of the parts.
        6. Return the reconstructed chunks based on the best grouping.

        Args:
            long_text (str): The text to be chunked. Must be a non-empty string.

        Returns:
            List[str]: A list of text chunks, each being a semantically coherent segment of the input `long_text`.

        Raises:
            TypeError: If `long_text` is not a string.
            ValueError: If `long_text` is an empty string.
            ValueError: If `long_text` is too short to be chunked.
            ValueError: If `n_chunk` is too small for the given text.
        """
        logger.warning("[BEG] split")
        logger.info("Configuration: %s", self.config)
        if not isinstance(long_text, str):
            raise TypeError(
                f"Expected 'long_text' to be str, got {type(long_text).__name__}."
            )

        # Sanitize argument `long_text`
        text = long_text.strip()
        if len(text) == 0:
            raise ValueError("Expect long_text to be non-empty string.")

        etc: int = estimate_token_count(text)
        if etc < self.config.chunk_size:
            if self.config.n_chunk == 1:
                return [text]
            raise ValueError("Text is too short to be chunked.")

        ideal_k = max(2, ceil(etc / self.config.chunk_size))
        if self.config.n_chunk < ideal_k:
            raise ValueError(
                "n_chunk is too small. We will end up with many chunks with size > chunk_size."
            )

        sentence_chunker = SentenceChunker()
        lines = sentence_chunker.split(text)
        parts: list[str] = []
        max_part_size: int = max(1, int(self.config.chunk_size * 0.5))
        for line in lines:
            _etc: int = estimate_token_count(line)
            if _etc < max_part_size:
                parts.append(line)
            else:
                # Resolve long sentence
                fgc = FixedGroupChunker(
                    FixedGroupChunkerConfig(
                        G=max(2, int(_etc // max_part_size)),
                        level="character",
                    )
                )
                new_lines = fgc.split(line)
                parts.extend(new_lines)

        n_part: int = len(parts)
        if n_part == self.config.n_chunk:
            # Assumption 1:
            #   Every part is well within the encoder's ctx_length
            # Assumption 2:
            #   Sentence chunker had already split the text into meaningful parts.
            logger.warning(
                "Optimization algorithm not ran. `n_part` is equal to `n_chunk`."
            )
            return parts

        # Transform individual parts into embedding
        logger.info("Embedding %d lines.", n_part)
        embeddings: list[list[float]] = []
        token_counts: list[int] = []
        for index in range(n_part):
            e, tc = self._encode(parts, index, index + 1)
            if e and tc is None:
                tc = 0
            embeddings.append(e)
            token_counts.append(tc)

        # Variables
        grouping = RandomInitializer(n_part, self.config.n_chunk).init()
        best_grouping = grouping
        iteration = 0
        best_score: float = 0
        logger.info("BEGIN Optimization")
        logger.warning("======= [%d] =======", iteration)
        logger.warning("Grouping: %s", grouping)
        while iteration < self.config.max_iteration:
            score = self.eval(embeddings, grouping, n_part, True)
            coverage = ChunkerMetrics.calculate_coverage(n_part, grouping)
            within_chunk_size: bool = all_within_chunk_size(
                parts, grouping, self.config.chunk_size
            )
            if (
                score > best_score and coverage >= self.config.min_coverage
                and within_chunk_size
            ):
                best_score = score
                best_grouping = grouping[:]
                logger.warning("Improved! Score: %.4f.", score)
                # logger.warning("Grouping: %s", grouping)

            iteration += 1
            logger.warning("======= [%d] =======", iteration)
            if random.random() < self.config.randomness:
                logger.warning("Randomly re-initialize the grouping.")
                grouping = RandomInitializer(n_part, self.config.n_chunk).init()
            else:
                logger.warning("Step forward the grouping.")
                grouping = self.step_forward(grouping, n_part)
            logger.info("Grouping: %s", grouping)

        # Wrap up
        score = self.eval(embeddings, grouping, n_part, True)
        coverage = ChunkerMetrics.calculate_coverage(n_part, grouping)
        if score > best_score and coverage >= self.config.min_coverage:
            best_score = score
            best_grouping = grouping[:]
            logger.warning("Improved! Score: %.4f.", score)
            # logger.warning("Grouping: %s", grouping)

        best_grouping.sort(key=lambda x: x[0])
        output: list[str] = []
        for g_start, g_end in best_grouping:
            reconstructed_chunk = reconstruct_chunk(lines[g_start:g_end])
            output.append(reconstructed_chunk)

        logger.warning("Best Grouping: %s", best_grouping)
        logger.warning("[END] find_best_grouping")
        return output


class AsyncSemanticChunker:
    """
    `AsyncSemanticChunker` is a class that implements the `AsyncSplitter` interface.
    
    Fundamentally similar with `SemanticChunker`, excepts calling any method that
    generates embeddings asynchronously.
    """
    C: int = 2

    def __init__(
        self,
        encoder: Encoder,
        config: SemanticChunkerConfig,
    ):
        self.__encoder = encoder
        self.__config = config

        # Cache Variables
        self.__pws_cache: dict[tuple[int, int], float] = {}
        self.__e_cache: dict[tuple[int, int], tuple[list[float], int]] = {}

    @property
    def config(self) -> SemanticChunkerConfig:
        return self.__config

    @property
    def encoder(self) -> Encoder:
        return self.__encoder

    @staticmethod
    def drop_duplicates(
        grouping: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        unique_set = set()
        for group in grouping:
            if group not in unique_set:
                unique_set.add(group)
        return [*unique_set]

    @staticmethod
    def calculate_number_of_combinations(N: int, C: int) -> int:
        if N == C:
            return 1

        if N < C or N < 2 * C:
            return 0

        return N - 2 * C + 1

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
        F = max(1, int(G * self.config.update_rate))
        N_COMBS = self.calculate_number_of_combinations(
            N_LINE, AsyncSemanticChunker.C
        )
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

                if right - left >= AsyncSemanticChunker.C and not found:
                    break

                loop_counter += 1

            if not found:
                new_grouping[point] = (left, right)

        return new_grouping

    @staticmethod
    def calculate_cosine_similarity(
        vec1: list[float], vec2: list[float]
    ) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1)**0.5
        norm2 = sum(b * b for b in vec2)**0.5
        similarity = dot_product / (
            norm1 * norm2
        ) if norm1 != 0 and norm2 != 0 else 0
        return similarity

    def calculate_pairwise_similarity(
        self,
        embeddings: list[list[float]],
        start: int,
        end: int,
    ) -> float:
        """
        Computes the average pairwise cosine similarity between lines within a group.

        Args:
            embeddings (List[List[float]]): Embeddings of all lines.
            start (int): Start index of the group.
            end (int): End index of the group.

        Returns:
            float: Average pairwise similarity score.

        range(start, end)
        x = sum(cosine_similarity(vec_i, vec_j)) where i != j
        y = end - start
        result = x / y
        """
        if end - start <= 1:
            return 0

        pairwise_similarities = []
        for vi in range(start, end):
            for vj in range(vi + 1, end):
                key = (vi, vj)
                if key not in self.__pws_cache:
                    self.__pws_cache[key] = self.calculate_cosine_similarity(
                        embeddings[vi], embeddings[vj]
                    )
                similarity = self.__pws_cache[key]
                # logger.info("%d vs %d => %.4f", vi, vj, similarity)
                pairwise_similarities.append(similarity)
        return (
            sum(pairwise_similarities) /
            len(pairwise_similarities) if pairwise_similarities else 0
        )

    async def _encode(self, lines: list[str], start: int,
                      end: int) -> tuple[list[float], int]:
        """Cached encoding.

        Args:
            lines (List[str]): A list of all lines.
            start (int): The start index
            end (int): The end index

        Returns:
            Tuple[List[float], int]: The corresponding embedding and token counts

        Notes:
        * Only embed the line when it's not found in the cache.
        """
        key = (start, end)
        if key not in self.__e_cache:
            self.__e_cache[key] = await self.__encoder.encode_v2_async(
                reconstruct_chunk(lines[start:end]) if end -
                start > 1 else lines[start]
            )
        return self.__e_cache[key]

    def eval(self, *args) -> float:
        """
        Evaluates the current chunking based on semantic coherence and other metrics.

        Args:
            args (list[Any]): Variable length argument list. Must contain:
                - embeddings (List[List[float]]): Embeddings of all lines.
                - grouping (List[Tuple[int, int]]): The current grouping of lines.
                - capacity (int): The total number of lines.
                - verbose (bool): Whether to log metrics.

        Returns:
            score (float): A score representing the quality of the chunking. 
            Higher score indicate better chunking.
            ```python
            score = cohesion - overlapped + coverage
            ```
        """
        assert len(args) >= 4, "Expect embeddings, grouping, capacity, verbose."
        embeddings, grouping, capacity, verbose, *_ = args
        cohesion: float = 0
        for g_start, g_end in grouping:
            cohesion += self.calculate_pairwise_similarity(
                embeddings, g_start, g_end
            )
        cohesion /= len(grouping) if grouping else 1

        overlapped = ChunkerMetrics.calculate_overlapped(capacity, grouping)
        coverage = ChunkerMetrics.calculate_coverage(capacity, grouping)

        score: float = cohesion - overlapped + coverage
        if verbose:
            logger.info("metrics/cohesion: %.4f", cohesion)
            logger.info("metrics/overlapped: %.4f", overlapped)
            logger.info("metrics/coverage: %.4f", coverage)
            logger.info("metrics/score: %.4f", score)
        return score

    async def split(self, long_text: str):
        """
        Splits the input `long_text` into semantically coherent chunks.
        This is the main entry point for `SemanticChunker`.

        ## High-level Steps:
        1. Run input validation.
        2. Split the text into sentences using `SentenceChunker`.
        3. For every sentence that is longer than `max_part_size`, 
            run `FixedGroupChunker` to split it into smaller parts.
        4. If the number of parts is equal to `n_chunk`, return the parts.
        5. Otherwise, run the optimization algorithm to find the best grouping of the parts.
        6. Return the reconstructed chunks based on the best grouping.

        Args:
            long_text (str): The text to be chunked. Must be a non-empty string.

        Returns:
            List[str]: A list of text chunks, each being a semantically coherent segment of the input `long_text`.

        Raises:
            TypeError: If `long_text` is not a string.
            ValueError: If `long_text` is an empty string.
            ValueError: If `long_text` is too short to be chunked.
            ValueError: If `n_chunk` is too small for the given text.
        """
        logger.warning("[BEG] split")
        logger.info("Configuration: %s", self.config)
        if not isinstance(long_text, str):
            raise TypeError(
                f"Expected 'long_text' to be str, got {type(long_text).__name__}."
            )

        # Sanitize argument `long_text`
        text = long_text.strip()
        if len(text) == 0:
            raise ValueError("Expect long_text to be non-empty string.")

        etc: int = estimate_token_count(text)
        if etc < self.config.chunk_size:
            if self.config.n_chunk == 1:
                return [text]
            raise ValueError("Text is too short to be chunked.")

        ideal_k = max(2, ceil(etc / self.config.chunk_size))
        if self.config.n_chunk < ideal_k:
            raise ValueError(
                "n_chunk is too small. We will end up with many chunks with size > chunk_size."
            )

        sentence_chunker = SentenceChunker()
        lines = sentence_chunker.split(text)
        parts: list[str] = []
        max_part_size: int = max(1, int(self.config.chunk_size * 0.5))
        for line in lines:
            _etc: int = estimate_token_count(line)
            if _etc < max_part_size:
                parts.append(line)
            else:
                # Resolve long sentence
                fgc = FixedGroupChunker(
                    FixedGroupChunkerConfig(
                        G=max(2, int(_etc // max_part_size)),
                        level="character",
                    )
                )
                new_lines = fgc.split(line)
                parts.extend(new_lines)

        n_part: int = len(parts)
        if n_part == self.config.n_chunk:
            # Assumption 1:
            #   Every part is well within the encoder's ctx_length
            # Assumption 2:
            #   Sentence chunker had already split the text into meaningful parts.
            logger.warning(
                "Optimization algorithm not ran. `n_part` is equal to `n_chunk`."
            )
            return parts

        # Transform individual parts into embedding
        logger.info("Embedding %d lines.", n_part)
        embeddings: list[list[float]] = []
        token_counts: list[int] = []
        for index in range(n_part):
            e, tc = await self._encode(parts, index, index + 1)
            if e and tc is None:
                tc = 0
            embeddings.append(e)
            token_counts.append(tc)

        # Variables
        grouping = RandomInitializer(n_part, self.config.n_chunk).init()
        best_grouping = grouping
        iteration = 0
        best_score: float = 0
        logger.info("BEGIN Optimization")
        logger.warning("======= [%d] =======", iteration)
        logger.warning("Grouping: %s", grouping)
        while iteration < self.config.max_iteration:
            score = self.eval(embeddings, grouping, n_part, True)
            coverage = ChunkerMetrics.calculate_coverage(n_part, grouping)
            within_chunk_size: bool = all_within_chunk_size(
                lines, grouping, self.config.chunk_size
            )
            if (
                score > best_score and coverage >= self.config.min_coverage
                and within_chunk_size
            ):
                best_score = score
                best_grouping = grouping[:]
                logger.warning("Improved! Score: %.4f.", score)
                # logger.warning("Grouping: %s", grouping)

            iteration += 1
            logger.warning("======= [%d] =======", iteration)
            if random.random() < self.config.randomness:
                logger.warning("Randomly re-initialize the grouping.")
                grouping = RandomInitializer(n_part, self.config.n_chunk).init()
            else:
                logger.warning("Step forward the grouping.")
                grouping = self.step_forward(grouping, n_part)
            logger.info("Grouping: %s", grouping)

        # Wrap up
        score = self.eval(embeddings, grouping, n_part, True)
        coverage = ChunkerMetrics.calculate_coverage(n_part, grouping)
        if score > best_score and coverage >= self.config.min_coverage:
            best_score = score
            best_grouping = grouping[:]
            logger.warning("Improved! Score: %.4f.", score)
            # logger.warning("Grouping: %s", grouping)

        best_grouping.sort(key=lambda x: x[0])
        output: list[str] = []
        for g_start, g_end in best_grouping:
            reconstructed_chunk = reconstruct_chunk(lines[g_start:g_end])
            output.append(reconstructed_chunk)

        logger.warning("Best Grouping: %s", best_grouping)
        logger.warning("[END] find_best_grouping")
        return output


# class SimulatedAnnealingSemanticChunker(SemanticChunker):
#     """
#     `SimulatedAnnealingSemanticChunker` enhances the `SemanticChunker` by integrating the Simulated Annealing
#     optimization technique to improve the quality of text chunking.

#     The simulated annealing approach allows the algorithm to escape `local optima` by probabilistically accepting
#     worse solutions based on the current temperature, thereby exploring a broader range of possible groupings
#     to achieve superior semantic coherence.

#     **Enhancements Over `SemanticChunker`:**
#     * **Simulated Annealing Parameters:**
#         * `temperature`: Controls the probability of accepting worse solutions during optimization.
#         * `cooling_rate`: Determines the rate at which the temperature decreases.
#         * `constants`: Weights for combining evaluation metrics (coverage, utilization rate, cohesion, wastage).

#     * **Enhanced Evaluation Function:**
#         * Combines multiple metrics—coverage, utilization rate, cohesion, and wastage—using customizable constants to compute the overall score.
#         * Incorporates group centroid similarity, increasing computational and financial costs.

#     **Cost Consideration:**
#     * Generating embeddings for each line and each group centroid may incur significant costs, especially when using paid encoder services or APIs.
#     * **Be mindful of your budget** when processing large texts or a high number of lines to avoid unexpected expenses.
#     * The additional computation for group centroid embeddings increases both computational and financial costs compared to the base `SemanticChunker`.

#     **Evaluation Metrics:**

#     - **Coverage:**
#         - Measures the proportion of lines included in the final grouping relative to the total number of lines.
#         - *Higher values indicate that more lines are effectively utilized in the chunks.*

#     - **Utilization Rate:**
#         - Assesses how efficiently the encoder's context length is used across all chunks.
#         - *Higher utilization rates signify better usage of the available context capacity.*

#     - **Cohesion:**
#         - Evaluates the semantic coherence within each chunk by averaging the cosine similarity between sentences and their respective group centroids.
#         - *Higher cohesion scores denote more semantically coherent groupings.*

#     - **Wastage:**
#         - Calculates the proportion of unused context capacity in the encoder across all chunks.
#         - *Lower wastage rates indicate more efficient use of the encoder's capacity.*

#     - **Overall Score:**
#         - Combines the above metrics using user-defined constants to compute a weighted sum, guiding the optimization process.
#         - *The choice of constants allows users to prioritize certain metrics over others based on specific requirements.*

#     **Parameters:**
#     - `encoder (Encoder)`: An encoder instance used to generate embeddings for text lines.
#         - **Note:** The encoder is used internally to compute embeddings, which are not exposed outside the `split` method.
#     - `config (dict)`: Configuration dictionary containing the following keys:
#         - `K (int)`: Number of groups to split the text into. Must be a positive integer.
#         - `MAX_ITERATION (int)`: Maximum number of iterations for the optimization process. Must be a positive integer.
#         - `update_rate (float, optional)`: Rate at which the grouping is updated during optimization. Must be between 0 and 1. Defaults to `0.5`.
#         - `min_coverage (float, optional)`: Minimum coverage required for a grouping to be considered valid. Must be between 0 and 1. Defaults to `0.8`.
#         - `temperature (float, optional)`: Initial temperature for the simulated annealing process. Must be between 0 and 1.0. Defaults to `1.0`.
#         - `cooling_rate (float, optional)`: Rate at which the temperature decreases during the simulated annealing process. Must be between 0 and 1.0. Defaults to `0.05`.
#         - `constants (tuple of float, optional)`: Weights for the evaluation metrics in the overall score calculation.
#             Must contain up to four float values (coverage, utilization, cohesion, wastage). Defaults to `(1.0, 1.0, 1.0, 1.0)`.

#     **Notes:**
#     * For more detailed information on the general algorithm and evaluation metrics, refer to the `SemanticChunker` documentation.
#     """

#     def __init__(
#         self,
#         encoder: Encoder,
#         config: dict,
#     ):
#         self.raise_if_invalid(config)
#         super().__init__(encoder=encoder, config=config)
#         self.__temperature: float = config.get("temperature", 1.0)
#         self.__cooling_rate: float = config.get("cooling_rate", 0.05)
#         self.__constants: tuple = config.get("constants", (1.0, 1.0, 1.0, 1.0))
#         while len(self.__constants) < 4:
#             self.__constants += (1.0, )
#         # Cache Variables
#         self.__gcs_cache: dict[tuple[int, int, int], float] = {}

#     @staticmethod
#     def raise_if_invalid(parameters: dict) -> None:
#         K: int = parameters.get("K", 10)
#         if K is not None and not isinstance(K, int):
#             raise TypeError(
#                 f"Expect K to be type 'int', got '{type(K).__name__}'."
#             )
#         if K <= 0:
#             raise ValueError(f"Expect K > 0, got {K}.")
#         MAX_ITERATION: int = parameters.get("MAX_ITERATION", 20)
#         if MAX_ITERATION is not None and not isinstance(MAX_ITERATION, int):
#             raise TypeError(
#                 f"Expect MAX_ITERATION to be type 'int', got '{type(MAX_ITERATION).__name__}'."
#             )
#         if MAX_ITERATION <= 0:
#             raise ValueError(f"Expect MAX_ITERATION > 0, got {MAX_ITERATION}.")

#         update_rate: Optional[float] = parameters.get("update_rate", None)
#         if update_rate is not None and not isinstance(update_rate, float):
#             raise TypeError(
#                 f"Expect update_rate to be type 'float', got '{type(update_rate).__name__}'."
#             )
#         if update_rate < 0 or update_rate > 1.0:
#             raise ValueError(
#                 f"Expect update_rate within the range of [0, 1.0], got '{update_rate}'."
#             )
#         temperature: Optional[float] = parameters.get("temperature", None)
#         if temperature is not None and not isinstance(temperature, float):
#             raise TypeError(
#                 f"Expect temperature to be type 'float', got '{type(temperature).__name__}'."
#             )
#         if temperature < 0 or temperature > 1.0:
#             raise ValueError(
#                 f"Expect temperature within the range of [0, 1.0], got '{temperature}'."
#             )
#         cooling_rate: float = parameters.get("cooling_rate", None)
#         if cooling_rate is not None and not isinstance(cooling_rate, float):
#             raise TypeError(
#                 f"Expect cooling_rate to be type 'float', got '{type(cooling_rate).__name__}'."
#             )
#         if cooling_rate < 0 or cooling_rate > 1.0:
#             raise ValueError(
#                 f"Expect cooling_rate within the range of [0, 1.0], got '{cooling_rate}'."
#             )
#         constants: tuple[
#             float, float, float,
#             float] = parameters.get("constants", (1.0, 1.0, 1.0, 1.0))
#         if constants is not None and not isinstance(constants, tuple):
#             raise TypeError(
#                 f"Expect constants to be type 'tuple', got '{type(constants).__name__}'."
#             )
#         if len(constants) > 4:
#             raise ValueError(
#                 f"Expect at most 4 values in constants, got {len(constants)}."
#             )
#         min_coverage: float = parameters.get("min_coverage", 0.8)
#         if min_coverage is not None and not isinstance(min_coverage, float):
#             raise TypeError(
#                 f"Expect min_coverage to be type 'float', got '{type(min_coverage).__name__}'."
#             )
#         if min_coverage <= 0 or min_coverage > 1:
#             raise ValueError(
#                 f"Expect min_coverage within the range of (0, 1.0], got '{min_coverage}'."
#             )

#     @property
#     def temperature(self) -> float:
#         return self.__temperature

#     @property
#     def cooling_rate(self) -> float:
#         return self.__cooling_rate

#     @property
#     def constants(self) -> tuple[float, float, float, float]:
#         return self.__constants

#     def cooldown(self) -> None:
#         # Need a more robust way to cool it down
#         self.__temperature -= self.__cooling_rate
#         self.__temperature = max(0, self.__temperature)

#     def optimize(self, input_list: list[tuple[int, int]],
#                  RIGHT_BOUND: int) -> list[tuple[int, int]]:
#         output_list: list[tuple[int, int]] = input_list[:]
#         k = len(input_list)
#         factor = min(1, int(k * self.update_rate))
#         for _ in range(factor):
#             point = random.randint(0, k - 1)
#             increment = random.randint(0, 1) == 0
#             reference_tuple = output_list[point]

#             if increment:
#                 left = reference_tuple[0]
#                 right = min(RIGHT_BOUND, reference_tuple[1] + 1)
#             else:
#                 left = max(0, reference_tuple[0] - 1)
#                 right = reference_tuple[1]
#             new_tuple = (left, right)
#             assert new_tuple[1] - new_tuple[0] >= 1
#             output_list[point] = new_tuple

#         # Handle duplicated combination
#         # Harder to have duplication with high capacity and low K
#         unique_list = self.drop_duplicates(output_list)
#         diff = k - len(unique_list)
#         if diff > 0:
#             for _ in range(diff):
#                 while True:
#                     # Find a random chunk within the 25 - 75 %
#                     # This might end up in a very large chunk!
#                     start = random.randint(RIGHT_BOUND // 4, RIGHT_BOUND // 2)
#                     end = random.randint(start, RIGHT_BOUND // 4 * 3)
#                     new_tuple = (start, end)
#                     if new_tuple not in unique_list:
#                         break
#                 unique_list.append(new_tuple)

#         return unique_list

#     @staticmethod
#     def calculate_cosine_similarity(
#         vec1: list[float], vec2: list[float]
#     ) -> float:
#         dot_product = sum(a * b for a, b in zip(vec1, vec2))
#         norm1 = sum(a * a for a in vec1)**0.5
#         norm2 = sum(b * b for b in vec2)**0.5
#         similarity = dot_product / (
#             norm1 * norm2
#         ) if norm1 != 0 and norm2 != 0 else 0
#         return similarity

#     def calculate_sentence_to_centroid_similarity(
#         self,
#         embeddings: list[list[float]],
#         start: int,
#         end: int,
#         group_embedding: list[float],
#     ) -> float:
#         if end - start <= 1:
#             return 0

#         pairwise_similarities = []
#         for vi in range(start, end):
#             key = (start, end, vi)
#             if key not in self.__gcs_cache:
#                 va = embeddings[vi]
#                 self.__gcs_cache[key] = self.calculate_cosine_similarity(
#                     group_embedding, va
#                 )
#             similarity = self.__gcs_cache[key]
#             pairwise_similarities.append(similarity)
#         return (
#             sum(pairwise_similarities) /
#             len(pairwise_similarities) if pairwise_similarities else 0
#         )

#     def eval(self, *args) -> float:
#         """
#         Evaluates the current grouping based on multiple metrics.

#         Args:
#             *args: Variable length argument list.
#                 Expected order: lines, tokens, embeddings, grouping, RIGHT_BOUND, ...

#         Returns:
#             float: The overall score combining coverage, utilization, cohesion, and wastage.
#         """
#         assert (
#             len(args) >= 5
#         ), "Expect lines, tokens, embeddings, grouping, RIGHT_BOUND."
#         lines, tokens, embeddings, grouping, RIGHT_BOUND, *_ = args
#         coverage = ChunkerMetrics.calculate_coverage(RIGHT_BOUND, grouping)
#         utilization = ChunkerMetrics.calculate_utilization_rate(
#             self.encoder.ctx_length, tokens, grouping
#         )
#         wastage = ChunkerMetrics.calculate_wastage_rate(
#             self.encoder.ctx_length, tokens, grouping
#         )
#         cohesion: float = 0
#         for g_start, g_end in grouping:
#             group_embedding, _ = self._encode(lines, g_start, g_end)
#             score = self.calculate_sentence_to_centroid_similarity(
#                 embeddings, g_start, g_end, group_embedding
#             )
#             cohesion += score
#         cohesion /= len(grouping) if grouping else 1
#         C1, C2, C3, C4 = self.constants

#         return coverage * C1 + utilization * C2 + cohesion * C3 - wastage * C4

#     def split(self, long_text: str):
#         """
#         Splits the input `long_text` into semantically coherent chunks.

#         Args:
#             long_text (str): The text to be chunked. Must be a non-empty string.

#         Returns:
#             List[str]: A list of text chunks, each being a semantically coherent segment of the input `long_text`.

#         Raises:
#             TypeError: If `long_text` is not a string.
#             ValueError: If `long_text` is an empty string.
#         """
#         logger.info("Chunker: SimulatedAnnealingSemanticChunker")
#         logger.info("CONFIG: %s", self.config)
#         logger.info(
#             "Encoder: %s, Context length: %d, Dimension: %d",
#             self.encoder.model_name,
#             self.encoder.ctx_length,
#             self.encoder.dimension,
#         )
#         if not isinstance(long_text, str):
#             raise TypeError(
#                 f"Expected 'long_text' to be str, got {type(long_text).__name__}."
#             )
#         # Sanitize argument `long_text`
#         text = long_text.strip()
#         if len(text) == 0:
#             raise ValueError("Expect long_text to be non-empty string.")

#         sentence_chunker = SentenceChunker({})
#         lines = sentence_chunker.split(text)
#         TOTAL_CAPACITY = len(lines)

#         K: int = self.config.get("K", 0)
#         if K == 0:
#             raise ValueError("Missing Argument: K")

#         if len(lines) < K:
#             return lines

#         # Transform individual parts into embedding
#         logger.info("Embedding %d lines.", TOTAL_CAPACITY)
#         embeddings: list[list[float]] = []
#         token_counts: list[int] = []
#         for index in range(TOTAL_CAPACITY):
#             e, tc = self._encode(lines, index, index + 1)
#             if e and tc is None:
#                 tc = 0
#             embeddings.append(e)
#             token_counts.append(tc)
#         # Separators are not included, therefore, this is only a close estimation.
#         total_tokens = sum(token_counts)
#         ideal_k = total_tokens // self.encoder.ctx_length
#         if K < ideal_k:
#             logger.warning(
#                 msg=
#                 f"{K} < {ideal_k}. Chunk longer than the encoder's ctx_length will be truncated."
#             )
#         if K == 1:
#             return [long_text]

#         MAX_ITERATION: int = self.config.get("MAX_ITERATION", 20)
#         # Initialization
#         logger.info("Initializing...")
#         initializer = RandomInitializer(TOTAL_CAPACITY, K)
#         grouping = initializer.init()
#         # [(i_start, i_end), (i+1_start, i+1_end), ..., (k-1_start, k-1_end), (k_start, k_end)]
#         best_group = grouping
#         iteration = 0
#         best_score: float = 0
#         MIN_COVERAGE: float = self.config.get("min_coverage", 0.8)
#         logger.info("BEGIN Optimization")
#         while iteration < MAX_ITERATION:
#             score: float = self.eval(
#                 lines, token_counts, embeddings, grouping, TOTAL_CAPACITY
#             )
#             coverage = ChunkerMetrics.calculate_coverage(
#                 TOTAL_CAPACITY, grouping
#             )
#             if score > best_score and coverage >= MIN_COVERAGE:
#                 logger.info(
#                     "[%d] Update best score to %.4ff, improved = %.4f\nGrouping: %s",
#                     iteration,
#                     score,
#                     score - best_score,
#                     grouping,
#                 )
#                 logger.info("Grouping: %s", grouping)
#                 best_score = score
#                 # Update best group
#                 best_group = grouping[:]
#             # Decide whether to revert
#             if best_score != score and random.uniform(0, 1) > self.temperature:
#                 grouping = best_group[:]
#             grouping = self.optimize(grouping, TOTAL_CAPACITY)
#             self.cooldown()
#             iteration += 1
#         logger.info("END Optimization")
#         logger.info("Best Score: %.4f", best_score)
#         logger.info(
#             "Coverage: %.4f",
#             ChunkerMetrics.calculate_coverage(TOTAL_CAPACITY, grouping),
#         )
#         # Bundle `lines` into `K` groups according to the discovered `best_group`
#         doc_list = []
#         best_group.sort(key=lambda g: g[0], reverse=False)
#         for g_start, g_end in best_group:
#             reconstructed_chunk = self.reconstruct_chunk(lines[g_start:g_end])
#             doc_list.append(reconstructed_chunk)
#         return doc_list
