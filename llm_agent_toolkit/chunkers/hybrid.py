import logging
import random
import charade
from functools import lru_cache

# Custom packages
from .._encoder import Encoder
from .._chunkers import Chunker, ChunkerMetrics, RandomInitializer
from .basic import SectionChunker, SentenceChunker

logger = logging.getLogger(__name__)


class HybridChunker(Chunker):
    """
    A “hybrid” semantic chunker that combines hierarchical splitting with
    a randomized grouping optimizer to generate text chunks fitting within
    an embedding model’s context window.

    Works in three phases:
      1. Section‑level split: use SectionChunker to break on high‑level
         delimiters (e.g. paragraphs).
      2. Sentence‑level fallback: if a section exceeds `chunk_size`, split
         into sentences and invoke `find_best_grouping()` to optimize K
         contiguous sentence spans for maximal cohesion and minimal overlap.
      3. Buffer‑accumulation: small sections are concatenated in a `temp`
         buffer to avoid tiny fragments, flush when adding them would
         exceed `chunk_size`.




    """

    DEFAULT_UPDATE_RATE: float = 0.5
    DEFAULT_CHUNK_SIZE: int = 512
    DEFAULT_MAX_ITERATION: int = 20
    DEFAULT_MIN_COVERAGE: float = 0.9

    def __init__(
        self,
        encoder: Encoder,
        config: dict,
    ):
        super().__init__(config)
        self.__encoder = encoder
        self.unpack_parameters(config)
        self.validate_parameters()

    @property
    def encoder(self) -> Encoder:
        return self.__encoder

    @property
    def update_rate(self) -> float:
        """The update rate of the chunker.

        Constraints:
        - Must be within (0.0, 1.0]

        ```
        # G (int): Number of groups
        # F (int): Numbers of groups to be updated

        F: int = max(1, int(G * update_rate))
        ```
        """
        return self.__update_rate

    @property
    def chunk_size(self) -> int:
        """The maximum length of each chunk.

        Counted in tokens.
        """
        return self.__chunk_size

    @property
    def max_iteration(self) -> int:
        """The maximum number of optimization steps.

        Constraints:
        - Must be > 0
        - Must be > `self.patient`
        """
        return self.__max_iteration

    @property
    def min_coverage(self) -> float:
        """The min coverage threshold.

        Constraints:
        - Must be within (0.0, 1.0]
        """
        return self.__min_coverage

    @property
    def temperature(self) -> float:
        """The threshold for triggering mutation.
        Higher temperature means less likely to mutate.

        Constraints:
        - Must be within [0.0, 1.0]
        """
        return self.__temperature

    @property
    def delta(self) -> float:
        """The minimum improvement threshold.

        Constraints:
        - Must be >= 0.0
        """
        return self.__delta

    @property
    def patient(self) -> int:
        """The number of consecutive non-improving iterations.

        Constraints:
        - Must be within [0, max_iteration]
        """
        return self.__patient

    def unpack_parameters(self, config: dict) -> None:
        self.__update_rate: float = config.get("update_rate", 0.5)
        self.__chunk_size: int = config.get("chunk_size", 512)
        self.__max_iteration: int = config.get("max_iteration", 20)
        self.__min_coverage: float = config.get("min_coverage", 0.8)
        self.__temperature: float = config.get("temperature", 1.0)
        self.__delta: float = config.get("delta", 0.00001)
        self.__patient: int = config.get("patient", 5)

    def validate_parameters(self) -> None:
        """Validate parameters.

        Raises:
        - ValueError: When parameters are in invalid range.
        """
        if self.update_rate <= 0.0 or self.update_rate > 1.0:
            raise ValueError(
                f"Expect update_rate within the range of (0.0, 1.0], got {self.update_rate}."
            )

        if self.chunk_size <= 0 or self.chunk_size > self.encoder.ctx_length:
            raise ValueError(
                f"Expect chunk_size within the range of (0, encoder.ctx_length], got {self.chunk_size}."
            )

        if self.min_coverage <= 0.0 or self.min_coverage > 1.0:
            raise ValueError(
                f"Expect min_coverage within the range of (0.0, 1.0], got {self.min_coverage}."
            )

        if self.temperature < 0.0 or self.temperature > 1.0:
            raise ValueError(
                f"Expect temperature within the range of [0.0, 1.0], got {self.temperature}."
            )

        if self.temperature < 0.0 or self.temperature > 1.0:
            raise ValueError(
                f"Expect temperature within the range of [0.0, 1.0], got {self.temperature}."
            )

        if self.delta < 0.0:
            raise ValueError(f"Expect delta >= 0.0, got {self.delta}.")

        if self.patient < 0 or self.patient > self.max_iteration:
            raise ValueError(
                f"Expect patient within the range of [0, max_iteration], got {self.patient}."
            )

    @staticmethod
    def drop_duplicates(grouping: list[tuple[int, int]]) -> list[tuple[int, int]]:
        unique_set = set()
        for group in grouping:
            if group not in unique_set:
                unique_set.add(group)
        return [*unique_set]

    def step_forward(
        self, input_list: list[tuple[int, int]], RIGHT_BOUND: int
    ) -> list[tuple[int, int]]:
        output_list: list[tuple[int, int]] = input_list[:]
        k = len(input_list)
        F = max(1, int(k * self.update_rate))
        # Update a random chunk `F` times
        for _ in range(F):
            # Make sure the chunk is not too small!
            left, right = 0, 0
            point = -1
            while right - left < 2:
                # Randomly select a chunk
                point = random.randint(0, k - 1)
                reference_tuple = output_list[point]
                # 0: decrement, 1: increment
                increment = random.randint(0, 1) == 0

                if increment:
                    left = reference_tuple[0]
                    right = min(RIGHT_BOUND, reference_tuple[1] + 1)
                else:
                    left = max(0, reference_tuple[0] - 1)
                    right = reference_tuple[1]

            assert point != -1
            output_list[point] = (left, right)

        # Handle duplicated combination
        # Harder to have duplication with high capacity and low K
        unique_list = self.drop_duplicates(output_list)
        diff = k - len(unique_list)
        if diff > 0:
            for _ in range(diff):
                while True:
                    # Make sure the chunk is not too small!
                    start, end = 0, 0
                    new_tuple = (0, 2)  # Assume RIGHT_BOUND >= 2
                    while end - start < 2 and new_tuple in unique_list:
                        # Find a random chunk within the 25 - 75 %
                        # This might end up in a very large chunk!
                        start = random.randint(RIGHT_BOUND // 4, RIGHT_BOUND // 2)
                        end = random.randint(start, RIGHT_BOUND // 4 * 3)
                        new_tuple = (start, end)
                    unique_list.append(new_tuple)

        return unique_list

    @staticmethod
    def calculate_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        similarity = dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0
        return similarity

    @lru_cache(maxsize=512)
    def str_to_embedding(self, text: str) -> list[float]:
        return self.__encoder.encode(text)

    @staticmethod
    def isascii(text: str) -> bool:
        byte_sentence = text.encode("utf-8")
        result = charade.detect(byte_sentence)
        return result["encoding"] == "ascii"

    def eval(self, *args) -> float:
        """
        Evaluates the current grouping based on pairwise similarity.

        Args:
            *args: Variable length argument list.

        Returns:
            float: Score.

        ### Calculations
        ```
        score: float = sum(positive_metrics) - sum(negative_metrics) * 0.25
        ```

        ### Positive Metrics:
          - Intra‑group cohesion (average pairwise similarity within each chunk)
          - Coverage (fraction of lines covered)

        ### Negative Metrics:
          - Inter‑group cohesion (similarity between different chunks)
          - Overlap (penalized if chunks share lines)
          - Overflow penalty (proportional to how much a chunk exceeds context length)
        """
        assert len(args) >= 3, "Expect lines, grouping, capacity."
        lines, grouping, capacity, *_ = args
        # Intra-group cohesion
        intra_group_cohesion: float = 0
        for g_start, g_end in grouping:
            if g_end - g_start < 2:
                # Only one line
                continue

            group_cohesion: float = 0.0
            for vi in range(g_start, g_end - 1):
                a_text = self.reconstruct_custom_chunk(
                    lines[g_start : vi + 1], "sentence"
                )
                b_text = self.reconstruct_custom_chunk(
                    lines[vi + 1 : g_end], "sentence"
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

        # Inter-group cohesion
        inter_group_cohesion: float = 0
        n_groups = len(grouping)
        n_compare = n_groups * (n_groups - 1) / 2
        for i in range(n_groups - 1):
            i_start, i_end = grouping[i]
            i_line = self.reconstruct_custom_chunk(lines[i_start:i_end], "sentence")
            i_embedding = self.str_to_embedding(i_line)
            for j in range(i + 1, n_groups):
                j_start, j_end = grouping[j]
                j_line = self.reconstruct_custom_chunk(lines[j_start:j_end], "sentence")
                j_embedding = self.str_to_embedding(j_line)
                cosine_similarity = self.calculate_cosine_similarity(
                    i_embedding, j_embedding
                )
                inter_group_cohesion += cosine_similarity
        inter_group_cohesion /= n_compare

        # Overflow
        overflow: float = 0.0
        for g_start, g_end in grouping:
            g_line = self.reconstruct_custom_chunk(lines[g_start:g_end], "sentence")
            if self.isascii(g_line):
                coeficient = 0.5
            else:
                coeficient = 0.75
            est_token_count = len(g_line) * coeficient
            g_overflow = max(0, (est_token_count * -self.chunk_size) / self.chunk_size)
            overflow += g_overflow
        overflow /= len(grouping)

        overlapped = ChunkerMetrics.calculate_overlapped(capacity, grouping)
        coverage = ChunkerMetrics.calculate_coverage(capacity, grouping)

        logger.warning("==== Metrics ====")
        logger.warning("POSITIVE")
        logger.warning("Intra-group Cohesion: %.4f", intra_group_cohesion)
        logger.warning("Coverage: %.4f", coverage)
        logger.warning("NEGATIVE")
        logger.warning("Inter-group Cohesion: %.4f", inter_group_cohesion)
        logger.warning("Overlapped: %.4f", overlapped)
        logger.warning("Overflow: %.4f", overflow)

        positive_metrics = [intra_group_cohesion, coverage]
        negative_metrics = [inter_group_cohesion, overlapped, overflow]
        return sum(positive_metrics) - sum(negative_metrics) * 0.25

    def split(self, long_text: str):
        logger.warning("[BEG] split")
        logger.warning("CONFIG: %s", self.config)
        logger.warning(
            "Encoder: %s, Context length: %d, Dimension: %d",
            self.encoder.model_name,
            self.encoder.ctx_length,
            self.encoder.dimension,
        )
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
                # 20 sentences per group
                initializer = RandomInitializer(
                    len(sentences),
                    max(
                        2,
                        int(est_tc // self.chunk_size),
                        len(sentences) // 20,
                    ),
                )
                grouping = initializer.init()
                grouping = self.find_best_grouping(sentences, grouping)
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
                    temp = self.reconstruct_custom_chunk([temp, section], "section")
                else:
                    temp = section
        if temp:
            output.append(temp)

        logger.warning("[END] split")
        return output

    def find_best_grouping(
        self, lines: list[str], grouping: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Explore the search space and find the best grouping.

        Args:
            lines (list[str]): A list of lines to be grouped.
            grouping (list[tuple[int, int]]): The initial grouping.

        Returns:
            list[tuple[int, int]]: The best grouping found.

        ## Optimization
        * **Algorithm**:
            - Hill‑climb
            - Random‑restart search: Aim to escape local optima
        * **Evaluation**:
            - Balanced sum of positive and negative metrics
        * **Termination**:
            - Early stopping if no minimum improvement after a certain number of iterations

        Notes:
        - min_coverage is a hard constraint
        """
        logger.warning("[BEG] find_best_grouping")

        # Variables
        iteration: int = 0
        best_score: float = 0.0
        best_grouping: list[tuple[int, int]] = grouping[:]
        non_improving_counter: int = 0
        L = len(lines)
        G = len(grouping)
        logger.warning("Lines: %d, Groups: %d", L, G)
        initializer = RandomInitializer(L, G)
        for line in lines:
            logger.warning(">>>> %d | %s", len(line), line)
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

            if random.random() >= self.temperature:
                grouping = initializer.init()
            else:
                grouping = self.step_forward(grouping, L)

        score: float = self.eval(lines, grouping, L)
        coverage: float = ChunkerMetrics.calculate_coverage(L, grouping)
        if score > best_score and coverage >= self.min_coverage:
            best_score = score
            best_grouping = grouping[:]
            logger.warning("Improved! Score: %.4f.", score)
            logger.warning("Grouping: %s", grouping)

        logger.warning("[END] find_best_grouping")
        # Sort the best grouping
        best_grouping.sort(key=lambda x: x[0])
        logger.warning("Best Grouping: %s", best_grouping)
        return best_grouping

    @staticmethod
    def reconstruct_custom_chunk(partial_chunk: list[str], level: str) -> str:
        if level == "section":
            return "\n\n".join([chunk.strip() for chunk in partial_chunk])
        return " ".join([chunk.strip() for chunk in partial_chunk])
