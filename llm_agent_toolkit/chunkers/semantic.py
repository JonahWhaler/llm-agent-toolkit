import re
import random
import logging

from .._chunkers import Chunker, ChunkerMetrics, RandomInitializer
from .._encoder import Encoder

logger = logging.getLogger(__name__)


class SemanticChunker(Chunker):
    def __init__(
        self,
        encoder: Encoder,
        config: dict,
    ):
        self.raise_if_invalid(config)
        super().__init__(config)
        self.__encoder = encoder
        self.__update_rate: float = config.get("update_rate", 0.5)

        # Cache Variables
        self.__pws_cache: dict[tuple[int, int], float] = {}
        self.__e_cache: dict[tuple[int, int], tuple[list[float], int]] = {}

    @property
    def encoder(self) -> Encoder:
        return self.__encoder

    @property
    def update_rate(self) -> float:
        return self.__update_rate

    @staticmethod
    def raise_if_invalid(parameters: dict) -> None:
        K: int = parameters.get("K", 10)
        if K is not None and not isinstance(K, int):
            raise TypeError(f"Expect K to be type 'int', got '{type(K).__name__}'.")
        if K <= 0:
            raise ValueError(f"Expect K > 0, got {K}.")
        MAX_ITERATION: int = parameters.get("MAX_ITERATION", 20)
        if MAX_ITERATION is not None and not isinstance(MAX_ITERATION, int):
            raise TypeError(
                f"Expect MAX_ITERATION to be type 'int', got '{type(MAX_ITERATION).__name__}'."
            )
        if MAX_ITERATION <= 0:
            raise ValueError(f"Expect MAX_ITERATION > 0, got {MAX_ITERATION}.")
        update_rate: float = parameters.get("update_rate", None)
        if update_rate is not None and not isinstance(update_rate, float):
            raise TypeError(
                f"Expect update_rate to be type 'float', got '{type(update_rate).__name__}'."
            )
        if 0 > update_rate > 1.0:
            raise ValueError(
                f"Expect update_rate within the range of [0, 1.0], got '{update_rate}'."
            )
        min_coverage: float = parameters.get("min_coverage", 0.8)
        if min_coverage is not None and not isinstance(min_coverage, float):
            raise TypeError(
                f"Expect min_coverage to be type 'float', got '{type(min_coverage).__name__}'."
            )
        if 0 >= min_coverage > 1:
            raise ValueError(
                f"Expect min_coverage within the range of (0, 1.0], got '{min_coverage}'."
            )

    @staticmethod
    def drop_duplicates(grouping: list[tuple[int, int]]) -> list[tuple[int, int]]:
        unique_set = set()
        for group in grouping:
            if group not in unique_set:
                unique_set.add(group)
        return [*unique_set]

    def optimize(
        self, input_list: list[tuple[int, int]], RIGHT_BOUND: int
    ) -> list[tuple[int, int]]:
        output_list: list[tuple[int, int]] = input_list[:]
        k = len(input_list)
        factor = min(1, int(k * self.update_rate))
        for _ in range(factor):
            point = random.randint(0, k - 1)
            increment = random.randint(0, 1) == 0
            reference_tuple = output_list[point]

            if increment:
                left = reference_tuple[0]
                right = min(RIGHT_BOUND, reference_tuple[1] + 1)
            else:
                left = max(0, reference_tuple[0] - 1)
                right = reference_tuple[1]
            new_tuple = (left, right)
            assert new_tuple[1] - new_tuple[0] >= 1
            output_list[point] = new_tuple

        # Handle duplicated combination
        # Harder to have duplication with high capacity and low K
        unique_list = self.drop_duplicates(output_list)
        diff = k - len(unique_list)
        if diff > 0:
            for _ in range(diff):
                while True:
                    # Find a random chunk within the 25 - 75 %
                    # This might end up in a very large chunk!
                    start = random.randint(RIGHT_BOUND // 4, RIGHT_BOUND // 2)
                    end = random.randint(start, RIGHT_BOUND // 4 * 3)
                    new_tuple = (start, end)
                    if new_tuple not in unique_list:
                        break
                unique_list.append(new_tuple)

        return unique_list

    @staticmethod
    def calculate_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        similarity = dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0
        return similarity

    def calculate_pairwise_similarity(
        self,
        embeddings: list[list[float]],
        start: int,
        end: int,
    ) -> float:
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
                # logger.info("%d vs %d => %f", vi, vj, similarity)
                pairwise_similarities.append(similarity)
        return (
            sum(pairwise_similarities) / len(pairwise_similarities)
            if pairwise_similarities
            else 0
        )

    def _encode(
        self, lines: list[str], start: int, end: int
    ) -> tuple[list[float], int]:
        key = (start, end)
        if key not in self.__e_cache:
            self.__e_cache[key] = self.__encoder.encode_v2(
                self.reconstruct_chunk(lines[start:end])
                if end - start > 1
                else lines[start]
            )
        return self.__e_cache[key]

    def eval(self, *args) -> float:
        embeddings, grouping, *_ = args
        cohesion: float = 0
        for g_start, g_end in grouping:
            cohesion += self.calculate_pairwise_similarity(embeddings, g_start, g_end)
        cohesion /= len(grouping)

        return cohesion

    def split(self, long_text: str):
        if not isinstance(long_text, str):
            raise TypeError(
                f"Expected 'long_text' to be str, got {type(long_text).__name__}."
            )
        # Sanitize argument `long_text`
        text = long_text.replace("\n\n", "\n").strip("\n ")  # Remove excessive newlines
        text = text.replace("\n", "\n")  # Convert viewable newline to readable newline
        if len(text) == 0:
            raise ValueError("Expect long_text to be non-empty string.")
        # Split long text into multiple parts. ("Hello! How are you?") => ["Hello", "!", "How are you", "?"]
        lines = re.split(r"([.?!\n\t])\s*", text)
        lines = list(filter(lambda line: line, lines))  # Remove invalid lines
        TOTAL_CAPACITY = len(lines)
        # Transform individual parts into embedding
        embeddings: list[list[float]] = []
        token_counts: list[int] = []
        for index in range(TOTAL_CAPACITY):
            e, tc = self._encode(lines, index, index + 1)
            if e and tc is None:
                tc = 0
            embeddings.append(e)
            token_counts.append(tc)
        # Separators are not included, therefore, this is only a close estimation.
        total_tokens = sum(token_counts)
        ideal_k = total_tokens // self.__encoder.ctx_length
        K: int = self.config.get("K", ideal_k)
        if K < ideal_k:
            logger.warning(
                msg=f"{K} < {ideal_k}. Chunk longer than the encoder's ctx_length will be truncated."
            )
        MAX_ITERATION: int = self.config.get("MAX_ITERATION", 20)
        # Initialization
        initializer = RandomInitializer(TOTAL_CAPACITY, K)
        grouping = initializer.init()
        # [(i_start, i_end), (i+1_start, i+1_end), ..., (k-1_start, k-1_end), (k_start, k_end)]
        best_group = grouping
        iteration = 0
        best_score: float = -100.0
        MIN_COVERAGE: float = self.config.get("min_coverage", 0.9)
        while iteration < MAX_ITERATION:
            # logger.info("Iteration [%d]/[%d]", iteration, MAX_ITERATION)
            score: float = self.eval(embeddings, grouping)
            if (
                score > best_score
                and ChunkerMetrics.calculate_coverage(TOTAL_CAPACITY, grouping)
                > MIN_COVERAGE
            ):
                logger.info("[%d] Update %f to %f", iteration, best_score, score)
                best_score = score
                # Update best group
                best_group = grouping[:]
            # Decide whether to revert
            if best_score != score:
                grouping = best_group[:]
            grouping = self.optimize(grouping, TOTAL_CAPACITY)
            iteration += 1
        print("Best Score: %f", best_score)
        print(
            "Coverage: %f", ChunkerMetrics.calculate_coverage(TOTAL_CAPACITY, grouping)
        )
        # Bundle `lines` into `K` groups according to the discovered `best_group`
        doc_list = []
        for g_start, g_end in best_group:
            reconstructed_chunk = self.reconstruct_chunk(lines[g_start:g_end])
            doc_list.append(reconstructed_chunk)
        return doc_list


class SimulatedAnnealingSemanticChunker(SemanticChunker):
    def __init__(
        self,
        encoder: Encoder,
        config: dict,
    ):
        self.raise_if_invalid(config)
        super().__init__(encoder=encoder, config=config)
        self.__temperature: float = config.get("temperature", 1.0)
        self.__cooling_rate: float = config.get("cooling_rate", 0.05)
        self.__constants: tuple = config.get("constants", (1.0, 1.0, 1.0, 1.0))
        while len(self.__constants) < 4:
            self.__constants += (1.0,)
        # Cache Variables
        self.__gcs_cache: dict[tuple[int, int, int], float] = {}

    @staticmethod
    def raise_if_invalid(parameters: dict) -> None:
        K: int = parameters.get("K", 10)
        if K is not None and not isinstance(K, int):
            raise TypeError(f"Expect K to be type 'int', got '{type(K).__name__}'.")
        if K <= 0:
            raise ValueError(f"Expect K > 0, got {K}.")
        MAX_ITERATION: int = parameters.get("MAX_ITERATION", 20)
        if MAX_ITERATION is not None and not isinstance(MAX_ITERATION, int):
            raise TypeError(
                f"Expect MAX_ITERATION to be type 'int', got '{type(MAX_ITERATION).__name__}'."
            )
        if MAX_ITERATION <= 0:
            raise ValueError(f"Expect MAX_ITERATION > 0, got {MAX_ITERATION}.")

        update_rate: float = parameters.get("update_rate", None)
        if update_rate is not None and not isinstance(update_rate, float):
            raise TypeError(
                f"Expect update_rate to be type 'float', got '{type(update_rate).__name__}'."
            )
        if 0 > update_rate > 1.0:
            raise ValueError(
                f"Expect update_rate within the range of [0, 1.0], got '{update_rate}'."
            )
        temperature: float = parameters.get("temperature", None)
        if temperature is not None and not isinstance(temperature, float):
            raise TypeError(
                f"Expect temperature to be type 'float', got '{type(temperature).__name__}'."
            )
        if 0 > temperature > 1.0:
            raise ValueError(
                f"Expect temperature within the range of [0, 1.0], got '{temperature}'."
            )
        cooling_rate: float = parameters.get("cooling_rate", None)
        if cooling_rate is not None and not isinstance(cooling_rate, float):
            raise TypeError(
                f"Expect cooling_rate to be type 'float', got '{type(cooling_rate).__name__}'."
            )
        if 0 > cooling_rate > 1.0:
            raise ValueError(
                f"Expect cooling_rate within the range of [0, 1.0], got '{cooling_rate}'."
            )
        constants: tuple[float, float, float, float] = parameters.get("constants", None)
        if constants is not None and not isinstance(constants, tuple):
            raise TypeError(
                f"Expect constants to be type 'tuple', got '{type(constants).__name__}'."
            )
        if len(constants) > 4:
            raise ValueError(
                f"Expect at most 4 values in constants, got {len(constants)}."
            )
        min_coverage: float = parameters.get("min_coverage", 0.8)
        if min_coverage is not None and not isinstance(min_coverage, float):
            raise TypeError(
                f"Expect min_coverage to be type 'float', got '{type(min_coverage).__name__}'."
            )
        if 0 >= min_coverage > 1:
            raise ValueError(
                f"Expect min_coverage within the range of (0, 1.0], got '{min_coverage}'."
            )

    @property
    def temperature(self) -> float:
        return self.__temperature

    @property
    def cooling_rate(self) -> float:
        return self.__cooling_rate

    @property
    def constants(self) -> tuple[float, float, float, float]:
        return self.__constants

    def cooldown(self) -> None:
        # Need a more robust way to cool it down
        self.__temperature -= self.__cooling_rate
        self.__temperature = max(0, self.__temperature)

    def optimize(
        self, input_list: list[tuple[int, int]], RIGHT_BOUND: int
    ) -> list[tuple[int, int]]:
        output_list: list[tuple[int, int]] = input_list[:]
        k = len(input_list)
        factor = min(1, int(k * self.update_rate))
        for _ in range(factor):
            point = random.randint(0, k - 1)
            increment = random.randint(0, 1) == 0
            reference_tuple = output_list[point]

            if increment:
                left = reference_tuple[0]
                right = min(RIGHT_BOUND, reference_tuple[1] + 1)
            else:
                left = max(0, reference_tuple[0] - 1)
                right = reference_tuple[1]
            new_tuple = (left, right)
            assert new_tuple[1] - new_tuple[0] >= 1
            output_list[point] = new_tuple

        # Handle duplicated combination
        # Harder to have duplication with high capacity and low K
        unique_list = self.drop_duplicates(output_list)
        diff = k - len(unique_list)
        if diff > 0:
            for _ in range(diff):
                while True:
                    # Find a random chunk within the 25 - 75 %
                    # This might end up in a very large chunk!
                    start = random.randint(RIGHT_BOUND // 4, RIGHT_BOUND // 2)
                    end = random.randint(start, RIGHT_BOUND // 4 * 3)
                    new_tuple = (start, end)
                    if new_tuple not in unique_list:
                        break
                unique_list.append(new_tuple)

        return unique_list

    @staticmethod
    def calculate_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        similarity = dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0
        return similarity

    def calculate_sentence_to_centroid_similarity(
        self,
        embeddings: list[list[float]],
        start: int,
        end: int,
        group_embedding: list[float],
    ) -> float:
        if end - start <= 1:
            return 0

        pairwise_similarities = []
        for vi in range(start, end):
            key = (start, end, vi)
            if key not in self.__gcs_cache:
                va = embeddings[vi]
                self.__gcs_cache[key] = self.calculate_cosine_similarity(
                    group_embedding, va
                )
            similarity = self.__gcs_cache[key]
            pairwise_similarities.append(similarity)
        return (
            sum(pairwise_similarities) / len(pairwise_similarities)
            if pairwise_similarities
            else 0
        )

    def eval(self, *args) -> float:
        lines, tokens, embeddings, grouping, RIGHT_BOUND, *_ = args
        coverage = ChunkerMetrics.calculate_coverage(RIGHT_BOUND, grouping)
        utilization = ChunkerMetrics.calculate_utilization_rate(
            self.__encoder.ctx_length, tokens, grouping
        )
        wastage = ChunkerMetrics.calculate_wastage_rate(
            self.__encoder.ctx_length, tokens, grouping
        )
        cohesion: float = 0
        for g_start, g_end in grouping:
            group_embedding, _ = self._encode(lines, g_start, g_end)
            score = self.calculate_sentence_to_centroid_similarity(
                embeddings, g_start, g_end, group_embedding
            )
            cohesion += score
        cohesion /= len(grouping)
        C1, C2, C3, C4 = self.constants

        return coverage * C1 + utilization * C2 + cohesion * C3 - wastage * C4

    def split(self, long_text: str):
        if not isinstance(long_text, str):
            raise TypeError(
                f"Expected 'long_text' to be str, got {type(long_text).__name__}."
            )
        # Sanitize argument `long_text`
        text = long_text.replace("\n\n", "\n").strip("\n ")  # Remove excessive newlines
        text = text.replace("\n", "\n")  # Convert viewable newline to readable newline
        if len(text) == 0:
            raise ValueError("Expect long_text to be non-empty string.")
        # Split long text into multiple parts. ("Hello! How are you?") => ["Hello", "!", "How are you", "?"]
        lines = re.split(r"([.?!\n\t])\s*", text)
        lines = list(filter(lambda line: line, lines))  # Remove invalid lines
        TOTAL_CAPACITY = len(lines)
        # Transform individual parts into embedding
        embeddings: list[list[float]] = []
        token_counts: list[int] = []
        for index in range(TOTAL_CAPACITY):
            e, tc = self._encode(lines, index, index + 1)
            if e and tc is None:
                tc = 0
            embeddings.append(e)
            token_counts.append(tc)
        # Separators are not included, therefore, this is only a close estimation.
        total_tokens = sum(token_counts)
        ideal_k = total_tokens // self.__encoder.ctx_length
        K: int = self.config.get("K", ideal_k)
        if K < ideal_k:
            logger.warning(
                msg=f"{K} < {ideal_k}. Chunk longer than the encoder's ctx_length will be truncated."
            )
        MAX_ITERATION: int = self.config.get("MAX_ITERATION", 20)
        # Initialization
        initializer = RandomInitializer(TOTAL_CAPACITY, K)
        grouping = initializer.init()
        # [(i_start, i_end), (i+1_start, i+1_end), ..., (k-1_start, k-1_end), (k_start, k_end)]
        best_group = grouping
        iteration = 0
        best_score: float = -100.0
        MIN_COVERAGE: float = self.config.get("min_coverage", 0.9)
        while iteration < MAX_ITERATION:
            logger.info("Iteration [%d]/[%d]", iteration, MAX_ITERATION)
            score: float = self.eval(
                lines, token_counts, embeddings, grouping, TOTAL_CAPACITY
            )
            if (
                score > best_score
                and ChunkerMetrics.calculate_coverage(TOTAL_CAPACITY, grouping)
                > MIN_COVERAGE
            ):
                logger.info("[%d] Update %f to %f", iteration, best_score, score)
                best_score = score
                # Update best group
                best_group = grouping[:]
            # Decide whether to revert
            if best_score != score and random.uniform(0, 1) > self.temperature:
                grouping = best_group[:]
            grouping = self.optimize(grouping, TOTAL_CAPACITY)
            self.cooldown()
            iteration += 1
        # Bundle `lines` into `K` groups according to the discovered `best_group`
        doc_list = []
        for g_start, g_end in best_group:
            reconstructed_chunk = self.reconstruct_chunk(lines[g_start:g_end])
            doc_list.append(reconstructed_chunk)
        return doc_list
