import re
import random
import logging

from .._chunkers import Chunker, ChunkerMetrics, RandomInitializer
from .._encoder import Encoder

logger = logging.getLogger(__name__)


class SemanticChunker(Chunker):
    """`SemanticChunker` is a concrete implementation of the `Chunker` abstract base class.

    It splits long text into semantically coherent chunks using a stochastic approach.
    The algorithm initializes with a random grouping of lines and iteratively modifies this grouping.
    If a modification improves the coherence score while maintaining minimum coverage, the change is accepted;
    otherwise, the grouping reverts to the best known state.

    **Key Features:**
    * **Stochastic Approach:**
        * Utilizes randomness in initial grouping and optimization steps to explore diverse groupings.
    * **Caching Mechanism:**
        * Caches pairwise similarities and embeddings to optimize performance by avoiding redundant computations.
    * **Efficient Embedding Management:**
        * Embeddings are computed internally for each line, which can be computationally expensive.
        * Embeddings are **not exposed** outside the `split` method and are solely used for the purpose of chunking the input text.

    **Algorithm Overview:**

    1. **Split Text into Lines:**
        - Divide the input `long_text` into N lines based on punctuation and whitespace.
          - Example: `"Hello! How are you?"` → `["Hello", "!", "How are you", "?"]`

    2. **Initial Grouping:**
        - Group the N lines into K initial groups.
          - Example: `[["Hello", "!"], ["How are you", "?"]]`

    3. **Iterative Optimization (Up to M Iterations):**
        - **3.1 Compute Score:**
            - Calculate the coherence score of the current grouping using the evaluation function.
        - **3.2 Update Best Grouping:**
            - If the current score is better than the best score and the grouping meets the minimum coverage requirement, update the best score and best grouping.
        - **3.3 Revert if Necessary:**
            - If the score does not improve, revert to the best known grouping.
        - **3.4 Optimize Grouping:**
            - Modify the current grouping randomly to explore new groupings.

    4. **Construct Final Chunks:**
        - Based on the best grouping found, reconstruct K coherent text chunks.

    **Evaluation Metrics:**

    - **Pairwise Similarity:**
        - Computes the cosine similarity between every pair of distinct lines within a group.
        - The average of these similarities indicates the semantic coherence of the group.
        - Higher values signify greater coherence.

    - **Average Pairwise Similarity:**
        - Averages the `Pairwise Similarity` scores across all groups.
        - Higher average values indicate that the overall grouping is more semantically coherent.

    **Parameters:**
    - `encoder (Encoder)`: An encoder instance used to generate embeddings for text lines.
    - `config (dict)`: Configuration dictionary containing the following keys:
        - `K (int)`: Number of groups to split the text into. Must be a positive integer.
        - `MAX_ITERATION (int)`: Maximum number of iterations for the optimization process. Must be a positive integer.
        - `update_rate (float, optional)`: Rate at which the grouping is updated during optimization. Must be between 0 and 1. Defaults to `0.5`.
        - `min_coverage (float, optional)`: Minimum coverage required for a grouping to be considered valid. Must be between 0 and 1. Defaults to `0.8`.

    **Cost Consideration:**
        * Generating embeddings for each line may incur significant costs, especially when using paid encoder services or APIs.
        * **Be mindful of your budget** when processing large texts or a high number of lines to avoid unexpected expenses.

    **Embedding Usage:**
        * Embeddings are **strictly used internally** within the `split` method for chunking purposes.
        * Embeddings are **not exposed** or intended for external use outside of the `SemanticChunker`'s internal processes.

    **Countering the Effect of Stochastic Nature:**
    * `min_coverage` ensures that a sufficient number of lines are included in the final grouping under this stochastic algorithm.
    * The `drop_duplicates` method is designed to remove duplicated groups, enhancing the diversity of groupings.

    **Notes:**
    * This implementation does not use statistical threshold to determine when to `break` apart sentences.
    * Does not have early termination mechanism, rely solely on `MAX_ITERATION`.
    """

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
        if update_rate < 0 or update_rate > 1.0:
            raise ValueError(
                f"Expect update_rate within the range of [0, 1.0], got '{update_rate}'."
            )
        min_coverage: float = parameters.get("min_coverage", 0.8)
        if min_coverage is not None and not isinstance(min_coverage, float):
            raise TypeError(
                f"Expect min_coverage to be type 'float', got '{type(min_coverage).__name__}'."
            )
        if min_coverage <= 0 or min_coverage > 1:
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
            sum(pairwise_similarities) / len(pairwise_similarities)
            if pairwise_similarities
            else 0
        )

    def _encode(
        self, lines: list[str], start: int, end: int
    ) -> tuple[list[float], int]:
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
                self.reconstruct_chunk(lines[start:end])
                if end - start > 1
                else lines[start]
            )
        return self.__e_cache[key]

    def eval(self, *args) -> float:
        """
        Evaluates the current grouping based on pairwise similarity.

        Args:
            *args: Variable length argument list. Expected to include embeddings and grouping.

        Returns:
            float: Cohesion score.
        """
        assert len(args) >= 2, "Expect embeddings, grouping."
        embeddings, grouping, *_ = args
        cohesion: float = 0
        for g_start, g_end in grouping:
            cohesion += self.calculate_pairwise_similarity(embeddings, g_start, g_end)
        cohesion /= len(grouping) if grouping else 1

        return cohesion

    def split(self, long_text: str):
        """
        Splits the input `long_text` into semantically coherent chunks.

        Args:
            long_text (str): The text to be chunked. Must be a non-empty string.

        Returns:
            List[str]: A list of text chunks, each being a semantically coherent segment of the input `long_text`.

        Raises:
            TypeError: If `long_text` is not a string.
            ValueError: If `long_text` is an empty string.
        """
        logger.info("Chunker: SemanticChunker")
        logger.info("CONFIG: %s", self.config)
        logger.info(
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
        # text = long_text.replace("\n\n", "\n").strip("\n ")  # Remove excessive newlines
        # text = text.replace("\n", "\n")  # Convert viewable newline to readable newline
        if len(text) == 0:
            raise ValueError("Expect long_text to be non-empty string.")
        # Split long text into multiple parts. ("Hello! How are you?") => ["Hello", "!", "How are you", "?"]
        lines = re.split(r"([.?!\n\t])\s*", text)
        lines = list(filter(lambda line: line, lines))  # Remove invalid lines
        TOTAL_CAPACITY = len(lines)
        # Transform individual parts into embedding
        logger.info("Embedding %d lines.", TOTAL_CAPACITY)
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
        ideal_k = total_tokens // self.encoder.ctx_length
        K: int = self.config.get("K", ideal_k)
        if K < ideal_k:
            logger.warning(
                msg=f"{K} < {ideal_k}. Chunk longer than the encoder's ctx_length will be truncated."
            )
        if K == 1:
            return [long_text]

        MAX_ITERATION: int = self.config.get("MAX_ITERATION", 20)
        # Initialization
        logger.info("Initializing...")
        initializer = RandomInitializer(TOTAL_CAPACITY, K)
        grouping = initializer.init()
        # [(i_start, i_end), (i+1_start, i+1_end), ..., (k-1_start, k-1_end), (k_start, k_end)]
        best_group = grouping
        iteration = 0
        best_score: float = -100.0
        MIN_COVERAGE: float = self.config.get("min_coverage", 0.9)
        logger.info("BEGIN Optimization")
        while iteration < MAX_ITERATION:
            # logger.info("Iteration [%d]/[%d]", iteration, MAX_ITERATION)
            score: float = self.eval(embeddings, grouping)
            if (
                score > best_score
                and ChunkerMetrics.calculate_coverage(TOTAL_CAPACITY, grouping)
                > MIN_COVERAGE
            ):
                logger.info(
                    "[%d] Update %.4ff to %.4ff, improved = %.4f",
                    iteration,
                    best_score,
                    score,
                    score - best_score,
                )
                logger.info("Grouping: %s", grouping)
                best_score = score
                # Update best group
                best_group = grouping[:]
            # Decide whether to revert
            if best_score != score:
                grouping = best_group[:]
            grouping = self.optimize(grouping, TOTAL_CAPACITY)
            iteration += 1
        logger.info("END Optimization")
        logger.info("Best Score: %.4f", best_score)
        logger.info(
            "Coverage: %.4f",
            ChunkerMetrics.calculate_coverage(TOTAL_CAPACITY, grouping),
        )
        # Bundle `lines` into `K` groups according to the discovered `best_group`
        doc_list = []
        best_group.sort(key=lambda g: g[0], reverse=False)
        for g_start, g_end in best_group:
            reconstructed_chunk = self.reconstruct_chunk(lines[g_start:g_end])
            doc_list.append(reconstructed_chunk)
        return doc_list


class SimulatedAnnealingSemanticChunker(SemanticChunker):
    """
    `SimulatedAnnealingSemanticChunker` enhances the `SemanticChunker` by integrating the Simulated Annealing
    optimization technique to improve the quality of text chunking.

    The simulated annealing approach allows the algorithm to escape `local optima` by probabilistically accepting
    worse solutions based on the current temperature, thereby exploring a broader range of possible groupings
    to achieve superior semantic coherence.

    **Enhancements Over `SemanticChunker`:**
    * **Simulated Annealing Parameters:**
        * `temperature`: Controls the probability of accepting worse solutions during optimization.
        * `cooling_rate`: Determines the rate at which the temperature decreases.
        * `constants`: Weights for combining evaluation metrics (coverage, utilization rate, cohesion, wastage).

    * **Enhanced Evaluation Function:**
        * Combines multiple metrics—coverage, utilization rate, cohesion, and wastage—using customizable constants to compute the overall score.
        * Incorporates group centroid similarity, increasing computational and financial costs.

    **Cost Consideration:**
    * Generating embeddings for each line and each group centroid may incur significant costs, especially when using paid encoder services or APIs.
    * **Be mindful of your budget** when processing large texts or a high number of lines to avoid unexpected expenses.
    * The additional computation for group centroid embeddings increases both computational and financial costs compared to the base `SemanticChunker`.

    **Evaluation Metrics:**

    - **Coverage:**
        - Measures the proportion of lines included in the final grouping relative to the total number of lines.
        - *Higher values indicate that more lines are effectively utilized in the chunks.*

    - **Utilization Rate:**
        - Assesses how efficiently the encoder's context length is used across all chunks.
        - *Higher utilization rates signify better usage of the available context capacity.*

    - **Cohesion:**
        - Evaluates the semantic coherence within each chunk by averaging the cosine similarity between sentences and their respective group centroids.
        - *Higher cohesion scores denote more semantically coherent groupings.*

    - **Wastage:**
        - Calculates the proportion of unused context capacity in the encoder across all chunks.
        - *Lower wastage rates indicate more efficient use of the encoder's capacity.*

    - **Overall Score:**
        - Combines the above metrics using user-defined constants to compute a weighted sum, guiding the optimization process.
        - *The choice of constants allows users to prioritize certain metrics over others based on specific requirements.*

    **Parameters:**
    - `encoder (Encoder)`: An encoder instance used to generate embeddings for text lines.
        - **Note:** The encoder is used internally to compute embeddings, which are not exposed outside the `split` method.
    - `config (dict)`: Configuration dictionary containing the following keys:
        - `K (int)`: Number of groups to split the text into. Must be a positive integer.
        - `MAX_ITERATION (int)`: Maximum number of iterations for the optimization process. Must be a positive integer.
        - `update_rate (float, optional)`: Rate at which the grouping is updated during optimization. Must be between 0 and 1. Defaults to `0.5`.
        - `min_coverage (float, optional)`: Minimum coverage required for a grouping to be considered valid. Must be between 0 and 1. Defaults to `0.8`.
        - `temperature (float, optional)`: Initial temperature for the simulated annealing process. Must be between 0 and 1.0. Defaults to `1.0`.
        - `cooling_rate (float, optional)`: Rate at which the temperature decreases during the simulated annealing process. Must be between 0 and 1.0. Defaults to `0.05`.
        - `constants (tuple of float, optional)`: Weights for the evaluation metrics in the overall score calculation.
            Must contain up to four float values (coverage, utilization, cohesion, wastage). Defaults to `(1.0, 1.0, 1.0, 1.0)`.

    **Notes:**
    * For more detailed information on the general algorithm and evaluation metrics, refer to the `SemanticChunker` documentation.
    """

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
        if update_rate < 0 or update_rate > 1.0:
            raise ValueError(
                f"Expect update_rate within the range of [0, 1.0], got '{update_rate}'."
            )
        temperature: float = parameters.get("temperature", None)
        if temperature is not None and not isinstance(temperature, float):
            raise TypeError(
                f"Expect temperature to be type 'float', got '{type(temperature).__name__}'."
            )
        if temperature < 0 or temperature > 1.0:
            raise ValueError(
                f"Expect temperature within the range of [0, 1.0], got '{temperature}'."
            )
        cooling_rate: float = parameters.get("cooling_rate", None)
        if cooling_rate is not None and not isinstance(cooling_rate, float):
            raise TypeError(
                f"Expect cooling_rate to be type 'float', got '{type(cooling_rate).__name__}'."
            )
        if cooling_rate < 0 or cooling_rate > 1.0:
            raise ValueError(
                f"Expect cooling_rate within the range of [0, 1.0], got '{cooling_rate}'."
            )
        constants: tuple[float, float, float, float] = parameters.get(
            "constants", (1.0, 1.0, 1.0, 1.0)
        )
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
        if min_coverage <= 0 or min_coverage > 1:
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
        """
        Evaluates the current grouping based on multiple metrics.

        Args:
            *args: Variable length argument list.
                Expected order: lines, tokens, embeddings, grouping, RIGHT_BOUND, ...

        Returns:
            float: The overall score combining coverage, utilization, cohesion, and wastage.
        """
        assert (
            len(args) >= 5
        ), "Expect lines, tokens, embeddings, grouping, RIGHT_BOUND."
        lines, tokens, embeddings, grouping, RIGHT_BOUND, *_ = args
        coverage = ChunkerMetrics.calculate_coverage(RIGHT_BOUND, grouping)
        utilization = ChunkerMetrics.calculate_utilization_rate(
            self.encoder.ctx_length, tokens, grouping
        )
        wastage = ChunkerMetrics.calculate_wastage_rate(
            self.encoder.ctx_length, tokens, grouping
        )
        cohesion: float = 0
        for g_start, g_end in grouping:
            group_embedding, _ = self._encode(lines, g_start, g_end)
            score = self.calculate_sentence_to_centroid_similarity(
                embeddings, g_start, g_end, group_embedding
            )
            cohesion += score
        cohesion /= len(grouping) if grouping else 1
        C1, C2, C3, C4 = self.constants

        return coverage * C1 + utilization * C2 + cohesion * C3 - wastage * C4

    def split(self, long_text: str):
        """
        Splits the input `long_text` into semantically coherent chunks.

        Args:
            long_text (str): The text to be chunked. Must be a non-empty string.

        Returns:
            List[str]: A list of text chunks, each being a semantically coherent segment of the input `long_text`.

        Raises:
            TypeError: If `long_text` is not a string.
            ValueError: If `long_text` is an empty string.
        """
        logger.info("Chunker: SimulatedAnnealingSemanticChunker")
        logger.info("CONFIG: %s", self.config)
        logger.info(
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
        # text = long_text.replace("\n\n", "\n").strip("\n ")  # Remove excessive newlines
        # text = text.replace("\n", "\n")  # Convert viewable newline to readable newline
        if len(text) == 0:
            raise ValueError("Expect long_text to be non-empty string.")
        # Split long text into multiple parts. ("Hello! How are you?") => ["Hello", "!", "How are you", "?"]
        lines = re.split(r"([.?!\n\t])\s*", text)
        lines = list(filter(lambda line: line, lines))  # Remove invalid lines
        TOTAL_CAPACITY = len(lines)
        # Transform individual parts into embedding
        logger.info("Embedding %d lines.", TOTAL_CAPACITY)
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
        ideal_k = total_tokens // self.encoder.ctx_length
        K: int = self.config.get("K", ideal_k)
        if K < ideal_k:
            logger.warning(
                msg=f"{K} < {ideal_k}. Chunk longer than the encoder's ctx_length will be truncated."
            )
        if K == 1:
            return [long_text]

        MAX_ITERATION: int = self.config.get("MAX_ITERATION", 20)
        # Initialization
        logger.info("Initializing...")
        initializer = RandomInitializer(TOTAL_CAPACITY, K)
        grouping = initializer.init()
        # [(i_start, i_end), (i+1_start, i+1_end), ..., (k-1_start, k-1_end), (k_start, k_end)]
        best_group = grouping
        iteration = 0
        best_score: float = -100.0
        MIN_COVERAGE: float = self.config.get("min_coverage", 0.8)
        logger.info("BEGIN Optimization")
        while iteration < MAX_ITERATION:
            # logger.info("Iteration [%d]/[%d]", iteration, MAX_ITERATION)
            score: float = self.eval(
                lines, token_counts, embeddings, grouping, TOTAL_CAPACITY
            )
            if (
                score > best_score
                and ChunkerMetrics.calculate_coverage(TOTAL_CAPACITY, grouping)
                > MIN_COVERAGE
            ):
                logger.info(
                    "[%d] Update %.4ff to %.4ff, improved = %.4f",
                    iteration,
                    best_score,
                    score,
                    score - best_score,
                )
                logger.info("Grouping: %s", grouping)
                best_score = score
                # Update best group
                best_group = grouping[:]
            # Decide whether to revert
            if best_score != score and random.uniform(0, 1) > self.temperature:
                grouping = best_group[:]
            grouping = self.optimize(grouping, TOTAL_CAPACITY)
            self.cooldown()
            iteration += 1
        logger.info("END Optimization")
        logger.info("Best Score: %.4f", best_score)
        logger.info(
            "Coverage: %.4f",
            ChunkerMetrics.calculate_coverage(TOTAL_CAPACITY, grouping),
        )
        # Bundle `lines` into `K` groups according to the discovered `best_group`
        doc_list = []
        best_group.sort(key=lambda g: g[0], reverse=False)
        for g_start, g_end in best_group:
            reconstructed_chunk = self.reconstruct_chunk(lines[g_start:g_end])
            doc_list.append(reconstructed_chunk)
        return doc_list
