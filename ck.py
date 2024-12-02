import re
from abc import abstractmethod, ABC
import random
from llm_agent_toolkit.encoder import OllamaEncoder
from llm_agent_toolkit._encoder import Encoder


class Chunker(ABC):
    def __init__(self, config: dict):
        self.__config = config

    @property
    def config(self) -> dict:
        return self.__config

    @abstractmethod
    def split(self, long_text: str) -> list[str]:
        raise NotImplementedError


class FixedSizeChunker(Chunker):
    def __init__(self, config: dict):
        super().__init__(config)
        print("Tada!!!")

    def split(self, long_text: str) -> list[str]:
        chunk_size: int = self.config.get("chunk_size", 1024)
        stride_rate: float = self.config.get("stride_rate", 1)
        stride: int = int(chunk_size * stride_rate)
        output_list = []
        for offset in range(0, len(long_text), stride):
            chunk = long_text[offset : offset + chunk_size]
            output_list.append(chunk)
        return output_list


class FixedGroupChunker(Chunker):
    def __init__(self, config: dict):
        super().__init__(config)
        print("Tada!!!")

    def split(self, long_text: str) -> list[str]:
        k: int = self.config.get("k", len(long_text) // 1024)
        text_list = long_text.split(". ")
        text_list = list(map(lambda x: x.strip("\n "), text_list))
        grouping = self.even_split(text_list, k)
        output_list = []
        offset = 0
        for g in grouping:
            chunk = ""
            for pointer in range(g):
                chunk += f"{text_list[offset + pointer]}. "
            output_list.append(chunk)
        return output_list

    @staticmethod
    def even_split(input_list: list, k: int) -> list:
        total_len = len(input_list)
        even_size: int = total_len // k
        remainer = total_len - (k * even_size)
        output_list = [even_size for _ in range(k)]
        output_list[-1] += remainer
        return output_list


class OptimizeStrategy(ABC):
    @abstractmethod
    def optimize(self, input_list: list[int]) -> list[int]:
        raise NotImplementedError


class OptimizeStrategyV2(ABC):
    @abstractmethod
    def optimize(
        self, input_list: list[tuple[int, int]], RIGHT_BOUND: int
    ) -> list[tuple[int, int]]:
        raise NotImplementedError


class SimulatedAnnealingStrategy(OptimizeStrategy):
    def __init__(self, initial_temp: float, cooling_rate: float):
        self.__initial_temp = initial_temp  # [0, 1]
        self.__colling_rate = cooling_rate  # [0, 1]

    @property
    def temperature(self) -> float:
        return self.__initial_temp

    @property
    def cooling_rate(self) -> float:
        return self.__colling_rate

    def cooldown(self) -> None:
        self.__initial_temp -= self.__colling_rate

    def optimize(self, input_list: list[int]) -> list[int]:
        output_list = input_list[:]
        k = len(input_list)
        factor = int(k * self.temperature)
        self.cooldown()
        for _ in range(factor):
            point = random.randint(0, k - 1)
            if point == (k - 1) and output_list[point] > 1:
                output_list[point] += 1
                output_list[0] -= 1
            elif output_list[point] > 1:
                output_list[point] += 1
                output_list[point + 1] -= 1
        assert sum(output_list) == sum(input_list)
        return output_list


class SimulatedAnnealingStrategyV2(OptimizeStrategyV2):
    def __init__(self, update_rate: float, initial_temp: float, cooling_rate: float):
        self.__update_rate = update_rate  # [0, 1]
        self.__temperature = initial_temp  # [0, 1]
        self.__cooling_rate = cooling_rate  # [0, 1]

    @property
    def update_rate(self) -> float:
        return self.__update_rate

    @property
    def temperature(self) -> float:
        return self.__temperature

    @property
    def cooling_rate(self) -> float:
        return self.__cooling_rate

    def cooldown(self) -> None:
        # Need a more robust way to cool it down
        self.__temperature -= self.__cooling_rate
        self.__temperature = max(0, self.__temperature)

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
        factor = int(k * self.update_rate)
        for _ in range(factor):
            point = random.randint(0, k - 1)
            increment = random.randint(0, 1) == 0
            reference_tuple = output_list[point]

            if increment:
                new_tuple = (
                    reference_tuple[0],
                    min(RIGHT_BOUND, reference_tuple[1] + 1),
                )
            else:
                new_tuple = (
                    max(0, reference_tuple[0] - 1),
                    reference_tuple[1],
                )
            assert new_tuple[1] - new_tuple[0] >= 1
            output_list[point] = new_tuple
        # Handle duplicated combination
        # Harder to have duplication with high capacity and low K
        unique_list = self.drop_duplicates(output_list)
        diff = k - len(unique_list)
        if diff > 0:
            for _ in range(diff):
                while True:
                    # This might end up in a very large chunk!
                    start = random.randint(RIGHT_BOUND // 4, RIGHT_BOUND // 2)
                    end = random.randint(start, RIGHT_BOUND // 4 * 3)
                    new_tuple = (start, end)  # Find a random chunk within the 25 - 75 %
                    if new_tuple not in unique_list:
                        break
                unique_list.append(new_tuple)
        return unique_list


class RandomStrategy(OptimizeStrategy):
    def optimize(self, input_list: list[int]) -> list[int]:
        remainer = sum(input_list)
        k = len(input_list)
        # Random Exploration, encourage even distribution
        # Does not support overlapping
        output_list = [0] * k
        while remainer > 0:
            even_size = remainer // k
            if even_size < 1:
                for _ in range(remainer):
                    index = random.randint(0, k - 1)
                    output_list[index] += 1
                break
            new_growth = [random.randint(1, even_size) for _ in range(k)]
            for index in range(k):
                output_list[index] += new_growth[index]
            remainer -= sum(new_growth)

        return output_list


class SemanticChunker(Chunker):
    def __init__(self, encoder: Encoder, strategy: OptimizeStrategy, config: dict):
        super().__init__(config=config)
        self.encoder = encoder
        self.strategy = strategy
        self.similarity_cache: dict = {}  # Cache for cosine similarity

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        similarity = dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0
        return similarity

    def group_coherance(
        self, embeddings: list[list[float]], start: int, end: int
    ) -> float:
        if end - start <= 1:
            return 0

        pairwise_similarities = []
        for vi in range(start, end):
            va = embeddings[vi]
            for vj in range(vi + 1, end):
                vb = embeddings[vj]
                key = (vi, vj)
                if key not in self.similarity_cache:
                    self.similarity_cache[key] = self.cosine_similarity(va, vb)
                similarity = self.similarity_cache[key]
                pairwise_similarities.append(similarity)
        return (
            sum(pairwise_similarities) / len(pairwise_similarities)
            if pairwise_similarities
            else 0
        )

    def split(self, long_text: str) -> list[str]:
        k: int = self.config.get("k", 10)
        lines = re.split(r"(?<=[.?!])\s+", long_text.strip())
        embeddings = [self.encoder.encode(line) for line in lines]
        full_len = len(lines)
        grouping = self.init(full_len, k)  # Initialize a random arrangement
        best_group = grouping
        max_iteration = 100
        iteration = 0
        best_score = -100.0
        while iteration < max_iteration:
            print(f"Iteration [{iteration}]/[{max_iteration}]")
            scores = []
            offset = 0
            for g in grouping:
                coherence_score = self.group_coherance(embeddings, offset, offset + g)
                scores.append(coherence_score)
                offset += g
            avg_score = sum(scores) / len(scores)
            if avg_score > best_score:
                print(f"Improved from {best_score} => {avg_score}")
                best_score = avg_score
                best_group = grouping[:]

            iteration += 1
            grouping = self.strategy.optimize(grouping)

        doc_list = []
        offset = 0
        for g in best_group:
            partial_chunks = lines[offset : offset + g]
            doc_list.append(" ".join(partial_chunks))  # Join back like this is BAD!
            offset += g
        return doc_list

    @staticmethod
    def init(total_capacity: int, k: int) -> list[int]:
        remainer = total_capacity
        # Random Exploration, encourage even distribution
        # Does not support overlapping
        output_list = [0] * k
        while remainer > 0:
            even_size = remainer // k
            if even_size < 1:
                for _ in range(remainer):
                    index = random.randint(0, k - 1)
                    output_list[index] += 1
                break
            new_growth = [random.randint(1, even_size) for _ in range(k)]
            for index in range(k):
                output_list[index] += new_growth[index]
            remainer -= sum(new_growth)

        return output_list


class SemanticChunkerV2(Chunker):
    """
    SemanticChunkerV2 is a text chunking utility that segments a long text into
    semantically coherent chunks. It uses simulated annealing to optimize the chunk boundaries
    and supports overlapping groups for improved context preservation.

    This class is particularly useful for preprocessing large texts in natural language
    processing tasks, such as summarization, topic modeling, or text classification.

    Attributes:
        encoder (Encoder): An encoder instance to generate embeddings for text segments.
        strategy (SimulatedAnnealingStrategyV2): An optimization strategy using simulated
            annealing to refine chunk boundaries.
        config (dict): A configuration dictionary containing hyperparameters such as the
            number of chunks (`k`), maximum iterations (`max_iteration`), and overlap size (`overlap`).

    Methods:
        init(total_capacity: int, k: int, overlap: int = 2) -> list[tuple[int, int]]:
            Initializes random group boundaries with optional overlapping.

        cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
            Computes the cosine similarity between two vectors.

        group_coherance(
            embeddings: list[list[float]], start: int, end: int, group_embedding: list[float]
        ) -> float:
            Computes the average semantic coherence of a group based on cosine similarity
            between the group's embedding and individual sentence embeddings.

        encode_with_cache(sentences: list[str], start: int, end: int) -> list[float]:
            Encodes a group of sentences into an embedding, using caching to avoid redundant computations.

        split(long_text: str) -> list[str]:
            Splits the input text into semantically coherent chunks based on the
            optimization strategy and returns a list of chunks.

    # TODO: Detect duplicated groups

    Usage:
        >>> from some_encoder_module import Encoder
        >>> from some_strategy_module import SimulatedAnnealingStrategyV2
        >>> config = {"k": 10, "max_iteration": 100, "overlap": 2}
        >>> encoder = Encoder()
        >>> strategy = SimulatedAnnealingStrategyV2(update_rate=0.5, initial_temp=1.0, cooling_rate=0.95)
        >>> chunker = SemanticChunkerV2(encoder, strategy, config)
        >>> long_text = "Hello! How are you? I'm fine. Thank you."
        >>> chunks = chunker.split(long_text)
        >>> print(chunks)
        ['Hello!', "How are you?", "I'm fine.", 'Thank you.']
    """

    def __init__(
        self, encoder: Encoder, strategy: SimulatedAnnealingStrategyV2, config: dict
    ):
        super().__init__(config=config)
        self.encoder = encoder
        self.strategy = strategy
        self.similarity_cache: dict = {}  # Cache for cosine similarity
        self.embedding_cache: dict = {}  # Cache for embedding

    @staticmethod
    def init(total_capacity: int, k: int) -> list[tuple[int, int]]:
        remainer = total_capacity
        # Random Exploration, encourage even distribution
        # Does not support overlapping
        init_list = [0] * k
        while remainer > 0:
            even_size = remainer // k
            if even_size < 1:
                for _ in range(remainer):
                    index = random.randint(0, k - 1)
                    init_list[index] += 1
                break
            new_growth = [random.randint(1, even_size) for _ in range(k)]
            for index in range(k):
                init_list[index] += new_growth[index]
            remainer -= sum(new_growth)
        offset = 0
        output_list: list[tuple[int, int]] = []
        for size in init_list:
            output_list.append((offset, offset + size))
            offset += size
        return output_list

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        similarity = dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0
        return similarity

    def group_coherance(
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
            if key not in self.similarity_cache:
                va = embeddings[vi]
                self.similarity_cache[key] = self.cosine_similarity(group_embedding, va)
            similarity = self.similarity_cache[key]
            pairwise_similarities.append(similarity)
        return (
            sum(pairwise_similarities) / len(pairwise_similarities)
            if pairwise_similarities
            else 0
        )

    def encode_with_cache(self, lines: list[str], start: int, end: int) -> list[float]:
        key = (start, end)
        if key not in self.embedding_cache:
            self.embedding_cache[key] = self.encoder.encode(
                self.reconstruct_chunk(lines[start:end])
            )
        return self.embedding_cache[key]

    def split(self, long_text: str) -> list[str]:
        K: int = self.config.get("k", 10)
        MAX_ITERATION: int = self.config.get("max_iteration", 20)
        text = long_text.replace("\n\n", "\n")  # Remove continuous newline
        # Split long text into multiple parts. ("Hello! How are you?") => ["Hello", "!", "How are you", "?"]
        lines = re.split(r"([.?!])\s*", text.strip())
        full_len = len(lines)
        # Transform individual parts into embedding
        embeddings: list[list[float]] = [
            self.encode_with_cache(lines, index, index) for index in range(full_len)
        ]
        # Initialization
        grouping = self.init(
            full_len, K
        )  # [(i_start, i_end), (i+1_start, i+1_end), ..., (k-1_start, k-1_end), (k_start, k_end)]
        best_group = grouping
        iteration = 0
        best_score = -100.0

        while iteration < MAX_ITERATION:
            print(f"Iteration [{iteration}]/[{MAX_ITERATION}]")
            score: float = self.evaluate(lines, embeddings, grouping)
            if score > best_score:
                best_score = score
                # Update best group
                best_group = grouping[:]
            # Decide whether to revert
            if best_score != score and random.uniform(0, 1) > self.strategy.temperature:
                grouping = best_group[:]
            grouping = self.strategy.optimize(grouping, full_len - 1)
            self.strategy.cooldown()
            iteration += 1
        # Bundle `lines` into `K` groups according to the discovered `best_group`
        doc_list = []
        for g_start, g_end in best_group:
            reconstructed_chunk = self.reconstruct_chunk(lines[g_start:g_end])
            doc_list.append(reconstructed_chunk)
        return doc_list

    def evaluate(
        self,
        lines: list[str],
        embeddings: list[list[float]],
        grouping: list[tuple[int, int]],
    ) -> float:
        scores = []
        for g in grouping:
            g_start, g_end = g
            group_embedding = self.encode_with_cache(lines, g_start, g_end)
            coherence_score = self.group_coherance(
                embeddings, g_start, g_end, group_embedding
            )
            scores.append(coherence_score)
        avg_score = sum(scores) / len(scores)
        return avg_score

    @staticmethod
    def reconstruct_chunk(partial_chunk: list[str]) -> str:
        reconstructed = ""

        for chunk in partial_chunk:
            if reconstructed and chunk not in {".", "?", "!"}:
                reconstructed += " "
            reconstructed += chunk
        return reconstructed


if __name__ == "__main__":
    # filepath = "./license.txt"
    # filepath = "./story.txt"
    filepath = "./research.md"

    with open(filepath, "r", encoding="utf-8") as freader:
        long_line = freader.read()
    long_line = long_line.replace("\n\n", "\n")
    enc = OllamaEncoder(
        connection_string="http://localhost:11434",
        model_name=OllamaEncoder.SUPPORTED_MODELS[0]["name"],
    )
    optimizer = SimulatedAnnealingStrategy(initial_temp=0.9, cooling_rate=0.05)
    chunker = SemanticChunker(encoder=enc, strategy=optimizer, config={"k": 20})

    results = chunker.split(long_line)

    with open("./research-groups.md", "w", encoding="utf-8") as writer:
        for gi, result in enumerate(results, start=1):
            writer.write(f"===== BEGIN G[{gi}] =====\n")
            writer.write(f"LEN = {len(result)}\n")
            writer.write(f"{result}\n")
            writer.write(f"===== END G[{gi}] =====\n")
