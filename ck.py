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
