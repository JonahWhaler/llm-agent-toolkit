import re
import random
import logging
from abc import abstractmethod, ABC

logger = logging.getLogger(__name__)


class ChunkerMetrics:
    @classmethod
    def calculate_utilization_rate(
        cls, CTX_LENGTH: int, token_counts: list[int], grouping: list[tuple[int, int]]
    ) -> float:
        utilization_scores = []
        for g_start, g_end in grouping:
            g_token_counts = sum(token_counts[g_start:g_end])
            if g_token_counts > CTX_LENGTH:
                # overflow
                utilization_score = 1.0
            else:
                utilization_score = g_token_counts / CTX_LENGTH

            utilization_scores.append(utilization_score)
        return sum(utilization_scores) / len(utilization_scores)

    @classmethod
    def calculate_wastage_rate(
        cls, CTX_LENGTH: int, token_counts: list[int], grouping: list[tuple[int, int]]
    ) -> float:
        wastage_scores = []
        for g_start, g_end in grouping:
            g_token_counts = sum(token_counts[g_start:g_end])
            wastage = g_token_counts - CTX_LENGTH
            if wastage > 0:
                wastage_rate = wastage / g_token_counts
            else:
                wastage_rate = 0
            wastage_scores.append(wastage_rate)
        return sum(wastage_scores) / len(wastage_scores)

    @classmethod
    def calculate_coverage(
        cls, capacity: int, grouping: list[tuple[int, int]]
    ) -> float:
        """Calculate the coverage.

        Returns:
            float: [0, 1.0]
        """
        # Initialize states
        rooms = [0 for _ in range(capacity)]
        for g_start, g_end in grouping:
            for i in range(g_start, g_end):
                rooms[i] += 1
        occupied = list(filter(lambda q: q != 0, rooms))
        coverage = len(occupied) / len(rooms)
        return coverage


class ChunkerInitializer:
    @classmethod
    def uniform(
        cls, total_capacity: int, k: int, resolution: str = "skip"
    ) -> list[tuple[int, int]]:
        """Initialize chunk groupings uniformly.
        Resolve with `resolution` when `total-capacity` cannot be evenly distributed into `k` groups.

        Args:
            - total_capacity (int): The total size of be divided into chunks.
            - k (int): The number of chunks to create.
            - resolution (str): Default = "skip", options = ["front", "back", "skip"]
        """
        chunk_size = total_capacity // k
        remainer = total_capacity - chunk_size * k
        output_list = []
        offset = 0
        for ki in range(k):
            right = offset + chunk_size
            if ki == 0 and resolution == "front":
                right += remainer
            elif ki == k - 1 and resolution == "back":
                right = total_capacity
            output_list.append((offset, min(right, total_capacity)))
            offset = right
        assert ChunkerMetrics.calculate_coverage(total_capacity, output_list) == 1.0
        return output_list

    @classmethod
    def overlap(cls, total_capacity: int, k: int) -> list[tuple[int, int]]:
        """Initialize chunk groupings with overlapping regions.

        Results are random and non-overlapping.

        Args:
            - total_capacity (int): The total size of be divided into chunks.
            - k (int): The number of chunks to create.
        """
        remainer = total_capacity
        # Initialize chunk sizes to zero
        init_list = [0] * k
        while remainer > 0:
            # Determine the maximum even size that can be allocated to each chunk
            even_size = remainer // k
            if even_size < 1:
                # If remaining capacity is less than the number of chunks,
                # distribute the remaining one by one randomly to chunks
                for _ in range(remainer):
                    index = random.randint(0, k - 1)
                    init_list[index] += 1
                break  # All remaining capacity has been distributed
            # Randomly decide how much to add to each chunk in this iteration
            new_growth = [random.randint(1, even_size) for _ in range(k)]
            # Add the new growth to each chunk's size
            for index in range(k):
                init_list[index] += new_growth[index]
            # Decrease the remaining capacity by the total allocated in this iteration
            remainer -= sum(new_growth)

        offset = 0
        output_list: list[tuple[int, int]] = []
        for size in init_list:
            output_list.append((offset, offset + size))
            offset += size

        assert ChunkerMetrics.calculate_coverage(total_capacity, output_list) == 1.0
        return output_list


class Chunker(ABC):
    def __init__(self, config: dict):
        self.__config = config

    @property
    def config(self) -> dict:
        return self.__config

    @abstractmethod
    def split(self, long_text: str) -> list[str]:
        raise NotImplementedError

    @staticmethod
    def reconstruct_chunk(partial_chunk: list[str]) -> str:
        reconstructed = ""

        for chunk in partial_chunk:
            if reconstructed and chunk not in {".", "?", "!"}:
                reconstructed += " "
            reconstructed += chunk
        return reconstructed

    @staticmethod
    def _split(long_text: str, pattern: str = r"([.?!])\s*") -> list[str]:
        return re.split(pattern, long_text.strip())


class FixedCharacterChunker(Chunker):
    def __init__(self, config: dict):
        super().__init__(config)

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

    def split(self, long_text: str) -> list[str]:
        k: int = self.config.get("k", len(long_text) // 1024)
        lines = self._split(long_text)
        grouping = ChunkerInitializer.uniform(len(lines), k, "back")
        output_list = []
        for g_start, g_end in grouping:
            chunk = lines[g_start:g_end]
            g_string = self.reconstruct_chunk(chunk)
            output_list.append(g_string)
        return output_list


class OptimizeStrategy(ABC):
    @abstractmethod
    def optimize(self, input_list: list[tuple[int, int]]) -> list[tuple[int, int]]:
        raise NotImplementedError


class SimulatedAnnealingStrategy(OptimizeStrategy):
    def __init__(
        self,
        update_rate: float,
        initial_temp: float,
        cooling_rate: float,
        constraints: dict,
    ):
        self.__update_rate = update_rate  # [0, 1]
        self.__temperature = initial_temp  # [0, 1]
        self.__cooling_rate = cooling_rate  # [0, 1]
        self.__constraints = constraints

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
        self,
        input_list: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        RIGHT_BOUND = self.__constraints.get(
            "RIGHT_BOUND", max([t[1] for t in input_list])
        )
        output_list: list[tuple[int, int]] = input_list[:]
        k = len(input_list)
        factor = int(k * self.update_rate)
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
                    # This might end up in a very large chunk!
                    start = random.randint(RIGHT_BOUND // 4, RIGHT_BOUND // 2)
                    end = random.randint(start, RIGHT_BOUND // 4 * 3)
                    new_tuple = (start, end)  # Find a random chunk within the 25 - 75 %
                    if new_tuple not in unique_list:
                        break
                unique_list.append(new_tuple)
        return unique_list
