import random
import logging
from abc import abstractmethod, ABC
from typing import runtime_checkable, Protocol

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


@runtime_checkable
class ChunkingInitializer(Protocol):
    def init(
        self,
    ) -> list[tuple[int, int]]: ...


class UniformInitializer:
    """Initialize chunk groupings uniformly.
    Resolve with `resolution` when `total-capacity` cannot be evenly distributed into `k` groups.

    Attributes:
        - total_capacity (int): The total size of be divided into chunks.
        - k (int): The number of chunks to create.
        - resolution (str): Default = "skip", options = ["front", "back", "skip"]

    Notes:
    * coverage may not equal to 1.0 when resolution is "skip"
    """

    def __init__(self, total_capacity: int, k: int, resolution: str = "skip"):
        self.total_capacity = total_capacity
        self.k = k
        self.resolution = resolution

    def init(self) -> list[tuple[int, int]]:
        chunk_size = self.total_capacity // self.k
        remainer = self.total_capacity - chunk_size * self.k
        output_list = []
        offset = 0
        for ki in range(self.k):
            right = offset + chunk_size
            if ki == 0 and self.resolution == "front":
                right += remainer
            elif ki == self.k - 1 and self.resolution == "back":
                right = self.total_capacity
            output_list.append((offset, min(right, self.total_capacity)))
            offset = right

        return output_list


class RandomInitializer:
    """Initialize chunk groupings with random overlapping regions.

    Attributes:
        - total_capacity (int): The total size of be divided into chunks.
        - k (int): The number of chunks to create.

    Notes:
    * Guarantee coverage of 1.0
    """

    def __init__(self, total_capacity: int, k: int):
        self.total_capacity = total_capacity
        self.k = k

    def init(self):
        remainer = self.total_capacity
        # Initialize chunk sizes to zero
        init_list = [0] * self.k
        while remainer > 0:
            # Determine the maximum even size that can be allocated to each chunk
            even_size = remainer // self.k
            if even_size < 1:
                # If remaining capacity is less than the number of chunks,
                # distribute the remaining one by one randomly to chunks
                for _ in range(remainer):
                    index = random.randint(0, self.k - 1)
                    init_list[index] += 1
                break  # All remaining capacity has been distributed
            # Randomly decide how much to add to each chunk in this iteration
            new_growth = [random.randint(1, even_size) for _ in range(self.k)]
            # Add the new growth to each chunk's size
            for index in range(self.k):
                init_list[index] += new_growth[index]
            # Decrease the remaining capacity by the total allocated in this iteration
            remainer -= sum(new_growth)

        offset = 0
        output_list: list[tuple[int, int]] = []  # type: ignore
        for size in init_list:
            output_list.append((offset, offset + size))
            offset += size

        assert (
            ChunkerMetrics.calculate_coverage(self.total_capacity, output_list) == 1.0
        )
        return output_list


class Chunker(ABC):
    """
    Abstract base class for text chunkers.

    The `Chunker` class provides a standardized interface and common utilities
    for splitting long texts into smaller, manageable chunks. Subclasses must
    implement the `split` method to define specific chunking strategies.

    Attributes:
        config (dict): Configuration parameters for the chunker.

    Methods:
        split(long_text: str) -> list[str]:
            Splits the provided long text into a list of smaller text chunks.
            Must be implemented by all subclasses.

        reconstruct_chunk(partial_chunk: list[str]) -> str:
            Reconstructs a single text string from a list of partial chunks.
            Ensures proper spacing and punctuation between chunks.
    """

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
        """
        Reconstructs a single text string from a list of partial chunks.

        This method ensures proper spacing between chunks and correctly handles punctuation.

        Args:
            partial_chunk (list[str]): A list of text segments to be combined.

        Returns:
            str: The reconstructed text string.
        """
        reconstructed = []
        previous_chunk = ""

        for chunk in partial_chunk:
            if previous_chunk:
                if "#" in chunk or "`" in chunk:
                    reconstructed.append("\n")
                elif (
                    chunk not in {".", "?", "!", "\n", "\t"} and previous_chunk != "\n"
                ):
                    reconstructed.append(" ")
            reconstructed.append(chunk)
            previous_chunk = chunk

        return "".join(reconstructed)
