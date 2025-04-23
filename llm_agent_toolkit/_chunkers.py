import random
import logging
from typing import runtime_checkable, Protocol

logger = logging.getLogger(__name__)


class ChunkerMetrics:

    @classmethod
    def calculate_utilization_rate(
        cls, CTX_LENGTH: int, token_counts: list[int],
        grouping: list[tuple[int, int]]
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
        cls, CTX_LENGTH: int, token_counts: list[int],
        grouping: list[tuple[int, int]]
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

    @classmethod
    def calculate_overlapped(
        cls, capacity: int, grouping: list[tuple[int, int]]
    ) -> float:
        b = capacity
        a = 0
        for g_start, g_end in grouping:
            a += g_end - g_start

        overlapped = (a - b) / b
        # Worse case scenario - a = K * b, result in overlapped = (K - 1)
        # Do this to map it to [0.0, 1.0] space.
        overlapped = max(0, overlapped / (len(grouping) - 1))
        return overlapped


@runtime_checkable
class ChunkingInitializer(Protocol):

    def init(self, ) -> list[tuple[int, int]]:
        ...


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
                break    # All remaining capacity has been distributed

            # Add the new growth to each chunk's size
            for index in range(self.k):
                # Randomly decide how much to add to each chunk in this iteration
                init_list[index] += random.randint(1, even_size)
            # Decrease the remaining capacity by the total allocated in this iteration
            remainer = self.total_capacity - sum(init_list)

        offset = 0
        output_list: list[tuple[int, int]] = []    # type: ignore
        for size in init_list:
            output_list.append((offset, offset + size))
            offset += size

        assert (
            ChunkerMetrics.calculate_coverage(self.total_capacity,
                                              output_list) == 1.0
        )
        return output_list


@runtime_checkable
class Splitter(Protocol):

    def split(self, long_text: str) -> list[str]:
        ...


@runtime_checkable
class AsyncSplitter(Protocol):

    async def split(self, long_text: str) -> list[str]:
        ...
