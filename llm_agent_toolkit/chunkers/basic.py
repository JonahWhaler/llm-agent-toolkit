import logging
import re

from .._chunkers import Chunker, UniformInitializer

logger = logging.getLogger(name=__name__)


class FixedCharacterChunker(Chunker):
    """FixedCharacterChunker splits text into fixed-size character chunks with optional overlapping.

    Configuration:
    - chunk size (int): (0, context_length of encoder], default = 512 characters.
    - stride_rate (float): (0, 1.0], default = 1.0

    Notes:
    - Expect int(chunk_size * stride_rate) >= 1
    """

    def __init__(self, config: dict):
        self.raise_if_invalid(config)
        super().__init__(config)

    @staticmethod
    def raise_if_invalid(parameters: dict) -> None:
        chunk_size: int = parameters.get("chunk_size", 512)
        if chunk_size is not None and not isinstance(chunk_size, int):
            raise TypeError(
                f"Expect chunk_size to be type 'int', got '{type(chunk_size).__name__}'."
            )
        if chunk_size <= 0:
            raise ValueError(f"Expect chunk_size > 0, got {chunk_size}.")
        stride_rate: float = parameters.get("stride_rate", 1.0)
        if stride_rate is not None and not isinstance(stride_rate, float):
            raise TypeError(
                f"Expect stride_rate to be type 'float', got '{type(stride_rate).__name__}'."
            )
        if stride_rate <= 0 > 1:
            raise ValueError(
                f"Expect stride_rate to be within (0, 1.0], got {stride_rate}."
            )
        if int(chunk_size * stride_rate) == 0:
            raise ValueError(
                "Expect stride >= 1. Please consider adjust chunk_size and stride_rate so that int(chunk_size * stride_rate) >= 1."
            )

    def split(self, long_text: str) -> list[str]:
        """Splits long text into fixed-size character chunks with optional overlapping.

        Args:
            long_text (str): The text to be split into chunks.

        Returns:
            list[str]: A list of text chunks.

        Raises:
            TypeError: If `long_text` is not type 'str'.
            ValueError: If `long_text` is an empty string.

        Notes:
        - If `chunk_size` is greater than `long_text`, the return list will have one chunk.
        """
        if not isinstance(long_text, str):
            raise TypeError(
                f"Expected 'long_text' to be str, got {type(long_text).__name__}."
            )
        text = long_text.replace("\n\n", "\n").strip("\n ")
        if len(text) == 0:
            raise ValueError("Expect long_text to be non-empty string.")

        chunk_size: int = self.config.get("chunk_size", 512)
        if chunk_size > len(text):
            logger.warning(
                "chunk_size (%d) is greater than > len(text) (%d), therefore, only 1 chunk is return.",
                chunk_size,
                len(text),
            )
            return [text]

        stride_rate: float = self.config.get("stride_rate", 1.0)
        stride: int = int(chunk_size * stride_rate)
        output_list = []
        for offset in range(0, len(text), stride):
            chunk = text[offset : offset + chunk_size]
            output_list.append(chunk)
        return output_list


class FixedGroupChunker(Chunker):
    """FixedGroupChunker splits text into K chunks.

    Configuration:
    - K (int): [1,]. Required field.
    - resolution (str): ["front", "back", "skip"], default = "back"
    - level (str): ["word", "character"], default = "character"

    Notes:
    - Does not guarantee token counts of each chunk.
    - Does nto guarantee the return list to have K chunks.
    """

    def __init__(self, config: dict):
        self.raise_if_invalid(config)
        super().__init__(config)

    @staticmethod
    def raise_if_invalid(parameters: dict) -> None:
        K: int = parameters.get("K", None)
        if K is not None and not isinstance(K, int):
            raise TypeError(f"Expect K to be type 'int', got '{type(K).__name__}'.")
        if K <= 0:
            raise ValueError(f"Expect K > 0, got {K}.")
        resolution: str = parameters.get("resolution", "back")
        if resolution is not None and not isinstance(resolution, str):
            raise TypeError(
                f"Expect resolution to be type 'str', got '{type(resolution).__name__}'."
            )
        if resolution not in ["front", "back", "skip"]:
            raise ValueError(
                f"Expect resolution to be either ['front', 'back', 'skip'], got {resolution}."
            )
        level: str = parameters.get("level", "character")
        if level is not None and not isinstance(level, str):
            raise TypeError(
                f"Expect level to be type 'str', got '{type(level).__name__}'."
            )
        if level not in ["word", "character"]:
            raise ValueError(
                f"Expect level to be either ['word', 'character'], got {level}."
            )

    def split(self, long_text: str) -> list[str]:
        """Splits long text into K chunks.

        Args:
            long_text (str): The text to be split into chunks.

        Returns:
            list[str]: A list of text chunks.

        Raises:
            TypeError: If `long_text` is not type 'str'.
            ValueError: If `long_text` is an empty string.

        Notes:
        - If `K` is greater than len(lines), the return list will have len(lines) chunks.
            **Therefore, user should not assume the return list to have K chunks**.
        """
        if not isinstance(long_text, str):
            raise TypeError(
                f"Expected 'long_text' to be str, got {type(long_text).__name__}."
            )
        # Invalid level will be caught at __init__
        level: str = self.config.get("level", "character")
        # Missing K will be caught at __init__
        K: int = self.config.get("K", 1)
        # Invalid resolution will be caught at __init__
        resolution: str = self.config.get("resolution", "back")
        # BEGIN
        # Sanitize argument `long_text`
        text = long_text.replace("\n\n", "\n").strip("\n ")  # Remove excessive newlines
        text = text.replace("\n", "\n")  # Convert viewable newline to readable newline
        if len(text) == 0:
            raise ValueError("Expect long_text to be non-empty string.")
        lines = (
            list(text) if level == "character" else re.split(r"([.?!\n\t])\s*", text)
        )
        lines = list(filter(lambda line: line, lines))  # Remove invalid lines
        if K > len(lines):
            logger.warning(
                "K (%d) is greater than > len(lines) (%d), therefore, only %d chunks are return.",
                K,
                len(lines),
                len(lines),
            )
            return lines

        initializer = UniformInitializer(len(lines), K, resolution)
        grouping = initializer.init()
        output_list: list[str] = []
        for g_start, g_end in grouping:
            chunk = lines[g_start:g_end]
            if level == "word":
                g_string = self.reconstruct_chunk(chunk)
            else:
                # reconstruct_chunk is not suitable for character-wise spliting.
                # Example:
                # original_text = "Happy World!"
                # lines = list(original_text)
                # print(lines) # ['H', 'a', 'p', 'p', 'y', ' ', 'W', 'o', 'r', 'l', 'd', '!']
                # reconstructed = self.reconstruct_chunk(lines)
                # print(reconstructed) # H a p p y  W o r l d!
                g_string = "".join(chunk)
            output_list.append(g_string)
        # END
        return output_list
