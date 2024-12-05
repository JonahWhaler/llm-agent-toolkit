import logging
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
    def __init__(self, config: dict):
        super().__init__(config)

    def split(self, long_text: str) -> list[str]:
        k: int = self.config.get("k", None)
        if k is None:
            raise KeyError("'K'")
        lines = self._split(long_text)
        initializer = UniformInitializer(len(lines), k, "back")
        grouping = initializer.init()
        output_list = []
        for g_start, g_end in grouping:
            chunk = lines[g_start:g_end]
            g_string = self.reconstruct_chunk(chunk)
            output_list.append(g_string)
        return output_list
