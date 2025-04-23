import logging
import re
from pydantic import BaseModel, field_validator
import charade
from .._chunkers import UniformInitializer
from .utility import reconstruct_chunk

logger = logging.getLogger(name=__name__)


class FixedCharacterChunkerConfig(BaseModel):
    """Configuration for FixedCharacterChunker.

    Configuration:
    - chunk size (int): 
        Maximum length of each chunk, default = 128 characters.
    - stride rate (float): 
        Stride rate of the sliding window, default = 1.0.

    Notes:
        * **Overlapping**: stride_rate < 1.0
        * **Non-overlapping**: stride_rate = 1.0
    """
    chunk_length: int = 128
    stride_rate: float = 1.0

    @field_validator("chunk_length")
    @classmethod
    def validate_chunk_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(
                f"Expect chunk_size to be greater than 0, got {value}."
            )
        return value

    @field_validator("stride_rate")
    @classmethod
    def validate_stride_rate(cls, value: float) -> float:
        if value <= 0 or value > 1.0:
            raise ValueError(
                f"Expect stride_rate to be in (0, 1.0], got {value}."
            )
        return value


class FixedCharacterChunker:
    """FixedCharacterChunker splits text into fixed-size character chunks with optional overlapping.

    Attributes:
        config (FixedCharacterChunkerConfig): Configuration for the chunker.
    """

    def __init__(self, config: FixedCharacterChunkerConfig):
        self.__config = config

    @property
    def config(self) -> FixedCharacterChunkerConfig:
        return self.__config

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

        text = long_text.strip("\n ")
        if len(text) == 0:
            raise ValueError("Expect long_text to be non-empty string.")

        if self.config.chunk_length > len(text):
            return [text]

        output_list = []
        stride: int = int(self.config.chunk_length * self.config.stride_rate)
        for offset in range(0, len(text), stride):
            chunk = text[offset:offset + self.config.chunk_length]
            output_list.append(chunk)
        return output_list


class FixedGroupChunkerConfig(BaseModel):
    """
    Configuration for FixedGroupChunker.

    Configuration:
    - K (int): Number of chunks, default = 1.
    - level (str): ["word", "character"], default = "character"
    """
    G: int = 1
    level: str = "character"

    @field_validator("G")
    @classmethod
    def validate_G(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"Expect G to be greater than 0, got {value}.")
        return value

    @field_validator("level")
    @classmethod
    def validate_level(cls, value: str) -> str:
        if value not in ["character", "word"]:
            raise ValueError(
                f"Expect level to be either ['character', 'word'], got {value}."
            )
        return value


class FixedGroupChunker:
    """FixedGroupChunker splits text into K chunks.

    Attributes:
        config (FixedGroupChunkerConfig): Configuration for the chunker.

    Constraints:
    - When `level` is "word", the resulting chunks may have varying lengths.
    - When `level` is "character", the resulting chunks may have malformed words.
    """

    def __init__(self, config: FixedGroupChunkerConfig):
        self.__config = config

    @property
    def config(self) -> FixedGroupChunkerConfig:
        return self.__config

    def split(self, long_text: str) -> list[str]:
        """Splits long text into K chunks.

        Args:
            long_text (str): The text to be split into chunks.

        Returns:
            list[str]: A list of text chunks.

        Raises:
            TypeError: If `long_text` is not type 'str'.
            ValueError: If `long_text` is an empty string.
            ValueError: If `G` is greater than the number of lines in `long_text`.
        """
        logger.warning("[BEG] split")
        logger.info("Configuration: %s", self.config)
        if not isinstance(long_text, str):
            raise TypeError(
                f"Expected 'long_text' to be str, got {type(long_text).__name__}."
            )

        # BEGIN
        # Sanitize argument `long_text`
        # Remove excessive newlines
        text = long_text.replace("\n\n", "\n").strip("\n ")

        if len(text) == 0:
            raise ValueError("Expect long_text to be non-empty string.")

        if self.config.level == "word":
            lines = re.split(r"(\s+)", text)
        else:
            lines = text

        lines = list(map(lambda line: line.strip(), lines))
        lines = list(filter(lambda line: line, lines))
        if self.config.G > len(lines):
            raise ValueError(
                f"Expect G to be less than or equal to the number of lines, "
                f"got {self.config.G} > {len(lines)}."
            )

        initializer = UniformInitializer(len(lines), self.config.G, "back")
        grouping = initializer.init()
        output_list: list[str] = []
        for g_start, g_end in grouping:
            chunk = lines[g_start:g_end]
            if self.config.level == "word":
                g_string = reconstruct_chunk(chunk)
            else:
                g_string = "".join(chunk)
            output_list.append(g_string)
        logger.warning("[END] split")
        return output_list


class SentenceChunker:
    """Split long texts at sentence level.

    Notes:
    * Works better on well punctuated/formatted plain text content.
    * Support both ascii and non-ascii text.

    Constraints:
    - 'audio.mp3' -> ['audio.', 'mp3']
    - Does not support single quote, double quote, braces and brackets.
    """

    @staticmethod
    def isascii(text: str) -> bool:
        byte_sentence = text.encode("utf-8")
        result = charade.detect(byte_sentence)
        return result["encoding"] == "ascii"

    @staticmethod
    def patch_punctuation(lines: list[str], punctuation: str) -> list[str]:
        """
        Patches the punctuation in the lines.
        
        Args:
            lines (list[str]): The list of lines to be patched.
            punctuation (str): The punctuation characters to be used for patching.
            
        Returns:
            new_lines (list[str]): 
                A list of lines with patched punctuation.
        """
        new_lines = []
        temp = ""
        for line in lines:
            if temp == "":
                temp = line
            elif line in punctuation:
                temp += line
                new_lines.append(temp)
                temp = ""
            else:
                temp = f"{temp} {line}"
        if temp:
            new_lines.append(temp)
        return new_lines

    def split(self, long_text: str) -> list[str]:
        """Splits long text into sentences.
        This method uses regex to identify sentence boundaries based on punctuation.
        It handles both ascii and non-ascii text.

        Args:
            long_text (str): The text to be split into sentences.
            
        Returns:
            new_lines (list[str]): A list of sentences.
            
        Raises:
            TypeError: If `long_text` is not type 'str'.
            ValueError: If `long_text` is an empty string.
        """
        logger.warning("[BEG] split")
        if not isinstance(long_text, str):
            raise TypeError(
                f"Expected 'long_text' to be str, got {type(long_text).__name__}."
            )

        text = long_text.strip("\n ")
        if len(text) == 0:
            raise ValueError("Expect long_text to be non-empty string.")

        if SentenceChunker.isascii(long_text):
            pattern = r"([\n.?!;])\s*"
            punctuation = '\n.;?!"'
        else:
            pattern = r"([\n.?!;。”？！；])\s*"
            punctuation = '\n？！；。”.;?!"'

        lines = re.split(pattern, long_text)
        lines = list(filter(lambda line: line, lines))
        new_lines = SentenceChunker.patch_punctuation(lines, punctuation)
        logger.warning("[END] split")
        return new_lines


class SectionChunker:
    """Split long texts at section/paragraph level.

    Constraints:
    * Works better on well punctuated/formatted plain text content.
    """

    def split(self, long_text: str) -> list[str]:
        """Splits long text into sections/paragraphs.
        This method uses regex to identify section boundaries based on newlines.

        Args:
            long_text (str): The text to be split into sections/paragraphs.

        Returns:
            new_lines (list[str]): A list of sections/paragraphs.

        Raises:
            TypeError: If `long_text` is not type 'str'.
            ValueError: If `long_text` is an empty string.
        """
        logger.warning("[BEG] split")
        if not isinstance(long_text, str):
            raise TypeError(
                f"Expected 'long_text' to be str, got {type(long_text).__name__}."
            )

        text = long_text.strip("\n ")
        if len(text) == 0:
            raise ValueError("Expect long_text to be non-empty string.")

        pattern = r"([\n]{2,})\s*"
        lines = re.split(pattern, long_text)
        lines = list(map(lambda line: line.strip(), lines))
        lines = list(filter(lambda line: line, lines))
        logger.warning("[END] split")
        return lines
