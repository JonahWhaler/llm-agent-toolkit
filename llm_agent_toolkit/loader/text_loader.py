import os
import asyncio
from .._loader import BaseLoader


class TextLoader(BaseLoader):
    """
    A stateless loader for reading plain text files efficiently without retaining large data in memory.

    `TextLoader` is a concrete implementation of the `BaseLoader` abstract base class. It provides
    both synchronous and asynchronous methods to load text data from specified file paths. Designed
    to minimize memory usage, `TextLoader` reads files in a way that avoids holding large data chunks
    in memory for extended periods.

    This loader supports the following plain text file formats:

    - `.txt`: Plain text files.

    - `.md`: Markdown files.

    - `.py`: Python source files.

    - `.html`: HTML files.

    - `.css`: CSS files.

    - `.js`: JavaScript files.

    - `.json`: JSON files.

    - `.csv`: Comma-separated values files (Note: Loading large CSV files may be inefficient).

    Attributes:
    ----------
    - SUPPORTED_EXTENSIONS (tuple): A tuple of supported file extensions.

    Methods:
    ----------
    - load(input_path: str) -> list[str]: Synchronously reads and returns the content of the specified text file.

    - load_async(input_path: str) -> list[str]: Asynchronously reads and returns the content of the specified text file.

    - raise_if_invalid(input_path: str) -> None: Validates the input file path and raises appropriate exceptions if invalid.

    Raises:
    ----------
    - ValueError: If the input path is invalid or the file format is unsupported.
    - FileNotFoundError: If the specified file does not exist.
    - Exception: Propagates any exception raised during the file reading process.

    Notes:
    ----------
    - Formats like `.pdf` and `.doc` are text-based but contain complex structures. It is recommended to use
      specialized parsers for these formats to extract text effectively.
    - While `.csv` files are supported, `TextLoader` may not be the most efficient choice for very large CSV
      files. Consider using dedicated CSV parsing libraries for better performance and functionality.
    """

    SUPPORTED_EXTENSIONS = (
        ".txt",
        ".md",
        ".py",
        ".html",
        ".css",
        ".js",
        ".json",
        ".csv",
    )

    def __init__(self, encoding: str = "utf-8"):
        """
        Initializes a new instance of `TextLoader` with the specified encoding.

        Parameters:
        ----------
        - encoding (str, optional): The character encoding to use when reading text files. Defaults to "utf-8".

        TODO: Validate the encoding parameter to ensure it is a valid character encoding.
        """
        self.__encoding = encoding

    @staticmethod
    def raise_if_invalid(input_path: str) -> None:
        """
        Validates the input file path.

        Parameters:
        ----------
        - input_path (str): The file path to validate.

        Returns:
        ----------
        - None

        Raises:
        ----------
        - ValueError: If the input path is not a non-empty string or if the file format is unsupported.
        - FileNotFoundError: If the specified file does not exist.
        """
        if not all(
            [
                input_path is not None,
                isinstance(input_path, str),
                input_path.strip() != "",
            ]
        ):
            raise ValueError("Invalid input path: Path must be a non-empty string.")

        _, ext = os.path.splitext(input_path)
        if ext.lower() not in TextLoader.SUPPORTED_EXTENSIONS:
            supported = ", ".join(TextLoader.SUPPORTED_EXTENSIONS)
            raise ValueError(
                f"Unsupported file format: '{ext}'. Supported formats are: {supported}."
            )

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: '{input_path}'.")

    def load(self, input_path: str) -> str:
        """
        Synchronously reads and returns the content of the specified text file.

        Parameters:
        ----------
        - input_path (str): The file path to load the text from.

        Returns:
        ----------
        - str: The content of the text file.

        Raises:
        ----------
        - ValueError: If the input path is invalid or the file format is unsupported.
        - FileNotFoundError: If the specified file does not exist.
        - Exception: If an error occurs during file reading.
        """
        TextLoader.raise_if_invalid(input_path)

        try:
            with open(input_path, "r", encoding=self.__encoding) as f:
                data = f.read()
            return data
        except Exception as e:
            raise e

    async def load_async(self, input_path: str) -> str:
        """
        Asynchronously reads and returns the content of the specified text file.

        Parameters:
        ----------
        - input_path (str): The file path to load the text from.

        Returns:
        ----------
        - str: The content of the text file.

        Raises:
        ----------
        - ValueError: If the input path is invalid or the file format is unsupported.
        - FileNotFoundError: If the specified file does not exist.
        - Exception: If an error occurs during file reading.
        """

        def read_file(path: str) -> str:
            with open(path, "r", encoding=self.__encoding) as reader:
                return reader.read()

        TextLoader.raise_if_invalid(input_path)

        try:
            return await asyncio.to_thread(read_file, input_path)
        except Exception as e:
            raise e
