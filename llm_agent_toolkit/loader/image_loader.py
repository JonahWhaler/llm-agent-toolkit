import os
import warnings

from .._loader import BaseLoader
from .._core import Core, I2T_Core
from .._util import MessageBlock


class ImageToTextLoader(BaseLoader):
    """
    A loader for processing image files and extracting their textual descriptions.

    `ImageToTextLoader` is a concrete implementation of the `BaseLoader` abstract base class.
    It provides both synchronous (`load`) and asynchronous (`load_async`) methods to process image files
    and return textual descriptions of their content.

    This loader supports the following image file formats:

    - `.png`: Portable Network Graphics.

    - `.jpg`: JPEG images.

    - `.jpeg`: JPEG images.

    - `.gif`: Graphics Interchange Format.

    - `.webp`: WebP images.

    Attributes:
    ----------
    - SUPPORTED_EXTENSIONS (tuple): A tuple of supported image file extensions.
    - __prompt (str): The prompt used to guide the image processing (e.g., "What's in the image?").
    - __core (I2T_Core): The core processing unit responsible for converting images to text.

    Methods:
    ----------
    - load(input_path: str) -> str: Synchronously processes the specified image file and returns its textual description.

    - load_async(input_path: str) -> str: Asynchronously processes the specified image file and returns its textual description.

    - raise_if_invalid(input_path: str) -> None: Validates the input file path and raises appropriate exceptions if invalid.

    Raises:
    ----------
    - InvalidInputPathError: If the input path is invalid (e.g., not a non-empty string).
    - UnsupportedFileFormatError: If the file format is unsupported.
    - FileNotFoundError: If the specified file does not exist.
    - Exception: Propagates any unexpected exceptions raised during processing.

    Notes:
    ----------
    - Ensure that the `I2T_Core` core is properly configured and initialized before using this loader.

    """

    SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp")

    def __init__(self, core: Core, prompt: str = "What's in the image?"):
        """
        Initializes a new instance of `ImageToTextLoader` with the specified core processing unit.

        Parameters:
        ----------
        - core (I2T_Core): An instance of `I2T_Core` responsible for converting images to text.
        - prompt (str, optional): The prompt to guide image processing. Defaults to "What's in the image?".

        Raises:
        ----------
        - TypeError: If the core is not an instance of `I2T_Core`.

        Warnings:
        ----------
        - If the core's configuration `n` is not 1.
        """
        if not isinstance(core, I2T_Core):
            raise TypeError(
                "Expect `core` to be an instance of `I2T_Core`, got: {}".format(
                    type(core)
                )
            )

        if core.config.return_n != 1:
            warnings.warn(
                "Configured to return {} responses from `core`. "
                "Only first response will be used.".format(core.config.return_n)
            )

        self.__prompt = prompt
        self.__core = core

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
        if ext.lower() not in ImageToTextLoader.SUPPORTED_EXTENSIONS:
            supported = ", ".join(ImageToTextLoader.SUPPORTED_EXTENSIONS)
            raise ValueError(
                f"Unsupported file format: '{ext}'. Supported formats are: {supported}."
            )

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: '{input_path}'.")

    def load(self, input_path: str) -> str:
        """
        Synchronously processes the specified image file and returns its textual description based on the prompt.

        Parameters:
        ----------
        - input_path (str): The file path of the image to process.

        Returns:
        ----------
        - str: The textual description of the image content.

        Raises:
        ----------
        - ValueError: If the input path is invalid or the file format is unsupported.
        - FileNotFoundError: If the specified file does not exist.
        - Exception: If an error occurs during image processing.
        """
        ImageToTextLoader.raise_if_invalid(input_path)

        try:
            responses: list[MessageBlock | dict] = self.__core.run(
                query=self.__prompt, context=None, filepath=input_path
            )
            return responses[-1]["content"]
        except Exception as e:
            raise e

    async def load_async(self, input_path: str) -> str:
        """
        Asynchronously processes the specified image file and returns its textual description based on the prompt.

        Parameters:
        ----------
        - input_path (str): The file path of the image to process.

        Returns:
        ----------
        - str: The textual description of the image content.

        Raises:
        ----------
        - ValueError: If the input path is invalid or the file format is unsupported.
        - FileNotFoundError: If the specified file does not exist.
        - Exception: If an error occurs during image processing.
        """
        ImageToTextLoader.raise_if_invalid(input_path)

        try:
            responses: list[MessageBlock | dict] = await self.__core.run_async(
                query=self.__prompt, context=None, filepath=input_path
            )
            return responses[-1]["content"]
        except Exception as e:
            raise e
