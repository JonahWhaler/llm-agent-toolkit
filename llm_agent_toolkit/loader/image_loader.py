import os
import logging

from .._loader import BaseLoader
from .._core import ImageInterpreter
from .._util import MessageBlock

logger = logging.getLogger(__name__)


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
    - __image_interpreter (ImageInterpreter): The core processing unit responsible for converting images to text.

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
    """

    SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp")

    def __init__(
        self, image_interpreter: ImageInterpreter, prompt: str = "What's in the image?"
    ):
        """
        Initializes a new instance of `ImageToTextLoader` with the specified core processing unit.

        Parameters:
        ----------
        - image_interpreter (ImageInterpreter): An instance of `ImageInterpreter` responsible for converting images to text.
        - prompt (str, optional): The prompt to guide image processing. Defaults to "What's in the image?".
        """

        self.__prompt = prompt
        self.__image_interpreter = image_interpreter

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
            input_path (str): The file path of the image to process.

        Returns:
        ----------
            str: The textual description of the image content.
                If return_n > 1, the variant are joined with '#######'.

        Raises:
        ----------
        - ValueError: If the input path is invalid or the file format is unsupported.
        - FileNotFoundError: If the specified file does not exist.
        - Exception: If an error occurs during image processing.
        """
        ImageToTextLoader.raise_if_invalid(input_path)

        try:
            responses: list[MessageBlock | dict] = self.__image_interpreter.interpret(
                query=self.__prompt, context=None, filepath=input_path
            )
            return self.post_processing(responses)
        except Exception as e:
            raise e

    async def load_async(self, input_path: str) -> str:
        """
        Asynchronously processes the specified image file and returns its textual description based on the prompt.

        Parameters:
        ----------
            input_path (str): The file path of the image to process.

        Returns:
        ----------
            str: The textual description of the image content.
                If return_n was > 1, the variant are joined with '#######'.

        Raises:
        ----------
        - ValueError: If the input path is invalid or the file format is unsupported.
        - FileNotFoundError: If the specified file does not exist.
        - Exception: If an error occurs during image processing.
        """
        ImageToTextLoader.raise_if_invalid(input_path)

        try:
            responses: list[MessageBlock | dict] = (
                await self.__image_interpreter.interpret_async(
                    query=self.__prompt, context=None, filepath=input_path
                )
            )
            return self.post_processing(responses)
        except Exception as e:
            raise e

    @staticmethod
    def post_processing(responses: list[MessageBlock | dict]) -> str:
        contents: list[str] = []
        for response in responses:
            contents.append(response["content"])
        return "#######".join(contents)
