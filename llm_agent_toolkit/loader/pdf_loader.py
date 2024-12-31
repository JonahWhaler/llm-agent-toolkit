import os
import io
import logging
from contextlib import contextmanager

# PyMuPDF
import fitz  # type: ignore

# from fitz import Page, Document
import pdfplumber

from .._loader import BaseLoader
from .._core import ImageInterpreter
from .._util import MessageBlock

logger = logging.getLogger(__name__)

"""
Dependencies:
----------
- pdfplumber==0.11.4
- PyMuPDF==1.24.11
"""


class PDFLoader(BaseLoader):
    """
    A loader for parsing PDF files and extracting text, links, images, and tables.

    `PDFLoader` is a concrete implementation of the `BaseLoader` abstract base class.
    It provides both synchronous (`load`) and asynchronous (`load_async`) methods to process PDF files
    and return their parsed text content.

    When the `text_only` flag is set to False, it uses the `core` to interpret the textual description of images
    in the PDF file.

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

    Notes:
    ----------
    - Ensure that the `ImageInterpreter` core is properly configured and initialized before using this loader.
    """

    SUPPORTED_EXTENSIONS = (".pdf",)

    def __init__(
        self,
        text_only: bool = True,
        tmp_directory: str | None = None,
        image_interpreter: ImageInterpreter | None = None,
    ):
        self.__image_interpreter = image_interpreter
        self.__tmp_directory = tmp_directory
        if not text_only:
            assert isinstance(tmp_directory, str)
            tmp_directory = tmp_directory.strip()
            if not tmp_directory:
                raise ValueError(
                    "Invalid temporary directory: Must be a non-empty string."
                )

            if not os.path.exists(tmp_directory):
                logger.warning(
                    "Temporary directory not exists. Will create one with name: %s",
                    tmp_directory,
                )
                os.makedirs(tmp_directory)

    @staticmethod
    def raise_if_invalid(input_path: str) -> None:
        if not all(
            [input_path is not None, isinstance(input_path, str), input_path != ""]
        ):
            raise ValueError("Invalid input path: Path must be a non-empty string.")

        if input_path[-4:] != ".pdf":
            raise ValueError("Unsupported file format: Must be a PDF file.")

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: '{input_path}'.")

    def load(self, input_path: str) -> str:
        PDFLoader.raise_if_invalid(input_path)

        try:
            # Elegant way
            # return asyncio.run(self.load_async(input_path))
            markdown_content = []

            # Extract text, links, and images using PyMuPDF
            with fitz.open(input_path) as doc:
                for page_number, page in enumerate(doc, start=1):  # type: ignore
                    markdown_content.append(f"# Page {page_number}\n")
                    markdown_content.append(page.get_text())

                    # Extract links if available
                    links_content = self.handle_links(page.get_links(), page_number)
                    markdown_content.extend(links_content)

                    # Extract images and their alt text if available
                    images_content = self.handle_images(
                        doc, page, page.get_images(), page_number
                    )
                    markdown_content.extend(images_content)

            tables_content = self.handle_tables(input_path)
            markdown_content.extend(tables_content)
            return "\n".join(markdown_content)
        except Exception as e:
            raise e

    async def load_async(self, input_path: str) -> str:
        PDFLoader.raise_if_invalid(input_path)

        try:
            markdown_content = []

            # Extract text, links, and images using PyMuPDF
            with fitz.open(input_path) as doc:
                for page_number, page in enumerate(doc, start=1):  # type: ignore
                    markdown_content.append(f"# Page {page_number}\n")
                    markdown_content.append(page.get_text())

                    # Extract links if available
                    links_content = self.handle_links(page.get_links(), page_number)
                    markdown_content.extend(links_content)

                    # Extract images and their alt text if available
                    images_content = await self.handle_images_async(
                        doc, page, page.get_images(), page_number
                    )
                    markdown_content.extend(images_content)

            tables_content = self.handle_tables(input_path)
            markdown_content.extend(tables_content)
            return "\n".join(markdown_content)
        except Exception as e:
            raise e

    def extract_img_description(self, image_bytes: bytes, image_name: str) -> str:
        if self.__image_interpreter is None:
            return "Image description not available"

        image_caption = (
            f"filename={image_name}. This is an attachment found in a pdf file."
        )
        with self.temporary_file(image_bytes, image_name) as tmp_path:
            responses: list[MessageBlock | dict] = self.__image_interpreter.interpret(
                query=image_caption, context=None, filepath=tmp_path
            )
            response = responses[0]
            if "content" in response:
                return response["content"]
            raise RuntimeError("content not found in MessageBlock.")

    async def extract_img_description_async(
        self, image_bytes: bytes, image_name: str
    ) -> str:
        if self.__image_interpreter is None:
            return "Image description not available"

        image_caption = (
            f"filename={image_name}. This is an attachment found in a pdf file."
        )
        with self.temporary_file(image_bytes, image_name) as tmp_path:
            responses: list[
                MessageBlock | dict
            ] = await self.__image_interpreter.interpret_async(
                query=image_caption, context=None, filepath=tmp_path
            )
            response = responses[0]
            if "content" in response:
                return response["content"]
            raise RuntimeError("content not found in MessageBlock.")

    @contextmanager
    def temporary_file(self, image_bytes: bytes, filename: str):
        tmp_path = f"{self.__tmp_directory}/{filename}"
        try:
            image_stream = io.BytesIO(image_bytes)
            image_stream.seek(0)
            with open(tmp_path, "wb") as f:
                f.write(image_bytes)
            yield tmp_path
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @staticmethod
    def handle_links(links: list, page_number: int) -> list[str]:
        if links is None:
            return []

        markdown_content = ["\n## Links\n"]

        for link in links:
            if "uri" in link:
                markdown_content.append(
                    f"- [Link on Page {page_number}]({link['uri']})\n"
                )

        return markdown_content

    def handle_images(self, doc, page, images: list, page_number: int) -> list[str]:
        if images is None:
            return []

        markdown_content = ["\n## Images\n"]
        for img_index, img in enumerate(images, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            if base_image:
                image_name = f"image_{page_number}_{img_index}.{base_image['ext']}"
                markdown_content.extend(
                    self.handle_image(
                        base_image["image"],
                        image_name,
                        img_index,
                        page_number,
                    )
                )

        return markdown_content

    async def handle_images_async(
        self, doc, page, images: list, page_number: int
    ) -> list[str]:
        if images is None:
            return []

        markdown_content = ["\n## Images\n"]
        for img_index, img in enumerate(images, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            if base_image:
                image_name = f"image_{page_number}_{img_index}.{base_image['ext']}"
                markdown_content.extend(
                    await self.handle_image_async(
                        base_image["image"],
                        image_name,
                        img_index,
                        page_number,
                    )
                )

        return markdown_content

    def handle_image(
        self, image_bytes, image_name, img_index, page_number
    ) -> list[str]:
        markdown_content = []

        if self.__image_interpreter:
            image_description = self.extract_img_description(image_bytes, image_name)
        else:
            image_description = "Image description not available"
        markdown_content.append(
            f"- Image {img_index} on Page {page_number}: {image_name}\n"
        )
        markdown_content.append(f"  Description: \n{image_description}\n")
        markdown_content.append(f"  [IMAGE ATTACHED: {image_name}]\n")

        return markdown_content

    async def handle_image_async(
        self, image_bytes, image_name, img_index, page_number
    ) -> list[str]:
        markdown_content = []

        if self.__image_interpreter:
            image_description = await self.extract_img_description_async(
                image_bytes, image_name
            )
        else:
            image_description = "Image description not available"
        markdown_content.append(
            f"- Image {img_index} on Page {page_number}: {image_name}\n"
        )
        markdown_content.append(f"  Description: \n{image_description}\n")
        markdown_content.append(f"  [IMAGE ATTACHED: {image_name}]\n")

        return markdown_content

    @staticmethod
    def handle_tables(input_path: str) -> list[str]:
        markdown_content = []

        # Extract tables using pdfplumber
        with pdfplumber.open(input_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for table_index, table in enumerate(tables, start=1):
                    if table:
                        markdown_content.append(
                            f"\n## Table {table_index} on Page {page_number}\n"
                        )
                        for row in table[:]:
                            markdown_content.append(
                                "| " + " | ".join(str(cell) for cell in row) + " |"
                            )
                        markdown_content.append("\n")

        return markdown_content
