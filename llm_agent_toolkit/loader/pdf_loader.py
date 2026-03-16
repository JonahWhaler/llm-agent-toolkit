import os
import mimetypes
import logging
from typing import Optional, Tuple, List
from functools import reduce

# PyMuPDF
import fitz  # type: ignore

from fitz import Page, Document

from .._loader import BaseLoader
from .._core import Core as TextModel, ImageInterpreter
from .._util import TokenUsage, UsagePurpose
from .utils import CustomCSVHandler, DefinedTask, DefinedTaskAsync

logger = logging.getLogger(__name__)

"""
Dependencies:
----------
- pdfplumber==0.11.4
- PyMuPDF==1.24.11
"""


class PDFLoader(BaseLoader):
    """
    A loader for parsing PDF files and extracting text, tables, images, and links.

    `PDFLoader` uses the `PyMuPDF` library to process PDF files, offering both synchronous (`load`)
    and asynchronous (`load_async`) methods to extract content into a Markdown format.

    It preserves the document structure by ordering text blocks and tables based on their
    vertical position on the page.

    ## Sub-features:
    - **Image Interpretation**: If an `ImageInterpreter` is provided, the loader extracts images,
      generates descriptions using the interpreter, and appends them in a reference section.
      Image locations are marked with anchors in the main text.
    - **Web Page Summarization**: If a `TextModel` (acting as a `web_interpreter`) is provided,
      it summarizes external links found in the PDF and includes the summaries in a
      reference section.
    - **Table Extraction**: Tables are detected and converted into Markdown format.

    Attributes:
    ----------
    - tmp_directory (str | None, optional): A path to a temporary directory for storing
      intermediate files, such as extracted images. Required if `text_only` is False.
      Defaults to None.
    - image_interpreter (ImageInterpreter | None, optional): An AI model for generating
      descriptions from images. Required for image processing. Defaults to None.
    - web_interpreter (TextModel | None, optional): An AI model for summarizing web pages
      from links. Required for link summarization. Defaults to None.

    Methods:
    ----------
    - load(input_path: str) -> str: Synchronously processes the specified PDF file and returns its
      content as a Markdown string.
    - load_async(input_path: str) -> str: Asynchronously processes the specified PDF file and
      returns its content as a Markdown string.

    Notes:
    ----------
    - This loader relies on `PyMuPDF` (`fitz`). Ensure it is installed.
    - For image and link processing, the corresponding interpreter models (`ImageInterpreter`, `TextModel`)
      must be properly configured and provided during initialization.
    """

    SUPPORTED_MIMETYPES: Tuple[str] = ("application/pdf",)

    def __init__(
        self,
        text_only: bool = True,
        tmp_directory: str | None = None,
        image_interpreter: ImageInterpreter | None = None,
        web_interpreter: TextModel | None = None,
    ):
        self.__image_interpreter = image_interpreter
        self.__web_interpreter = web_interpreter
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
        self.__token_usage_list: List[TokenUsage] = []

    def log_usage(self, value: TokenUsage):
        self.__token_usage_list.append(value)

    def show_usage(self) -> dict[str, TokenUsage]:

        output = {}
        for p in UsagePurpose:
            xp_list = list(filter(lambda x: x.purpose is p, self.__token_usage_list))
            xp_total = reduce(lambda a, b: a + b, xp_list)
            output[p.value] = xp_total
        return output

    @classmethod
    def pre_load(cls, input_path: Optional[str] = None):
        if input_path is None:
            raise ValueError("File path is not set.")

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File {input_path} does not exist.")

        mt = mimetypes.guess_type(input_path)
        mime_type, encoding = mt
        if mime_type not in cls.SUPPORTED_MIMETYPES:
            raise ValueError(f"Expect *.docx file.")

    def handle_links(self, links: list[dict]) -> list[str]:
        markdown_content = []
        if links is None:
            return markdown_content

        for link in links:
            if "uri" in link:
                markdown_content.append(f"### {link['uri']}")
                if self.__web_interpreter:
                    site_summary, usage = DefinedTask.summarize_site(
                        self.__web_interpreter, link["uri"]
                    )
                    self.log_usage(usage)
                else:
                    site_summary = "Web page summary not available"
                markdown_content.append(f"**Summary**: {site_summary}\n")
        return markdown_content

    async def handle_links_async(self, links: list[dict]) -> list[str]:
        markdown_content = []
        if links is None:
            return markdown_content

        for link in links:
            if "uri" in link:
                markdown_content.append(f"### {link['uri']}")
                if self.__web_interpreter:
                    site_summary, usage = await DefinedTaskAsync.summarize_site(
                        self.__web_interpreter, link["uri"]
                    )
                    self.log_usage(usage)
                else:
                    site_summary = "Web page summary not available"
                markdown_content.append(f"**Summary**: {site_summary}\n")

        return markdown_content if len(markdown_content) > 0 else []

    def handle_images(self, doc: Document, images: list, page_number: int) -> list[str]:
        if not images:
            return []

        """
        # Validation Step
        """
        if self.__image_interpreter:
            if not isinstance(self.__tmp_directory, str):
                raise ValueError(
                    "Invalid temporary directory: Must be a non-empty string."
                )

            if not os.path.exists(self.__tmp_directory):
                raise FileNotFoundError(
                    f"Temporary directory not exists: {self.__tmp_directory}"
                )

            if not os.path.isdir(self.__tmp_directory):
                raise ValueError(f"Invalid temporary directory: {self.__tmp_directory}")

        def handle_image(image_bytes, image_name) -> str:
            """Custom handler for getting the image description synchrounously."""
            if self.__image_interpreter is None:
                return "Image description not available"

            assert self.__tmp_directory

            image_description, usage = DefinedTask.interpret_image(
                self.__image_interpreter,
                image_bytes,
                image_name,
                self.__tmp_directory,
            )
            self.log_usage(usage)
            return f"**Description**: \n{image_description}\n"

        """
        # Actual Processing            
        """
        markdown_content = []
        for img_index, img in enumerate(images, start=1):
            # Determine if we have an xref or raw block data
            # get_images() usually provides a tuple/list where index 0 is xref
            # get_text("dict") provides a dict where "image" is bytes

            image_data = None
            ext = "png"  # default

            if isinstance(img, dict) and "image" in img:
                # Case: From get_text("dict") -> it's already bytes
                if isinstance(img["image"], bytes):
                    image_data = img["image"]
                    ext = img.get("ext", "png")
                # Case: It might be a dict with an xref key
                elif isinstance(img["image"], int):
                    base_image = doc.extract_image(img["image"])
                    image_data = base_image["image"]
                    ext = base_image["ext"]

            elif isinstance(img, (list, tuple)):
                # Case: From page.get_images() -> first element is xref
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_data = base_image["image"]
                ext = base_image["ext"]

            if image_data:
                image_name = f"image_{page_number}_{img_index}.{ext}"
                # Pass the extracted bytes to your sub-handler
                markdown_content.append(
                    handle_image(
                        image_data,
                        image_name,
                    )
                )

        return markdown_content

    async def handle_images_async(
        self, doc: Document, images: list, page_number: int
    ) -> list[str]:
        if not images:
            return []

        """
        # Validation Step
        """
        if self.__image_interpreter:
            if not isinstance(self.__tmp_directory, str):
                raise ValueError(
                    "Invalid temporary directory: Must be a non-empty string."
                )

            if not os.path.exists(self.__tmp_directory):
                raise FileNotFoundError(
                    f"Temporary directory not exists: {self.__tmp_directory}"
                )

            if not os.path.isdir(self.__tmp_directory):
                raise ValueError(f"Invalid temporary directory: {self.__tmp_directory}")

        async def handle_image(image_bytes, image_name) -> str:
            """Custom handler for getting the image description asynchrounously."""
            if self.__image_interpreter is None:
                return "Image description not available"

            assert self.__tmp_directory

            image_description, usage = await DefinedTaskAsync.interpret_image(
                self.__image_interpreter,
                image_bytes,
                image_name,
                self.__tmp_directory,
            )
            self.log_usage(usage)
            return f"**Description**: \n{image_description}\n"

        """
        # Actual Processing            
        """
        markdown_content = []
        for img_index, img in enumerate(images, start=1):
            # Determine if we have an xref or raw block data
            # get_images() usually provides a tuple/list where index 0 is xref
            # get_text("dict") provides a dict where "image" is bytes

            image_data = None
            ext = "png"  # default

            if isinstance(img, dict) and "image" in img:
                # Case: From get_text("dict") -> it's already bytes
                if isinstance(img["image"], bytes):
                    image_data = img["image"]
                    ext = img.get("ext", "png")
                # Case: It might be a dict with an xref key
                elif isinstance(img["image"], int):
                    base_image = doc.extract_image(img["image"])
                    image_data = base_image["image"]
                    ext = base_image["ext"]

            elif isinstance(img, (list, tuple)):
                # Case: From page.get_images() -> first element is xref
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_data = base_image["image"]
                ext = base_image["ext"]

            if image_data:
                image_name = f"image_{page_number}_{img_index}.{ext}"
                # Pass the extracted bytes to your sub-handler
                markdown_content.append(
                    await handle_image(
                        image_data,
                        image_name,
                    )
                )

        return markdown_content

    def load(self, input_path: Optional[str] = None) -> str:
        PDFLoader.pre_load(input_path)
        assert input_path

        page_number = 0
        doc_content = []
        table_counter = 1
        try:
            page_content = []
            image_metadata = []
            markdown_content: list[tuple[tuple[float, float, float, float], str]] = []
            rects = []
            with fitz.open(input_path) as doc:
                for page in doc:
                    page_number += 1
                    page_content.append(f"# Page {page_number}\n")

                    # Handle tables in the page
                    tables = page.find_tables()  # type: ignore
                    for table in tables.tables:
                        data = table.extract()
                        table_content = CustomCSVHandler._rows_to_csv_string(data, ",")
                        table_content = (
                            f"\n*Table {table_counter}*\n\n"
                            + CustomCSVHandler.csv_to_markdown(table_content)
                            + "\n"
                        )
                        markdown_content.append((table.bbox, table_content))
                        rects.append(fitz.Rect(table.bbox))
                        table_counter += 1

                    # Extract structured blocks (Text = 0, Image = 1)
                    page_dict = page.get_text("dict")  # type: ignore
                    blocks = page_dict.get("blocks", [])
                    for b_idx, block in enumerate(blocks):
                        block_rect = fitz.Rect(block["bbox"])
                        if any([block_rect.intersects(rect) for rect in rects]):
                            continue

                        # CASE 1: BLOCK IS TEXT
                        if block["type"] == 0:
                            block_content = f""
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    block_content += span["text"]
                                block_content += "\n"
                            # Break after block - paragraph
                            block_content += "\n"
                            markdown_content.append((block["bbox"], block_content))
                            continue

                        # CASE 2: BLOCK IS AN IMAGE
                        if block["type"] == 1:
                            anchor_id = f"img_p{page_number}_b{b_idx}"

                            # Insert Anchor exactly where the image sits in the flow
                            markdown_content.append(
                                (block["bbox"], f"> [image@{anchor_id}](#{anchor_id})")
                            )

                            # Store the metadata for the end of the file
                            # Note: block contains image binary/bbox info
                            desc = self.handle_images(doc, [block], page_number)
                            image_metadata.append(f"### {anchor_id}\n")
                            desc = [f"{d}\n\n" for d in desc]
                            image_metadata.extend(desc)
                            continue

                    # Arrange markdown content
                    markdown_content.sort(key=lambda x: x[0][1], reverse=False)  # type: ignore
                    content: list[str] = list(map(lambda x: x[1], markdown_content))
                    markdown_content.clear()
                    page_content.extend(content)

                    # Append Image Descriptions at the Footer
                    if image_metadata:
                        page_content.append(f"## Image Reference Appendix\n")
                        page_content.extend(image_metadata)
                        image_metadata.clear()

                    # Handle Links separately (links usually don't have block types in dict)
                    links = self.handle_links(page.get_links())  # type: ignore
                    if links:
                        page_content.append(f"## Link Reference Appendix\n")
                        page_content.extend(links)

                doc_content.append("\n".join(page_content))
                page_content.clear()
            return "".join(doc_content)
        except Exception as e:
            logger.error(f"[PDFLoader.load] => {str(e)}")
            raise e

    async def load_async(self, input_path: Optional[str] = None) -> str:
        PDFLoader.pre_load(input_path)
        assert input_path

        page_number = 0
        doc_content = []
        table_counter = 1
        try:
            page_content = []
            image_metadata = []
            markdown_content: list[tuple[tuple[float, float, float, float], str]] = []
            rects = []
            with fitz.open(input_path) as doc:
                for page in doc:
                    page_number += 1
                    page_content.append(f"# Page {page_number}\n")

                    # Handle tables in the page
                    tables = page.find_tables()  # type: ignore
                    for table in tables.tables:
                        data = table.extract()
                        table_content = CustomCSVHandler._rows_to_csv_string(data, ",")
                        table_content = (
                            f"\n*Table {table_counter}*\n\n"
                            + CustomCSVHandler.csv_to_markdown(table_content)
                            + "\n"
                        )
                        markdown_content.append((table.bbox, table_content))
                        rects.append(fitz.Rect(table.bbox))
                        table_counter += 1

                    # Extract structured blocks (Text = 0, Image = 1)
                    page_dict = page.get_text("dict")  # type: ignore
                    blocks = page_dict.get("blocks", [])
                    for b_idx, block in enumerate(blocks):
                        block_rect = fitz.Rect(block["bbox"])
                        if any([block_rect.intersects(rect) for rect in rects]):
                            continue

                        # CASE 1: BLOCK IS TEXT
                        if block["type"] == 0:
                            block_content = f""
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    block_content += span["text"]
                                block_content += "\n"
                            # Break after block - paragraph
                            block_content += "\n"
                            markdown_content.append((block["bbox"], block_content))
                            continue

                        # CASE 2: BLOCK IS AN IMAGE
                        if block["type"] == 1:
                            anchor_id = f"img_p{page_number}_b{b_idx}"

                            # Insert Anchor exactly where the image sits in the flow
                            markdown_content.append(
                                (block["bbox"], f"> [image@{anchor_id}](#{anchor_id})")
                            )

                            # Store the metadata for the end of the file
                            # Note: block contains image binary/bbox info
                            desc = await self.handle_images_async(
                                doc, [block], page_number
                            )
                            image_metadata.append(f"### {anchor_id}\n")
                            desc = [f"{d}\n\n" for d in desc]
                            image_metadata.extend(desc)
                            continue

                    # Arrange markdown content
                    markdown_content.sort(key=lambda x: x[0][1], reverse=False)  # type: ignore
                    content: list[str] = list(map(lambda x: x[1], markdown_content))
                    markdown_content.clear()
                    page_content.extend(content)

                    # Append Image Descriptions at the Footer
                    if image_metadata:
                        page_content.append(f"## Image Reference Appendix\n")
                        page_content.extend(image_metadata)
                        image_metadata.clear()

                    # Handle Links separately (links usually don't have block types in dict)
                    links = await self.handle_links_async(page.get_links())  # type: ignore
                    if links:
                        page_content.append(f"## Link Reference Appendix\n")
                        page_content.extend(links)

                doc_content.append("\n".join(page_content))
                page_content.clear()
            return "".join(doc_content)
        except Exception as e:
            logger.error(f"[PDFLoader.load_async] => {str(e)}")
            raise e
