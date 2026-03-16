import os
import io
import logging
import mimetypes
from typing import Optional, Tuple, List, Any
from contextlib import contextmanager

# python-docx
import docx
from docx.document import Document as DocxDocument

# from docx.image.exceptions import UnrecognizedImageError
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph

from .._loader import BaseLoader
from .._core import ImageInterpreter, Core as TextModel
from .._util import TokenUsage, UsagePurpose
from .utils import DefinedTask, DefinedTaskAsync
from functools import reduce

logger = logging.getLogger(__name__)


class MsWordLoader(BaseLoader):
    """
    A loader for parsing MS Word files and extracting text, tables, images, and links.

    `MsWordLoader` uses the `python-docx` library to process MS Word files, offering both synchronous (`load`)
    and asynchronous (`load_async`) methods to extract content into a Markdown format.

    It preserves the document structure by iterating through every children objects.
    Followed by rendering them according to their type.

    ## Sub-features:
    - **Image Interpretation**: If an `ImageInterpreter` is provided, the loader extracts images,
      generates descriptions using the interpreter, and appends them in a reference section.
      Image locations are marked with anchors in the main text.

    - **Web Page Summarization**: If a `TextModel` (acting as a `web_interpreter`) is provided,
      it summarizes external links found in the MS Word document and includes the summaries in a
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
    - load(input_path: str) -> str: Synchronously processes the specified MS Word file and returns its
      content as a Markdown string.
    - load_async(input_path: str) -> str: Asynchronously processes the specified MS Word file and
      returns its content as a Markdown string.

    Notes:
    ----------
    - This loader relies on `python-docx` (`docx`). Ensure it is installed.
    - For image and link processing, the corresponding interpreter models (`ImageInterpreter`, `TextModel`)
      must be properly configured and provided during initialization.
    """

    SUPPORTED_MIMETYPES: Tuple[str] = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    def __init__(
        self,
        text_only: bool = True,
        tmp_directory: str | None = None,
        image_interpreter: ImageInterpreter | None = None,
        text_model: TextModel | None = None,
    ) -> None:
        # self.__text_only = text_only
        self.__image_interpreter = image_interpreter
        self.__web_interpreter = text_model
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

    @staticmethod
    def iter_block_items(document):
        """
        Generate a sequence of Paragraph and Table objects in document order.
        This is the key to iterating through the document's content blocks
        from top to bottom.
        """
        # The document's body element contains all the block-level content.
        parent_elm = document.element.body
        if parent_elm is None:
            return
        # Iterate through the children of the body element.
        for child in parent_elm.iterchildren():
            # A child can be a paragraph...
            if isinstance(child, CT_P):
                yield DocxParagraph(child, document)
            # ...or a table.
            elif isinstance(child, CT_Tbl):
                yield DocxTable(child, document)

    def load(self, input_path: str) -> str:
        MsWordLoader.pre_load(input_path)
        assert input_path
        assert self.__tmp_directory

        mydoc: DocxDocument = docx.Document(input_path)

        image_counter = 1

        md_lines = []
        img_lines: List[str] = []
        link_lines: List[str] = []

        rels = mydoc.part.rels
        for block in self.iter_block_items(mydoc):
            if isinstance(block, DocxParagraph):
                # Extract inlinne hyperlinks via xml xpath
                paragraph_text = block.text

                for hyperlink in block._p.xpath(".//w:hyperlink"):
                    rId = hyperlink.get(
                        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                    )
                    if rId and rId in rels:
                        rel = rels[rId]
                        # Ensure it's an external URL, not an internal document bookmark
                        if "hyperlink" in rel.reltype:
                            url = rel.target_ref
                            # Extract the visible text inside the hyperlink tag
                            link_text = "".join(
                                node.text
                                for node in hyperlink.xpath(".//w:t")
                                if node.text
                            )

                            if link_text and url:
                                # Pragmatic inline replacement: swap the raw text for Markdown
                                paragraph_text = paragraph_text.replace(
                                    link_text, f"[{link_text}]({url})", 1
                                )
                                if self.__web_interpreter:
                                    site_summary, usage = DefinedTask.summarize_site(
                                        self.__web_interpreter, url
                                    )
                                    self.log_usage(usage)
                                else:
                                    site_summary = "Web page summary not available"
                                link_lines.append(f"### URL: {url}")
                                link_lines.append(f"**Summary**: \n{site_summary}\n")

                md_lines.append(f"\n{paragraph_text}\n")
                # Images are contained within paragraphs as inline shapes within runs
                for run in block.runs:
                    # Find all blips (which contain image references) in the run
                    for rId in run.element.xpath(".//a:blip/@r:embed"):
                        # Get the image part from the document's part
                        image_part = mydoc.part.related_parts[rId]

                        # The image part has the image data and content type
                        image_bytes = image_part.blob
                        content_type = image_part.content_type

                        # Determine the file extension and create a filename
                        extension = content_type.split("/")[-1]
                        image_filename = f"image{image_counter}.{extension}"
                        image_counter += 1

                        md_lines.append(f"![{image_filename}](#)")

                        if self.__image_interpreter is None:
                            image_description = "Image description not available"
                        else:
                            image_description, usage = DefinedTask.interpret_image(
                                self.__image_interpreter,
                                image_bytes,
                                image_filename,
                                self.__tmp_directory,
                            )
                            self.log_usage(usage)

                        img_lines.append(f"### {image_filename}")
                        img_lines.append(f"**Description**: \n{image_description}\n")

            elif isinstance(block, DocxTable):
                md_lines.append("TABLE:")
                for i, row in enumerate(block.rows):
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)

                    md_lines.append(row_text)
                    if i == 0:
                        md_lines.append("| --- " * len(row.cells) + "|")

        output_string = "\n".join(md_lines)
        if len(img_lines):
            output_string += "\n## Image Reference Appendix\n"
            output_string += "\n".join(img_lines)

        if len(link_lines):
            output_string += "\n## Link Reference Appendix\n"
            output_string += "\n".join(link_lines)

        return output_string

    async def load_async(self, input_path: str) -> str:
        MsWordLoader.pre_load(input_path)
        assert input_path
        assert self.__tmp_directory

        mydoc: DocxDocument = docx.Document(input_path)

        image_counter = 1

        md_lines = []
        img_lines: List[str] = []
        link_lines: List[str] = []

        rels = mydoc.part.rels
        for block in self.iter_block_items(mydoc):
            if isinstance(block, DocxParagraph):
                # Extract inlinne hyperlinks via xml xpath
                paragraph_text = block.text

                for hyperlink in block._p.xpath(".//w:hyperlink"):
                    rId = hyperlink.get(
                        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                    )
                    if rId and rId in rels:
                        rel = rels[rId]
                        # Ensure it's an external URL, not an internal document bookmark
                        if "hyperlink" in rel.reltype:
                            url = rel.target_ref
                            # Extract the visible text inside the hyperlink tag
                            link_text = "".join(
                                node.text
                                for node in hyperlink.xpath(".//w:t")
                                if node.text
                            )

                            if link_text and url:
                                # Pragmatic inline replacement: swap the raw text for Markdown
                                paragraph_text = paragraph_text.replace(
                                    link_text, f"[{link_text}]({url})", 1
                                )
                                if self.__web_interpreter is None:
                                    site_summary = "Web page summary not available"
                                else:
                                    site_summary, usage = (
                                        await DefinedTaskAsync.summarize_site(
                                            self.__web_interpreter, url
                                        )
                                    )
                                    self.log_usage(usage)

                                link_lines.append(f"### URL: {url}")
                                link_lines.append(f"**Summary**: \n{site_summary}\n")

                md_lines.append(f"\n{paragraph_text}\n")
                # Images are contained within paragraphs as inline shapes within runs
                for run in block.runs:
                    # Find all blips (which contain image references) in the run
                    for rId in run.element.xpath(".//a:blip/@r:embed"):
                        # Get the image part from the document's part
                        image_part = mydoc.part.related_parts[rId]

                        # The image part has the image data and content type
                        image_bytes = image_part.blob
                        content_type = image_part.content_type

                        # Determine the file extension and create a filename
                        extension = content_type.split("/")[-1]
                        image_filename = f"image{image_counter}.{extension}"
                        image_counter += 1

                        md_lines.append(f"![{image_filename}](#)")

                        if self.__image_interpreter is None:
                            image_description = "Image description not available"
                        else:
                            image_description, usage = (
                                await DefinedTaskAsync.interpret_image(
                                    self.__image_interpreter,
                                    image_bytes,
                                    image_filename,
                                    self.__tmp_directory,
                                )
                            )
                            self.log_usage(usage)
                        img_lines.append(f"### {image_filename}")
                        img_lines.append(f"**Description**: \n{image_description}\n")

            elif isinstance(block, DocxTable):
                md_lines.append("TABLE:")
                for i, row in enumerate(block.rows):
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)

                    md_lines.append(row_text)
                    if i == 0:
                        md_lines.append("| --- " * len(row.cells) + "|")

        output_string = "\n".join(md_lines)
        if len(img_lines):
            output_string += "\n## Image Reference Appendix\n"
            output_string += "\n".join(img_lines)

        if len(link_lines):
            output_string += "\n## Link Reference Appendix\n"
            output_string += "\n".join(link_lines)

        return output_string
