import os
import io
import logging
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
from .._core import ImageInterpreter, Core as TextModel, MessageBlock, TokenUsage

logger = logging.getLogger(__name__)


class MsWordLoader(BaseLoader):
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

    def extract_img_description(self, image_bytes: bytes, image_name: str) -> str:
        if self.__image_interpreter is None:
            return "Image description not available"

        image_caption = (
            f"filename={image_name}. This is an attachment found in a pdf file."
        )
        with self.temporary_file(image_bytes, image_name) as tmp_path:
            ai_response: Tuple[List[MessageBlock | dict[str, Any]], TokenUsage] = (
                self.__image_interpreter.interpret(
                    query=image_caption, context=None, filepath=tmp_path
                )
            )
            responses, usage = ai_response
            if responses:
                return responses[0]["content"]

            raise RuntimeError("Expect at least one response on image interpretation.")
        
    async def extract_img_description_async(self, image_bytes: bytes, image_name: str) -> str:
        if self.__image_interpreter is None:
            return "Image description not available"

        image_caption = (
            f"filename={image_name}. This is an attachment found in a pdf file."
        )
        with self.temporary_file(image_bytes, image_name) as tmp_path:
            ai_response: Tuple[List[MessageBlock | dict[str, Any]], TokenUsage] = (
                await self.__image_interpreter.interpret_async(
                    query=image_caption, context=None, filepath=tmp_path
                )
            )
            responses, usage = ai_response
            if responses:
                return responses[0]["content"]

            raise RuntimeError("Expect at least one response on image interpretation.")
        
    def retrieve_site_summary(self, url: str) -> str:
        if self.__web_interpreter is None:
            return "Web page summary not available"

        query = f"site={url}"
        ai_response = self.__web_interpreter.run(query, None)
        responses, usage = ai_response
        if responses:
            return responses[0]["content"]

        raise RuntimeError("Expect at least one response on site summarization.")
    
    async def retrieve_site_summary_async(self, url: str) -> str:
        if self.__web_interpreter is None:
            return "Web page summary not available"

        query = f"site={url}"
        ai_response = await self.__web_interpreter.run_async(query, None)
        responses, usage = ai_response
        if responses:
            return responses[0]["content"]

        raise RuntimeError("Expect at least one response on site summarization.")
    
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
        assert input_path
        assert self.__tmp_directory

        mydoc: DocxDocument = docx.Document(input_path)

        image_counter = 1

        md_lines = []
        img_lines: list[str] = []
        link_lines: list[str] = []

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
                                site_summary = self.retrieve_site_summary(url)
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

                        image_description = self.extract_img_description(
                            image_bytes, image_name=image_filename
                        )

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
        assert input_path
        assert self.__tmp_directory

        mydoc: DocxDocument = docx.Document(input_path)

        image_counter = 1

        md_lines = []
        img_lines: list[str] = []
        link_lines: list[str] = []

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
                                site_summary = await self.retrieve_site_summary_async(url)
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

                        image_description = await self.extract_img_description_async(
                            image_bytes, image_name=image_filename
                        )

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
    