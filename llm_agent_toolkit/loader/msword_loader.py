from llm_agent_toolkit._loader import BaseLoader
from llm_agent_toolkit._core import Core, I2T_Core
from llm_agent_toolkit._util import OpenAIMessage
import os
import warnings
from contextlib import contextmanager
from docx import Document
import zipfile
from io import StringIO, BytesIO
import re

"""
Dependencies:
----------
- python-docx==1.1.2
"""


class MsWordLoader(BaseLoader):
    def __init__(
        self,
        text_only: bool = True,
        tmp_directory: str | None = None,
        core: Core | None = None,
    ):
        self.__core = core
        self.__tmp_directory = tmp_directory
        if not text_only:
            assert isinstance(core, I2T_Core)
            if core.config.n != 1:
                warnings.warn(
                    "Configured to return {} responses from `core`. "
                    "Only first response will be used.".format(core.config.n)
                )
            if not os.path.exists(tmp_directory):
                warnings.warn(
                    "Temporary directory not exists. "
                    "Will create one with name: {}".format(tmp_directory)
                )
                os.makedirs(tmp_directory)

    @staticmethod
    def raise_if_invalid(input_path: str) -> None:
        if not all([input_path is not None, type(input_path) is str, input_path != ""]):
            raise ValueError("Invalid input path: Path must be a non-empty string.")

        if input_path[-5:] != ".docx":
            raise ValueError("Unsupported file format: Must be a DOCX file.")

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: '{input_path}'.")

    def load(self, input_path: str) -> str:
        MsWordLoader.raise_if_invalid(input_path)

        markdown_content = []

        doc = Document(input_path)

        # Handle text content
        markdown_content.extend(self.extract_text_content(doc))

        # Handle tables
        markdown_content.extend(self.extract_table_contet(doc))

        # Handle images
        markdown_content.extend(self.extract_image_content(input_path))

        return "\n".join(markdown_content)

    async def load_async(self, input_path: str) -> str:
        MsWordLoader.raise_if_invalid(input_path)

        markdown_content = []

        doc = Document(input_path)

        # Handle text content
        markdown_content.extend(self.extract_text_content(doc))

        # Handle tables
        markdown_content.extend(self.extract_table_contet(doc))

        # Handle images
        markdown_content.extend(await self.extract_image_content_async(input_path))

        return "\n".join(markdown_content)

    @staticmethod
    def extract_text_content(doc: Document) -> list[str]:
        markdown_content = []

        # Iterate through all elements in the document
        for para in doc.paragraphs:

            if para.style.name.startswith("Heading"):
                # Handle Headings
                level = int(re.search(r"\d+", para.style.name).group(0))
                markdown_content.append(f"{'#' * level} {para.text}")
            elif para.text.strip():
                # Handle regular paragraphs and text
                markdown_content.append(para.text)

        return markdown_content

    @staticmethod
    def extract_table_contet(doc: Document) -> list[str]:
        markdown_content = []

        # Extract tables
        for table in doc.tables:
            markdown_content.append("\n")
            for row_index, row in enumerate(table.rows):
                row_data = [
                    f"| {MsWordLoader.get_cell_content_with_formatting(cell)} "
                    for cell in row.cells
                ]
                markdown_content.append("".join(row_data) + "|")
                # Create a separator for header row (assuming first row is header)
                if row_index == 0:
                    header_separator = "".join(["| --- " for _ in row.cells])
                    markdown_content.append(header_separator + "|")
            # Add a newline after the table
            markdown_content.append("\n")

        if len(markdown_content) > 0:
            markdown_content.insert(0, f"\n# Tables\n")

        return markdown_content

    @staticmethod
    def get_cell_content_with_formatting(cell):
        content = StringIO()
        for para in cell.paragraphs:
            for run in para.runs:
                if run.bold:
                    content.write(f"**{run.text}**")
                elif run.italic:
                    content.write(f"*{run.text}*")
                else:
                    content.write(run.text)
            content.write("\n")
        return content.getvalue().strip()

    @staticmethod
    def extract_alt_text_dict(docx: zipfile.ZipFile) -> dict[str, str]:
        """
        Extracts alt text for images in a DOCX file.

        Parameters:
        ----------
        - docx: zipfile.ZipFile: The ZipFile object representing the DOCX file

        Returns:
        ----------
        * dict[str, str]: Dictionary mapping image file names to their alt text

        """
        from xml.etree import ElementTree as ET

        image_alt_texts = {}
        # Parse the XML document to get image alt text
        if "word/document.xml" in docx.namelist():
            document_xml = docx.read("word/document.xml")
            root = ET.fromstring(document_xml)

            # Find all elements with cNvPr to identify inserted images

            for elem in root.iter():
                if elem.tag.endswith("cNvPr"):
                    r_id = elem.attrib.get("id")
                    alt_text = elem.attrib.get("descr", "Alt text not available")
                    ele_name = elem.attrib.get("name", str(r_id))  # Picture {index}
                    if ele_name and alt_text:
                        image_alt_texts[ele_name] = (
                            alt_text  # Add alt text if available
                        )

        return image_alt_texts

    @staticmethod
    def get_alt_by_name(d: dict[str, str], key1: str, key2: str) -> str:
        # This is needed because `extract_alt_text_dict` keys follow the pattern `Picture {index}` or `{index}`
        # key1: `Picture {index}`
        # key2: `{index}`
        return d.get(key1, d.get(key2, "Alt text not available"))

    def extract_image_content(self, input_path: str) -> list[str]:
        markdown_content = []

        with zipfile.ZipFile(input_path, "r") as docx:
            image_alt_texts: dict[str, str] = self.extract_alt_text_dict(docx)

            # Iterate through the files in the archive
            file_startswith_word_media_lst = list(
                filter(lambda f: f.startswith("word/media/"), docx.namelist())
            )
            for counter, file in enumerate(file_startswith_word_media_lst, start=1):
                # Extract the corresponding alt text
                # Assumption: Images are captured in the same order as `extract_alt_text_dict`
                alt_text = self.get_alt_by_name(
                    image_alt_texts, key1=f"Picture {counter}", key2=str(counter)
                )

                if self.__core is None:
                    image_description = "Image description not available"
                else:
                    image_data = docx.read(file)
                    image_name = os.path.basename(file)  # image{index}.png
                    with self.temporary_file(image_data, image_name) as image_path:
                        responses: list[OpenAIMessage | dict] = self.__core.run(
                            query="Describe this image",
                            context=None,
                            filepath=image_path,
                        )
                        if isinstance(responses[0], OpenAIMessage):
                            image_description = responses[0].content
                        elif isinstance(responses[0], dict):
                            image_description = responses[0]["content"]
                markdown_content.append(
                    f"## {os.path.basename(file)}\nDescription: {image_description}\n\nAlt Text: {alt_text}\n"
                )
        if len(markdown_content) > 0:
            markdown_content.insert(0, f"\n# Images\n")

        return markdown_content

    async def extract_image_content_async(self, input_path: str) -> list[str]:
        markdown_content = []

        with zipfile.ZipFile(input_path, "r") as docx:
            image_alt_texts: dict[str, str] = self.extract_alt_text_dict(docx)

            # Iterate through the files in the archive
            file_startswith_word_media_lst = list(
                filter(lambda f: f.startswith("word/media/"), docx.namelist())
            )
            for counter, file in enumerate(file_startswith_word_media_lst, start=1):
                # Extract the corresponding alt text
                # Assumption: Images are captured in the same order as `extract_alt_text_dict`
                alt_text = self.get_alt_by_name(
                    image_alt_texts, key1=f"Picture {counter}", key2=str(counter)
                )

                if self.__core is None:
                    image_description = "Image description not available"
                else:
                    image_data = docx.read(file)
                    image_name = os.path.basename(file)  # image{index}.png
                    with self.temporary_file(image_data, image_name) as image_path:
                        responses: list[OpenAIMessage | dict] = await self.__core.run_async(
                            query="Describe this image",
                            context=None,
                            filepath=image_path,
                        )
                        if isinstance(responses[0], OpenAIMessage):
                            image_description = responses[0].content
                        elif isinstance(responses[0], dict):
                            image_description = responses[0]["content"]
                markdown_content.append(
                    f"## {os.path.basename(file)}\nDescription: {image_description}\n\nAlt Text: {alt_text}\n"
                )
        if len(markdown_content) > 0:
            markdown_content.insert(0, f"\n# Images\n")

        return markdown_content

    @contextmanager
    def temporary_file(self, image_bytes: bytes, filename: str):
        tmp_path = f"{self.__tmp_directory}/{filename}"
        try:
            image_stream = BytesIO(image_bytes)
            image_stream.seek(0)
            with open(tmp_path, "wb") as f:
                f.write(image_bytes)
            yield tmp_path
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @contextmanager
    def increment_later(self, counter: int):
        yield counter
        return counter + 1
