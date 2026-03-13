import io
import csv
import os
from typing import List, Tuple, Any
from contextlib import contextmanager

from .._core import ImageInterpreter, Core as TextModel, MessageBlock, TokenUsage


class CustomCSVHandler:

    @staticmethod
    def csv_to_markdown(
        content: str, delimiter: str = ",", has_header: bool = True
    ) -> str:
        """Convert CSV table (String) to Markdown table.

        Args:
            content (str): The CSV table to be converted.
            delimiter (str, optional): The character used to separate values. Defaults to ",".
            has_header (bool, optional): Whether the CSV table has a header row. Defaults to True.

        Returns:
            str: The resulting Markdown table.
        """
        import csv

        csv_reader = csv.reader(io.StringIO(content), delimiter=delimiter)
        rows_list = list(csv_reader)

        formatted_rows = []
        if has_header:
            header = [cell.replace("|", "\\|") for cell in rows_list[0]]
            formatted_rows.append(f"| {' | '.join(header)} |")
            formatted_rows.append(f"| {' | '.join(['---'] * len(header))} |")
            data = rows_list[1:]
        else:
            data = rows_list

        for row in data:
            clean_row = [cell.replace("|", "\\|") for cell in row]
            formatted_rows.append(f"| {' | '.join(clean_row)} |")

        return "\n".join(formatted_rows)

    @staticmethod
    def _rows_to_csv_string(rows: List[List[str]], delimiter: str) -> str:
        """Convert row values into CSV string.

        Args:
            rows (List[List[str]]): The data rows to be converted.
            delimiter (str): The character used to separate values.

        Returns:
            str: The resulting CSV string.
        """
        output = io.StringIO()
        writer = csv.writer(output, delimiter=delimiter)
        writer.writerows(rows)
        return output.getvalue()


class DefinedTask:

    @staticmethod
    @contextmanager
    def temporary_file(image_bytes: bytes, filename: str, tmp_directory: str):
        tmp_path = f"{tmp_directory}/{filename}"
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
    def interpret_image(
        model: ImageInterpreter, image_bytes: bytes, image_name: str, tmp_directory: str
    ) -> str:
        if model is None:
            return "Image description not available"

        query = f"Please describe this attached image."
        with DefinedTask.temporary_file(
            image_bytes, image_name, tmp_directory
        ) as tmp_path:
            ai_response: Tuple[List[MessageBlock | dict[str, Any]], TokenUsage] = (
                model.interpret(query=query, context=None, filepath=tmp_path)
            )
            responses, usage = ai_response
            if responses:
                return responses[0]["content"]

            raise RuntimeError("Expect at least one response on image interpretation.")

    @staticmethod
    def summarize_site(model: TextModel, url: str) -> str:
        if model is None:
            return "Web page summary not available"

        query = f"site={url}"
        ai_response = model.run(query, None)
        responses, usage = ai_response
        if responses:
            return responses[0]["content"]

        raise RuntimeError("Expect at least one response on site summarization.")


class DefinedTaskAsync:

    @staticmethod
    @contextmanager
    def temporary_file(image_bytes: bytes, filename: str, tmp_directory: str):
        tmp_path = f"{tmp_directory}/{filename}"
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
    async def interpret_image(
        model, image_bytes: bytes, image_name: str, tmp_directory: str
    ) -> str:
        if model is None:
            return "Image description not available"

        query = f"Please describe this attached image."
        with DefinedTaskAsync.temporary_file(
            image_bytes, image_name, tmp_directory
        ) as tmp_path:
            ai_response: Tuple[List[MessageBlock | dict[str, Any]], TokenUsage] = (
                await model.interpret_async(
                    query=query, context=None, filepath=tmp_path
                )
            )
            responses, usage = ai_response
            if responses:
                return responses[0]["content"]

            raise RuntimeError("Expect at least one response on image interpretation.")

    @staticmethod
    async def summarize_site(model: TextModel, url: str) -> str:
        if model is None:
            return "Web page summary not available"

        query = f"site={url}"
        ai_response = await model.run_async(query, None)
        responses, usage = ai_response
        if responses:
            return responses[0]["content"]

        raise RuntimeError("Expect at least one response on site summarization.")
