import os
import json
import logging
from math import ceil
from typing import Any

# External Packages
import openai
import tiktoken
from PIL import Image

# Internal Packages
from ..._util import CreatorRole, MessageBlock
from ..._tool import ToolMetadata

logger = logging.getLogger(__name__)


class OpenAICore:
    """`OpenAICore` is designed to be the base class for any `Core` class aiming to integrate with OpenAI's API.
    It offer functionality to check whether the desired model is offered by OpenAI.

    Methods:
    * __available(None) -> bool
    * build_profile(model_name: str) -> dict[str, bool | int | str]
    * calculate_token_count(
            msgs: list[MessageBlock | dict[str, Any]], tools: list[ToolMetadata] | None = None,
            images: list[str] | None = None, image_detail: str | None = None
        ) -> int
    * load_csv(cls, input_path: str) -> None
    * calculate_image_tokens(width: int, height: int) -> int
    * resize(image_path: str, detail: str) -> tuple[bool, str | None]
    * determine_valid_size(width: int, height: int) -> tuple[int, int]
    """

    csv_path: str | None = None

    def __init__(self, model_name: str):
        self.__model_name = model_name
        if not self.__available():
            raise ValueError("%s is not available in OpenAI's model listing.")

    def __available(self) -> bool:
        try:
            client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])
            for model in client.models.list():
                if self.__model_name == model.id:
                    return True
            return False
        except Exception as e:
            logger.error("Exception: %s", e)
            raise

    @staticmethod
    def build_profile(model_name: str) -> dict[str, bool | int | str]:
        """
        Build the profile dict based on information found in ./llm_agent_toolkit/core/open_ai/openai.csv

        These are the models which the developer has experience with.
        If `model_name` is not found in the csv file, default value will be applied.
        """
        profile: dict[str, bool | int | str] = {"name": model_name}

        # If OpenAI.csv_path is set
        if OpenAICore.csv_path:
            with open(OpenAICore.csv_path, "r", encoding="utf-8") as csv:
                header = csv.readline()
                columns = header.strip().split(",")
                while True:
                    line = csv.readline()
                    if not line:
                        break
                    values = line.strip().split(",")
                    if values[0] == model_name:
                        for column, value in zip(columns[1:], values[1:]):
                            if column in ["context_length", "max_output_tokens"]:
                                profile[column] = int(value)
                            elif column == "remarks":
                                profile[column] = value
                            elif value == "TRUE":
                                profile[column] = True
                            else:
                                profile[column] = False
                        break

        # If OpenAI.csv_path is not set
        # Assign default values
        if "text_generation" not in profile:
            # Assume supported
            profile["text_generation"] = True
        if profile["text_generation"]:
            if "context_length" not in profile:
                # Most supported context length
                profile["context_length"] = 4096
            if "tool" not in profile:
                # Assume supported
                profile["tool"] = True

        return profile

    @classmethod
    def load_csv(cls, input_path: str):
        COLUMNS_STRING = "name,context_length,max_output_tokens,text_generation,tool,text_input,image_input,audio_input,text_output,image_output,audio_output,remarks"
        EXPECTED_COLUMNS = set(COLUMNS_STRING.split(","))
        # Begin validation
        with open(input_path, "r", encoding="utf-8") as csv:
            header = csv.readline()
            header = header.strip()
            columns = header.split(",")
            # Expect no columns is missing
            diff = EXPECTED_COLUMNS.difference(set(columns))
            if diff:
                raise ValueError(f"Missing columns in {input_path}: {', '.join(diff)}")
            # Expect all columns are in exact order
            if header != COLUMNS_STRING:
                raise ValueError(
                    f"Invalid header in {input_path}: \n{header}\n{COLUMNS_STRING}"
                )

            for line in csv:
                values = line.strip().split(",")
                name: str = values[0]
                for column, value in zip(columns, values):
                    if column in ["name", "remarks"]:
                        assert isinstance(
                            value, str
                        ), f"{name}.{column} must be a string."
                    elif column in ["context_length", "max_output_tokens"] and value:
                        try:
                            _ = int(value)
                        except ValueError:
                            print(f"{name}.{column} must be an integer.")
                            raise
                    elif value:
                        assert value.lower() in [
                            "true",
                            "false",
                        ], f"{name}.{column} must be a boolean."
        # End validation
        OpenAICore.csv_path = input_path

    def calculate_token_count(
        self,
        msgs: list[MessageBlock | dict[str, Any]],
        tools: list[ToolMetadata] | None = None,
        images: list[str] | None = None,
        image_detail: str | None = None,
    ) -> int:
        """Calculate the token count for the given messages and tools.
        Call tiktoken to calculate the token count.

        Args:
            msgs (list[MessageBlock | dict[str, Any]]): A list of messages.
            tools (list[ToolMetadata] | None, optional): A list of tools. Defaults to None.
            images (list[str] | None): A list of images. Defaults to None.
            image_detail (str | None): The level of detail of the image. Defaults to None.

        Returns:
            int: The token count.
        """
        token_count: int = 0
        encoding = tiktoken.encoding_for_model(self.__model_name)
        for msg in msgs:
            # Incase the dict does not comply with the MessageBlock format
            if "content" in msg and msg["content"]:
                if not isinstance(msg["content"], list):
                    token_count += len(encoding.encode(msg["content"]))
                # Skip images
            if "role" in msg and msg["role"] == CreatorRole.FUNCTION.value:
                if "name" in msg:
                    token_count += len(encoding.encode(msg["name"]))

        if tools:
            for tool in tools:
                token_count += len(encoding.encode(json.dumps(tool)))

        if images:
            for image_path in images:
                if image_detail == "low":
                    token_count += 85
                else:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        token_count += self.calculate_image_tokens(width, height)
        return token_count

    @staticmethod
    def calculate_image_tokens(width: int, height: int) -> int:
        """
        Calculate the token needed to process the image.

        **Args:**
            width (int): Width of the image.
            height (int): Height of the image.

        **Returns:**
            int: The token needed to process the image.

        **Notes:**
        * https://platform.openai.com/docs/guides/vision/calculating-costs
        * TODO: Caching.
        """
        if width == 512 and height == 512:
            return 85

        _width, _height = OpenAICore.determine_valid_size(width, height)
        tiles_width = ceil(_width / 512)
        tiles_height = ceil(_height / 512)
        total_tokens = 85 + 170 * (tiles_width * tiles_height)
        return total_tokens

    @staticmethod
    def resize(image_path: str, detail: str = "low") -> tuple[bool, str | None]:
        """
        Resize the input image to OpenAI acceptable image size.

        **Args:**
            image_path (str): The path of the image.
            detail (str): Level of detail.

        **Returns:**
            tuple[bool, str | None]:
                * 1st element (bool): True if it was resized else False
                * new path (str | None): Path to the resized image if it was resized else None
        """
        with Image.open(image_path) as img:
            if detail == "low":
                suggested_size = (512, 512)
            else:
                width, height = img.size
                suggested_size = OpenAICore.determine_valid_size(width, height)
            if suggested_size != img.size:
                img.resize(size=suggested_size, resample=Image.Resampling.BILINEAR)
                basename = os.path.basename(image_path)
                newpath = f"resized_{basename}"
                img.save(newpath)
                return True, newpath
        return False, None

    @staticmethod
    def determine_valid_size(width: int, height: int) -> tuple[int, int]:
        """
        Determine an acceptable image size.

        **Args:**
            width (int): Width of the image.
            height (int): Height of the image.

        **Returns:**
            tuple[int, int]:
                * width (int): New width
                * height (int): New height

        **Notes:**
        * TODO: Caching.
        """
        if width > 2048 or height > 2048:
            aspect_ratio = width / height
            if aspect_ratio > 1:
                width, height = 2048, int(2048 / aspect_ratio)
            else:
                width, height = int(2048 * aspect_ratio), 2048

        if width >= height and height > 768:
            width, height = int((768 / height) * width), 768
        elif height > width and width > 768:
            width, height = 768, int((768 / width) * height)

        tiles_width = ceil(width / 512)
        tiles_height = ceil(height / 512)
        return (tiles_width, tiles_height)


TOOL_PROMPT = """
Utilize tools to solve the problems. 
Results from tools will be kept in the context. 
Calling the tools repeatedly is highly discouraged.
"""
