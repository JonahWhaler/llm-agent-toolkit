import os
import json
import logging
import openai
import tiktoken

from ..._util import CreatorRole, MessageBlock
from ..._tool import ToolMetadata

logger = logging.getLogger(__name__)


class OpenAICore:
    """`OpenAICore` is designed to be the base class for any `Core` class aiming to integrate with OpenAI's API.
    It offer functionality to check whether the desired model is offered by OpenAI.

    Methods:
    * __available(None) -> bool
    * build_profile(model_name: str) -> dict[str, bool | int | str]
    * calculate_token_count(msgs: list[MessageBlock | dict], tools: list[ToolMetadata] | None = None)
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
            diff = EXPECTED_COLUMNS - set(columns)
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
        self, msgs: list[MessageBlock | dict], tools: list[ToolMetadata] | None = None
    ) -> int:
        """Calculate the token count for the given messages and tools.
        Call tiktoken to calculate the token count.

        Args:
            msgs (list[MessageBlock | dict]): A list of messages.
            tools (list[ToolMetadata] | None, optional): A list of tools. Defaults to None.

        Returns:
            int: The token count.
        """
        token_count: int = 0
        encoding = tiktoken.encoding_for_model(self.__model_name)
        for msg in msgs:
            # Incase the dict does not comply with the MessageBlock format
            if "content" in msg and msg["content"]:
                token_count += len(encoding.encode(msg["content"]))
            if "role" in msg and msg["role"] == CreatorRole.FUNCTION.value:
                token_count += len(encoding.encode(msg["name"]))

        if tools:
            for tool in tools:
                token_count += len(encoding.encode(json.dumps(tool)))

        return token_count


TOOL_PROMPT = """
Utilize tools to solve the problems. 
Results from tools will be kept in the context. 
Calling the tools repeatedly is highly discouraged.
"""
