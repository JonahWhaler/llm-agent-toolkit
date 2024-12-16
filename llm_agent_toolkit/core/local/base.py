import json
import logging
import ollama

from ..._util import CreatorRole, MessageBlock
from ..._tool import ToolMetadata

logger = logging.getLogger(__name__)


class OllamaCore:
    """`OllamaCore` is designed to be the base class for any `Core` class aiming to itegrate with LLM through Ollama.
    It offer functionality to pull the desired model from Ollama's server if it's not available locally.

    Attributes:
    * CONN_STRING (str)

    Methods:
    * __available(None) -> bool
    * __try_pull_model(None) -> None
    * build_profile(model_name: str) -> dict[str, bool | int | str]
    * calculate_token_count(msgs: list[MessageBlock | dict], tools: list[ToolMetadata] | None = None)
    """

    csv_path: str | None = None

    def __init__(self, connection_string: str, model_name: str):
        self.__connection_string = connection_string
        self.__model_name = model_name
        if not self.__available():
            self.__try_pull_model()

    @property
    def CONN_STRING(self) -> str:
        return self.__connection_string

    def __available(self) -> bool:
        try:
            client = ollama.Client(host=self.CONN_STRING)
            lst = list(client.list())[0]
            _, m, *_ = lst
            for _m in m:
                if _m.model == self.__model_name:
                    logger.info("Found %s => %s", self.__model_name, _m)
                    return True
            return False
        except ollama.RequestError as ore:
            logger.error("RequestError: %s", str(ore))
            raise
        except Exception as e:
            logger.error("Exception: %s", str(e))
            raise

    def __try_pull_model(self):
        """
        Attempt to pull the required model from ollama's server.

        **Raises:**
            ollama.ResponseError: pull model manifest: file does not exist
        """
        try:
            client = ollama.Client(host=self.CONN_STRING)
            _ = client.pull(self.__model_name, stream=False)
        except ollama.RequestError as oreqe:
            logger.error("RequestError: %s", str(oreqe))
            raise
        except ollama.ResponseError as orespe:
            logger.error("ResponseError: %s", str(orespe))
            raise
        except Exception as e:
            logger.error("Exception: %s (%s)", str(e), type(e))
            raise

    @staticmethod
    def build_profile(model_name: str) -> dict[str, bool | int | str]:
        """
        Build the profile dict based on information found OllamaCore.csv_path

        These are the models which the developer has experience with.
        If `model_name` is not found in the csv file, default value will be applied.
        """
        profile: dict[str, bool | int | str] = {"name": model_name}
        # If OllamaCore.csv_path is set
        if OllamaCore.csv_path:
            with open(OllamaCore.csv_path, "r", encoding="utf-8") as csv:
                header = csv.readline()
                columns = header.strip().split(",")
                while True:
                    line = csv.readline()
                    if not line:
                        break
                    values = line.strip().split(",")
                    if values[0] == model_name:
                        for column, value in zip(columns[1:], values[1:]):
                            if column == "context_length":
                                profile[column] = int(value)
                            elif column == "max_output_tokens":
                                profile[column] = 2048 if value == "" else int(value)
                            elif column == "remarks":
                                profile[column] = value
                            elif value == "TRUE":
                                profile[column] = True
                            else:
                                profile[column] = False
                        break
        # If OllamaCore.csv_path is not set or some fields are missing
        # Assign default values
        if "context_length" not in profile:
            # Most supported context length
            profile["context_length"] = 2048
        if "tool" not in profile:
            # Assume supported
            profile["tool"] = True
        if "text_generation" not in profile:
            # Assume supported
            profile["text_generation"] = True

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
        OllamaCore.csv_path = input_path

    def calculate_token_count(
        self, msgs: list[MessageBlock | dict], tools: list[ToolMetadata] | None = None
    ) -> int:
        """Calculate the token count for the given messages and tools.
        Efficient but not accurate. Child classes should implement a more accurate version.

        Args:
            msgs (list[MessageBlock | dict]): A list of messages.
            tools (list[ToolMetadata] | None, optional): A list of tools. Defaults to None.

        Returns:
            int: The token count. Number of characters divided by 2.

        Notes:
        * Decided to divide by 2 because my usecase most like involve using utf-8 encoding.
        """
        character_count: int = 0
        for msg in msgs:
            # Incase the dict does not comply with the MessageBlock format
            if "content" in msg and msg["content"]:
                character_count += len(msg["content"])
            if "role" in msg and msg["role"] == CreatorRole.TOOL.value:
                character_count += len(msg["name"])

        if tools:
            for tool in tools:
                character_count += len(json.dumps(tool))

        return character_count // 2


TOOL_PROMPT = """
Utilize tools to solve the problems. 
Results from tools will be kept in the context. 
Calling the tools repeatedly is highly discouraged.
"""
