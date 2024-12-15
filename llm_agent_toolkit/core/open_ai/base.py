import os
import logging
import openai

logger = logging.getLogger(__name__)


class OpenAICore:
    """`OpenAICore` is designed to be the base class for any `Core` class aiming to integrate with OpenAI's API.
    It offer functionality to check whether the desired model is offered by OpenAI.

    Methods:
    * __available(None) -> bool
    * build_profile(model_name: str) -> dict[str, bool | int | str]
    """

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
        with open(
            "./llm_agent_toolkit/core/open_ai/openai.csv", "r", encoding="utf-8"
        ) as csv:
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


TOOL_PROMPT = """
Utilize tools to solve the problems. 
Results from tools will be kept in the context. 
Calling the tools repeatedly is highly discouraged.
"""
