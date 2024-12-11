import logging
import ollama

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
    """

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
        Build the profile dict based on information found in ./llm_agent_toolkit/core/local/ollama.csv

        These are the models which the developer has experience with.
        If `model_name` is not found in the csv file, default value will be applied.

        Call .set_context_length to set the context length, default value is 2048.
        """
        logger.info("Building profile...")
        profile: dict[str, bool | int | str] = {"name": model_name}
        with open(
            "./llm_agent_toolkit/core/local/ollama.csv", "r", encoding="utf-8"
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
                        if column == "context_length":
                            profile[column] = int(value)
                        elif column == "remarks":
                            profile[column] = value
                        elif value == "TRUE":
                            profile[column] = True
                        else:
                            profile[column] = False
                    break

        if "context_length" not in profile:
            # Most supported context length
            profile["context_length"] = 2048
        if "tool" not in profile:
            # Assume supported
            profile["tool"] = True
        if "text_generation" not in profile:
            # Assume supported
            profile["text_generation"] = True
        logger.info("Profile ready")
        return profile