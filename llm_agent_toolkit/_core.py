from abc import abstractmethod, ABC

from ._util import (
    ChatCompletionConfig,
    ImageGenerationConfig,
    OpenAIMessage,
    ContextMessage
)


class Core(ABC):
    # TODO: Allow structured profile
    def __init__(
            self,
            system_prompt: str,
            model_name: str,
            config: ChatCompletionConfig | ImageGenerationConfig = ChatCompletionConfig(),
            tools: list | None = None
    ):
        self.__system_prompt = system_prompt
        self.__model_name = model_name
        self.__config = config
        self.__tools = tools

    @property
    def system_prompt(self):
        return self.__system_prompt

    @property
    def model_name(self):
        return self.__model_name

    @property
    def config(self):
        return self.__config

    @property
    def tools(self):
        return self.__tools

    @abstractmethod
    async def run_async(
            self,
            query: str,
            context: list[ContextMessage | dict] | None,
            **kwargs
    ) -> list[OpenAIMessage | dict]:
        raise NotImplementedError

    @abstractmethod
    def run(
            self,
            query: str,
            context: list[ContextMessage | dict] | None,
            **kwargs
    ) -> list[OpenAIMessage | dict]:
        raise NotImplementedError


class I2T_Core(ABC, Core):

    @staticmethod
    @abstractmethod
    def get_image_url(filepath: str) -> str:
        raise NotImplementedError

    @abstractmethod
    async def run_async(
            self,
            query: str,
            context: list[ContextMessage | dict] | None,
            **kwargs
    ) -> list[OpenAIMessage | dict]:
        raise NotImplementedError

    @abstractmethod
    def run(
            self,
            query: str,
            context: list[ContextMessage | dict] | None,
            **kwargs
    ) -> list[OpenAIMessage | dict]:
        raise NotImplementedError


class A2T_Core(ABC, Core):

    @staticmethod
    @abstractmethod
    def to_chunks(input_path: str, tmp_directory: str, config: dict) -> str:
        raise NotImplementedError
