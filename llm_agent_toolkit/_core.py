from abc import abstractmethod, ABC
import openai
import os
from ._util import (
    ChatCompletionConfig, ImageGenerationConfig, OpenAIMessage, OpenAIFunction,
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
