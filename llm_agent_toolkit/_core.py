"""
Description:
- This file defines the abstract base class `Core`, which is the core of the LLM agent toolkit.
- `I2T_Core` and `A2T_Core` are the abstract subclasses of `Core` for image-to-text and audio-to-text LLM models respectively.
"""

from abc import abstractmethod, ABC

from ._util import (
    ChatCompletionConfig,
    ModelConfig,
    ImageGenerationConfig,
    MessageBlock,
    TranscriptionConfig,
)
from ._tool import Tool


class Core(ABC):
    """
    Abstract base class for the core of the LLM agent toolkit.

    Attr:
    - system_prompt: str: The system prompt for the LLM model.
    - model_name: str: The name of the LLM model.
    - config: ChatCompletionConfig | ImageGenerationConfig: The configuration for the LLM model.
    - tools: list: The tools available for the LLM model.

    Notes:
    - TODO: Allow structured profile
    """

    def __init__(
        self,
        system_prompt: str,
        config: ChatCompletionConfig | ImageGenerationConfig | TranscriptionConfig,
        tools: list[Tool] | None = None,
    ):
        self.__system_prompt = system_prompt
        self.__config = config
        self.__tools = tools

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the LLM model."""
        return self.__system_prompt

    @property
    def model_name(self) -> str:
        """Return the name of the LLM model."""
        return self.__config.name

    @property
    def config(
        self,
    ) -> (
        ModelConfig | ChatCompletionConfig | ImageGenerationConfig | TranscriptionConfig
    ):
        """Return the configuration for the LLM model."""
        return self.__config

    @property
    def tools(self) -> list[Tool] | None:
        """Return the tools available for the LLM model."""
        return self.__tools

    @abstractmethod
    async def run_async(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        """Asynchronously run the LLM model with the given query and context."""
        raise NotImplementedError

    @abstractmethod
    def run(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        """Synchronously run the LLM model with the given query and context."""
        raise NotImplementedError


class I2T_Core(Core, ABC):
    """
    Abstract class for image-to-text LLM models, inherits from `Core`.

    Abstract methods:
    - get_image_url(filepath: str) -> str:
        Returns the URL of the image from the specified file path.
    - run_async(query: str, context: list[ContextMessage | dict] | None, **kwargs) -> list[OpenAIMessage | dict]:
        Asynchronously run the LLM model with the given query and context.
    - run(query: str, context: list[ContextMessage | dict] | None, **kwargs) -> list[OpenAIMessage | dict]:
        Synchronously run the LLM model with the given query and context.
    """

    def __init__(
        self,
        system_prompt: str,
        config: ChatCompletionConfig,
        tools: list | None = None,
    ):
        Core.__init__(
            self,
            system_prompt=system_prompt,
            config=config,
            tools=tools,
        )

    @staticmethod
    @abstractmethod
    def get_image_url(filepath: str) -> str:
        """Return the image url extracted from the file path."""
        raise NotImplementedError

    @abstractmethod
    async def run_async(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        """Asynchronously run the LLM model with the given query and context."""
        raise NotImplementedError

    @abstractmethod
    def run(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        """Synchronously run the LLM model with the given query and context."""
        raise NotImplementedError


class A2T_Core(Core, ABC):
    """
    Abstract class for audio-to-text LLM models, inherits from `Core`.

    Abstract methods:
    - to_chunks(input_path: str, tmp_directory: str, config: dict) -> str:
        Split the audio file into multiple chunks.
    - run_async(query: str, context: list[ContextMessage | dict] | None, **kwargs) -> list[OpenAIMessage | dict]:
        Asynchronously run the LLM model with the given query and context.
    - run(query: str, context: list[ContextMessage | dict] | None, **kwargs) -> list[OpenAIMessage | dict]:
        Synchronously run the LLM model with the given query and context.
    """

    def __init__(
        self,
        system_prompt: str,
        config: TranscriptionConfig,
        tools: list | None = None,
    ):
        Core.__init__(
            self,
            system_prompt=system_prompt,
            config=config,
            tools=tools,
        )

    @abstractmethod
    async def run_async(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        """Asynchronously run the LLM model with the given query and context."""
        raise NotImplementedError

    @abstractmethod
    def run(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        """Synchronously run the LLM model with the given query and context."""
        raise NotImplementedError
