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

    Notes:
    - TODO: Allow structured profile
    """

    def __init__(
        self,
        system_prompt: str,
        config: ChatCompletionConfig | ImageGenerationConfig | TranscriptionConfig,
    ):
        self.__system_prompt = system_prompt
        self.__config = config

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the LLM model."""
        return self.__system_prompt

    @property
    def model_name(self) -> str:
        """Return the name of the LLM model."""
        return self.__config.name

    @property
    def profile(self) -> dict[str, bool | int | str]:
        return {}

    @property
    def config(
        self,
    ) -> (
        ModelConfig | ChatCompletionConfig | ImageGenerationConfig | TranscriptionConfig
    ):
        """Return the configuration for the LLM model."""
        return self.__config

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


class TextGenerator(ABC):
    @property
    @abstractmethod
    def context_length(self) -> int:
        raise NotImplementedError

    @context_length.setter
    @abstractmethod
    def context_length(self, value):
        raise NotImplementedError


class ToolSupport(ABC):
    def __init__(self, tools: list[Tool] | None = None):
        self.__tools = tools

    @property
    def tools(self) -> list[Tool] | None:
        """Return the tools available for the LLM model."""
        return self.__tools

    @abstractmethod
    def call_tools(self, selected_tools: list) -> list[MessageBlock | dict]:
        raise NotImplementedError

    @abstractmethod
    async def call_tools_async(self, selected_tools: list) -> list[MessageBlock | dict]:
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
    ):
        Core.__init__(
            self,
            system_prompt=system_prompt,
            config=config,
        )

    @staticmethod
    @abstractmethod
    def get_image_url(filepath: str) -> str:
        """Return the image url extracted from the file path."""
        raise NotImplementedError


class A2T_Core(Core, ABC):
    """
    Abstract class for audio-to-text LLM models, inherits from `Core`.

    Abstract methods:
    - run_async(query: str, context: list[ContextMessage | dict] | None, **kwargs) -> list[OpenAIMessage | dict]:
        Asynchronously run the LLM model with the given query and context.
    - run(query: str, context: list[ContextMessage | dict] | None, **kwargs) -> list[OpenAIMessage | dict]:
        Synchronously run the LLM model with the given query and context.
    """

    def __init__(
        self,
        system_prompt: str,
        config: TranscriptionConfig,
    ):
        assert isinstance(config, TranscriptionConfig)
        Core.__init__(self, system_prompt=system_prompt, config=config)
