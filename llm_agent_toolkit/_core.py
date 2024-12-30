"""
Abstract Base Classes: `Core`
Interfaces: `ImageInterpreter` and `ToolSupport`.

Assumptions:
* Any classes that inherit `Core` are capable of handling chat completion.
"""

from abc import abstractmethod, ABC
from typing import Any
from ._util import ChatCompletionConfig, MessageBlock
from ._tool import Tool


class Core(ABC):
    """
    Abstract base class for the core of the LLM agent toolkit.

    Attr:
    - system_prompt: str: The system prompt for the LLM model.
    - model_name: str: The name of the LLM model.
    - config: ChatCompletionConfig: The configuration for the LLM model, it define the boundary of the execution.
    - profile: The profile of the chosen LLM model, it define the boundary of the LLM.
    """

    def __init__(
        self,
        system_prompt: str,
        config: ChatCompletionConfig,
    ):
        self.__system_prompt = system_prompt
        self.__config = config
        self.__profile: dict[str, bool | int | str] = {}

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
        """
        Profile is mostly for view purpose only,
        except the context_length which might be used to control the input to the LLM.
        """
        return self.__profile

    @profile.setter
    def profile(self, value: dict[str, bool | int | str]):
        self.__profile = value

    @property
    def config(
        self,
    ) -> ChatCompletionConfig:
        """Return the configuration for the LLM model."""
        return self.__config

    @property
    def context_length(self) -> int:
        ret = self.profile.get("context_length", 2048)
        assert isinstance(ret, int)
        return ret

    @context_length.setter
    def context_length(self, value: int):
        """
        Set the context length.
        It shall be the user's responsiblity to ensure this is a model supported context length.

        Args:
            context_length (int): Context length to be set.

        Returns:
            None

        Raises:
            TypeError: If context_length is not type int.
            ValueError: If context_length is <= 0.
        """
        if not isinstance(value, int):
            raise TypeError(
                f"Expect context_length to be type 'int', got '{type(value).__name__}'."
            )
        if value <= 0:
            raise ValueError("Expect context_length > 0.")

        self.profile["context_length"] = value

    @property
    def max_output_tokens(self) -> int:
        ret = self.profile.get("max_output_tokens", 2048)
        assert isinstance(ret, int)
        return ret

    @max_output_tokens.setter
    def max_output_tokens(self, value: int):
        """
        Set the max output tokens.
        It shall be the user's responsiblity to ensure this is a model supported max output tokens.

        Args:
            max_output_tokens (int): Max output tokens to be set.

        Returns:
            None

        Raises:
            TypeError: If max_output_tokens is not type int.
            ValueError: If max_output_tokens is <= 0.
        """
        if not isinstance(value, int):
            raise TypeError(
                f"Expect max_output_tokens to be type 'int', got '{type(value).__name__}'."
            )
        if value <= 0:
            raise ValueError("Expect max_output_tokens > 0.")

        self.profile["max_output_tokens"] = value

    @abstractmethod
    async def run_async(
        self, query: str, context: list[MessageBlock | dict[str, Any]] | None, **kwargs
    ) -> list[MessageBlock | dict[str, Any]]:
        """Asynchronously run the LLM model with the given query and context."""
        raise NotImplementedError

    @abstractmethod
    def run(
        self, query: str, context: list[MessageBlock | dict[str, Any]] | None, **kwargs
    ) -> list[MessageBlock | dict[str, Any]]:
        """Synchronously run the LLM model with the given query and context."""
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


class ImageInterpreter(ABC):
    """
    Abstract class for image-to-text LLM models.

    Abstract methods:
    - interpret_async(query: str, context: list[ContextMessage | dict] | None, filepath: str, **kwargs) -> list[OpenAIMessage | dict]:
        Asynchronously run the LLM model to interpret the image in `filepath`.
    - interpret(query: str, context: list[ContextMessage | dict] | None, filepath: str, **kwargs) -> list[OpenAIMessage | dict]:
        Synchronously run the LLM model to interpret the image in `filepath`.
    """

    @abstractmethod
    async def interpret_async(
        self,
        query: str,
        context: list[MessageBlock | dict[str, Any]] | None,
        filepath: str,
        **kwargs,
    ) -> list[MessageBlock | dict[str, Any]]:
        """Asynchronously run the LLM model to interpret the image in `filepath`.

        Use this method to explicitly express the intention to interpret an image.
        """
        raise NotImplementedError

    @abstractmethod
    def interpret(
        self,
        query: str,
        context: list[MessageBlock | dict[str, Any]] | None,
        filepath: str,
        **kwargs,
    ) -> list[MessageBlock | dict[str, Any]]:
        """Synchronously run the LLM model to interpret the image in `filepath`.

        Use this method to explicitly express the intention to interpret an image.
        """
        raise NotImplementedError
