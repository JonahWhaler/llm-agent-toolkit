import json
import logging
from typing import Any, Optional, Type, TypeVar
import ollama
from pydantic import BaseModel

from ..._core import Core
from ..._util import (
    CreatorRole,
    ChatCompletionConfig,
    MessageBlock,
    ResponseMode,
    TokenUsage,
)
from .base import OllamaCore

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class T2TSO_OLM_Core(Core, OllamaCore):
    """
    `T2TSO_OLM_Core` is a child class of `OllamaCore` and a concrete implementation of abstract base class `Core`.

    This class facilitates both synchronous and asynchronous communication with Ollama's API, enabling interaction
    with large language models (LLMs) for text generation tasks. It is particularly suited for chaining operations
    where structured outputs are required.

    Methods:
    - run(query: str, context: list[MessageBlock | dict[str, Any]] | None, **kwargs) -> tuple[list[MessageBlock | dict[str, Any]], TokenUsage]:
        Synchronously run the LLM model with the given query and context.
    - run_async(query: str, context: list[MessageBlock | dict[str, Any]] | None, **kwargs) -> tuple[list[MessageBlock | dict[str, Any]], TokenUsage]:
        Asynchronously run the LLM model with the given query and context.

    Notes:
    - The caller is responsible for memory management, output parsing and error handling.
    - If model is not available locally, an attempt to pull it from Ollama's server will be made.
    - The caller is responsible for choosing models that support `Structured Ouput`.
    - Configurable Parameters: `context_length` and `max_output_tokens`.
    - Best suited for chaining operations where structured data flow is essential.
    - https://ollama.com/blog/structured-outputs
    """

    def __init__(
        self,
        connection_string: str,
        system_prompt: str,
        config: ChatCompletionConfig,
    ):
        Core.__init__(self, system_prompt, config)
        OllamaCore.__init__(self, connection_string, config.name)
        self.profile = self.build_profile(model_name=config.name)

    def validate(
        self, response_mode: Optional[ResponseMode], response_format: Optional[Type[T]]
    ) -> None:
        if response_mode:
            if not isinstance(response_mode, ResponseMode):
                raise TypeError(
                    f"Expect mode to be an instance of 'ResponseMode', but got '{type(response_mode).__name__}'."
                )
            if response_mode is response_mode.SO:
                if response_format is None:
                    raise TypeError(
                        "Expect format to be a subclass of 'BaseModel', but got 'NoneType'."
                    )
                if not issubclass(response_format, BaseModel):
                    raise TypeError(
                        f"Expect format to be a subclass of 'BaseModel', but got '{type(response_format).__name__}'."
                    )

    async def run_async(
        self,
        query: str,
        context: list[dict | MessageBlock] | None,
        **kwargs,
    ) -> tuple[list[MessageBlock | dict[str, Any]], TokenUsage]:
        """
        Asynchronously generate text based on the given query and context.

        Args:
            query (str): The query to generate text for.
            context (list): A list of context messages or dictionaries.
            **kwargs: Additional keyword arguments:
                * `mode` (ResponseMode | None): Ouput mode.
                * `format` (BaseModel | None): Output structure.

        Returns:
            list[MessageBlock|dict]:
                * The list of messages generated by the LLM model.
                    Only 1 element in the list, it's content can be decoded by `json.loads()`.
            TokenUsage: The recorded token usage.

        Raises:
            TypeError:
                * `mode` is not type `ResponseMode`.
                * `format` is not a subclass of `BaseModel` when `mode` is `ResponseMode.SO`.
            ValueError:
                * If max_output_tokens <= 0. Theres no capacity left for text-generation.
                * If `filepath` is not a supported image format.
        """
        response_mode: Optional[ResponseMode] = kwargs.get("mode", ResponseMode.DEFAULT)
        response_format: Optional[Type[T]] = kwargs.get("format")  # type: ignore
        self.validate(response_mode, response_format)  # Raise an exception if invalid

        msgs: list[dict[str, Any] | MessageBlock] = [
            {"role": CreatorRole.SYSTEM.value, "content": self.system_prompt}
        ]

        if context:
            msgs.extend(context)
        msgs.append({"role": CreatorRole.USER.value, "content": query})

        # Determine the maximum number of tokens allowed for the response
        MAX_TOKENS = min(self.config.max_tokens, self.context_length)
        MAX_OUTPUT_TOKENS = min(
            MAX_TOKENS, self.max_output_tokens, self.config.max_output_tokens
        )
        prompt_token_count = self.calculate_token_count(msgs, None, None)
        max_output_tokens = min(
            MAX_OUTPUT_TOKENS,
            self.context_length - prompt_token_count,
        )

        try:
            client = ollama.AsyncClient(host=self.CONN_STRING)
            if max_output_tokens <= 0:
                logger.warning("Prompt token count: %d", prompt_token_count)
                raise ValueError("max_output_tokens <= 0")

            if response_mode is ResponseMode.SO and response_format:
                response = await client.chat(
                    model=self.model_name,
                    messages=msgs,
                    tools=None,
                    stream=False,
                    format=response_format.model_json_schema(),
                    options={
                        "temperature": self.config.temperature,
                        "num_predict": max_output_tokens,
                    },
                )
            elif response_mode is ResponseMode.JSON:
                response = await client.chat(
                    model=self.model_name,
                    messages=msgs,
                    tools=None,
                    stream=False,
                    format="json",
                    options={
                        "temperature": self.config.temperature,
                        "num_predict": max_output_tokens,
                    },
                )
            else:
                # response_mode is ResponseMode.DEFAULT
                response = await client.chat(
                    model=self.model_name,
                    messages=msgs,
                    tools=None,
                    stream=False,
                    options={
                        "temperature": self.config.temperature,
                        "num_predict": max_output_tokens,
                    },
                )

            llm_generated_content: str = response["message"]["content"]

            token_usage = self.update_usage(response)  # type: ignore

            if llm_generated_content:
                if response_mode is not ResponseMode.DEFAULT:
                    try:
                        # Validate JSON format
                        _ = json.loads(llm_generated_content)
                        content = llm_generated_content
                    except json.JSONDecodeError as decode_error:
                        e = {"error": str(decode_error), "text": llm_generated_content}
                        content = json.dumps(e)
                else:
                    content = llm_generated_content

                return [
                    {"role": CreatorRole.ASSISTANT.value, "content": content}
                ], token_usage
            raise RuntimeError("Content not available.")
        except Exception as e:
            logger.error("Exception: %s", e, exc_info=True, stack_info=True)
            raise

    def run(
        self, query: str, context: list[MessageBlock | dict[str, Any]] | None, **kwargs
    ) -> tuple[list[MessageBlock | dict[str, Any]], TokenUsage]:
        """
        Synchronously generate text based on the given query and context.

        Args:
            query (str): The query to generate text for.
            context (list): A list of context messages or dictionaries.
            **kwargs: Additional keyword arguments:
                * `mode` (ResponseMode | None): Ouput mode.
                * `format` (BaseModel | None): Output structure.

        Returns:
            list[MessageBlock|dict]:
                * The list of messages generated by the LLM model.
                    Only 1 element in the list, it's content can be decoded by `json.loads()`.
            TokenUsage: The recorded token usage.

        Raises:
            TypeError:
                * `mode` is not type `ResponseMode`.
                * `format` is not a subclass of `BaseModel` when `mode` is `ResponseMode.SO`.
            ValueError:
                * If max_output_tokens <= 0. Theres no capacity left for text-generation.
                * If `filepath` is not a supported image format.
        """
        response_mode: Optional[ResponseMode] = kwargs.get("mode", ResponseMode.DEFAULT)
        response_format: Optional[Type[T]] = kwargs.get("format")  # type: ignore
        self.validate(response_mode, response_format)  # Raise an exception if invalid

        msgs: list[dict[str, Any] | MessageBlock] = [
            {"role": CreatorRole.SYSTEM.value, "content": self.system_prompt}
        ]

        if context:
            msgs.extend(context)
        msgs.append({"role": CreatorRole.USER.value, "content": query})

        # Determine the maximum number of tokens allowed for the response
        MAX_TOKENS = min(self.config.max_tokens, self.context_length)
        MAX_OUTPUT_TOKENS = min(
            MAX_TOKENS, self.max_output_tokens, self.config.max_output_tokens
        )
        prompt_token_count = self.calculate_token_count(msgs, None, None)
        max_output_tokens = min(
            MAX_OUTPUT_TOKENS,
            self.context_length - prompt_token_count,
        )

        try:
            client = ollama.Client(host=self.CONN_STRING)
            if max_output_tokens <= 0:
                logger.warning("Prompt token count: %d", prompt_token_count)
                raise ValueError("max_output_tokens <= 0")

            if response_mode is ResponseMode.SO and response_format:
                response = client.chat(
                    model=self.model_name,
                    messages=msgs,
                    tools=None,
                    stream=False,
                    format=response_format.model_json_schema(),
                    options={
                        "temperature": self.config.temperature,
                        "num_predict": max_output_tokens,
                    },
                )
            elif response_mode is ResponseMode.JSON:
                response = client.chat(
                    model=self.model_name,
                    messages=msgs,
                    tools=None,
                    stream=False,
                    format="json",
                    options={
                        "temperature": self.config.temperature,
                        "num_predict": max_output_tokens,
                    },
                )
            else:
                # response_mode is ResponseMode.DEFAULT
                response = client.chat(
                    model=self.model_name,
                    messages=msgs,
                    tools=None,
                    stream=False,
                    options={
                        "temperature": self.config.temperature,
                        "num_predict": max_output_tokens,
                    },
                )

            llm_generated_content: str = response["message"]["content"]

            token_usage = self.update_usage(response)  # type: ignore

            if llm_generated_content:
                if response_mode is not ResponseMode.DEFAULT:
                    try:
                        # Validate JSON format
                        _ = json.loads(llm_generated_content)
                        content = llm_generated_content
                    except json.JSONDecodeError as decode_error:
                        e = {"error": str(decode_error), "text": llm_generated_content}
                        content = json.dumps(e)
                else:
                    content = llm_generated_content

                return [
                    {"role": CreatorRole.ASSISTANT.value, "content": content}
                ], token_usage
            raise RuntimeError("Content not available.")
        except Exception as e:
            logger.error("Exception: %s", e, exc_info=True, stack_info=True)
            raise
