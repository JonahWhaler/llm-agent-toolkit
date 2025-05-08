import os
import logging
import json
import time
import asyncio
from typing import Any, Optional, Type, TypeVar
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

# External Packages
from google import genai
from google.genai import types
from pydantic import BaseModel

# Self Defined Packages
from ..._util import (
    CreatorRole,
    ChatCompletionConfig,
    MessageBlock,
    ResponseMode,
    TokenUsage,
)
from ..._core import Core, ImageInterpreter
from .base import GeminiCore

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class GMN_StructuredOutput_Core(Core, GeminiCore, ImageInterpreter):
    """
    `GMN_StructuredOutput_Core` is a multimodal interface with image interpretation
    and structured output support.

    It provides a unified interface to work with Gemini LLM model.

    Attributes:
        SUPPORTED_IMAGE_FORMATS (tuple[str]): A tuple of supported image formats.
        MAX_ATTEMPT (int): The maximum number of retry attempts for API calls.
        DELAY_FACTOR (float): The factor by which the delay increases after each retry.
        MAX_DELAY (float): The maximum delay between retries.

    **Methods**:
        run(query: str, context: list[MessageBlock | dict[str, Any]] | None, **kwargs) -> tuple[list[MessageBlock | dict], TokenUsage]:
            Synchronously run the LLM model with the given query and context.
        run_async(query: str, context: list[MessageBlock | dict[str, Any]] | None, **kwargs) -> tuple[list[MessageBlock | dict], TokenUsage]:
            Asynchronously run the LLM model with the given query and context.
        interpret(query: str, context: list[MessageBlock | dict[str, Any]] | None, filepath: str, **kwargs) -> tuple[list[MessageBlock | dict], TokenUsage]:
            Synchronously interpret the given image.
        interpret_async(query: str, context: list[MessageBlock | dict[str, Any]] | None, filepath: str, **kwargs) -> tuple[list[MessageBlock | dict], TokenUsage]:
            Asynchronously interpret the given image.

    **Features**:
    - Image interpretation:
        PNG, JPEG, JPG, WEBP
    - Structured output:
        May define the expected structure in the prompt or provide it as response_schema.

    **Notes**:
    - Only supports single-turn execution.
    - Input: Text & Image only
    - Output: Text only
    - Best suited for chaining operations where structured data flow is essential.
    - https://ai.google.dev/gemini-api/docs/structured-output
    """

    SUPPORTED_IMAGE_FORMATS = (".png", ".jpeg", ".jpg", ".webp")
    MAX_ATTEMPT: int = 5
    DELAY_FACTOR: float = 1.5
    MAX_DELAY: float = 60.0

    def __init__(
        self,
        system_prompt: str,
        config: ChatCompletionConfig,
    ):
        Core.__init__(self, system_prompt, config)
        GeminiCore.__init__(self, config.name)
        self.profile = self.build_profile(model_name=config.name)

    def custom_config(
        self,
        max_output_tokens: int,
        response_mode: Optional[ResponseMode] = None,
        response_format: Optional[Type[T]] = None,
    ) -> types.GenerateContentConfig:
        """Adapter function.

        Transform custom ChatCompletionConfig -> types.GenerationContentConfig
        """
        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=self.config.temperature,
            max_output_tokens=max_output_tokens,
        )
        if response_mode == ResponseMode.SO:
            config.response_mime_type = "application/json"
            config.response_schema = response_format
        elif response_mode == ResponseMode.JSON:
            config.response_mime_type = "application/json"
        return config

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

    def run(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> tuple[list[MessageBlock | dict], TokenUsage]:
        """
        Synchronously run the LLM model with the given query and context.

        Args:
            query (str): The query to be processed by the LLM model.
            context (list[MessageBlock | dict] | None): The context to be used for the LLM model.
            **kwargs: Additional keyword arguments.
                * `filepath` (str | None): Path to the image file.
                * `mode` (ResponseMode | None): Ouput mode.
                * `format` (BaseModel | None): Output structure.

        Returns:
            output (tuple[list[MessageBlock | dict], TokenUsage]):
                1. The list of messages generated by the LLM model.
                2. The recorded token usage.

        **Notes**:
        * Single-turn execution.
        """
        response_mode: Optional[ResponseMode] = kwargs.get("mode", ResponseMode.DEFAULT)
        response_format: Optional[Type[T]] = kwargs.get("format")  # type: ignore
        self.validate(response_mode, response_format)  # Raise an exception if invalid

        filepath: str | None = kwargs.get("filepath", None)
        if filepath:
            ext = os.path.splitext(filepath)[-1]
            if ext not in GMN_StructuredOutput_Core.SUPPORTED_IMAGE_FORMATS:
                raise ValueError(f"Unsupported image type: {ext}")

        msgs: list[types.Content] = self.preprocessing(query, context, filepath)

        MAX_TOKENS = min(self.config.max_tokens, self.context_length)
        MAX_OUTPUT_TOKENS = min(
            MAX_TOKENS, self.max_output_tokens, self.config.max_output_tokens
        )
        attempt: int = 1
        delay: float = 5.0
        while attempt < GMN_StructuredOutput_Core.MAX_ATTEMPT:
            logger.debug("Attempt %d", attempt)
            messages = deepcopy(msgs)
            prompt_token_count = self.calculate_token_count(
                self.model_name,
                self.system_prompt,
                messages,
                imgs=None if filepath is None else [filepath],
            )
            max_output_tokens = min(
                MAX_OUTPUT_TOKENS,
                self.context_length - prompt_token_count,
            )

            config = self.custom_config(
                max_output_tokens, response_mode, response_format
            )
            try:
                client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=messages,  # type: ignore
                    config=config,
                )

                token_usage = self.update_usage(
                    response.usage_metadata, token_usage=None
                )

                candidates: Optional[list[types.Candidate]] = response.candidates
                if candidates is None or len(candidates) == 0:
                    raise RuntimeError(
                        f"Malformed response (No candidates): {response}"
                    )

                candidate = candidates[0]
                finish_reason = candidate.finish_reason
                content: Optional[types.Content] = candidate.content
                if content is None:
                    raise RuntimeError(f"Malformed response (No content): {candidate}")

                texts = self.get_texts(content)
                if finish_reason == types.FinishReason.STOP and texts:
                    messages.append(
                        types.Content(
                            role=CreatorRole.MODEL.value,
                            parts=[types.Part.from_text(text=text) for text in texts],
                        )
                    )
                elif finish_reason == types.FinishReason.MAX_TOKENS and texts:
                    logger.warning("Terminated due to length.")
                    e = {
                        "error": "Early Termination: Length",
                        "text": "\n".join(texts),
                    }
                    messages.append(
                        types.Content(
                            role=CreatorRole.MODEL.value,
                            parts=[
                                types.Part.from_text(
                                    text=json.dumps(e, ensure_ascii=False)
                                )
                            ],
                        )
                    )
                else:
                    logger.warning("Malformed response: %s", response)
                    logger.warning("Config: %s", self.config)
                    raise RuntimeError(f"Terminated: {finish_reason}")

                output = self.postprocessing(messages[-1:])
                return output, token_usage
            except Exception as e:
                if "502 Bad Gateway" in str(e):
                    logger.warning("RateLimitError: %s", e)
                    warn_msg = f"[{attempt}] Retrying in {delay} seconds..."
                    logger.warning(warn_msg)
                    time.sleep(delay)
                    attempt += 1
                    delay = delay * GMN_StructuredOutput_Core.DELAY_FACTOR
                    delay = min(GMN_StructuredOutput_Core.MAX_DELAY, delay)
                    continue

                raise

        raise RuntimeError("Max re-attempt reached")

    async def run_async(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> tuple[list[MessageBlock | dict], TokenUsage]:
        """
        Synchronously run the LLM model with the given query and context.

        Args:
            query (str): The query to be processed by the LLM model.
            context (list[MessageBlock | dict] | None): The context to be used for the LLM model.
            **kwargs: Additional keyword arguments.
                * `filepath` (str | None): Path to the image file.
                * `mode` (ResponseMode | None): Ouput mode.
                * `format` (BaseModel | None): Output structure.

        Returns:
            output (tuple[list[MessageBlock | dict], TokenUsage]):
                1. The list of messages generated by the LLM model.
                2. The recorded token usage.

        **Notes**:
        * Single-turn execution.
        """
        response_mode: Optional[ResponseMode] = kwargs.get("mode", ResponseMode.DEFAULT)
        response_format: Optional[Type[T]] = kwargs.get("format")  # type: ignore
        self.validate(response_mode, response_format)  # Raise an exception if invalid

        filepath: str | None = kwargs.get("filepath", None)
        if filepath:
            ext = os.path.splitext(filepath)[-1]
            if ext not in GMN_StructuredOutput_Core.SUPPORTED_IMAGE_FORMATS:
                raise ValueError(f"Unsupported image type: {ext}")

        msgs: list[types.Content] = self.preprocessing(query, context, filepath)

        MAX_TOKENS = min(self.config.max_tokens, self.context_length)
        MAX_OUTPUT_TOKENS = min(
            MAX_TOKENS, self.max_output_tokens, self.config.max_output_tokens
        )
        attempt: int = 1
        delay: float = 5.0
        while attempt < GMN_StructuredOutput_Core.MAX_ATTEMPT:
            logger.debug("Attempt %d", attempt)
            messages = deepcopy(msgs)
            prompt_token_count = self.calculate_token_count(
                self.model_name,
                self.system_prompt,
                messages,
                imgs=None if filepath is None else [filepath],
            )
            max_output_tokens = min(
                MAX_OUTPUT_TOKENS,
                self.context_length - prompt_token_count,
            )

            config = self.custom_config(
                max_output_tokens, response_mode, response_format
            )
            try:
                response = await self.acall(self.model_name, config, messages)

                token_usage = self.update_usage(
                    response.usage_metadata, token_usage=None
                )

                candidates: Optional[list[types.Candidate]] = response.candidates
                if candidates is None or len(candidates) == 0:
                    raise RuntimeError(
                        f"Malformed response (No candidates): {response}"
                    )

                candidate = candidates[0]
                finish_reason = candidate.finish_reason
                content: Optional[types.Content] = candidate.content
                if content is None:
                    raise RuntimeError(f"Malformed response (No content): {candidate}")

                texts = self.get_texts(content)
                if finish_reason == types.FinishReason.STOP and texts:
                    messages.append(
                        types.Content(
                            role=CreatorRole.MODEL.value,
                            parts=[types.Part.from_text(text=text) for text in texts],
                        )
                    )
                elif finish_reason == types.FinishReason.MAX_TOKENS and texts:
                    logger.warning("Terminated due to length.")
                    e = {
                        "error": "Early Termination: Length",
                        "text": "\n".join(texts),
                    }
                    messages.append(
                        types.Content(
                            role=CreatorRole.MODEL.value,
                            parts=[
                                types.Part.from_text(
                                    text=json.dumps(e, ensure_ascii=False)
                                )
                            ],
                        )
                    )
                else:
                    logger.warning("Malformed response: %s", response)
                    logger.warning("Config: %s", self.config)
                    raise RuntimeError(f"Terminated: {finish_reason}")

                output = self.postprocessing(messages[-1:])
                return output, token_usage
            except Exception as e:
                if "502 Bad Gateway" in str(e):
                    logger.warning("RateLimitError: %s", e)
                    warn_msg = f"[{attempt}] Retrying in {delay} seconds..."
                    logger.warning(warn_msg)
                    time.sleep(delay)
                    attempt += 1
                    delay = delay * GMN_StructuredOutput_Core.DELAY_FACTOR
                    delay = min(GMN_StructuredOutput_Core.MAX_DELAY, delay)
                    continue

                raise

        raise RuntimeError("Max re-attempt reached")

    @staticmethod
    async def acall(
        model_name: str, config: types.GenerateContentConfig, msgs: list[types.Content]
    ):
        """Use this to make the `generate_content` method asynchronous."""
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        with ThreadPoolExecutor() as executor:
            future = executor.submit(
                client.models.generate_content,
                model=model_name,
                contents=msgs,  # type: ignore
                config=config,
            )
            response = await asyncio.wrap_future(future)  # Makes the future awaitable
            return response

    def interpret(
        self,
        query: str,
        context: list[MessageBlock | dict[str, Any]] | None,
        filepath: str,
        **kwargs,
    ) -> tuple[list[MessageBlock | dict], TokenUsage]:
        return self.run(query=query, context=context, filepath=filepath, **kwargs)

    async def interpret_async(
        self,
        query: str,
        context: list[MessageBlock | dict[str, Any]] | None,
        filepath: str,
        **kwargs,
    ) -> tuple[list[MessageBlock | dict], TokenUsage]:
        return await self.run_async(
            query=query, context=context, filepath=filepath, **kwargs
        )
