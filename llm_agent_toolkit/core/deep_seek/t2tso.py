import os
import logging
import json
import time
import asyncio
from typing import Any, Optional, TypeVar
from copy import deepcopy

# from math import ceil

import openai
from openai import RateLimitError
from pydantic import BaseModel

from ..._core import Core
from ..._util import (
    CreatorRole,
    ChatCompletionConfig,
    MessageBlock,
    ResponseMode,
    TokenUsage,
)

from .base import DeepSeekCore

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class T2TSO_DS_Core(Core, DeepSeekCore):
    """
    `T2TSO_DS_Core` is a concrete implementation of abstract class `Core` and `DeepSeekCore`
    to provide chat completion capabilities with JSON output support.

    It includes methods for running asynchronous and synchronous
    execution, and handling retries with progressive backoff in case of errors.

    Attributes:
        MAX_ATTEMPT (int): The maximum number of retry attempts for API calls.
        DELAY_FACTOR (float): The factor by which the delay increases after each retry.
        MAX_DELAY (float): The maximum delay between retries.

    **Methods**:
        run_async(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> tuple[list[MessageBlock | dict], TokenUsage]:
            Asynchronously runs the LLM model with the given query and context, handling
            retries and token usage.
        run(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> tuple[list[MessageBlock | dict], TokenUsage]:
            Synchronously runs the LLM model with the given query and context, handling
            retries and token usage.

    **Notes**:
        - The class supports progressive backoff for retrying API calls in case of
          RateLimitError or other exceptions.
        - If structured JSON output is expected:
            * Define expected structure in the prompt
            * Set mode as ResponseMode.JSON
        - config.max_iteration hardcode as 1
    """

    MAX_ATTEMPT: int = 5
    DELAY_FACTOR: float = 1.5
    MAX_DELAY: float = 60.0

    def __init__(self, system_prompt: str, config: ChatCompletionConfig):
        Core.__init__(self, system_prompt, config)
        DeepSeekCore.__init__(self)
        self.profile = self.build_profile()

    def validate(
        self,
        response_mode: Optional[ResponseMode],
        # response_format: Optional[Type[T]] | None = None,
    ) -> None:
        if response_mode:
            if not isinstance(response_mode, ResponseMode):
                raise TypeError(
                    f"Expect mode to be an instance of 'ResponseMode', but got '{type(response_mode).__name__}'."
                )
            if response_mode is response_mode.SO:
                raise ValueError("Deepseek does not support ResponseMode.SO.")
                # if response_format is None:
                #     raise TypeError(
                #         "Expect format to be a subclass of 'BaseModel', but got 'NoneType'."
                #     )
                # if not issubclass(response_format, BaseModel):
                #     raise TypeError(
                #         f"Expect format to be a subclass of 'BaseModel', but got '{type(response_format).__name__}'."
                #     )

    async def run_async(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> tuple[list[MessageBlock | dict], TokenUsage]:
        response_mode: Optional[ResponseMode] = kwargs.get("mode", ResponseMode.DEFAULT)
        # response_format: Optional[Type[T]] = kwargs.get("format")  # type: ignore
        # Raise an exception if invalid
        self.validate(response_mode)

        msgs: list[MessageBlock | dict[str, Any]] = [
            {"role": CreatorRole.SYSTEM.value, "content": self.system_prompt}
        ]

        if context:
            msgs.extend(context)

        msgs.append(MessageBlock(role=CreatorRole.USER.value, content=query))

        # Determine the maximum number of tokens allowed for the response
        MAX_TOKENS = min(self.config.max_tokens, self.context_length)
        MAX_OUTPUT_TOKENS = min(
            MAX_TOKENS, self.max_output_tokens, self.config.max_output_tokens
        )

        attempt: int = 1
        delay: float = 5.0

        while attempt < T2TSO_DS_Core.MAX_ATTEMPT:
            logger.debug("Attempt %d", attempt)
            messages = deepcopy(msgs)
            prompt_token_count = self.calculate_token_count(msgs, None)
            max_output_tokens = min(
                MAX_OUTPUT_TOKENS,
                self.context_length - prompt_token_count,
            )
            if max_output_tokens <= 0:
                raise ValueError(
                    f"max_output_tokens <= 0. Prompt token count: {prompt_token_count}"
                )
            try:
                client = openai.AsyncOpenAI(
                    api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url=os.environ["DEEPSEEK_BASE_URL"],
                )

                if response_mode is ResponseMode.JSON:
                    response = await client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,  # type: ignore
                        frequency_penalty=0.5,
                        max_tokens=max_output_tokens,
                        temperature=self.config.temperature,
                        n=self.config.return_n,
                        response_format={"type": "json_object"},  # type: ignore
                    )
                else:
                    # response_mode is ResponseMode.DEFAULT
                    response = await client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,  # type: ignore
                        frequency_penalty=0.5,
                        max_tokens=max_output_tokens,
                        temperature=self.config.temperature,
                        n=self.config.return_n,
                    )

                token_usage = self.update_usage(response.usage)

                choice = response.choices[0]
                finish_reason = choice.finish_reason
                content = choice.message.content
                if finish_reason == "stop" and content:
                    if response_mode is not ResponseMode.DEFAULT:
                        try:
                            _ = json.loads(content)
                            output_string = content
                        except json.JSONDecodeError as decode_error:
                            e = {"error": str(decode_error), "text": content}
                            output_string = json.dumps(e, ensure_ascii=False)
                    else:
                        output_string = content

                    output: list[dict | MessageBlock] = [
                        MessageBlock(
                            role=CreatorRole.ASSISTANT.value, content=output_string
                        )
                    ]
                    return output, token_usage

                if finish_reason == "length" and content:
                    e = {"error": "Early Termination: Length", "text": content}
                    output: list[dict | MessageBlock] = [
                        MessageBlock(
                            role=CreatorRole.ASSISTANT.value,
                            content=json.dumps(e, ensure_ascii=False),
                        )
                    ]
                    return output, token_usage

                logger.warning("Malformed response: %s", response)
                logger.warning("Config: %s", self.config)
                raise RuntimeError(f"Terminated: {finish_reason}")
            except RateLimitError as rle:
                logger.warning("RateLimitError: %s", rle)
                warn_msg = f"[{attempt}] Retrying in {delay} seconds..."
                logger.warning(warn_msg)
                await asyncio.sleep(delay)
                attempt += 1
                delay = delay * T2TSO_DS_Core.DELAY_FACTOR
                delay = min(T2TSO_DS_Core.MAX_DELAY, delay)
                continue
            except Exception as e:
                logger.error("Exception: %s", e, exc_info=True, stack_info=True)
                raise

        raise RuntimeError("Max re-attempt reached")

    def run(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> tuple[list[MessageBlock | dict], TokenUsage]:
        response_mode: Optional[ResponseMode] = kwargs.get("mode", ResponseMode.DEFAULT)
        # response_format: Optional[Type[T]] = kwargs.get("format")  # type: ignore
        # Raise an exception if invalid
        self.validate(response_mode)

        msgs: list[MessageBlock | dict[str, Any]] = [
            {"role": CreatorRole.SYSTEM.value, "content": self.system_prompt}
        ]

        if context:
            msgs.extend(context)

        msgs.append(MessageBlock(role=CreatorRole.USER.value, content=query))

        # Determine the maximum number of tokens allowed for the response
        MAX_TOKENS = min(self.config.max_tokens, self.context_length)
        MAX_OUTPUT_TOKENS = min(
            MAX_TOKENS, self.max_output_tokens, self.config.max_output_tokens
        )

        attempt: int = 1
        delay: float = 5.0

        while attempt < T2TSO_DS_Core.MAX_ATTEMPT:
            logger.debug("Attempt %d", attempt)
            messages = deepcopy(msgs)
            prompt_token_count = self.calculate_token_count(msgs, None)
            max_output_tokens = min(
                MAX_OUTPUT_TOKENS,
                self.context_length - prompt_token_count,
            )
            if max_output_tokens <= 0:
                raise ValueError(
                    f"max_output_tokens <= 0. Prompt token count: {prompt_token_count}"
                )
            try:
                client = openai.OpenAI(
                    api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url=os.environ["DEEPSEEK_BASE_URL"],
                )

                if response_mode is ResponseMode.JSON:
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,  # type: ignore
                        frequency_penalty=0.5,
                        max_tokens=max_output_tokens,
                        temperature=self.config.temperature,
                        n=self.config.return_n,
                        response_format={"type": "json_object"},  # type: ignore
                    )
                else:
                    # response_mode is ResponseMode.DEFAULT
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,  # type: ignore
                        frequency_penalty=0.5,
                        max_tokens=max_output_tokens,
                        temperature=self.config.temperature,
                        n=self.config.return_n,
                    )

                token_usage = self.update_usage(response.usage)

                choice = response.choices[0]
                finish_reason = choice.finish_reason
                content = choice.message.content
                if finish_reason == "stop" and content:
                    if response_mode is not ResponseMode.DEFAULT:
                        try:
                            _ = json.loads(content)
                            output_string = content
                        except json.JSONDecodeError as decode_error:
                            e = {"error": str(decode_error), "text": content}
                            output_string = json.dumps(e)
                    else:
                        output_string = content

                    output: list[dict | MessageBlock] = [
                        MessageBlock(
                            role=CreatorRole.ASSISTANT.value, content=output_string
                        )
                    ]
                    return output, token_usage

                if finish_reason == "length" and content:
                    e = {"error": "Early Termination: Length", "text": content}
                    output: list[dict | MessageBlock] = [
                        MessageBlock(
                            role=CreatorRole.ASSISTANT.value,
                            content=json.dumps(e, ensure_ascii=False),
                        )
                    ]
                    return output, token_usage

                logger.warning("Malformed response: %s", response)
                logger.warning("Config: %s", self.config)
                raise RuntimeError(f"Terminated: {finish_reason}")

            except RateLimitError as rle:
                logger.warning("RateLimitError: %s", rle)
                warn_msg = f"[{attempt}] Retrying in {delay} seconds..."
                logger.warning(warn_msg)
                time.sleep(delay)
                attempt += 1
                delay = delay * T2TSO_DS_Core.DELAY_FACTOR
                delay = min(T2TSO_DS_Core.MAX_DELAY, delay)
                continue
            except Exception as e:
                logger.error("Exception: %s", e, exc_info=True, stack_info=True)
                raise

        raise RuntimeError("Max re-attempt reached")
