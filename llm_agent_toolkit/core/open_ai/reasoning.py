import os
import logging
import json
import time
import asyncio
from typing import Any
from copy import deepcopy

import openai
from openai import RateLimitError

from ..._core import Core
from ..._util import CreatorRole, ChatCompletionConfig, MessageBlock, TokenUsage
from .base import OpenAICore

logger = logging.getLogger(__name__)


class Reasoning_Core(Core, OpenAICore):
    """
    `Reasoning_Core` is a chat completion core for models with reasoning capabilities.

    It is designed to work with OpenAI's API and only limited to models that support reasoning.

    It includes methods for running asynchronous and synchronous execution, and
    handling retries with progressive backoff in case of errors.

    Attributes:
        SUPPORTED_MODELS (tuple[str]): A tuple of supported model names.
        SUPPORTED_EFFORT (tuple[str]): A tuple of supported reasoning efforts.
        MAX_ATTEMPT (int): The maximum number of retry attempts for API calls.
        DELAY_FACTOR (float): The factor by which the delay increases after each retry.
        MAX_DELAY (float): The maximum delay between retries.

    Methods:
    - run(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> tuple[list[MessageBlock | dict], TokenUsage]:
        Synchronously run the LLM model with the given query and context.
    - run_async(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> tuple[list[MessageBlock | dict], TokenUsage]:
        Asynchronously run the LLM model with the given query and context.

    **Notes**:
    - Backend URL: https://api.openai.com
    - `temperature` has to be 1.0
    - o1-mini is deprecated: https://platform.openai.com/docs/models/o1-mini
    """

    SUPPORTED_MODELS = ("o4-mini", "o3-mini")
    SUPPORTED_EFFORT = ("low", "medium", "high")
    MAX_ATTEMPT: int = 5
    DELAY_FACTOR: float = 1.5
    MAX_DELAY: float = 60.0

    def __init__(
        self,
        system_prompt: str,
        config: ChatCompletionConfig,
        reasoning_effort: str = "low",
    ):
        if config.name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"{config.name} is not supported. Supported models: {self.SUPPORTED_MODELS}"
            )
        if config.temperature != 1.0:
            raise ValueError("temperature has to be 1.0")

        Core.__init__(self, system_prompt, config)
        OpenAICore.__init__(self, config.name)
        self.profile = self.build_profile(config.name)
        if reasoning_effort not in self.SUPPORTED_EFFORT:
            raise ValueError(
                f"{reasoning_effort} is not supported. Supported effort: {self.SUPPORTED_EFFORT}"
            )
        self.reasoning_effort = reasoning_effort

    async def run_async(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> tuple[list[MessageBlock | dict[str, Any]], TokenUsage]:
        """
        Asynchronously run the LLM model with the given query and context.

        Args:
            query (str): The query to be processed by the LLM model.
            context (list[MessageBlock | dict] | None): The context to be used for the LLM model.
            **kwargs: Additional keyword arguments.

        Returns:
            response_tuple (tuple[list[MessageBlock | dict], TokenUsage]):
            * output: The list of messages generated by the LLM model.
            * usage: The recorded token usage.

        **Notes**:
        * system_prompt: Provided in USER role.
        * max_tokens: max_completion_tokens
        * temperature: Hardcode as 1.0
        * frequency_penalty: Not supported
        * reasoning_effort: Not all reasoning models support this parameter
        """
        msgs: list[MessageBlock | dict] = [
            MessageBlock(role=CreatorRole.USER.value, content=self.system_prompt)
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

        while attempt < Reasoning_Core.MAX_ATTEMPT:
            logger.debug("Attempt %d", attempt)
            messages = deepcopy(msgs)
            prompt_token_count = self.calculate_token_count(msgs, None, None, None)
            max_output_tokens = min(
                MAX_OUTPUT_TOKENS,
                self.context_length - prompt_token_count,
            )

            if max_output_tokens <= 0:
                raise ValueError(
                    f"max_output_tokens <= 0. Prompt token count: {prompt_token_count}"
                )

            try:
                client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
                response = await client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,  # type: ignore
                    # frequency_penalty=0.5,
                    max_completion_tokens=max_output_tokens,
                    temperature=self.config.temperature,
                    n=self.config.return_n,
                    reasoning_effort=self.reasoning_effort,  # type: ignore
                )

                token_usage = self.update_usage(response.usage)

                choice = response.choices[0]
                finish_reason = choice.finish_reason
                content = choice.message.content
                if finish_reason == "stop" and content:
                    output: list[MessageBlock | dict] = [
                        {
                            "role": CreatorRole.ASSISTANT.value,
                            "content": content,
                        }
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
                delay = delay * Reasoning_Core.DELAY_FACTOR
                delay = min(Reasoning_Core.MAX_DELAY, delay)
                continue
            except Exception as e:
                logger.error("Exception: %s", e, exc_info=True, stack_info=True)
                raise

        raise RuntimeError("Max re-attempt reached")

    def run(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> tuple[list[MessageBlock | dict[str, Any]], TokenUsage]:
        """
        Synchronously run the LLM model with the given query and context.

        Args:
            query (str): The query to be processed by the LLM model.
            context (list[MessageBlock | dict] | None): The context to be used for the LLM model.
            **kwargs: Additional keyword arguments.

        Returns:
            response_tuple (tuple[list[MessageBlock | dict], TokenUsage]):
            * output: The list of messages generated by the LLM model.
            * usage: The recorded token usage.

        **Notes**:
        * system_prompt: Provided in USER role.
        * max_tokens: max_completion_tokens
        * temperature: Hardcode as 1.0
        * frequency_penalty: Not supported
        * reasoning_effort: Not all reasoning models support this parameter
        """
        msgs: list[MessageBlock | dict] = [
            MessageBlock(role=CreatorRole.USER.value, content=self.system_prompt)
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

        while attempt < Reasoning_Core.MAX_ATTEMPT:
            logger.debug("Attempt %d", attempt)
            messages = deepcopy(msgs)
            prompt_token_count = self.calculate_token_count(msgs, None, None, None)
            max_output_tokens = min(
                MAX_OUTPUT_TOKENS,
                self.context_length - prompt_token_count,
            )

            if max_output_tokens <= 0:
                raise ValueError(
                    f"max_output_tokens <= 0. Prompt token count: {prompt_token_count}"
                )

            try:
                client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,  # type: ignore
                    # frequency_penalty=0.5,
                    max_completion_tokens=max_output_tokens,
                    temperature=self.config.temperature,
                    n=self.config.return_n,
                    reasoning_effort=self.reasoning_effort,  # type: ignore
                )

                token_usage = self.update_usage(response.usage)

                choice = response.choices[0]
                finish_reason = choice.finish_reason
                content = choice.message.content
                if finish_reason == "stop" and content:
                    output: list[MessageBlock | dict] = [
                        {
                            "role": CreatorRole.ASSISTANT.value,
                            "content": content,
                        }
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
                delay = delay * Reasoning_Core.DELAY_FACTOR
                delay = min(Reasoning_Core.MAX_DELAY, delay)
                continue
            except Exception as e:
                logger.error("Exception: %s", e, exc_info=True, stack_info=True)
                raise

        raise RuntimeError("Max re-attempt reached")
