import os
import logging
import json
from typing import Any, Optional, Type, TypeVar
from math import ceil

import openai
from pydantic import BaseModel

from ..._core import Core
from ..._util import CreatorRole, ChatCompletionConfig, MessageBlock, ResponseMode

from .base import DeepSeekCore

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class T2TSO_DS_Core(Core, DeepSeekCore):
    def __init__(self, system_prompt: str, config: ChatCompletionConfig):
        Core.__init__(self, system_prompt, config)
        DeepSeekCore.__init__(self)
        self.profile = self.build_profile()

    def validate(
        self,
        response_mode: Optional[ResponseMode],
        response_format: Optional[Type[T]] | None = None,
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
    ) -> list[MessageBlock | dict]:
        response_mode: Optional[ResponseMode] = kwargs.get("mode", ResponseMode.DEFAULT)
        # response_format: Optional[Type[T]] = kwargs.get("format")  # type: ignore
        self.validate(response_mode, None)  # Raise an exception if invalid

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
        prompt_token_count = self.calculate_token_count(msgs, None)
        max_output_tokens = min(
            MAX_OUTPUT_TOKENS,
            self.context_length - prompt_token_count,
        )

        accumulated_token_count = 0  # Accumulated token count across iterations

        try:
            client = openai.AsyncOpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url=os.environ["DEEPSEEK_BASE_URL"],
            )
            if max_output_tokens <= 0:
                logger.warning("Prompt token count: %d", prompt_token_count)
                raise ValueError("max_output_tokens <= 0")

            if response_mode is ResponseMode.JSON:
                response = await client.chat.completions.create(
                    model=self.model_name,
                    messages=msgs,  # type: ignore
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
                    messages=msgs,  # type: ignore
                    frequency_penalty=0.5,
                    max_tokens=max_output_tokens,
                    temperature=self.config.temperature,
                    n=self.config.return_n,
                )

            choice = response.choices[0]
            _content = getattr(choice.message, "content", "Not Available")

            accumulated_token_count += (
                response.usage.total_tokens if response.usage else 0
            )

            if _content:
                if response_mode is not ResponseMode.DEFAULT:
                    try:
                        _ = json.loads(_content)
                        content = _content
                    except json.JSONDecodeError as decode_error:
                        e = {"error": str(decode_error)}
                        content = json.dumps(e)
                else:
                    content = _content
                return [{"role": CreatorRole.ASSISTANT.value, "content": content}]
            raise RuntimeError(f"Content not available. Reason: {choice.finish_reason}")
        except Exception as e:
            logger.error("Exception: %s", e)
            raise

    def run(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        response_mode: Optional[ResponseMode] = kwargs.get("mode", ResponseMode.DEFAULT)
        # response_format: Optional[Type[T]] = kwargs.get("format")  # type: ignore
        self.validate(response_mode, None)  # Raise an exception if invalid

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
        prompt_token_count = self.calculate_token_count(msgs, None)
        max_output_tokens = min(
            MAX_OUTPUT_TOKENS,
            self.context_length - prompt_token_count,
        )

        accumulated_token_count = 0  # Accumulated token count across iterations

        try:
            client = openai.Client(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url=os.environ["DEEPSEEK_BASE_URL"],
            )
            if max_output_tokens <= 0:
                logger.warning("Prompt token count: %d", prompt_token_count)
                raise ValueError("max_output_tokens <= 0")

            if response_mode is ResponseMode.JSON:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=msgs,  # type: ignore
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
                    messages=msgs,  # type: ignore
                    frequency_penalty=0.5,
                    max_tokens=max_output_tokens,
                    temperature=self.config.temperature,
                    n=self.config.return_n,
                )

            choice = response.choices[0]
            _content = getattr(choice.message, "content", "Not Available")

            accumulated_token_count += (
                response.usage.total_tokens if response.usage else 0
            )

            if _content:
                if response_mode is not ResponseMode.DEFAULT:
                    try:
                        _ = json.loads(_content)
                        content = _content
                    except json.JSONDecodeError as decode_error:
                        e = {"error": str(decode_error)}
                        content = json.dumps(e)
                else:
                    content = _content
                return [{"role": CreatorRole.ASSISTANT.value, "content": content}]
            raise RuntimeError(f"Content not available. Reason: {choice.finish_reason}")
        except Exception as e:
            logger.error("Exception: %s", e)
            raise
