import os
import logging
import time
import asyncio
import json
from typing import Optional
from copy import deepcopy
import openai
from openai import RateLimitError

from ..._core import Core
from ..._util import CreatorRole, ChatCompletionConfig, MessageBlock, TokenUsage
from .base import DeepSeekCore

logger = logging.getLogger(__name__)


class Reasoner_Core(Core, DeepSeekCore):
    """
    `Reasoner_Core` is a concrete implementation of abstract class `Core` and `DeepSeekCore`
    to provide chat completion capabilities with reasoning support.

    It includes methods for preprocessing input, running asynchronous and synchronous
    execution, and handling retries with progressive backoff in case of errors.

    Attributes:
        SUPPORTED_MODELS (str): The name of the supported reasoning model.
        MAX_ATTEMPT (int): The maximum number of retry attempts for API calls.
        DELAY_FACTOR (float): The factor by which the delay increases after each retry.
        MAX_DELAY (float): The maximum delay between retries.

    **Methods**:
        preprocessing(system_prompt: str, query: str, context: Optional[list[MessageBlock | dict]]) -> list[MessageBlock | dict]:
            Preprocesses the input messages to be sent to the LLM model, ensuring proper
            interleaving of user and assistant roles.
        run_async(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> tuple[list[MessageBlock | dict], TokenUsage]:
            Asynchronously runs the LLM model with the given query and context, handling
            retries and token usage.
        run(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> tuple[list[MessageBlock | dict], TokenUsage]:
            Synchronously runs the LLM model with the given query and context, handling
            retries and token usage.

    **Notes**:
        - The class supports progressive backoff for retrying API calls in case of
          RateLimitError or other exceptions.
        - The `include_rc` parameter determines whether reasoning content is included
          in the output.
        - config.max_iteration hardcode as 1

    **Constraints**:
        - **features**:
            deepseek-reasoner does not support function calling、JSON output
        - **parameters**:
            deepseek-reasoner does not accept temperature、top_p、presence_penalty、frequency_penalty、logprobs、top_logprobs
    """

    SUPPORTED_MODELS = "deepseek-reasoner"
    MAX_ATTEMPT: int = 5
    DELAY_FACTOR: float = 1.5
    MAX_DELAY: float = 60.0

    def __init__(
        self,
        system_prompt: str,
        config: ChatCompletionConfig,
    ):
        if config.name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"{config.name} is not supported. Supported models: {self.SUPPORTED_MODELS}"
            )
        Core.__init__(self, system_prompt, config)
        DeepSeekCore.__init__(self)
        self.profile = self.build_profile(config.name)

    @staticmethod
    def preprocessing(
        system_prompt: str, query: str, context: Optional[list[MessageBlock | dict]]
    ) -> list[MessageBlock | dict]:
        """
        Preprocess the input messages to be sent to the LLM model.

        Rules:
        1. System instruction is added under the USER role.
        2. Ensure user/assistant role are interleaved.

        Args:
            system_prompt (str): The system prompt to be sent to the LLM model.
            query (str): The query to be processed by the LLM model.
            context (list[MessageBlock | dict] | None): The context to be used for the LLM model.

        Returns:
            list[MessageBlock | dict]: The preprocessed messages to be sent to the LLM model.
        """
        msgs: list[MessageBlock | dict] = [
            MessageBlock(role=CreatorRole.SYSTEM.value, content=system_prompt)
        ]

        if context:
            msgs.extend(context)

        msgs.append(MessageBlock(role=CreatorRole.USER.value, content=query))

        outputs: list[MessageBlock | dict] = []

        a_role = msgs[0]["role"]
        content = msgs[0]["content"]

        for msg in msgs[1:]:
            if msg["role"] == a_role:
                content += "\n" + msg["content"]
            else:
                outputs.append({"role": a_role, "content": content})
                a_role = msg["role"]
                content = msg["content"]

        outputs.append({"role": a_role, "content": content})
        return outputs

    async def run_async(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> tuple[list[MessageBlock | dict], TokenUsage]:
        """
        Asynchronously run the LLM model with the given query and context.

        Args:
            query (str): The query to be processed by the LLM model.
            context (list[MessageBlock | dict] | None): The context to be used for the LLM model.
            include_rc (bool): Whether to include `reasoning content` in the output, default is True.
            **kwargs: Additional keyword arguments.

        Returns:
            list[MessageBlock | dict]: The list of messages generated by the LLM model.
            TokenUsage: The recorded token usage.

        Notes:
        * max_tokens -> max_completion_tokens
        """
        include_rc: bool = kwargs.get("include_rc", True)
        msgs: list[MessageBlock | dict] = self.preprocessing(
            self.system_prompt, query, context
        )

        # Determine the maximum number of tokens allowed for the response
        MAX_TOKENS = min(self.config.max_tokens, self.context_length)
        MAX_OUTPUT_TOKENS = min(
            MAX_TOKENS, self.max_output_tokens, self.config.max_output_tokens
        )

        attempt: int = 1
        delay: float = 5.0

        while attempt < Reasoner_Core.MAX_ATTEMPT:
            logger.debug("Attempt %d", attempt)
            messages = deepcopy(msgs)
            prompt_token_count = self.calculate_token_count(messages, tools=None)
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
                response = await client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,  # type: ignore
                    max_tokens=max_output_tokens,
                )

                token_usage = self.update_usage(response.usage)

                choice = response.choices[0]
                finish_reason = choice.finish_reason
                content = choice.message.content
                if finish_reason == "stop" and content:
                    reasoning_content = getattr(
                        choice.message, "reasoning_content", None
                    )
                    response_string = content
                    if reasoning_content and include_rc:
                        response_string = (
                            f"<REASONING>\n{reasoning_content}\n</REASONING>\n"
                            + response_string
                        )

                    output: list[dict | MessageBlock] = [
                        MessageBlock(
                            role=CreatorRole.ASSISTANT.value, content=response_string
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
                delay = delay * Reasoner_Core.DELAY_FACTOR
                delay = min(Reasoner_Core.MAX_DELAY, delay)
                continue
            except Exception as e:
                logger.error("Exception: %s", e, exc_info=True, stack_info=True)
                raise

        raise RuntimeError("Max re-attempt reached")

    def run(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> tuple[list[MessageBlock | dict], TokenUsage]:
        """
        Synchronously generate text based on the given query and context.

        Args:
            query (str): The query to generate text for.
            context (list): A list of context messages or dictionaries.
            include_rc (bool): Whether to include `reasoning content` in the output, default is True.
            **kwargs: Additional keyword arguments.

        Returns:
            list[MessageBlock | dict]: The list of messages generated by the LLM model.
            TokenUsage: The recorded token usage.

        Notes:
        * max_tokens -> max_completion_tokens
        """
        include_rc: bool = kwargs.get("include_rc", True)
        msgs: list[MessageBlock | dict] = self.preprocessing(
            self.system_prompt, query, context
        )

        # Determine the maximum number of tokens allowed for the response
        MAX_TOKENS = min(self.config.max_tokens, self.context_length)
        MAX_OUTPUT_TOKENS = min(
            MAX_TOKENS, self.max_output_tokens, self.config.max_output_tokens
        )

        attempt: int = 1
        delay: float = 5.0

        while attempt < Reasoner_Core.MAX_ATTEMPT:
            logger.debug("Attempt %d", attempt)
            messages = deepcopy(msgs)
            prompt_token_count = self.calculate_token_count(messages, tools=None)
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
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,  # type: ignore
                    max_tokens=max_output_tokens,
                )

                token_usage = self.update_usage(response.usage)

                choice = response.choices[0]
                finish_reason = choice.finish_reason
                content = choice.message.content
                if finish_reason == "stop" and content:
                    reasoning_content = getattr(
                        choice.message, "reasoning_content", None
                    )
                    response_string = content
                    if reasoning_content and include_rc:
                        response_string = (
                            f"<REASONING>\n{reasoning_content}\n</REASONING>\n"
                            + response_string
                        )

                    output: list[dict | MessageBlock] = [
                        MessageBlock(
                            role=CreatorRole.ASSISTANT.value, content=response_string
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
                delay = delay * Reasoner_Core.DELAY_FACTOR
                delay = min(Reasoner_Core.MAX_DELAY, delay)
                continue
            except Exception as e:
                logger.error("Exception: %s", e, exc_info=True, stack_info=True)
                raise

        raise RuntimeError("Max re-attempt reached")
