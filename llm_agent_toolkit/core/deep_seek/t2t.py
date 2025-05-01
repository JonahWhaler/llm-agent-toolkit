import os
import logging
import time
import asyncio
import json
from copy import deepcopy

import openai
from openai import RateLimitError

from ..._core import Core, ToolSupport
from ..._util import CreatorRole, ChatCompletionConfig, MessageBlock, TokenUsage
from ..._tool import Tool, ToolMetadata
from .base import DeepSeekCore, TOOL_PROMPT

logger = logging.getLogger(__name__)


class T2T_DS_Core(Core, DeepSeekCore, ToolSupport):
    """
    `T2T_DS_Core` is a concrete implementation of abstract class `Core`, `DeepSeekCore` and `ToolSupport`
    to provide chat completion capabilities with function calling support.

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
        call_tools(selected_tools: list) -> tuple[list[MessageBlock|dict], TokenUsage]:
            Synchronously call selected tools.
        call_tools_async(selected_tools: list) -> tuple[list[MessageBlock|dict], TokenUsage]:
            Asynchronously call selected tools.

    **Notes**:
        - The class supports progressive backoff for retrying API calls in case of
          RateLimitError or other exceptions.
    """

    MAX_ATTEMPT: int = 5
    DELAY_FACTOR: float = 1.5
    MAX_DELAY: float = 60.0

    def __init__(
        self,
        system_prompt: str,
        config: ChatCompletionConfig,
        tools: list[Tool] | None = None,
    ):
        Core.__init__(self, system_prompt, config)
        DeepSeekCore.__init__(self)
        ToolSupport.__init__(self, tools)
        self.profile = self.build_profile()

    async def run_async(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> tuple[list[MessageBlock | dict], TokenUsage]:
        msgs: list[MessageBlock | dict] = [
            MessageBlock(role=CreatorRole.SYSTEM.value, content=self.system_prompt)
        ]

        if context:
            msgs.extend(context)
        msgs.append(MessageBlock(role=CreatorRole.USER.value, content=query))

        tools: list[ToolMetadata] | None = None
        if self.tools:
            tools = [tool.info for tool in self.tools]
            msgs.append(
                MessageBlock(role=CreatorRole.SYSTEM.value, content=TOOL_PROMPT)
            )

        # later use this to skip the preloaded messages
        number_of_inputs = len(msgs)

        # Determine the maximum number of tokens allowed for the response
        MAX_TOKENS = min(self.config.max_tokens, self.context_length)
        MAX_OUTPUT_TOKENS = min(
            MAX_TOKENS, self.max_output_tokens, self.config.max_output_tokens
        )

        attempt: int = 1
        delay: float = 5.0

        while attempt < T2T_DS_Core.MAX_ATTEMPT:
            logger.debug("Attempt %d", attempt)
            messages = deepcopy(msgs)

            prompt_token_count = self.calculate_token_count(msgs, tools)
            max_output_tokens = min(
                MAX_OUTPUT_TOKENS,
                self.context_length - prompt_token_count,
            )

            iteration, solved = 0, False
            token_usage = TokenUsage(input_tokens=0, output_tokens=0)
            try:
                client = openai.AsyncOpenAI(
                    api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url=os.environ["DEEPSEEK_BASE_URL"],
                )
                while (
                    not solved
                    and max_output_tokens > 0
                    and iteration < self.config.max_iteration
                    and token_usage.total_tokens < MAX_TOKENS
                ):
                    logger.debug("Iteration: [%d]", iteration)
                    response = await client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,  # type: ignore
                        frequency_penalty=0.5,
                        max_tokens=max_output_tokens,
                        temperature=self.config.temperature,
                        n=self.config.return_n,
                        tools=tools,  # type: ignore
                    )

                    token_usage = self.update_usage(response.usage, token_usage)

                    choice = response.choices[0]
                    finish_reason = choice.finish_reason
                    content = choice.message.content
                    tool_calls = choice.message.tool_calls

                    if finish_reason == "tool_calls" and tool_calls:
                        output, ttku = await self.call_tools_async(tool_calls)
                        if output:
                            messages.append(choice.message)  # type: ignore
                            messages.extend(output)
                            token_usage += ttku

                        prompt_token_count = self.calculate_token_count(messages, tools)
                        max_output_tokens = min(
                            MAX_OUTPUT_TOKENS,
                            self.context_length - prompt_token_count,
                        )
                        iteration += 1
                    elif finish_reason == "stop" and content:
                        messages.append(
                            MessageBlock(
                                role=CreatorRole.ASSISTANT.value, content=content
                            )
                        )
                        solved = True
                    elif finish_reason == "length" and content:
                        logger.warning("Terminated due to length.")
                        e = {"error": "Early Termination: Length", "text": content}
                        messages.append(
                            MessageBlock(
                                role=CreatorRole.ASSISTANT.value,
                                content=json.dumps(e, ensure_ascii=False),
                            )
                        )
                        solved = True
                    else:
                        logger.warning("Malformed response: %s", response)
                        logger.warning("Config: %s", self.config)
                        raise RuntimeError(f"Terminated: {finish_reason}")

                # End while
                if not solved:
                    warning_message = "Warning: "
                    if iteration == self.config.max_iteration:
                        warning_message += f"Maximum iteration reached. {iteration}/{self.config.max_iteration}\n"
                    elif token_usage.total_tokens >= MAX_TOKENS:
                        warning_message += f"Maximum token count reached. {token_usage.total_tokens}/{MAX_TOKENS}\n"
                    elif max_output_tokens <= 0:
                        warning_message += f"Maximum output tokens <= 0. {prompt_token_count}/{self.context_length}\n"
                    else:
                        warning_message += "Unknown reason"
                    raise RuntimeError(warning_message)

                # Return only the generated messages
                generated_msgs = messages[number_of_inputs:]
                filtered_msgs = list(
                    filter(
                        lambda msg: isinstance(msg, dict) or type(msg) is MessageBlock,
                        generated_msgs,
                    )
                )
                return filtered_msgs, token_usage
            except RateLimitError as rle:
                logger.warning("RateLimitError: %s", rle)
                warn_msg = f"[{attempt}] Retrying in {delay} seconds..."
                logger.warning(warn_msg)
                await asyncio.sleep(delay)
                attempt += 1
                delay = delay * T2T_DS_Core.DELAY_FACTOR
                delay = min(T2T_DS_Core.MAX_DELAY, delay)
                continue
            except Exception as e:
                logger.error("Exception: %s", e, exc_info=True, stack_info=True)
                raise
        raise RuntimeError("Max re-attempt reached")

    def run(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> tuple[list[MessageBlock | dict], TokenUsage]:
        msgs: list[MessageBlock | dict] = [
            MessageBlock(role=CreatorRole.SYSTEM.value, content=self.system_prompt)
        ]

        if context:
            msgs.extend(context)
        msgs.append(MessageBlock(role=CreatorRole.USER.value, content=query))

        tools: list[ToolMetadata] | None = None
        if self.tools:
            tools = [tool.info for tool in self.tools]
            msgs.append(
                MessageBlock(role=CreatorRole.SYSTEM.value, content=TOOL_PROMPT)
            )

        # later use this to skip the preloaded messages
        number_of_inputs = len(msgs)

        # Determine the maximum number of tokens allowed for the response
        MAX_TOKENS = min(self.config.max_tokens, self.context_length)
        MAX_OUTPUT_TOKENS = min(
            MAX_TOKENS, self.max_output_tokens, self.config.max_output_tokens
        )

        attempt: int = 1
        delay: float = 5.0

        while attempt < T2T_DS_Core.MAX_ATTEMPT:
            logger.debug("Attempt %d", attempt)
            messages = deepcopy(msgs)

            prompt_token_count = self.calculate_token_count(msgs, tools)
            max_output_tokens = min(
                MAX_OUTPUT_TOKENS,
                self.context_length - prompt_token_count,
            )

            iteration, solved = 0, False
            token_usage = TokenUsage(input_tokens=0, output_tokens=0)
            try:
                client = openai.OpenAI(
                    api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url=os.environ["DEEPSEEK_BASE_URL"],
                )
                while (
                    not solved
                    and max_output_tokens > 0
                    and iteration < self.config.max_iteration
                    and token_usage.total_tokens < MAX_TOKENS
                ):
                    logger.debug("Iteration: [%d]", iteration)
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,  # type: ignore
                        frequency_penalty=0.5,
                        max_tokens=max_output_tokens,
                        temperature=self.config.temperature,
                        n=self.config.return_n,
                        tools=tools,  # type: ignore
                    )

                    token_usage = self.update_usage(response.usage, token_usage)

                    choice = response.choices[0]
                    finish_reason = choice.finish_reason
                    content = choice.message.content
                    tool_calls = choice.message.tool_calls

                    if finish_reason == "tool_calls" and tool_calls:
                        output, ttku = self.call_tools(tool_calls)
                        if output:
                            messages.append(choice.message)  # type: ignore
                            messages.extend(output)
                            token_usage += ttku

                        prompt_token_count = self.calculate_token_count(messages, tools)
                        max_output_tokens = min(
                            MAX_OUTPUT_TOKENS,
                            self.context_length - prompt_token_count,
                        )
                        iteration += 1
                    elif finish_reason == "stop" and content:
                        messages.append(
                            MessageBlock(
                                role=CreatorRole.ASSISTANT.value, content=content
                            )
                        )
                        solved = True
                    elif finish_reason == "length" and content:
                        logger.warning("Terminated due to length.")
                        e = {"error": "Early Termination: Length", "text": content}
                        messages.append(
                            MessageBlock(
                                role=CreatorRole.ASSISTANT.value,
                                content=json.dumps(e, ensure_ascii=False),
                            )
                        )
                        solved = True
                    else:
                        logger.warning("Malformed response: %s", response)
                        logger.warning("Config: %s", self.config)
                        raise RuntimeError(f"Terminated: {finish_reason}")

                # End while
                if not solved:
                    warning_message = "Warning: "
                    if iteration == self.config.max_iteration:
                        warning_message += f"Maximum iteration reached. {iteration}/{self.config.max_iteration}\n"
                    elif token_usage.total_tokens >= MAX_TOKENS:
                        warning_message += f"Maximum token count reached. {token_usage.total_tokens}/{MAX_TOKENS}\n"
                    elif max_output_tokens <= 0:
                        warning_message += f"Maximum output tokens <= 0. {prompt_token_count}/{self.context_length}\n"
                    else:
                        warning_message += "Unknown reason"
                    raise RuntimeError(warning_message)

                # Return only the generated messages
                generated_msgs = messages[number_of_inputs:]
                filtered_msgs = list(
                    filter(
                        lambda msg: isinstance(msg, dict) or type(msg) is MessageBlock,
                        generated_msgs,
                    )
                )
                return filtered_msgs, token_usage
            except RateLimitError as rle:
                logger.warning("RateLimitError: %s", rle)
                warn_msg = f"[{attempt}] Retrying in {delay} seconds..."
                logger.warning(warn_msg)
                time.sleep(delay)
                attempt += 1
                delay = delay * T2T_DS_Core.DELAY_FACTOR
                delay = min(T2T_DS_Core.MAX_DELAY, delay)
                continue
            except Exception as e:
                logger.error("Exception: %s", e, exc_info=True, stack_info=True)
                raise
        raise RuntimeError("Max re-attempt reached")

    async def call_tools_async(
        self, selected_tools: list
    ) -> tuple[list[MessageBlock | dict], TokenUsage]:
        """
        Asynchronously call every selected tools.

        Args:
            selected_tools (list): A list of selected tools.

        Returns:
            list: A list of messages generated by the tools.
            TokenUsage: The recorded token usage.

        Notes:
            - If more than one tool is selected, they are executed independently and separately.
            - Tools chaining is not supported.
            - Does not raise exception on failed tool execution, an error message is returned instead to guide the calling LLM.
        """
        output: list[MessageBlock | dict] = []
        token_usage = TokenUsage()
        for tool_call in selected_tools:
            for tool in self.tools:  # type: ignore
                if tool.token_usage.total_tokens > 0:
                    tool.reset_token_usage()

                if tool.info["function"]["name"] != tool_call.function.name:
                    continue

                args = tool_call.function.arguments
                try:
                    result = await tool.run_async(args)
                    output.append(
                        {
                            "role": CreatorRole.TOOL.value,
                            "content": f"{tool_call.function.name}({args}) => {result}",
                            "tool_call_id": tool_call.id,
                        }
                    )
                except Exception as e:
                    output.append(
                        {
                            "role": CreatorRole.TOOL.value,
                            "content": str(e),
                            "tool_call_id": tool_call.id,
                        }
                    )
                finally:
                    token_usage += tool.token_usage

                break

        return output, token_usage

    def call_tools(
        self, selected_tools: list
    ) -> tuple[list[MessageBlock | dict], TokenUsage]:
        """
        Synchronously call every selected tools.

        Args:
            selected_tools (list): A list of selected tools.

        Returns:
            list: A list of messages generated by the tools.
            TokenUsage: The recorded token usage.

        Notes:
            - If more than one tool is selected, they are executed independently and separately.
            - Tools chaining is not supported.
            - Does not raise exception on failed tool execution, an error message is returned instead to guide the calling LLM.
        """
        output: list[MessageBlock | dict] = []
        token_usage = TokenUsage()
        for tool_call in selected_tools:
            for tool in self.tools:  # type: ignore
                if tool.token_usage.total_tokens > 0:
                    tool.reset_token_usage()

                if tool.info["function"]["name"] != tool_call.function.name:
                    continue
                args = tool_call.function.arguments
                try:
                    result = tool.run(args)
                    output.append(
                        {
                            "role": CreatorRole.TOOL.value,
                            "content": f"{tool_call.function.name}({args}) => {result}",
                            "tool_call_id": tool_call.id,
                        }
                    )
                except Exception as e:
                    output.append(
                        {
                            "role": CreatorRole.TOOL.value,
                            "content": str(e),
                            "tool_call_id": tool_call.id,
                        }
                    )
                finally:
                    token_usage += tool.token_usage

                break

        return output, token_usage
