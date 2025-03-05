import os
import logging

import openai

from ..._core import Core, ToolSupport
from ..._util import CreatorRole, ChatCompletionConfig, MessageBlock, TokenUsage
from ..._tool import Tool, ToolMetadata
from .base import DeepSeekCore, TOOL_PROMPT

logger = logging.getLogger(__name__)


class T2T_DS_Core(Core, DeepSeekCore, ToolSupport):
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

        tools_metadata: list[ToolMetadata] | None = None
        if self.tools:
            tools_metadata = [tool.info for tool in self.tools]
            msgs.append(
                MessageBlock(role=CreatorRole.SYSTEM.value, content=TOOL_PROMPT)
            )

        NUMBER_OF_PRIMERS = len(msgs)  # later use this to skip the preloaded messages

        # Determine the maximum number of tokens allowed for the response
        MAX_TOKENS = min(self.config.max_tokens, self.context_length)
        MAX_OUTPUT_TOKENS = min(
            MAX_TOKENS, self.max_output_tokens, self.config.max_output_tokens
        )
        prompt_token_count = self.calculate_token_count(msgs, tools_metadata)
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
                # logger.debug("Iteration: [%d]", iteration)
                if tools_metadata and iteration + 1 == self.config.max_iteration:
                    # Force the llm to provide answer
                    tools_metadata = None
                    msgs.remove(
                        {"role": CreatorRole.SYSTEM.value, "content": TOOL_PROMPT}
                    )

                response = await client.chat.completions.create(
                    model=self.model_name,
                    messages=msgs,  # type: ignore
                    frequency_penalty=0.5,
                    max_tokens=max_output_tokens,
                    temperature=self.config.temperature,
                    n=self.config.return_n,
                    tools=tools_metadata,  # type: ignore
                )
                choice = response.choices[0]
                _content = getattr(choice.message, "content", "Not Available")

                if _content:
                    msgs.append(
                        MessageBlock(role=CreatorRole.ASSISTANT.value, content=_content)
                    )

                tool_calls = choice.message.tool_calls
                if tool_calls:
                    output, ttku = await self.call_tools_async(tool_calls)
                    if output:
                        msgs.append(choice.message)  # type: ignore
                        msgs.extend(output)
                        token_usage += ttku

                solved = choice.finish_reason == "stop"
                prompt_token_count = self.calculate_token_count(msgs, tools_metadata)
                max_output_tokens = min(
                    MAX_OUTPUT_TOKENS,
                    self.context_length - prompt_token_count,
                )
                iteration += 1
                token_usage = self.update_usage(response.usage, token_usage)

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
            else:
                msgs[-1]["role"] = CreatorRole.ASSISTANT.value

            generated_msgs = msgs[
                NUMBER_OF_PRIMERS:
            ]  # Return only the generated messages
            filtered_msgs = list(
                filter(
                    lambda msg: isinstance(msg, dict) or type(msg) is MessageBlock,
                    generated_msgs,
                )
            )
            return filtered_msgs, token_usage
        except Exception as e:
            logger.error("Exception: %s", e, exc_info=True, stack_info=True)
            raise

    def run(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> tuple[list[MessageBlock | dict], TokenUsage]:
        msgs: list[MessageBlock | dict] = [
            MessageBlock(role=CreatorRole.SYSTEM.value, content=self.system_prompt)
        ]

        if context:
            msgs.extend(context)
        msgs.append(MessageBlock(role=CreatorRole.USER.value, content=query))

        tools_metadata: list[ToolMetadata] | None = None
        if self.tools:
            tools_metadata = [tool.info for tool in self.tools]
            msgs.append(
                MessageBlock(role=CreatorRole.SYSTEM.value, content=TOOL_PROMPT)
            )

        NUMBER_OF_PRIMERS = len(msgs)  # later use this to skip the preloaded messages

        # Determine the maximum number of tokens allowed for the response
        MAX_TOKENS = min(self.config.max_tokens, self.context_length)
        MAX_OUTPUT_TOKENS = min(
            MAX_TOKENS, self.max_output_tokens, self.config.max_output_tokens
        )
        prompt_token_count = self.calculate_token_count(msgs, tools_metadata)
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
                # logger.debug("Iteration: [%d]", iteration)
                if tools_metadata and iteration + 1 == self.config.max_iteration:
                    # Force the llm to provide answer
                    tools_metadata = None
                    msgs.remove(
                        {"role": CreatorRole.SYSTEM.value, "content": TOOL_PROMPT}
                    )

                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=msgs,  # type: ignore
                    frequency_penalty=0.5,
                    max_tokens=max_output_tokens,
                    temperature=self.config.temperature,
                    n=self.config.return_n,
                    tools=tools_metadata,  # type: ignore
                )
                choice = response.choices[0]

                _content = getattr(choice.message, "content", None)

                if _content:
                    msgs.append(
                        MessageBlock(role=CreatorRole.ASSISTANT.value, content=_content)
                    )

                tool_calls = choice.message.tool_calls
                if tool_calls:
                    output, ttku = self.call_tools(tool_calls)
                    if output:
                        msgs.append(choice.message)  # type: ignore
                        msgs.extend(output)
                        token_usage += ttku

                solved = choice.finish_reason == "stop"
                prompt_token_count = self.calculate_token_count(msgs, tools_metadata)
                max_output_tokens = min(
                    MAX_OUTPUT_TOKENS,
                    self.context_length - prompt_token_count,
                )
                iteration += 1
                token_usage = self.update_usage(response.usage, token_usage)

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
            else:
                msgs[-1]["role"] = CreatorRole.ASSISTANT.value

            generated_msgs = msgs[
                NUMBER_OF_PRIMERS:
            ]  # Return only the generated messages

            filtered_msgs = list(
                filter(
                    lambda msg: isinstance(msg, dict),
                    generated_msgs,
                )
            )

            return filtered_msgs, token_usage
        except Exception as e:
            logger.error("Exception: %s", e, exc_info=True, stack_info=True)
            raise

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
                            "content": f"({args}) => {result}",
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
                            "content": f"({args}) => {result}",
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
