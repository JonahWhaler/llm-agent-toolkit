import os
import logging
import base64
import openai
from ..._core import Core, ToolSupport, ImageInterpreter
from ..._util import CreatorRole, ChatCompletionConfig, MessageBlock, TokenUsage
from ..._tool import Tool, ToolMetadata
from .base import OpenAICore, TOOL_PROMPT

logger = logging.getLogger(__name__)


class I2T_OAI_Core(Core, OpenAICore, ImageInterpreter, ToolSupport):
    """
    `I2T_OAI_Core` is a concrete implementation of abstract base classes `TextGenerator`, `ImageInterpreter`, and `ToolSupport`.
    `I2T_OAI_Core` is also a child class of `OpenAICore`.

    It facilitates synchronous and asynchronous communication with OpenAI's API to interpret images.

    **Methods:**
    - run(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> tuple[list[MessageBlock | dict], TokenUsage]:
        Synchronously run the LLM model to interpret images.
    - run_async(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> tuple[list[MessageBlock | dict], TokenUsage]:
        Asynchronously run the LLM model to interpret images.
    - interpret(query: str, context: list[MessageBlock | dict] | None, filepath: str, **kwargs) -> tuple[list[MessageBlock | dict], TokenUsage]:
        Synchronously interpret the given image.
    - interpret_async(query: str, context: list[MessageBlock | dict] | None, filepath: str, **kwargs) -> tuple[list[MessageBlock | dict], TokenUsage]:
        Asynchronously interpret the given image.
    - get_image_url(filepath: str) -> str:
        Returns the URL of the image from the specified file path.
    - call_tools_async(selected_tools: list) -> list[MessageBlock | dict]:
        Asynchronously call tools.
    - call_tools(selected_tools: list) -> list[MessageBlock | dict]:
        Synchronously call tools.

    **Notes:**
    - Supported image format: .png, .jpeg, .jpg, .gif, .webp
    - The caller is responsible for memory management, output parsing and error handling.
    - If model is not available under OpenAI's listing, raise ValueError.
    - `context_length` is configurable.
    - `max_output_tokens` is configurable.
    """

    SUPPORTED_IMAGE_FORMATS = (".png", ".jpeg", ".jpg", ".gif", ".webp")

    def __init__(
        self,
        system_prompt: str,
        config: ChatCompletionConfig,
        tools: list[Tool] | None = None,
    ):
        Core.__init__(self, system_prompt, config)
        OpenAICore.__init__(self, config.name)
        ToolSupport.__init__(self, tools)
        self.profile = self.build_profile(config.name)
        if tools and self.profile.get("tool", False) is False:
            logger.warning("Tool might not work on this %s", self.model_name)
        if self.profile.get("image_input", False) is False:
            logger.warning("Vision might not work on this %s", self.model_name)

    async def run_async(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> tuple[list[MessageBlock | dict], TokenUsage]:
        """
        Asynchronously run the LLM model to interpret images.

        Args:
            query (str): The query to be interpreted.
            context (list[MessageBlock | dict] | None): The context to be used for the query.
            filepath (str): The path to the image file to be interpreted.
            **kwargs: Additional keyword arguments.

        Returns:
            list[MessageBlock | dict]: The list of messages generated by the LLM model.
            TokenUsage: The recorded token usage.

        Notes:
        * Early Termination Condition:
                * If a solution is found.
                * If the maximum iteration is reached.
                * If the accumulated token count is greater than or equal to the maximum token count.
                * If the maximum output tokens are less than or equal to zero.
        """
        msgs: list[MessageBlock | dict] = [
            MessageBlock(role=CreatorRole.SYSTEM.value, content=self.system_prompt)
        ]

        if context:
            msgs.extend(context)

        filepath: str | None = kwargs.get("filepath", None)
        if filepath:
            # detail hardcode as "high"
            resized, newpath = self.resize(filepath, "high")
            if resized and newpath:
                img_url = self.get_image_url(newpath)
                os.remove(newpath)
            else:
                img_url = self.get_image_url(filepath)

            msgs.append(
                {
                    "role": CreatorRole.USER.value,
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {"url": img_url, "detail": "high"},
                        },
                    ],  # type: ignore
                }
            )
        else:
            msgs.append(MessageBlock(role=CreatorRole.USER.value, content=query))

        tools_metadata: list[ToolMetadata] | None = None
        if self.tools:
            tools_metadata = [tool.info for tool in self.tools]
            msgs.append(
                MessageBlock(role=CreatorRole.SYSTEM.value, content=TOOL_PROMPT)
            )

        NUMBER_OF_PRIMERS = len(msgs)  # later use this to skip the preloaded messages

        MAX_TOKENS = min(self.config.max_tokens, self.context_length)
        MAX_OUTPUT_TOKENS = min(
            MAX_TOKENS, self.max_output_tokens, self.config.max_output_tokens
        )
        prompt_token_count = self.calculate_token_count(
            msgs,
            tools_metadata,
            images=[filepath] if filepath else None,
            image_detail="high" if filepath else None,
        )
        max_output_tokens = min(
            MAX_OUTPUT_TOKENS,
            self.context_length - prompt_token_count,
        )

        token_usage = TokenUsage(input_tokens=0, output_tokens=0)
        iteration, solved = 0, False

        try:
            client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
            while (
                not solved
                and max_output_tokens > 0
                and iteration < self.config.max_iteration
                and token_usage.total_tokens < MAX_TOKENS
            ):
                # logger.debug("\n\nIteration: %d", iteration)
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
                    output = await self.call_tools_async(tool_calls)
                    msgs.extend(output)

                solved = tool_calls is None
                prompt_token_count = self.calculate_token_count(
                    msgs,
                    tools_metadata,
                    images=[filepath] if filepath else None,
                    image_detail="high" if filepath else None,
                )
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
            return msgs[
                NUMBER_OF_PRIMERS:
            ], token_usage  # Return only the generated messages
        except Exception as e:
            logger.error("Exception: %s", e, exc_info=True, stack_info=True)
            raise

    def run(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> tuple[list[MessageBlock | dict], TokenUsage]:
        """
        Synchronously run the LLM model to interpret images.

        Args:
            query (str): The query to be interpreted.
            context (list[MessageBlock | dict] | None): The context to be used for the query.
            filepath (str): The path to the image file to be interpreted.
            **kwargs: Additional keyword arguments.

        Returns:
            list[MessageBlock | dict]: The list of messages generated by the LLM model.
            TokenUsage: The recorded token usage.

        Notes:
        * Early Termination Condition:
                * If a solution is found.
                * If the maximum iteration is reached.
                * If the accumulated token count is greater than or equal to the maximum token count.
                * If the maximum output tokens are less than or equal to zero.
        """
        msgs: list[MessageBlock | dict] = [
            MessageBlock(role=CreatorRole.SYSTEM.value, content=self.system_prompt)
        ]

        if context:
            msgs.extend(context)

        filepath: str | None = kwargs.get("filepath", None)
        if filepath:
            # detail hardcode as "high"
            resized, newpath = self.resize(filepath, "high")
            if resized and newpath:
                img_url = self.get_image_url(newpath)
                os.remove(newpath)
            else:
                img_url = self.get_image_url(filepath)

            msgs.append(
                {
                    "role": CreatorRole.USER.value,
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {"url": img_url, "detail": "high"},
                        },
                    ],  # type: ignore
                }
            )
        else:
            msgs.append(MessageBlock(role=CreatorRole.USER.value, content=query))

        tools_metadata: list[ToolMetadata] | None = None
        if self.tools:
            tools_metadata = [tool.info for tool in self.tools]
            msgs.append(
                MessageBlock(role=CreatorRole.SYSTEM.value, content=TOOL_PROMPT)
            )

        NUMBER_OF_PRIMERS = len(msgs)  # later use this to skip the preloaded messages

        MAX_TOKENS = min(self.config.max_tokens, self.context_length)
        MAX_OUTPUT_TOKENS = min(
            MAX_TOKENS, self.max_output_tokens, self.config.max_output_tokens
        )
        prompt_token_count = self.calculate_token_count(
            msgs,
            tools_metadata,
            images=[filepath] if filepath else None,
            image_detail="high" if filepath else None,
        )
        max_output_tokens = min(
            MAX_OUTPUT_TOKENS,
            self.context_length - prompt_token_count,
        )

        iteration, solved = 0, False
        token_usage = TokenUsage(input_tokens=0, output_tokens=0)

        try:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            while (
                not solved
                and max_output_tokens > 0
                and iteration < self.config.max_iteration
                and token_usage.total_tokens < MAX_TOKENS
            ):
                # logger.debug("\n\nIteration: %d", iteration)
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
                _content = getattr(choice.message, "content", "Not Available")
                if _content:
                    msgs.append(
                        MessageBlock(role=CreatorRole.ASSISTANT.value, content=_content)
                    )

                tool_calls = choice.message.tool_calls
                if tool_calls:
                    output = self.call_tools(tool_calls)
                    msgs.extend(output)

                solved = tool_calls is None
                prompt_token_count = self.calculate_token_count(
                    msgs,
                    tools_metadata,
                    images=[filepath] if filepath else None,
                    image_detail="high" if filepath else None,
                )
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
            return msgs[
                NUMBER_OF_PRIMERS:
            ], token_usage  # Return only the generated messages
        except Exception as e:
            logger.error("Exception: %s", e, exc_info=True, stack_info=True)
            raise

    @staticmethod
    def get_image_url(filepath: str):
        ext = os.path.splitext(filepath)[-1]
        if ext not in I2T_OAI_Core.SUPPORTED_IMAGE_FORMATS:
            raise ValueError(f"Unsupported image type: {ext}")
        ext = ext[1:] if ext != ".jpg" else "jpeg"
        try:
            with open(filepath, "rb") as f:
                encoded_image = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/{ext};base64,{encoded_image}"
        except FileNotFoundError as fnfe:
            logger.error("FileNotFoundError: %s", fnfe, exc_info=True, stack_info=True)
            raise
        except Exception as e:
            logger.error("Exception: %s", e, exc_info=True, stack_info=True)
            raise

    async def call_tools_async(self, selected_tools: list) -> list[MessageBlock | dict]:
        """
        Asynchronously call every selected tools.

        Args:
            selected_tools (list): A list of selected tools.

        Returns:
            list: A list of messages generated by the tools.

        Notes:
            - If more than one tool is selected, they are executed independently and separately.
            - Tools chaining is not supported.
            - Does not raise exception on failed tool execution, an error message is returned instead to guide the calling LLM.
        """
        output: list[MessageBlock | dict] = []
        for tool_call in selected_tools:
            for tool in self.tools:  # type: ignore
                if tool.info["function"]["name"] != tool_call.function.name:
                    continue
                args = tool_call.function.arguments
                try:
                    result = await tool.run_async(args)
                    output.append(
                        MessageBlock(
                            role=CreatorRole.FUNCTION.value,
                            content=f"({args}) => {result}",
                            name=tool_call.function.name,
                        )
                    )
                except Exception as e:
                    output.append(
                        MessageBlock(
                            role=CreatorRole.FUNCTION.value,
                            content=f"({args}) => {e}",
                            name=tool_call.function.name,
                        )
                    )
                break

        return output

    def call_tools(self, selected_tools: list) -> list[MessageBlock | dict]:
        """
        Synchronously call every selected tools.

        Args:
            selected_tools (list): A list of selected tools.

        Returns:
            list: A list of messages generated by the tools.

        Notes:
            - If more than one tool is selected, they are executed independently and separately.
            - Tools chaining is not supported.
            - Does not raise exception on failed tool execution, an error message is returned instead to guide the calling LLM.
        """
        output: list[MessageBlock | dict] = []
        for tool_call in selected_tools:
            for tool in self.tools:  # type: ignore
                if tool.info["function"]["name"] != tool_call.function.name:
                    continue
                args = tool_call.function.arguments
                try:
                    result = tool.run(args)
                    output.append(
                        MessageBlock(
                            role=CreatorRole.FUNCTION.value,
                            content=f"({args}) => {result}",
                            name=tool_call.function.name,
                        )
                    )
                except Exception as e:
                    output.append(
                        MessageBlock(
                            role=CreatorRole.FUNCTION.value,
                            content=f"({args}) => {e}",
                            name=tool_call.function.name,
                        )
                    )
                break

        return output

    def interpret(
        self,
        query: str,
        context: list[MessageBlock | dict] | None,
        filepath: str,
        **kwargs,
    ):
        return self.run(query=query, context=context, filepath=filepath, **kwargs)

    async def interpret_async(
        self,
        query: str,
        context: list[MessageBlock | dict] | None,
        filepath: str,
        **kwargs,
    ):
        return await self.run_async(
            query=query, context=context, filepath=filepath, **kwargs
        )
