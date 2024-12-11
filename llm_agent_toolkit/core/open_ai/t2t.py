import os
import logging
import openai
from ..._core import Core, TextGenerator, ToolSupport
from ..._util import (
    CreatorRole,
    ChatCompletionConfig,
    MessageBlock,
)
from ..._tool import Tool
from .base import OpenAICore, TOOL_PROMPT

logger = logging.getLogger(__name__)


class T2T_OAI_Core(Core, OpenAICore, TextGenerator, ToolSupport):
    """
    `T2T_OAI_Core` is a concrete implementation of the `Core` abstract class.
    `T2T_OAI_Core` is also a child class of `OpenAICore`.
    It facilitates synchronous and asynchronous communication with OpenAI's API.

    Methods:
    - run(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> list[MessageBlock | dict]:
        Synchronously run the LLM model with the given query and context.
    - run_async(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> list[MessageBlock | dict]:
        Asynchronously run the LLM model with the given query and context.

    Notes:
    - Loop until a solution is found, or maximum iteration or token count is reached.
    - The caller is responsible for memory management, output parsing and error handling.
    - If model is not available under OpenAI's listing, raise ValueError.
    - `context_length` is configurable.
    """

    def __init__(
        self,
        system_prompt: str,
        config: ChatCompletionConfig,
        tools: list[Tool] | None = None,
    ):
        assert isinstance(config, ChatCompletionConfig)
        Core.__init__(self, system_prompt, config)
        OpenAICore.__init__(self, config.name)
        ToolSupport.__init__(self, tools)
        self.__profile = self.build_profile(config.name)
        if tools and self.profile["tool"] is False:
            logger.warning("Tool might not work on this %s", self.model_name)

    @property
    def context_length(self) -> int:
        return self.profile["context_length"]

    @context_length.setter
    def context_length(self, value):
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

        self.__profile["context_length"] = value

    @property
    def profile(self) -> dict:
        """
        Profile is mostly for view purpose only,
        except the context_length which might be used to control the input to the LLM.
        """
        return self.__profile

    async def run_async(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        """
        Asynchronously run the LLM model with the given query and context.

        Args:
        - query (str): The query to be processed by the LLM model.
        - context (list[MessageBlock | dict] | None): The context to be used for the LLM model.

        Returns:
        - list[MessageBlock | dict]: The output of the LLM model.
        """
        msgs: list[MessageBlock | dict] = [
            MessageBlock(role=CreatorRole.SYSTEM.value, content=self.system_prompt)
        ]
        if context is not None:
            msgs.extend(context)
        msgs.append(MessageBlock(role=CreatorRole.USER.value, content=query))
        if self.tools is not None:
            tools_metadata = []
            for tool in self.tools:
                tools_metadata.append(tool.info)
            msgs.append(
                MessageBlock(role=CreatorRole.SYSTEM.value, content=TOOL_PROMPT)
            )
        else:
            tools_metadata = None
        number_of_primers = len(msgs)
        if isinstance(self.config, ChatCompletionConfig):
            temperature = self.config.temperature
            max_tokens = self.config.max_tokens
        else:
            temperature = 0.7
            max_tokens = 8192

        max_tokens = min(max_tokens, self.context_length)

        iteration = 0
        token_count = 0
        solved = False
        try:
            client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
            while iteration < self.config.max_iteration and token_count < max_tokens:
                # print(f"\n\nIteration: {iteration}")
                response = await client.chat.completions.create(
                    model=self.model_name,
                    messages=msgs,  # type: ignore
                    frequency_penalty=0.5,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=self.config.return_n,
                    functions=tools_metadata,  # type: ignore
                )
                if response.usage:
                    token_count += response.usage.total_tokens
                choice = response.choices[0]
                _content = getattr(choice.message, "content", "Not Available")
                msgs.append(MessageBlock(role=choice.message.role, content=_content))

                tool_calls = choice.message.tool_calls

                if tool_calls is None:
                    solved = True
                    break

                output = await self.call_tools_async(tool_calls)

                msgs.extend(output)
                iteration += 1

            if not solved:
                if iteration == self.config.max_iteration:
                    logger.warning(
                        "Maximum iteration reached. %d/%d",
                        iteration,
                        self.config.max_iteration,
                    )
                elif token_count >= max_tokens:
                    logger.warning(
                        "Maximum token count reached. %d/%d", token_count, max_tokens
                    )
            return msgs[number_of_primers:]  # Return only the generated messages
        except Exception as e:
            logger.error("Error: %s", e)
            raise

    def run(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        """
        Synchronously generate text based on the given query and context.

        Args:
            query (str): The query to generate text for.
            context (list): A list of context messages or dictionaries.
            **kwargs: Additional keyword arguments.

        Returns:
        - list[MessageBlock | dict]: The output of the LLM model.
        """
        msgs: list[MessageBlock | dict] = [
            MessageBlock(role=CreatorRole.SYSTEM.value, content=self.system_prompt)
        ]
        if context is not None:
            msgs.extend(context)
        msgs.append(MessageBlock(role=CreatorRole.USER.value, content=query))
        if self.tools is not None:
            tools_metadata = []
            for tool in self.tools:
                tools_metadata.append(tool.info)
            msgs.append(
                MessageBlock(role=CreatorRole.SYSTEM.value, content=TOOL_PROMPT)
            )
        else:
            tools_metadata = None
        number_of_primers = len(msgs)
        if isinstance(self.config, ChatCompletionConfig):
            temperature = self.config.temperature
            max_tokens = self.config.max_tokens
        else:
            temperature = 0.7
            max_tokens = 8192

        max_tokens = min(max_tokens, self.context_length)

        iteration = 0
        token_count = 0
        solved = False
        try:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            while iteration < self.config.max_iteration and token_count < max_tokens:
                # print(f"\n\nIteration: {iteration}")
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=msgs,  # type: ignore
                    frequency_penalty=0.5,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=self.config.return_n,
                    tools=tools_metadata,  # type: ignore
                )
                if response.usage:
                    token_count += response.usage.total_tokens
                choice = response.choices[0]
                _content = getattr(choice.message, "content", "Not Available")
                if _content:
                    msgs.append(
                        MessageBlock(role=choice.message.role, content=_content)
                    )

                tool_calls = choice.message.tool_calls

                if tool_calls is None:
                    solved = True
                    break

                output = self.call_tools(tool_calls)

                msgs.extend(output)
                iteration += 1

            if not solved:
                if iteration == self.config.max_iteration:
                    logger.warning(
                        "Maximum iteration reached. %d/%d",
                        iteration,
                        self.config.max_iteration,
                    )
                elif token_count >= max_tokens:
                    logger.warning(
                        "Maximum token count reached. %d/%d", token_count, max_tokens
                    )
            return msgs[number_of_primers:]  # Return only the generated messages
        except Exception as e:
            logger.error("Error: %s", e)
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
