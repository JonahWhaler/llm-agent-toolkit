import os
import warnings
import openai
from ..._core import Core
from ..._util import (
    CreatorRole,
    ChatCompletionConfig,
    MessageBlock,
)
from ..._tool import Tool


class T2T_OAI_Core(Core):
    """
    `T2T_OAI_Core` is a concrete implementation of the `Core` abstract class.
    It facilitates synchronous and asynchronous communication with OpenAI's API.

    Methods:
    - run(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> list[MessageBlock | dict]:
        Synchronously run the LLM model with the given query and context.
    - run_async(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> list[MessageBlock | dict]:
        Asynchronously run the LLM model with the given query and context.

    Notes:
    - Loop until a solution is found, or maximum iteration or token count is reached.
    - The caller is responsible for memory management, output parsing and error handling.
    """

    def __init__(
        self,
        system_prompt: str,
        config: ChatCompletionConfig,
        tools: list[Tool] | None = None,
    ):
        assert isinstance(config, ChatCompletionConfig)
        super().__init__(system_prompt, config, tools)

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
            for ctx in context:
                msgs.append(ctx)
        msgs.append({"role": CreatorRole.USER.value, "content": query})
        tools_metadata: list = []
        if self.tools is not None:
            for tool in self.tools:
                tools_metadata.append(tool.info)

        if isinstance(self.config, ChatCompletionConfig):
            temperature = self.config.temperature
            max_tokens = self.config.max_tokens
        else:
            temperature = 0.7
            max_tokens = 4096
        iteration = 0
        token_count = 0
        solved = False
        try:
            client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
            while iteration < self.config.max_iteration and token_count < max_tokens:
                print(f"\n\nIteration: {iteration}")
                response = await client.chat.completions.create(
                    model=self.model_name,
                    messages=msgs,  # type: ignore
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=self.config.return_n,
                    functions=tools_metadata,  # type: ignore
                )
                if response.usage:
                    token_count += response.usage.total_tokens
                choice = response.choices[0]
                _content = getattr(choice.message, "content", "Not Available")
                msgs.append(
                    {
                        "role": CreatorRole.ASSISTANT.value,
                        "content": _content,
                    }
                )

                tool_calls = choice.message.tool_calls

                if tool_calls is None:
                    solved = True
                    break

                output = await self.__call_tools_async(tool_calls)

                msgs.extend(output)
                iteration += 1

            if not solved:
                if iteration == self.config.max_iteration:
                    warnings.warn(
                        f"Maximum iteration reached. {iteration}/{self.config.max_iteration}"
                    )
                elif token_count >= max_tokens:
                    warnings.warn(
                        f"Maximum token count reached. {token_count}/{max_tokens}"
                    )
            return msgs
        except Exception as e:
            print(f"run_async: {e}")
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
            for ctx in context:
                msgs.append(ctx)
        msgs.append({"role": CreatorRole.USER.value, "content": query})
        tools_metadata: list = []
        if self.tools is not None:
            for tool in self.tools:
                tools_metadata.append(tool.info)

        if isinstance(self.config, ChatCompletionConfig):
            temperature = self.config.temperature
            max_tokens = self.config.max_tokens
        else:
            temperature = 0.7
            max_tokens = 4096
        iteration = 0
        token_count = 0
        solved = False
        try:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            while iteration < self.config.max_iteration and token_count < max_tokens:
                print(f"\n\nIteration: {iteration}")
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=msgs,  # type: ignore
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
                        {
                            "role": CreatorRole.ASSISTANT.value,
                            "content": _content,
                        }
                    )

                tool_calls = choice.message.tool_calls

                if tool_calls is None:
                    solved = True
                    break

                output = self.__call_tools(tool_calls)

                msgs.extend(output)
                iteration += 1

            if not solved:
                if iteration == self.config.max_iteration:
                    warnings.warn(
                        f"Maximum iteration reached. {iteration}/{self.config.max_iteration}"
                    )
                elif token_count >= max_tokens:
                    warnings.warn(
                        f"Maximum token count reached. {token_count}/{max_tokens}"
                    )
            return msgs
        except Exception as e:
            print(f"run: {e}")
            raise

    async def __call_tools_async(
        self, selectd_tools: list
    ) -> list[MessageBlock | dict]:
        output: list[MessageBlock | dict] = []
        for tool_call in selectd_tools:
            for tool in self.tools:  # type: ignore
                if tool.info["function"]["name"] != tool_call.function.name:
                    continue
                args = tool_call.function.arguments
                try:
                    result = await tool.run_async(args)
                    output.append(
                        {
                            "role": CreatorRole.FUNCTION.value,
                            "name": tool_call.function.name,
                            "content": f"({args}) => {result}",
                        }
                    )
                except Exception as e:
                    output.append(
                        {
                            "role": CreatorRole.FUNCTION.value,
                            "name": tool_call.function.name,
                            "content": f"({args}) => {e}",
                        }
                    )
                break

        return output

    def __call_tools(self, selectd_tools: list) -> list[MessageBlock | dict]:
        output: list[MessageBlock | dict] = []
        for tool_call in selectd_tools:
            for tool in self.tools:  # type: ignore
                if tool.info["function"]["name"] != tool_call.function.name:
                    continue
                args = tool_call.function.arguments
                try:
                    result = tool.run(args)
                    output.append(
                        {
                            "role": CreatorRole.FUNCTION.value,
                            "name": tool_call.function.name,
                            "content": f"({args}) => {result}",
                        }
                    )
                except Exception as e:
                    output.append(
                        {
                            "role": CreatorRole.FUNCTION.value,
                            "name": tool_call.function.name,
                            "content": f"({args}) => {e}",
                        }
                    )
                break

        return output
