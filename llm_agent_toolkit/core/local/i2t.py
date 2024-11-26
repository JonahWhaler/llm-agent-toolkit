import warnings
import ollama
from ..._core import I2T_Core
from ..._util import (
    CreatorRole,
    ChatCompletionConfig,
    MessageBlock,
)


TOOL_PROMPT = """
Utilize tools to solve the problems. 
Results from tools will be kept in the context. 
Calling the tools repeatedly is highly discouraged.
"""


class I2T_OLM_Core(I2T_Core):
    def __init__(
        self,
        connection_string: str,
        system_prompt: str,
        config: ChatCompletionConfig,
        tools: list | None = None,
    ):
        super().__init__(system_prompt, config, None)
        self.__connection_string = connection_string
        if tools is not None:
            warnings.warn("Tools may not be supported by vision models.")

    @property
    def CONN_STRING(self) -> str:
        return self.__connection_string

    @staticmethod
    def get_image_url(filepath: str) -> str:
        return ""

    def run(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        msgs: list[MessageBlock | dict] = [
            MessageBlock(role=CreatorRole.SYSTEM.value, content=self.system_prompt)
        ]

        if context is not None:
            msgs.extend(context)

        filepath: str | None = kwargs.get("filepath", None)
        if filepath is not None:
            msgs.append(
                {"role": CreatorRole.USER.value, "content": query, "images": [filepath]}
            )
        number_of_primers = len(msgs)
        if self.tools is not None:
            tools_metadata = []
            for tool in self.tools:
                tools_metadata.append(tool.info)
            msgs.append(
                MessageBlock(role=CreatorRole.SYSTEM.value, content=TOOL_PROMPT)
            )
        else:
            tools_metadata = None

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
            client = ollama.Client(host=self.CONN_STRING)
            while iteration < self.config.max_iteration and token_count < max_tokens:
                print(f"\n\nIteration: {iteration}")
                response = client.chat(
                    model=self.model_name,
                    messages=msgs,
                    tools=tools_metadata,
                    stream=False,
                    options={"temperature": temperature, "num_predict": max_tokens},
                )
                llm_generated_content = response["message"]["content"]
                if llm_generated_content != "":
                    msgs.append(
                        MessageBlock(
                            role=CreatorRole.ASSISTANT.value,
                            content=llm_generated_content,
                        )
                    )

                tool_calls = response["message"]["tool_calls"]
                if tool_calls is None:
                    solved = True
                    break

                output = self.__call_tools(tool_calls)
                msgs.extend(output)

                token_count += response["prompt_eval_count"] + response["eval_count"]
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
            return msgs[number_of_primers:]  # Return only the generated messages
        except Exception as e:
            print(f"run: {e}")
            raise

    async def run_async(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        msgs: list[MessageBlock | dict] = [
            MessageBlock(role=CreatorRole.SYSTEM.value, content=self.system_prompt)
        ]

        if context is not None:
            msgs.extend(context)

        filepath: str | None = kwargs.get("filepath", None)
        if filepath is not None:
            msgs.append(
                {
                    "role": CreatorRole.USER.value,
                    "content": query,
                    "images": [filepath],
                }
            )
        number_of_primers = len(msgs)
        if self.tools is not None:
            tools_metadata = []
            for tool in self.tools:
                tools_metadata.append(tool.info)
            msgs.append(
                MessageBlock(role=CreatorRole.SYSTEM.value, content=TOOL_PROMPT)
            )
        else:
            tools_metadata = None

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
            client = ollama.AsyncClient(host=self.CONN_STRING)
            while iteration < self.config.max_iteration and token_count < max_tokens:
                print(f"\n\nIteration: {iteration}")
                response = await client.chat(
                    model=self.model_name,
                    messages=msgs,
                    tools=tools_metadata,
                    stream=False,
                    options={"temperature": temperature, "num_predict": max_tokens},
                )
                llm_generated_content = response["message"]["content"]
                if llm_generated_content != "":
                    msgs.append(
                        MessageBlock(
                            role=CreatorRole.ASSISTANT.value,
                            content=llm_generated_content,
                        )
                    )

                tool_calls = response["message"]["tool_calls"]
                if tool_calls is None:
                    solved = True
                    break

                output = await self.__call_tools_async(tool_calls)
                msgs.extend(output)
                token_count += response["prompt_eval_count"] + response["eval_count"]
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
            return msgs[number_of_primers:]  # Return only the generated messages
        except Exception as e:
            print(f"run: {e}")
            raise

    async def __call_tools_async(
        self, selectd_tools: list
    ) -> list[MessageBlock | dict]:
        """
        Asynchronously call every selected tools.

        Args:
            selectd_tools (list): A list of selected tools.

        Returns:
            list: A list of messages generated by the tools.

        Notes:
            - If more than one tool is selected, they are executed independently and separately.
            - Tools chaining is not supported.
            - Does not raise exception on failed tool execution, an error message is returned instead to guide the calling LLM.
        """
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
        """
        Synchronously call every selected tools.

        Args:
            selectd_tools (list): A list of selected tools.

        Returns:
            list: A list of messages generated by the tools.

        Notes:
            - If more than one tool is selected, they are executed independently and separately.
            - Tools chaining is not supported.
            - Does not raise exception on failed tool execution, an error message is returned instead to guide the calling LLM.
        """
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