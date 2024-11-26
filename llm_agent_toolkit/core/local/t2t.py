# import os
import json
import warnings
import logging
import ollama
from ..._core import Core
from ..._util import (
    CreatorRole,
    ChatCompletionConfig,
    MessageBlock,
)

from ..._tool import Tool


logger = logging.getLogger(__name__)


TOOL_PROMPT = """
Utilize tools to solve the problems. 
Results from tools will be kept in the context. 
Calling the tools repeatedly is highly discouraged.
"""


class T2T_OLM_Core(Core):
    def __init__(
        self,
        connection_string: str,
        system_prompt: str,
        config: ChatCompletionConfig,
        tools: list[Tool] | None = None,
    ):
        assert isinstance(config, ChatCompletionConfig)
        super().__init__(system_prompt, config, tools)
        self.__connection_string = connection_string

    @property
    def CONN_STRING(self) -> str:
        return self.__connection_string

    def run(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
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
            logger.info(TOOL_PROMPT)
        else:
            tools_metadata = None
        number_of_primers = len(msgs)
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
            logger.error("Error: %s", e)
            raise

    async def run_async(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
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
            logger.info(TOOL_PROMPT)
        else:
            tools_metadata = None
        number_of_primers = len(msgs)
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
            logger.error("Error: %s", e)
            raise

    async def __call_tools_async(
        self, selected_tools: list
    ) -> list[MessageBlock | dict]:
        output: list[MessageBlock | dict] = []

        for tool_call in selected_tools:
            for tool in self.tools:  # type: ignore
                if tool.info["function"]["name"] != tool_call.function.name:
                    continue
                args = json.dumps(tool_call.function.arguments)
                try:
                    result = await tool.run_async(args)
                    output.append(
                        {
                            "role": CreatorRole.TOOL.value,
                            "name": tool_call.function.name,
                            "content": f"({args}) => {result}",
                        }
                    )
                except Exception as e:
                    output.append(
                        MessageBlock(
                            role=CreatorRole.TOOL.value,
                            content=f"({args}) => {e}",
                            name=tool_call.function.name,
                        )
                    )

        return output

    def __call_tools(self, selected_tools: list) -> list[MessageBlock | dict]:
        output: list[MessageBlock | dict] = []

        for tool_call in selected_tools:
            for tool in self.tools:  # type: ignore
                if tool.info["function"]["name"] != tool_call.function.name:
                    continue
                args = json.dumps(tool_call.function.arguments)
                try:
                    result = tool.run(args)
                    output.append(
                        {
                            "role": CreatorRole.TOOL.value,
                            "name": tool_call.function.name,
                            "content": f"({args}) => {result}",
                        }
                    )
                except Exception as e:
                    output.append(
                        MessageBlock(
                            role=CreatorRole.TOOL.value,
                            content=f"({args}) => {e}",
                            name=tool_call.function.name,
                        )
                    )

        return output