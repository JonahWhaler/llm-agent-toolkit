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
    """
    `T2T_OLM_Core` is a concrete implementation of the `Core` abstract class.
    It facilitates synchronous and asynchronous communication with ollama's API.

    Methods:
    - run(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> list[MessageBlock | dict]:
        Synchronously run the LLM model with the given query and context.
    - run_async(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> list[MessageBlock | dict]:
        Asynchronously run the LLM model with the given query and context.

    Notes:
    - Loop until a solution is found, or maximum iteration or token count is reached.
    - The caller is responsible for memory management, output parsing and error handling.
    - The caller is responsible for choosing models that support `Tools`.
    - If model is not available locally, pull it from Ollama's server.
    - `context_length` is configurable.
    """

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
        if not self.__available():
            self.__try_pull_model()
        self.__profile = self.__build_profile(model_name=config.name)
        if tools and self.profile["tool"] is False:
            logger.warning("Tool might not work on this %s", self.model_name)

    def __available(self) -> bool:
        try:
            client = ollama.Client(host=self.CONN_STRING)
            lst = list(client.list())[0]
            _, m, *_ = lst
            for _m in m:
                if _m.model == self.model_name:
                    logger.info("Found %s => %s", self.model_name, _m)
                    return True
            return False
        except ollama.RequestError as ore:
            logger.error("RequestError: %s", str(ore))
            raise
        except Exception as e:
            logger.error("Exception: %s", str(e))
            raise

    def __try_pull_model(self):
        """
        Attempt to pull the required model from ollama's server.

        **Raises:**
            ollama.ResponseError: pull model manifest: file does not exist
        """
        try:
            client = ollama.Client(host=self.CONN_STRING)
            _ = client.pull(self.model_name, stream=False)
        except ollama.RequestError as oreqe:
            logger.error("RequestError: %s", str(oreqe))
            raise
        except ollama.ResponseError as orespe:
            logger.error("ResponseError: %s", str(orespe))
            raise
        except Exception as e:
            logger.error("Exception: %s (%s)", str(e), type(e))
            raise

    @staticmethod
    def __build_profile(model_name: str) -> dict[str, bool | int | str]:
        """
        Build the profile dict based on information found in ./llm_agent_toolkit/core/local/ollama.csv

        These are the models which the developer has experience with.
        If `model_name` is not found in the csv file, default value will be applied.

        Call .set_context_length to set the context length, default value is 2048.
        """
        logger.info("Building profile...")
        profile: dict[str, bool | int | str] = {"name": model_name}
        with open(
            "./llm_agent_toolkit/core/local/ollama.csv", "r", encoding="utf-8"
        ) as csv:
            header = csv.readline()
            columns = header.strip().split(",")
            while True:
                line = csv.readline()
                if not line:
                    break
                values = line.strip().split(",")
                if values[0] == model_name:
                    for column, value in zip(columns[1:], values[1:]):
                        if column == "context_length":
                            profile[column] = int(value)
                        elif column == "remarks":
                            profile[column] = value
                        elif value == "TRUE":
                            profile[column] = True
                        else:
                            profile[column] = False
                    break

        if "context_length" not in profile:
            # Most supported context length
            profile["context_length"] = 2048
        if "tool" not in profile:
            # Assume supported
            profile["tool"] = True
        if "text_generation" not in profile:
            # Assume supported
            profile["text_generation"] = True
        logger.info("Profile ready")
        return profile

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

    @property
    def CONN_STRING(self) -> str:
        return self.__connection_string

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

        max_tokens = min(max_tokens, self.context_length)

        iteration = 0
        token_count = 0
        solved = False
        try:
            client = ollama.Client(host=self.CONN_STRING)
            while iteration < self.config.max_iteration and token_count < max_tokens:
                # print(f"\n\nIteration: {iteration}")
                response = client.chat(
                    model=self.model_name,
                    messages=msgs,
                    tools=tools_metadata,
                    stream=False,
                    options={"temperature": temperature, "num_predict": max_tokens},
                )
                token_count += response["eval_count"] + response["prompt_eval_count"]

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

        max_tokens = min(max_tokens, self.context_length)

        iteration = 0
        token_count = 0
        solved = False
        try:
            client = ollama.AsyncClient(host=self.CONN_STRING)
            while iteration < self.config.max_iteration and token_count < max_tokens:
                # print(f"\n\nIteration: {iteration}")
                response = await client.chat(
                    model=self.model_name,
                    messages=msgs,
                    tools=tools_metadata,
                    stream=False,
                    options={"temperature": temperature, "num_predict": max_tokens},
                )
                token_count += response["eval_count"] + response["prompt_eval_count"]

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

    async def __call_tools_async(
        self, selected_tools: list
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
