import os
import warnings
import base64
import openai
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


class I2T_OAI_Core(I2T_Core):
    """
    `I2T_OAI_Core` is a concrete implementation of the `I2T_Core` abstract base class.
    It facilitates synchronous and asynchronous communication with OpenAI's API to interpret images.

    **Methods:**
    - run(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> list[MessageBlock | dict]:
        Synchronously run the LLM model to interpret images.
    - run_async(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> list[MessageBlock | dict]:
        Asynchronously run the LLM model to interpret images.
    - get_image_url(filepath: str) -> str:
        Returns the URL of the image from the specified file path.
    - __call_tools_async(selectd_tools: list) -> list[MessageBlock | dict]:
        Asynchronously call tools.
    - __call_tools(selectd_tools: list) -> list[MessageBlock | dict]:
        Synchronously call tools.

    **Notes:**
    - Supported image format: .png, .jpeg, .gif, .webp
    - Tools are supported.
    """

    SUPPORTED_IMAGE_FORMATS = (".png", ".jpeg", ".gif", ".webp")

    def __init__(
        self,
        system_prompt: str,
        config: ChatCompletionConfig,
        tools: list | None = None,
    ):
        assert isinstance(config, ChatCompletionConfig)
        super().__init__(system_prompt, config, tools)

    async def run_async(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        """
        Synchronously run the LLM model to interpret images.

        Args:
            query (str): The query to be interpreted.
            context (list[MessageBlock | dict] | None): The context to be used for the query.
            filepath (str): The path to the image file to be interpreted.

        Returns:
            list[MessageBlock | dict]: The list of messages generated by the LLM model.
        """
        msgs: list[MessageBlock | dict] = [
            MessageBlock(role=CreatorRole.SYSTEM.value, content=self.system_prompt)
        ]

        if context is not None:
            msgs.extend(context)

        filepath: str | None = kwargs.get("filepath", None)
        if filepath is not None:
            img_url = self.get_image_url(filepath)
            msgs.append(
                {
                    "role": CreatorRole.USER.value,
                    "content": [{"type": "image_url", "image_url": {"url": img_url}}],  # type: ignore
                }
            )
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
            max_tokens = 4096
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
                msgs.append(
                    MessageBlock(role=CreatorRole.ASSISTANT.value, content=_content)
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
            return msgs[number_of_primers:]  # Return only the generated messages
        except Exception as e:
            # print(f"run: {e}")
            raise

    def run(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        """
        Synchronously run the LLM model to interpret images.

        Args:
            query (str): The query to be interpreted.
            context (list[MessageBlock | dict] | None): The context to be used for the query.
            filepath (str): The path to the image file to be interpreted.

        Returns:
            list[MessageBlock | dict]: The list of messages generated by the LLM model.
        """
        msgs: list[MessageBlock | dict] = [
            MessageBlock(role=CreatorRole.SYSTEM.value, content=self.system_prompt)
        ]

        if context is not None:
            msgs.extend(context)

        filepath: str | None = kwargs.get("filepath", None)
        if filepath is not None:
            img_url = self.get_image_url(filepath)
            msgs.append(
                {
                    "role": CreatorRole.USER.value,
                    "content": [{"type": "image_url", "image_url": {"url": img_url}}],  # type: ignore
                }
            )
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
            max_tokens = 4096
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
                        MessageBlock(role=CreatorRole.ASSISTANT.value, content=_content)
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
            return msgs[number_of_primers:]  # Return only the generated messages
        except Exception as e:
            # print(f"run: {e}")
            raise

    @staticmethod
    def get_image_url(filepath: str):
        ext = filepath.split(".")[-1].lower()
        ext = "jpeg" if ext == "jpg" else ext
        if ext not in I2T_OAI_Core.SUPPORTED_IMAGE_FORMATS:
            raise ValueError(f"Unsupported image type: {ext}")
        prefix = f"data:image/{ext};base64"
        try:
            with open(filepath, "rb") as f:
                encoded_image = base64.b64encode(f.read()).decode("utf-8")
                return f"{prefix},{encoded_image}"
        except Exception as e:
            # print(f"get_image_url: {e}")
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
