import os
import logging
import asyncio
import json
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types
from ..._core import Core, ImageInterpreter, ToolSupport
from ..._util import CreatorRole, ChatCompletionConfig, MessageBlock, TokenUsage
from ..._tool import Tool
from .base import GeminiCore, TOOL_PROMPT

logger = logging.getLogger(__name__)


class I2T_GMN_Core_W_Tool(Core, GeminiCore, ToolSupport, ImageInterpreter):
    """
    `I2T_GMN_Core_W_Tool` is abstract base classes `Core`, `ToolSupport`, `ImageInterpreter` and `GeminiCore`.
    It facilitates synchronous and asynchronous communication with Gemini's API.

    Methods:
    - run(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> tuple[list[MessageBlock | dict], TokenUsage]:
        Synchronously run the LLM model with the given query and context.
    - run_async(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> tuple[list[MessageBlock | dict], TokenUsage]:
        Asynchronously run the LLM model with the given query and context.
    - call_tools_async(selected_tools: list) -> tuple[list[MessageBlock | dict], TokenUsage]:
        Asynchronously call tools.
    - call_tools(selected_tools: list) -> tuple[list[MessageBlock | dict], TokenUsage]:
        Synchronously call tools.
    - interpret(query: str, context: list[MessageBlock | dict] | None, filepath: str, **kwargs) -> tuple[list[MessageBlock | dict], TokenUsage]:
        Synchronously interpret the given image.
    - interpret_async(query: str, context: list[MessageBlock | dict] | None, filepath: str, **kwargs) -> tuple[list[MessageBlock | dict], TokenUsage]:
        Asynchronously interpret the given image.

    **Notes:**
    - The caller is responsible for memory management, output parsing and error handling.
    - If model is not available under Gemini's listing, raise ValueError.
    - `context_length` is configurable.
    - `max_output_tokens` is configurable.
    - When MALFORMED_FUNCTION_CALL is encountered, it's probably the selected model's inability to handle the function call. The program will unload the provided tool and a warning will be logged.
    - If not solve, a warning will be logged.
    """

    SUPPORTED_IMAGE_FORMATS = (".png", ".jpeg", ".jpg", ".webp")

    def __init__(
        self,
        system_prompt: str,
        config: ChatCompletionConfig,
        tools: list[Tool] | None = None,
    ):
        Core.__init__(self, system_prompt, config)
        GeminiCore.__init__(self, config.name)
        ToolSupport.__init__(self, tools)
        self.profile = self.build_profile(config.name)

    def gemini_compatible_tool_definition(self):
        tools = []

        if self.tools is None:
            return None

        for tool in self.tools:
            f_info: dict = tool.info["function"]

            properties = {}
            for key, value in f_info["parameters"]["properties"].items():
                properties[key] = types.Schema(
                    type=value["type"],
                    description=value["description"],
                )
            required = []
            if "required" in f_info["parameters"]:
                required = f_info["parameters"]["required"]

            t = types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=f_info["name"],
                        description=f_info["description"],
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties=properties,
                            required=required,
                        ),
                    )
                ]
            )
            tools.append(t)

        return tools

    def custom_config(
        self, max_output_tokens: int, use_tool: bool = False
    ) -> types.GenerateContentConfig:
        """Adapter function.

        Transform custom ChatCompletionConfig -> types.GenerationContentConfig
        """
        si = self.system_prompt
        tools = None
        if use_tool:
            tools = self.gemini_compatible_tool_definition()
            if tools:
                si += f"\n{TOOL_PROMPT}"

        config = types.GenerateContentConfig(
            system_instruction=si,
            temperature=self.config.temperature,
            max_output_tokens=max_output_tokens,
            tools=tools if use_tool else None,
        )

        return config

    def run(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> tuple[list[MessageBlock | dict[str, Any]], TokenUsage]:
        """
        Synchronously run the LLM model with the given query and context.

        Args:
            query (str): The query to be processed by the LLM model.
            context (list[MessageBlock | dict] | None): The context to be used for the LLM model.
            **kwargs: Additional keyword arguments.

        Returns:
            list[MessageBlock | dict]: The list of messages generated by the LLM model.
            TokenUsage: The recorded token usage.
        """
        filepath: str | None = kwargs.get("filepath", None)
        if filepath:
            ext = os.path.splitext(filepath)[-1]
            if ext not in I2T_GMN_Core_W_Tool.SUPPORTED_IMAGE_FORMATS:
                raise ValueError(f"Unsupported image type: {ext}")

        msgs: list[types.Content] = self.preprocessing(query, context, filepath)
        NUMBER_OF_PRIMERS = len(msgs)  # later use this to skip the preloaded messages

        MAX_TOKENS = min(self.config.max_tokens, self.context_length)
        MAX_OUTPUT_TOKENS = min(
            MAX_TOKENS, self.max_output_tokens, self.config.max_output_tokens
        )
        prompt_token_count = self.calculate_token_count(
            self.model_name,
            self.system_prompt,
            msgs,
            imgs=None if filepath is None else [filepath],
        )
        max_output_tokens = min(
            MAX_OUTPUT_TOKENS,
            self.context_length - prompt_token_count,
        )

        config = self.custom_config(max_output_tokens, True)
        iteration, solved = 0, False
        token_usage = TokenUsage(input_tokens=0, output_tokens=0)
        response = None
        try:
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            while (
                not solved
                and max_output_tokens > 0
                and iteration < self.config.max_iteration
                and token_usage.total_tokens < MAX_TOKENS
            ):
                logger.debug("\nIteration [%d]", iteration)
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=msgs,  # type: ignore
                    config=config,
                )
                finish_reason = self.get_finish_reason(response)
                if finish_reason == types.FinishReason.MALFORMED_FUNCTION_CALL:
                    logger.warning("Malformed function call. Unload tools.")
                    config = self.custom_config(max_output_tokens, use_tool=False)

                response_text = self.get_response_text(response)
                if response_text:
                    msgs.append(
                        types.Content(
                            role=CreatorRole.MODEL.value,
                            parts=[types.Part.from_text(text=response_text)],
                        )
                    )

                function_call = self.get_function_call(response)
                if function_call:
                    tool_outputs, tool_token_usage = self.call_tools(
                        selected_tools=[function_call]
                    )
                    f_c_p = self.bind_function_call_response(
                        [function_call], tool_outputs
                    )
                    msgs.extend(f_c_p)
                    token_usage += tool_token_usage

                solved = (
                    function_call is None and finish_reason == types.FinishReason.STOP
                )
                prompt_token_count = self.calculate_token_count(
                    self.model_name,
                    self.system_prompt,
                    msgs,
                    imgs=None if filepath is None else [filepath],
                )
                max_output_tokens = min(
                    MAX_OUTPUT_TOKENS,
                    self.context_length - prompt_token_count,
                )
                iteration += 1
                if iteration == self.config.max_iteration - 1:
                    config = self.custom_config(max_output_tokens, use_tool=False)

                token_usage = self.update_usage(
                    response.usage_metadata, token_usage=token_usage
                )
            # End while

            if not solved:
                warning_message = self.warning_message(
                    iteration,
                    self.config.max_iteration,
                    token_usage,
                    MAX_TOKENS,
                    max_output_tokens,
                )
                logger.warning(warning_message)
                # raise RuntimeError(warning_message)

            output = self.postprocessing(msgs[NUMBER_OF_PRIMERS:])
            return output, token_usage  # Return only the generated messages messages
        except Exception as e:
            if response:
                logger.warning("Response: %s", response)
            logger.error("Exception: %s", e, exc_info=True, stack_info=True)
            raise

    @staticmethod
    async def acall(
        model_name: str, config: types.GenerateContentConfig, msgs: list[types.Content]
    ):
        """Use this to make the `generate_content` method asynchronous."""
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        with ThreadPoolExecutor() as executor:
            future = executor.submit(
                client.models.generate_content,
                model=model_name,
                contents=msgs,  # type: ignore
                config=config,
            )
            response = await asyncio.wrap_future(future)  # Makes the future awaitable
            return response

    async def run_async(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> tuple[list[MessageBlock | dict[str, Any]], TokenUsage]:
        """
        Asynchronously run the LLM model with the given query and context.

        Args:
            query (str): The query to be processed by the LLM model.
            context (list[MessageBlock | dict] | None): The context to be used for the LLM model.
            **kwargs: Additional keyword arguments.

        Returns:
            list[MessageBlock | dict]: The list of messages generated by the LLM model.
            TokenUsage: The recorded token usage.
        """
        filepath: str | None = kwargs.get("filepath", None)
        if filepath:
            ext = os.path.splitext(filepath)[-1]
            if ext not in I2T_GMN_Core_W_Tool.SUPPORTED_IMAGE_FORMATS:
                raise ValueError(f"Unsupported image type: {ext}")

        msgs: list[types.Content] = self.preprocessing(query, context, filepath)
        NUMBER_OF_PRIMERS = len(msgs)  # later use this to skip the preloaded messages

        MAX_TOKENS = min(self.config.max_tokens, self.context_length)
        MAX_OUTPUT_TOKENS = min(
            MAX_TOKENS, self.max_output_tokens, self.config.max_output_tokens
        )
        prompt_token_count = self.calculate_token_count(
            self.model_name,
            self.system_prompt,
            msgs,
            imgs=None if filepath is None else [filepath],
        )
        max_output_tokens = min(
            MAX_OUTPUT_TOKENS,
            self.context_length - prompt_token_count,
        )

        config = self.custom_config(max_output_tokens, use_tool=True)
        iteration, solved = 0, False
        token_usage = TokenUsage(input_tokens=0, output_tokens=0)
        response = None
        try:
            while (
                not solved
                and max_output_tokens > 0
                and iteration < self.config.max_iteration
                and token_usage.total_tokens < MAX_TOKENS
            ):
                logger.debug("\nIteration [%d]", iteration)
                response = await self.acall(self.model_name, config, msgs)
                finish_reason = self.get_finish_reason(response)
                if finish_reason == types.FinishReason.MALFORMED_FUNCTION_CALL:
                    logger.warning("Malformed function call. Unload tools.")
                    config = self.custom_config(max_output_tokens, use_tool=False)

                response_text = self.get_response_text(response)
                if response_text:
                    msgs.append(
                        types.Content(
                            role=CreatorRole.MODEL.value,
                            parts=[types.Part.from_text(text=response_text)],
                        )
                    )

                function_call = self.get_function_call(response)
                if function_call:
                    tool_outputs, tool_token_usage = await self.call_tools_async(
                        selected_tools=[function_call]
                    )
                    f_c_p: list[types.Content] = self.bind_function_call_response(
                        [function_call], tool_outputs
                    )
                    msgs.extend(f_c_p)
                    token_usage += tool_token_usage

                solved = (
                    function_call is None and finish_reason == types.FinishReason.STOP
                )
                prompt_token_count = self.calculate_token_count(
                    self.model_name,
                    self.system_prompt,
                    msgs,
                    imgs=None if filepath is None else [filepath],
                )
                max_output_tokens = min(
                    MAX_OUTPUT_TOKENS,
                    self.context_length - prompt_token_count,
                )
                iteration += 1
                if iteration == self.config.max_iteration - 1:
                    config = self.custom_config(max_output_tokens, use_tool=False)

                token_usage = self.update_usage(
                    response.usage_metadata, token_usage=token_usage
                )
            # End while

            if not solved:
                warning_message = self.warning_message(
                    iteration,
                    self.config.max_iteration,
                    token_usage,
                    MAX_TOKENS,
                    max_output_tokens,
                )
                logger.warning(warning_message)
                # raise RuntimeError(warning_message)

            output = self.postprocessing(msgs[NUMBER_OF_PRIMERS:])
            return output, token_usage  # Return only the generated messages messages
        except Exception as e:
            if response:
                logger.warning("Response: %s", response)
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

                if tool.info["function"]["name"] != tool_call["name"]:
                    continue
                args = tool_call["arguments"]
                args_str = "JSONDecodeError"
                try:
                    args_str = json.dumps(args)
                    result = await tool.run_async(args_str)
                    output.append(
                        MessageBlock(
                            role=CreatorRole.USER.value,
                            content=f"Function {tool_call['name']} called. Args: {args_str} -> Result: {result}",
                        )
                    )
                except json.JSONDecodeError as jde:
                    output.append(
                        {
                            "role": CreatorRole.USER.value,
                            "content": f"Function {tool_call['name']} called. Failed: JSONDecodeError|{str(jde)}",
                        }
                    )
                except Exception as e:
                    output.append(
                        {
                            "role": CreatorRole.USER.value,
                            "content": f"Function {tool_call['name']} called. Args: {args_str} -> Failed: {str(e)}",
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

                if tool.info["function"]["name"] != tool_call["name"]:
                    continue
                args = tool_call["arguments"]

                logger.warning(
                    "Calling %s with %s", tool.info["function"]["name"], args
                )
                try:
                    result = tool.run(json.dumps(args))
                    output.append(
                        MessageBlock(
                            role=CreatorRole.USER.value,
                            content=f"Function {tool_call['name']} called. Result: {result}",
                        )
                    )
                except Exception as e:
                    output.append(
                        {
                            "role": CreatorRole.USER.value,
                            "content": f"Function {tool_call['name']} called. Failed: {str(e)}",
                        }
                    )
                finally:
                    token_usage += tool.token_usage
                    break

        return output, token_usage

    @staticmethod
    def bind_function_call_response(
        fcalls: list[dict[str, Any]], fresps: list[MessageBlock | dict]
    ) -> list[types.Content]:
        """
        Adapter function to bind function call and function_call_response.
        """
        output: list[types.Content] = []
        for fc, fr in zip(fcalls, fresps):
            output.append(
                types.Content(
                    role=CreatorRole.MODEL.value,
                    parts=[
                        types.Part.from_function_call(
                            name=fc["name"],
                            args=fc["arguments"],
                        )
                    ],
                )
            )
            output.append(
                types.Content(
                    role=CreatorRole.USER.value,
                    parts=[types.Part.from_text(text=fr["content"])],
                )
            )
        return output

    def interpret(
        self,
        query: str,
        context: list[MessageBlock | dict] | None,
        filepath: str,
        **kwargs,
    ) -> tuple[list[MessageBlock | dict], TokenUsage]:
        ext = os.path.splitext(filepath)[-1]
        if ext not in I2T_GMN_Core_W_Tool.SUPPORTED_IMAGE_FORMATS:
            raise ValueError(f"Unsupported image type: {ext}")

        return self.run(query=query, context=context, filepath=filepath, **kwargs)

    async def interpret_async(
        self,
        query: str,
        context: list[MessageBlock | dict] | None,
        filepath: str,
        **kwargs,
    ) -> tuple[list[MessageBlock | dict], TokenUsage]:
        ext = os.path.splitext(filepath)[-1]
        if ext not in I2T_GMN_Core_W_Tool.SUPPORTED_IMAGE_FORMATS:
            raise ValueError(f"Unsupported image type: {ext}")

        return await self.run_async(
            query=query, context=context, filepath=filepath, **kwargs
        )
