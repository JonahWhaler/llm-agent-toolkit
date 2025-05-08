import os
import logging
import time
import asyncio
import json
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from google import genai
from google.genai import types
from ..._core import Core, ImageInterpreter, ToolSupport
from ..._util import CreatorRole, ChatCompletionConfig, MessageBlock, TokenUsage
from ..._tool import Tool
from .base import GeminiCore, TOOL_PROMPT

logger = logging.getLogger(__name__)


class I2T_GMN_Core_W_Tool(Core, GeminiCore, ToolSupport, ImageInterpreter):
    """
    `I2T_GMN_Core_W_Tool` provides a unified multimodal interface to Gemini LLM models.

    Attributes:
        SUPPORTED_IMAGE_FORMATS (tuple[str]): A tuple of supported image formats.
        MAX_ATTEMPT (int): The maximum number of retry attempts for API calls.
        DELAY_FACTOR (float): The factor by which the delay increases after each retry.
        MAX_DELAY (float): The maximum delay between retries.

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

    **Features**:
    - Image interpretation: PNG, JPEG, JPG, WEBP
    - Function Calling: Provide function declaration and execute LLM selected functions locally.

    **Notes:**
    - Input: Text & Image only
    - Output: Text only
    """

    SUPPORTED_IMAGE_FORMATS = (".png", ".jpeg", ".jpg", ".webp")
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
            output (tuple[list[MessageBlock | dict], TokenUsage]):
                1. The list of messages generated by the LLM model.
                2. The recorded token usage.
        """
        filepath: str | None = kwargs.get("filepath", None)
        if filepath:
            ext = os.path.splitext(filepath)[-1]
            if ext not in I2T_GMN_Core_W_Tool.SUPPORTED_IMAGE_FORMATS:
                raise ValueError(f"Unsupported image type: {ext}")

        msgs: list[types.Content] = self.preprocessing(query, context, filepath)
        # later use this to skip the preloaded messages
        number_of_inputs = len(msgs)

        MAX_TOKENS = min(self.config.max_tokens, self.context_length)
        MAX_OUTPUT_TOKENS = min(
            MAX_TOKENS, self.max_output_tokens, self.config.max_output_tokens
        )

        attempt: int = 1
        delay: float = 5.0
        while attempt < I2T_GMN_Core_W_Tool.MAX_ATTEMPT:
            logger.debug("Attempt %d", attempt)
            messages = deepcopy(msgs)
            prompt_token_count = self.calculate_token_count(
                self.model_name,
                self.system_prompt,
                messages,
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
                        contents=messages,  # type: ignore
                        config=config,
                    )
                    token_usage = self.update_usage(
                        response.usage_metadata, token_usage=token_usage
                    )

                    candidates: Optional[list[types.Candidate]] = response.candidates
                    if candidates is None or len(candidates) == 0:
                        raise RuntimeError(
                            f"Malformed response (No candidates): {response}"
                        )

                    candidate = candidates[0]
                    finish_reason = candidate.finish_reason
                    content: Optional[types.Content] = candidate.content
                    if content is None:
                        raise RuntimeError(
                            f"Malformed response (No content): {candidate}"
                        )

                    texts = self.get_texts(content)
                    functs = self.get_functs(content)
                    logger.warning("Finish Reason: %s", finish_reason)
                    logger.warning("Texts: %s", texts)
                    logger.warning("Functions: %s", functs)
                    if finish_reason == types.FinishReason.STOP and functs:
                        logger.info("Function call detected.")
                        if texts:
                            messages.append(
                                types.Content(
                                    role=CreatorRole.MODEL.value,
                                    parts=[
                                        types.Part.from_text(text=text)
                                        for text in texts
                                    ],
                                )
                            )
                        tool_outputs, tool_token_usage = self.call_tools(
                            selected_tools=functs
                        )
                        f_c_p = self.bind_function_call_response(functs, tool_outputs)
                        messages.extend(f_c_p)
                        token_usage += tool_token_usage
                        iteration += 1
                        prompt_token_count = self.calculate_token_count(
                            self.model_name,
                            self.system_prompt,
                            messages,
                            imgs=None if filepath is None else [filepath],
                        )
                        max_output_tokens = min(
                            MAX_OUTPUT_TOKENS,
                            self.context_length - prompt_token_count,
                        )
                    elif finish_reason == types.FinishReason.STOP and texts:
                        logger.info("SOLVED")
                        messages.append(
                            types.Content(
                                role=CreatorRole.MODEL.value,
                                parts=[
                                    types.Part.from_text(text=text) for text in texts
                                ],
                            )
                        )
                        solved = True
                    elif finish_reason == types.FinishReason.MAX_TOKENS and texts:
                        logger.warning("Terminated due to length.")
                        e = {
                            "error": "Early Termination: Length",
                            "text": "\n".join(texts),
                        }
                        messages.append(
                            types.Content(
                                role=CreatorRole.MODEL.value,
                                parts=[
                                    types.Part.from_text(
                                        text=json.dumps(e, ensure_ascii=False)
                                    )
                                ],
                            )
                        )
                        solved = True
                    else:
                        logger.warning("Malformed response: %s", response)
                        logger.warning("Config: %s", self.config)
                        raise RuntimeError(f"Terminated: {finish_reason}")
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

                # Return only the generated messages messages
                output = self.postprocessing(messages[number_of_inputs:])
                return output, token_usage
            except Exception as e:
                if "502 Bad Gateway" in str(e):
                    logger.warning("RateLimitError: %s", e)
                    warn_msg = f"[{attempt}] Retrying in {delay} seconds..."
                    logger.warning(warn_msg)
                    time.sleep(delay)
                    attempt += 1
                    delay = delay * I2T_GMN_Core_W_Tool.DELAY_FACTOR
                    delay = min(I2T_GMN_Core_W_Tool.MAX_DELAY, delay)
                    continue

                raise

        raise RuntimeError("Max re-attempt reached")

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
            output (tuple[list[MessageBlock | dict], TokenUsage]):
                1. The list of messages generated by the LLM model.
                2. The recorded token usage.
        """
        filepath: str | None = kwargs.get("filepath", None)
        if filepath:
            ext = os.path.splitext(filepath)[-1]
            if ext not in I2T_GMN_Core_W_Tool.SUPPORTED_IMAGE_FORMATS:
                raise ValueError(f"Unsupported image type: {ext}")

        msgs: list[types.Content] = self.preprocessing(query, context, filepath)
        # later use this to skip the preloaded messages
        number_of_inputs = len(msgs)

        MAX_TOKENS = min(self.config.max_tokens, self.context_length)
        MAX_OUTPUT_TOKENS = min(
            MAX_TOKENS, self.max_output_tokens, self.config.max_output_tokens
        )

        attempt: int = 1
        delay: float = 5.0
        while attempt < I2T_GMN_Core_W_Tool.MAX_ATTEMPT:
            logger.debug("Attempt %d", attempt)
            messages = deepcopy(msgs)
            prompt_token_count = self.calculate_token_count(
                self.model_name,
                self.system_prompt,
                messages,
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
                while (
                    not solved
                    and max_output_tokens > 0
                    and iteration < self.config.max_iteration
                    and token_usage.total_tokens < MAX_TOKENS
                ):
                    logger.debug("\nIteration [%d]", iteration)
                    response = await self.acall(self.model_name, config, messages)
                    token_usage = self.update_usage(
                        response.usage_metadata, token_usage=token_usage
                    )

                    candidates: Optional[list[types.Candidate]] = response.candidates
                    if candidates is None or len(candidates) == 0:
                        raise RuntimeError(
                            f"Malformed response (No candidates): {response}"
                        )

                    candidate = candidates[0]
                    finish_reason = candidate.finish_reason
                    content: Optional[types.Content] = candidate.content
                    if content is None:
                        raise RuntimeError(
                            f"Malformed response (No content): {candidate}"
                        )

                    texts = self.get_texts(content)
                    functs = self.get_functs(content)
                    logger.warning("Finish Reason: %s", finish_reason)
                    logger.warning("Texts: %s", texts)
                    logger.warning("Functions: %s", functs)
                    if finish_reason == types.FinishReason.STOP and functs:
                        logger.info("Function call detected.")
                        if texts:
                            messages.append(
                                types.Content(
                                    role=CreatorRole.MODEL.value,
                                    parts=[
                                        types.Part.from_text(text=text)
                                        for text in texts
                                    ],
                                )
                            )
                        tool_outputs, tool_token_usage = self.call_tools(
                            selected_tools=functs
                        )
                        f_c_p = self.bind_function_call_response(functs, tool_outputs)
                        messages.extend(f_c_p)
                        token_usage += tool_token_usage
                        iteration += 1
                        prompt_token_count = self.calculate_token_count(
                            self.model_name,
                            self.system_prompt,
                            messages,
                            imgs=None if filepath is None else [filepath],
                        )
                        max_output_tokens = min(
                            MAX_OUTPUT_TOKENS,
                            self.context_length - prompt_token_count,
                        )
                    elif finish_reason == types.FinishReason.STOP and texts:
                        logger.info("SOLVED")
                        messages.append(
                            types.Content(
                                role=CreatorRole.MODEL.value,
                                parts=[
                                    types.Part.from_text(text=text) for text in texts
                                ],
                            )
                        )
                        solved = True
                    elif finish_reason == types.FinishReason.MAX_TOKENS and texts:
                        logger.warning("Terminated due to length.")
                        e = {
                            "error": "Early Termination: Length",
                            "text": "\n".join(texts),
                        }
                        messages.append(
                            types.Content(
                                role=CreatorRole.MODEL.value,
                                parts=[
                                    types.Part.from_text(
                                        text=json.dumps(e, ensure_ascii=False)
                                    )
                                ],
                            )
                        )
                        solved = True
                    else:
                        logger.warning("Malformed response: %s", response)
                        logger.warning("Config: %s", self.config)
                        raise RuntimeError(f"Terminated: {finish_reason}")
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

                # Return only the generated messages messages
                output = self.postprocessing(messages[number_of_inputs:])
                return output, token_usage
            except Exception as e:
                if "502 Bad Gateway" in str(e):
                    logger.warning("RateLimitError: %s", e)
                    warn_msg = f"[{attempt}] Retrying in {delay} seconds..."
                    logger.warning(warn_msg)
                    time.sleep(delay)
                    attempt += 1
                    delay = delay * I2T_GMN_Core_W_Tool.DELAY_FACTOR
                    delay = min(I2T_GMN_Core_W_Tool.MAX_DELAY, delay)
                    continue

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
            output (tuple[list[MessageBlock | dict], TokenUsage]):
                1. A list of messages generated by the tools.
                2. The recorded token usage.

        Notes:
            - If more than one tool is selected, they are executed independently and separately.
            - Static tools chaining is not supported.
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
                    args_str = json.dumps(args, ensure_ascii=False)
                    result = await tool.run_async(args_str)
                    output_string = f"{tool_call['name']}({args_str}) -> {result}"
                    output.append(
                        MessageBlock(
                            role=CreatorRole.USER.value,
                            content=output_string,
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
                            "content": f"{tool_call['name']}({args_str}) -> {e}",
                        }
                    )
                finally:
                    token_usage += tool.token_usage

        return output, token_usage

    def call_tools(
        self, selected_tools: list
    ) -> tuple[list[MessageBlock | dict], TokenUsage]:
        """
        Synchronously call every selected tools.

        Args:
            selected_tools (list): A list of selected tools.

        Returns:
            output (tuple[list[MessageBlock | dict], TokenUsage]):
                1. A list of messages generated by the tools.
                2. The recorded token usage.

        Notes:
            - If more than one tool is selected, they are executed independently and separately.
            - Static tools chaining is not supported.
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
                    args_str = json.dumps(args, ensure_ascii=False)
                    result = tool.run(args_str)
                    output_string = f"{tool_call['name']}({args_str}) -> {result}"
                    output.append(
                        MessageBlock(
                            role=CreatorRole.USER.value,
                            content=output_string,
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
                            "content": f"{tool_call['name']}({args_str}) -> {e}",
                        }
                    )
                finally:
                    token_usage += tool.token_usage

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
