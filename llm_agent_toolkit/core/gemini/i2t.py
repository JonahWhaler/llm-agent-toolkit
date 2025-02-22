import os
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types
from ..._core import Core, ImageInterpreter
from ..._util import CreatorRole, ChatCompletionConfig, MessageBlock
from .base import GeminiCore

logger = logging.getLogger(__name__)


class I2T_GMN_Core(Core, GeminiCore, ImageInterpreter):
    """
    `I2T_GMN_Core` is abstract base classes `Core`, `GeminiCore` and `ImageInterpreter`.
    It facilitates synchronous and asynchronous communication with Gemini's API.

    Methods:
    - run(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> list[MessageBlock | dict]:
        Synchronously run the LLM model with the given query and context.
    - run_async(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> list[MessageBlock | dict]:
        Asynchronously run the LLM model with the given query and context.
    - interpret(query: str, context: list[MessageBlock | dict] | None, filepath: str, **kwargs) -> list[MessageBlock | dict]:
        Synchronously interpret the given image.
    - interpret_async(query: str, context: list[MessageBlock | dict] | None, filepath: str, **kwargs) -> list[MessageBlock | dict]:
        Asynchronously interpret the given image.
    """

    SUPPORTED_IMAGE_FORMATS = (".png", ".jpeg", ".jpg", ".webp")

    def __init__(self, system_prompt: str, config: ChatCompletionConfig):
        Core.__init__(self, system_prompt, config)
        GeminiCore.__init__(self, config.name)
        self.profile = self.build_profile(config.name)

    def custom_config(self, max_output_tokens: int) -> types.GenerateContentConfig:
        """Adapter function.

        Transform custom ChatCompletionConfig -> types.GenerationContentConfig
        """
        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=self.config.temperature,
            max_output_tokens=max_output_tokens,
            # frequency_penalty=0.5,
        )
        return config

    def run(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        """
        Synchronously run the LLM model with the given query and context.

        Args:
            query (str): The query to be processed by the LLM model.
            context (list[MessageBlock | dict] | None): The context to be used for the LLM model.
            **kwargs: Additional keyword arguments.

        Returns:
            list[MessageBlock | dict]: The list of messages generated by the LLM model.

        Notes:
        * Single-Turn Execution.
        """
        msgs: list[types.Content] = []
        output: list[MessageBlock | dict] = []

        if context:
            for ctx in context:
                _role = ctx["role"]
                if _role == "system":
                    # This can happend when user force an system message into the context
                    _role = "model"
                msgs.append(
                    types.Content(
                        role=_role,
                        parts=[types.Part.from_text(text=ctx["content"])],
                    )
                )

        parts = []
        filepath: str | None = kwargs.get("filepath", None)
        if filepath:
            with open(filepath, "rb") as f:
                data = f.read()
            ext = os.path.splitext(filepath)[-1][1:]
            ext = "jpeg" if ext == "jpg" else ext
            parts.append(types.Part.from_bytes(data=data, mime_type=f"image/{ext}"))

        parts.append(types.Part.from_text(text=query))
        msgs.append(
            types.Content(
                role="user",
                parts=parts,
            )
        )
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

        # accumulated_token_count = 0  # Accumulated token count across iterations
        config = self.custom_config(max_output_tokens)
        try:
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            response = client.models.generate_content(
                model=self.model_name,
                contents=msgs,  # type: ignore
                config=config,
            )

            usage = response.usage_metadata
            if usage:
                logger.info("Usage: %s", usage)

            response_text = getattr(response, "text", None)
            if response_text is None:
                raise RuntimeError("response.text is None")

            output.append(
                {"role": CreatorRole.ASSISTANT.value, "content": response_text}
            )
            return output
        except Exception as e:
            logger.error("Exception: %s", e)
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
            usage = response.usage_metadata

            response_text = getattr(response, "text", None)
            return usage, response_text

    async def run_async(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        """
        Asynchronously run the LLM model with the given query and context.

        Args:
            query (str): The query to be processed by the LLM model.
            context (list[MessageBlock | dict] | None): The context to be used for the LLM model.
            **kwargs: Additional keyword arguments.

        Returns:
            list[MessageBlock | dict]: The list of messages generated by the LLM model.

        Notes:
        * Single-Turn Execution.
        """
        msgs: list[types.Content] = []
        output: list[MessageBlock | dict] = []

        if context:
            for ctx in context:
                _role = ctx["role"]
                if _role == "system":
                    # This can happend when user force an system message into the context
                    _role = "model"
                msgs.append(
                    types.Content(
                        role=_role,
                        parts=[types.Part.from_text(text=ctx["content"])],
                    )
                )
        parts = []
        filepath: str | None = kwargs.get("filepath", None)
        if filepath:
            with open(filepath, "rb") as f:
                data = f.read()
            ext = os.path.splitext(filepath)[-1][1:]
            ext = "jpeg" if ext == "jpg" else ext
            parts.append(types.Part.from_bytes(data=data, mime_type=f"image/{ext}"))

        parts.append(types.Part.from_text(text=query))
        msgs.append(
            types.Content(
                role="user",
                parts=parts,
            )
        )

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

        # accumulated_token_count = 0  # Accumulated token count across iterations
        config = self.custom_config(max_output_tokens)
        try:
            usage, response_text = await self.acall(self.model_name, config, msgs)
            if usage:
                logger.info("Usage: %s", usage)

            if response_text is None:
                raise RuntimeError("response.text is None")

            output.append(
                {"role": CreatorRole.ASSISTANT.value, "content": response_text}
            )
            return output
        except Exception as e:
            logger.error("Exception: %s", e)
            raise

    def interpret(
        self,
        query: str,
        context: list[MessageBlock | dict] | None,
        filepath: str,
        **kwargs,
    ):
        ext = os.path.splitext(filepath)[-1]
        if ext not in I2T_GMN_Core.SUPPORTED_IMAGE_FORMATS:
            raise ValueError(f"Unsupported image type: {ext}")

        return self.run(query=query, context=context, filepath=filepath, **kwargs)

    async def interpret_async(
        self,
        query: str,
        context: list[MessageBlock | dict] | None,
        filepath: str,
        **kwargs,
    ):
        ext = os.path.splitext(filepath)[-1]
        if ext not in I2T_GMN_Core.SUPPORTED_IMAGE_FORMATS:
            raise ValueError(f"Unsupported image type: {ext}")

        return await self.run_async(
            query=query, context=context, filepath=filepath, **kwargs
        )
