import os
import logging
from typing import Any

import ollama

from ..._core import Core, ImageInterpreter
from ..._util import CreatorRole, ChatCompletionConfig, MessageBlock, TokenUsage
from .base import OllamaCore

logger = logging.getLogger(__name__)


class I2T_OLM_Core(Core, OllamaCore, ImageInterpreter):  # , ToolSupport
    """
    `I2T_OLM_Core` is a concrete implementation of the abstract base classes `Core` and `ImageInterpreter`.
    `I2T_OLM_Core` is also a child class of `OllamaCore`.

    It facilitates synchronous and asynchronous communication with Ollama's API to interpret images.

    **Methods:**
    - run(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> tuple[list[MessageBlock | dict], TokenUsage]:
        Synchronously run the LLM model to interpret images.
    - run_async(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> tuple[list[MessageBlock | dict], TokenUsage]:
        Asynchronously run the LLM model to interpret images.
    - interpret(query: str, context: list[MessageBlock | dict] | None, filepath: str, **kwargs) -> tuple[list[MessageBlock | dict], TokenUsage]:
        Synchronously interpret the given image.
    - interpret_async(query: str, context: list[MessageBlock | dict] | None, filepath: str, **kwargs) -> tuple[list[MessageBlock | dict], TokenUsage]:
        Asynchronously interpret the given image.

    **Notes:**
    - Supported image format: .png, .jpeg, .jpg, .gif, .webp
    - Tools are not supported. At the point of implementing, Ollama have not release models that support both vision and tool.
    - The caller is responsible for memory management, output parsing and error handling.
    - The caller is responsible for choosing models that support `Vision`.
    - If model is not available locally, an attempt to pull it from Ollama's server will be made.
    - `context_length` is configurable.
    - `max_output_tokens` is configurable.
    """

    SUPPORTED_IMAGE_FORMATS = (".png", ".jpeg", ".jpg", ".gif", ".webp")

    def __init__(
        self,
        connection_string: str,
        system_prompt: str,
        config: ChatCompletionConfig,
    ):
        assert isinstance(config, ChatCompletionConfig)
        Core.__init__(self, system_prompt, config)
        OllamaCore.__init__(self, connection_string, config.name)
        self.profile = self.build_profile(model_name=config.name)
        if self.profile.get("image_input", False) is False:
            logger.warning("Vision might not work on this %s", self.model_name)

    async def run_async(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> tuple[list[MessageBlock | dict[str, Any]], TokenUsage]:
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
        * Early Termination Condition: Not applicable, only run one iteration.
        """
        msgs: list[MessageBlock | dict] = [
            MessageBlock(role=CreatorRole.SYSTEM.value, content=self.system_prompt)
        ]

        if context:
            msgs.extend(context)

        filepath: str | None = kwargs.get("filepath", None)
        if filepath:
            ext = os.path.splitext(filepath)[-1]
            ext = ext.lower()
            if ext not in I2T_OLM_Core.SUPPORTED_IMAGE_FORMATS:
                raise ValueError(f"Unsupported image type: {ext}")
            msgs.append(
                {"role": CreatorRole.USER.value, "content": query, "images": [filepath]}
            )
        else:
            msgs.append(MessageBlock(role=CreatorRole.USER.value, content=query))

        NUMBER_OF_PRIMERS = len(msgs)  # later use this to skip the preloaded messages

        MAX_TOKENS = min(self.config.max_tokens, self.context_length)
        MAX_OUTPUT_TOKENS = min(
            MAX_TOKENS, self.max_output_tokens, self.config.max_output_tokens
        )
        prompt_token_count = self.calculate_token_count(
            msgs, None, images=[filepath] if filepath else None
        )
        max_output_tokens = min(
            MAX_OUTPUT_TOKENS,
            self.context_length - prompt_token_count,
        )
        if max_output_tokens <= 0:
            logger.warning("Prompt token count: %d", prompt_token_count)
            raise ValueError("max_output_tokens <= 0")
        try:
            client = ollama.AsyncClient(host=self.CONN_STRING)
            response = await client.chat(
                model=self.model_name,
                messages=msgs,
                tools=None,
                stream=False,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": max_output_tokens,
                },
            )

            llm_generated_content = response["message"]["content"]
            msgs.append(
                MessageBlock(
                    role=CreatorRole.ASSISTANT.value,
                    content=llm_generated_content,
                )
            )
            token_usage = self.update_usage(response)  # type: ignore

            return msgs[
                NUMBER_OF_PRIMERS:
            ], token_usage  # Return only the generated messages
        except Exception as e:
            logger.error("Exception: %s", e, exc_info=True, stack_info=True)
            raise

    def run(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> tuple[list[MessageBlock | dict[str, Any]], TokenUsage]:
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
        * Early Termination Condition: Not applicable, only run one iteration.
        """
        msgs: list[MessageBlock | dict] = [
            MessageBlock(role=CreatorRole.SYSTEM.value, content=self.system_prompt)
        ]

        if context:
            msgs.extend(context)

        filepath: str | None = kwargs.get("filepath", None)
        if filepath:
            ext = os.path.splitext(filepath)[-1]
            ext = ext.lower()
            if ext not in I2T_OLM_Core.SUPPORTED_IMAGE_FORMATS:
                raise ValueError(f"Unsupported image type: {ext}")
            msgs.append(
                {"role": CreatorRole.USER.value, "content": query, "images": [filepath]}
            )
        else:
            msgs.append(MessageBlock(role=CreatorRole.USER.value, content=query))

        NUMBER_OF_PRIMERS = len(msgs)  # later use this to skip the preloaded messages

        MAX_TOKENS = min(self.config.max_tokens, self.context_length)
        MAX_OUTPUT_TOKENS = min(
            MAX_TOKENS, self.max_output_tokens, self.config.max_output_tokens
        )
        prompt_token_count = self.calculate_token_count(
            msgs, None, images=[filepath] if filepath else None
        )
        max_output_tokens = min(
            MAX_OUTPUT_TOKENS,
            self.context_length - prompt_token_count,
        )
        if max_output_tokens <= 0:
            logger.warning("Prompt token count: %d", prompt_token_count)
            raise ValueError("max_output_tokens <= 0")

        try:
            client = ollama.Client(host=self.CONN_STRING)
            response = client.chat(
                model=self.model_name,
                messages=msgs,
                tools=None,
                stream=False,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": max_output_tokens,
                },
            )

            llm_generated_content = response["message"]["content"]
            msgs.append(
                MessageBlock(
                    role=CreatorRole.ASSISTANT.value,
                    content=llm_generated_content,
                )
            )

            token_usage = self.update_usage(response)  # type: ignore

            return msgs[
                NUMBER_OF_PRIMERS:
            ], token_usage  # Return only the generated messages
        except Exception as e:
            logger.error("Exception: %s", e, exc_info=True, stack_info=True)
            raise

    def interpret(
        self,
        query: str,
        context: list[MessageBlock | dict] | None,
        filepath: str,
        **kwargs,
    ) -> tuple[list[MessageBlock | dict[str, Any]], TokenUsage]:
        return self.run(query=query, context=context, filepath=filepath, **kwargs)

    async def interpret_async(
        self,
        query: str,
        context: list[MessageBlock | dict] | None,
        filepath: str,
        **kwargs,
    ) -> tuple[list[MessageBlock | dict[str, Any]], TokenUsage]:
        return await self.run_async(
            query=query, context=context, filepath=filepath, **kwargs
        )
