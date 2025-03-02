import os
import logging
import openai
from ..._core import Core
from ..._util import (
    CreatorRole,
    ChatCompletionConfig,
    MessageBlock,
)
from .base import DeepSeekCore

logger = logging.getLogger(__name__)


class O1Beta_DS_Core(Core, DeepSeekCore):
    SUPPORTED_MODELS = "deepseek-reasoner"

    def __init__(
        self,
        system_prompt: str,
        config: ChatCompletionConfig,
    ):
        if config.name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"{config.name} is not supported. Supported models: {self.SUPPORTED_MODELS}"
            )
        Core.__init__(self, system_prompt, config)
        DeepSeekCore.__init__(self)
        self.profile = self.build_profile(config.name)

    async def run_async(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        """
        Asynchronously run the LLM model with the given query and context.

        Args:
            query (str): The query to be processed by the LLM model.
            context (list[MessageBlock | dict] | None): The context to be used for the LLM model.
            include_rc (bool): Whether to include `reasoning content` in the output, default is True.
            **kwargs: Additional keyword arguments.

        Returns:
            list[MessageBlock | dict]: The list of messages generated by the LLM model.

        Notes:
        * No system prompt!
        * max_tokens -> max_completion_tokens
        """
        # MessageBlock(role=CreatorRole.USER.value, content=self.system_prompt)
        include_rc: bool = kwargs.get("include_rc", True)
        msgs: list[MessageBlock | dict] = []

        if context:
            msgs.extend(context)
        msgs.append(MessageBlock(role=CreatorRole.USER.value, content=query))

        # Determine the maximum number of tokens allowed for the response
        MAX_TOKENS = min(self.config.max_tokens, self.context_length)
        MAX_OUTPUT_TOKENS = min(
            MAX_TOKENS, self.max_output_tokens, self.config.max_output_tokens
        )
        prompt_token_count = self.calculate_token_count(msgs, tools=None)
        max_output_tokens = min(
            MAX_OUTPUT_TOKENS,
            self.context_length - prompt_token_count,
        )

        accumulated_token_count = 0  # Accumulated token count across iterations

        if max_output_tokens <= 0:
            logger.warning("Prompt token count: %d", prompt_token_count)
            raise ValueError("max_output_tokens <= 0")

        try:
            client = openai.AsyncOpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url=os.environ["DEEPSEEK_BASE_URL"],
            )
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=msgs,  # type: ignore
                max_tokens=max_output_tokens,
            )

            choice = response.choices[0]
            _content = getattr(choice.message, "content", None)
            _reasoning_content = getattr(choice.message, "reasoning_content", None)

            accumulated_token_count += (
                response.usage.total_tokens if response.usage else 0
            )
            logger.info("Usage: %s", response.usage)
            if _content:
                response_string = _content
                if _reasoning_content and include_rc:
                    response_string = (
                        f"<COT>\n{_reasoning_content}\n</COT>\n" + response_string
                    )
                return [
                    {"role": CreatorRole.ASSISTANT.value, "content": response_string}
                ]

            failed_reason = choice.finish_reason
            raise RuntimeError(failed_reason)
        except Exception as e:
            logger.error("Exception: %s", e)
            raise

    def run(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        """
        Synchronously generate text based on the given query and context.

        Args:
            query (str): The query to generate text for.
            context (list): A list of context messages or dictionaries.
            include_rc (bool): Whether to include `reasoning content` in the output, default is True.
            **kwargs: Additional keyword arguments.

        Returns:
            list[MessageBlock | dict]: The list of messages generated by the LLM model.

        Notes:
        * No system prompt!
        * max_tokens -> max_completion_tokens
        """
        # MessageBlock(role=CreatorRole.USER.value, content=self.system_prompt)
        include_rc: bool = kwargs.get("include_rc", True)
        msgs: list[MessageBlock | dict] = []

        if context:
            msgs.extend(context)
        msgs.append(MessageBlock(role=CreatorRole.USER.value, content=query))

        # Determine the maximum number of tokens allowed for the response
        MAX_TOKENS = min(self.config.max_tokens, self.context_length)
        MAX_OUTPUT_TOKENS = min(
            MAX_TOKENS, self.max_output_tokens, self.config.max_output_tokens
        )
        prompt_token_count = self.calculate_token_count(msgs, tools=None)
        max_output_tokens = min(
            MAX_OUTPUT_TOKENS,
            self.context_length - prompt_token_count,
        )

        accumulated_token_count = 0  # Accumulated token count across iterations

        if max_output_tokens <= 0:
            logger.warning("Prompt token count: %d", prompt_token_count)
            raise ValueError("max_output_tokens <= 0")
        logger.info("max output tokens: %d", max_output_tokens)
        logger.info("prompt token count: %d", prompt_token_count)

        try:
            client = openai.OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url=os.environ["DEEPSEEK_BASE_URL"],
            )

            response = client.chat.completions.create(
                model=self.model_name,
                messages=msgs,  # type: ignore
                max_tokens=max_output_tokens,
            )

            choice = response.choices[0]
            _content = getattr(choice.message, "content", None)
            _reasoning_content = getattr(choice.message, "reasoning_content", None)

            accumulated_token_count += (
                response.usage.total_tokens if response.usage else 0
            )
            logger.info("Usage: %s", response.usage)
            if _content:
                response_string = _content
                if _reasoning_content and include_rc:
                    response_string = (
                        f"<COT>\n{_reasoning_content}\n</COT>\n" + response_string
                    )  # re.sub(r"<COT>.*?</COT>\n*", "", x, flags=re.DOTALL)
                return [
                    {"role": CreatorRole.ASSISTANT.value, "content": response_string}
                ]
            failed_reason = choice.finish_reason
            raise RuntimeError(failed_reason)
        except RuntimeError as rte:
            logger.error("RuntimeError: %s", rte)
            raise
        except Exception as e:
            logger.error("Exception: %s", e)
            raise
