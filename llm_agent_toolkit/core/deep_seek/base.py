import json
import logging
from math import ceil
from typing import Any

import openai
from ..._util import MessageBlock, TokenUsage
from ..._tool import ToolMetadata

logger = logging.getLogger(__name__)


class DeepSeekCore:
    def __init__(
        self,
    ):
        pass

    @staticmethod
    def build_profile(model_name: str = "deepseek-chat") -> dict[str, bool | int | str]:
        if model_name == "deepseek-reasoner":
            return {
                "context_length": 64_000,
                "max_output_tokens": 8_000,
                "text_generation": True,
                "tool": False,
            }  # MAX COT TOKEN = 32K
        return {
            "context_length": 64_000,
            "max_output_tokens": 8_000,
            "text_generation": True,
            "tool": True,
        }

    def calculate_token_count(
        self,
        msgs: list[MessageBlock | dict[str, Any]],
        tools: list[ToolMetadata] | None = None,
    ) -> int:
        """Calculate the token count for the given messages.
        Efficient but not accurate. Child classes should implement a more accurate version.

        Args:
            msgs (list[MessageBlock | dict[str, Any]]): A list of messages.
            tools (list[ToolMetadata] | None): A list of tool description, default to None.

        Returns:
            int: The token count.

        Notes:
        * https://api-docs.deepseek.com/quick_start/token_usage
        * Remove whitespaces
        """
        CONVERSION_FACTOR = 0.6
        character_count: int = 0
        for msg in msgs:
            # Incase the dict does not comply with the MessageBlock format
            if not isinstance(msg, dict):
                continue
            if "content" in msg and msg["content"]:
                cleaned_content = msg["content"].replace("\n", "")
                cleaned_content = cleaned_content.replace(" ", "")
                character_count += len(cleaned_content)
            # if "role" in msg and msg["role"] == CreatorRole.TOOL.value:
            #     if "name" in msg:
            #         character_count += len(msg["name"])

        if tools:
            for tool in tools:
                character_count += len(json.dumps(tool))

        text_token_count = ceil(character_count * CONVERSION_FACTOR)
        logger.info("Token Estimation:\nText: %d", text_token_count)
        return text_token_count

    @staticmethod
    def update_usage(
        completion_usage: openai.types.CompletionUsage | None,
        token_usage: TokenUsage | None = None,
    ) -> TokenUsage:
        """Transforms CompletionUsage to TokenUsage. This is a adapter function."""
        if completion_usage is None:
            raise RuntimeError("Response Usage is None.")

        if token_usage is None:
            token_usage = TokenUsage(
                input_tokens=completion_usage.prompt_tokens,
                output_tokens=completion_usage.completion_tokens,
            )
        else:
            token_usage.input_tokens += completion_usage.prompt_tokens
            token_usage.output_tokens += completion_usage.completion_tokens
        logger.debug("Token Usage: %s", token_usage)
        return token_usage


TOOL_PROMPT = """
**Utilize tools** to solve the problems. 
You will be called iteratively with progressively updated context.
Use the result stored in the context to solve the problem.
Therefore, break the problem into smaller sub-problems and use the result to solve them.
Calling the tools repeatedly is highly discouraged.
"""
