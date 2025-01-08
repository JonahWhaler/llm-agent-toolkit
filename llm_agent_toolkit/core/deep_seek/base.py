import json
import logging
from math import ceil
from typing import Any

from ..._util import MessageBlock
from ..._tool import ToolMetadata

logger = logging.getLogger(__name__)


class DeepSeekCore:
    def __init__(
        self,
    ):
        pass

    @staticmethod
    def build_profile() -> dict[str, bool | int | str]:
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

        Returns:
            int: The token count.

        Notes:
        * Set `CONVERSION_FACTOR` as 2 because my usecase most like involve using utf-8 encoding.
        """
        CONVERSION_RATE = 2
        character_count: int = 0
        for msg in msgs:
            # Incase the dict does not comply with the MessageBlock format
            if not isinstance(msg, dict):
                continue
            if "content" in msg and msg["content"]:
                character_count += len(msg["content"])
            # if "role" in msg and msg["role"] == CreatorRole.TOOL.value:
            #     if "name" in msg:
            #         character_count += len(msg["name"])

        if tools:
            for tool in tools:
                character_count += len(json.dumps(tool))

        return ceil(character_count // CONVERSION_RATE)


TOOL_PROMPT = """
**Utilize tools** to solve the problems. 
You will be called iteratively with progressively updated context.
Use the result stored in the context to solve the problem.
Therefore, break the problem into smaller sub-problems and use the result to solve them.
Calling the tools repeatedly is highly discouraged.
"""
