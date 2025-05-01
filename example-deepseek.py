"""This file only shows that the listed functions are working.
It does not means the results are correct.
Please do not take this as tests.
"""

import asyncio
import json
import logging

from dotenv import load_dotenv

from llm_agent_toolkit import ChatCompletionConfig, ResponseMode
from llm_agent_toolkit.core.deep_seek import (
    Text_to_Text,
    Text_to_Text_SO,
    Reasoner_Core,
)
from llm_agent_toolkit.tool import LazyTool

logging.basicConfig(
    filename=r"./dev/log/example-deepseek.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def execute_t2t_wo_tool(model_name: str, prompt: str) -> None:
    """
    Code snippet of calling LLM without tools.
    """
    llm = Text_to_Text(
        system_prompt="You are Whales, faithful AI assistant.",
        config=ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=1,
            max_tokens=2048,
            max_output_tokens=256,
            temperature=0.7,
        ),
        tools=None,
    )
    results, token_usage = llm.run(query=prompt, context=None)
    logger.info("Token Usage: %s", token_usage)
    logger.info("Prompt:\t\t%s", prompt)
    for result in results:
        output_string = f"{result['role']:15s}:\t{result['content']}"
        logger.info(output_string)


async def async_t2t_wo_tool(model_name: str, prompt: str) -> None:
    """
    Code snippet of calling LLM without tools.
    """
    llm = Text_to_Text(
        system_prompt="You are Whales, faithful AI assistant.",
        config=ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=1,
            max_tokens=4096,
            max_output_tokens=256,
            temperature=0.7,
        ),
        tools=None,
    )
    results, token_usage = await llm.run_async(query=prompt, context=None)
    logger.info("Token Usage: %s", token_usage)
    logger.info("Prompt:\t\t%s", prompt)
    for result in results:
        output_string = f"{result['role']:15s}:\t{result['content']}"
        logger.info(output_string)


def execute_t2t_w_tool(model_name: str, prompt: str) -> None:
    """
    Code snippet of calling LLM with tools.
    """

    def adder(a: int, b: int) -> int:
        """Add a with b.

        Args:
            a (int): The first number.
            b (int): The second number.

        Returns:
            int (int): The sum of a and b
        """
        return a + b

    async def divider(a: int, b: int) -> float:
        """Divide a by b.

        Args:
            a (int): The first number.
            b (int): The second number.

        Returns:
            float: the division of a and b

        Raises:
            ValueError: When b is 0
        """
        if b == 0:
            raise ValueError("Division by zero.")
        return a / b

    def multiplier(a: float, b: float) -> float:
        """Multiply a by b.,

        Args:
            a (float): The first number.
            b (float): The second number.

        Returns:
            float: The mulplication of a and b
        """
        return a * b

    add_tool = LazyTool(adder, is_coroutine_function=False)
    div_tool = LazyTool(divider, is_coroutine_function=True)
    mul_tool = LazyTool(multiplier, is_coroutine_function=False)

    llm = Text_to_Text(
        system_prompt="You are Whales, faithful AI math assistant.",
        config=ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=7,
            max_tokens=2048,
            max_output_tokens=256,
            temperature=0.2,
        ),
        tools=[add_tool, div_tool, mul_tool],
    )
    results, token_usage = llm.run(query=prompt, context=None)
    logger.info("Token Usage: %s", token_usage)
    logger.info("Prompt:\t\t%s", prompt)
    for result in results:
        output_string = f"{result['role']:15s}:\t{result['content']}"
        logger.info(output_string)


async def async_t2t_w_tool(model_name: str, prompt: str) -> None:
    """
    Code snippet of calling LLM with tools.
    """

    def adder(a: int, b: int) -> int:
        """Add a with b.

        Args:
            a (int): The first number.
            b (int): The second number.

        Returns:
            int (int): The sum of a and b
        """
        return a + b

    async def divider(a: int, b: int) -> float:
        """Divide a by b.

        Args:
            a (int): The first number.
            b (int): The second number.

        Returns:
            float: the division of a and b

        Raises:
            ValueError: When b is 0
        """
        if b == 0:
            raise ValueError("Division by zero.")
        return a / b

    def multiplier(a: float, b: float) -> float:
        """Multiply a by b.,

        Args:
            a (float): The first number.
            b (float): The second number.

        Returns:
            float: The mulplication of a and b
        """
        return a * b

    add_tool = LazyTool(adder, is_coroutine_function=False)
    div_tool = LazyTool(divider, is_coroutine_function=True)
    mul_tool = LazyTool(multiplier, is_coroutine_function=False)

    llm = Text_to_Text(
        system_prompt="You are Whales, faithful AI math assistant.",
        config=ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=7,
            max_tokens=2048,
            max_output_tokens=256,
            temperature=0.2,
        ),
        tools=[add_tool, div_tool, mul_tool],
    )
    results, token_usage = await llm.run_async(query=prompt, context=None)
    logger.info("Token Usage: %s", token_usage)
    logger.info("Prompt:\t\t%s", prompt)
    for result in results:
        output_string = f"{result['role']:15s}:\t{result['content']}"
        logger.info(output_string)


SPROMPT = """
Task: Assist user in whatever he/she ask for.

Response Schema:
---
{
    'question': 'User\'s Question',
    'answer': 'Assistant\'s reply'
}
---

Note:
Alway response in JSON format without additional comments or explanation.
"""


def execute_t2tso(model_name: str, prompt: str) -> None:
    """
    Code snippet of calling LLM and expect a structured output.
    """
    # max_iteration = 1
    llm = Text_to_Text_SO(
        system_prompt=SPROMPT,
        config=ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=1,
            max_tokens=4096,
            max_output_tokens=256,
            temperature=1.0,
        ),
    )
    results, token_usage = llm.run(query=prompt, context=None, mode=ResponseMode.JSON)
    logger.info("Token Usage: %s", token_usage)
    logger.info("Prompt:\t\t%s", prompt)

    text_content = results[-1]["content"]
    try:
        json_content = json.loads(text_content)
        for k, v in json_content.items():
            logger.info("%s: %s", k, v)
    except json.JSONDecodeError:
        logger.error("JSONDecodeError: %s", text_content)


async def async_t2tso(model_name: str, prompt: str) -> None:
    """
    Code snippet of calling LLM and expect a structured output.
    """
    # max_iteration = 1
    llm = Text_to_Text_SO(
        system_prompt=SPROMPT,
        config=ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=1,
            max_tokens=4096,
            max_output_tokens=256,
            temperature=1.0,
        ),
    )
    results, token_usage = await llm.run_async(
        query=prompt, context=None, mode=ResponseMode.JSON
    )
    logger.info("Token Usage: %s", token_usage)
    logger.info("Prompt:\t\t%s", prompt)

    text_content = results[-1]["content"]
    try:
        json_content = json.loads(text_content)
        for k, v in json_content.items():
            logger.info("%s: %s", k, v)
    except json.JSONDecodeError:
        logger.error("JSONDecodeError: %s", text_content)


def execute_reasoner_core(model_name: str, prompt: str) -> None:
    """
    Code snippet of calling reasoning LLM.
    """
    # max_iteration = 1
    llm = Reasoner_Core(
        system_prompt="You are deep thinker.",
        config=ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=1,
            max_tokens=4096,
            max_output_tokens=256,
            temperature=1.0,
        ),
    )
    results, token_usage = llm.run(query=prompt, context=None)
    logger.info("Token Usage: %s", token_usage)
    logger.info("Prompt:\t\t%s", prompt)
    for result in results:
        output_string = f"{result['role']:15s}:\t{result['content']}"
        logger.info(output_string)


async def async_reasoner_core(model_name: str, prompt: str) -> None:
    """
    Code snippet of calling reasoning LLM.
    """
    # max_iteration = 1
    llm = Reasoner_Core(
        system_prompt="You are deep thinker.",
        config=ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=1,
            max_tokens=4096,
            max_output_tokens=256,
            temperature=1.0,
        ),
    )
    results, token_usage = await llm.run_async(query=prompt, context=None)
    logger.info("Token Usage: %s", token_usage)
    logger.info("Prompt:\t\t%s", prompt)
    for result in results:
        output_string = f"{result['role']:15s}:\t{result['content']}"
        logger.info(output_string)


def synchronous_tasks() -> None:
    CHAT_COMPLETION_MODEL = "deepseek-chat"
    REASONER_MODEL = "deepseek-reasoner"

    logger.info("======= Synchronous tasks =======")
    execute_t2t_wo_tool(
        CHAT_COMPLETION_MODEL,
        "What is expected when two object meet each other in infinite speed and force?",
    )
    execute_t2t_w_tool(CHAT_COMPLETION_MODEL, "Solve 13 * 17 + 25 / 5 = ?")
    execute_t2tso(
        CHAT_COMPLETION_MODEL,
        "Is DeepSeek V3 a suitable model for content moderation of social media platform?",
    )
    execute_t2tso(CHAT_COMPLETION_MODEL, "Ulala~")
    execute_reasoner_core(REASONER_MODEL, "Solve 13 * 17 + 25 / 5 = ?")


async def asynchronous_tasks() -> None:
    CHAT_COMPLETION_MODEL = "deepseek-chat"
    REASONER_MODEL = "deepseek-reasoner"
    logger.info("======= Asynchronous tasks =======")
    tasks = [
        async_t2t_wo_tool(
            CHAT_COMPLETION_MODEL,
            "What is expected when two object meet each other in infinite speed and force?",
        ),
        async_t2t_w_tool(CHAT_COMPLETION_MODEL, "Solve 13 * 17 + 25 / 5 = ?"),
        async_t2tso(
            CHAT_COMPLETION_MODEL,
            "Is DeepSeek V3 a suitable model for content moderation of social media platform?",
        ),
        async_t2tso(CHAT_COMPLETION_MODEL, "Ulala~"),
        async_reasoner_core(REASONER_MODEL, "Solve 13 * 17 + 25 / 5 = ?"),
    ]
    await asyncio.gather(*tasks)


def try_deepseek_examples() -> None:
    synchronous_tasks()
    asyncio.run(asynchronous_tasks())


if __name__ == "__main__":
    load_dotenv()
    try_deepseek_examples()
