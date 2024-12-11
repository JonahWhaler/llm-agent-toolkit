"""This file only shows that the listed functions are working.
It does not means the results are correct.
Please do not take this as tests.
"""

import asyncio
import logging
from dotenv import load_dotenv

logging.basicConfig(
    filename="./snippet/output/example-openai.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

FILEPATH = "./dev/classroom.jpg"


def adder(a: int, b: int) -> int:
    """Add a with b.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int (int): Results
    """
    return a + b


async def divider(a: int, b: int) -> float:
    """Divide a by b.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        float: Results

    Raises:
        ValueError: When b is 0
    """
    if b == 0:
        raise ValueError("Division by zero.")
    return a / b


def exec_t2t_wo_tool():
    from llm_agent_toolkit import ChatCompletionConfig
    from llm_agent_toolkit.core.open_ai import Text_to_Text

    SYSTEM_PROMPT = "You are Whales, faithful AI assistant."
    QUERY = "What can you do for me?"
    MODEL_NAME = "gpt-4o-mini"

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        return_n=1,
        max_iteration=1,
        max_tokens=4096,
        temperature=0.7,
    )
    llm = Text_to_Text(
        system_prompt=SYSTEM_PROMPT,
        config=config,
        tools=None,
    )
    results = llm.run(query=QUERY, context=None)
    logger.info("Query: %s", QUERY)
    for result in results:
        logger.info(">>>> %s\n", result)


async def aexec_t2t_wo_tool():
    from llm_agent_toolkit import ChatCompletionConfig
    from llm_agent_toolkit.core.open_ai import Text_to_Text

    SYSTEM_PROMPT = "You are Whales, faithful AI assistant."
    QUERY = "What can you do for me?"
    MODEL_NAME = "gpt-4o-mini"

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        return_n=1,
        max_iteration=1,
        max_tokens=4096,
        temperature=0.7,
    )
    llm = Text_to_Text(
        system_prompt=SYSTEM_PROMPT,
        config=config,
        tools=None,
    )
    results = await llm.run_async(query=QUERY, context=None)
    logger.info("Query: %s", QUERY)
    for result in results:
        logger.info(">>>> %s\n", result)


def exec_t2t_w_tool():
    from llm_agent_toolkit import ChatCompletionConfig
    from llm_agent_toolkit.core.open_ai import Text_to_Text
    from llm_agent_toolkit.tool import LazyTool

    SYSTEM_PROMPT = "You are Whales, faithful AI assistant."
    QUERY = "10 + 5 / 5 = ?"
    MODEL_NAME = "gpt-4o-mini"

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        return_n=1,
        max_iteration=5,
        max_tokens=4096,
        temperature=0.7,
    )
    add_tool = LazyTool(adder, is_coroutine_function=False)
    div_tool = LazyTool(divider, is_coroutine_function=True)

    llm = Text_to_Text(
        system_prompt=SYSTEM_PROMPT,
        config=config,
        tools=[add_tool, div_tool],
    )
    results = llm.run(query=QUERY, context=None)
    logger.info("Query: %s", QUERY)
    for result in results:
        logger.info(">>>> %s\n", result)


async def aexec_t2t_w_tool():
    from llm_agent_toolkit import ChatCompletionConfig
    from llm_agent_toolkit.core.open_ai import Text_to_Text
    from llm_agent_toolkit.tool import LazyTool

    SYSTEM_PROMPT = "You are Whales, faithful AI assistant."
    QUERY = "10 + 5 / 5 = ?"
    MODEL_NAME = "gpt-4o-mini"

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        return_n=1,
        max_iteration=5,
        max_tokens=4096,
        temperature=0.7,
    )
    add_tool = LazyTool(adder, is_coroutine_function=False)
    div_tool = LazyTool(divider, is_coroutine_function=True)

    llm = Text_to_Text(
        system_prompt=SYSTEM_PROMPT,
        config=config,
        tools=[add_tool, div_tool],
    )
    results = await llm.run_async(query=QUERY, context=None)
    logger.info("Query: %s", QUERY)
    for result in results:
        logger.info(">>>> %s\n", result)


def exec_i2t():
    from llm_agent_toolkit import ChatCompletionConfig
    from llm_agent_toolkit.core.open_ai import Image_to_Text

    SYSTEM_PROMPT = "You are Whales, faithful AI assistant."
    QUERY = "What's in the image?"
    MODEL_NAME = "gpt-4o-mini"

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        return_n=1,
        max_iteration=5,
        max_tokens=4096,
        temperature=0.7,
    )

    llm = Image_to_Text(
        system_prompt=SYSTEM_PROMPT,
        config=config,
        tools=None,
    )
    results = llm.run(query=QUERY, context=None, filepath=FILEPATH)
    logger.info("Query: %s", QUERY)
    for result in results:
        logger.info(">>>> %s\n", result)


async def aexec_i2t():
    from llm_agent_toolkit import ChatCompletionConfig
    from llm_agent_toolkit.core.open_ai import Image_to_Text

    SYSTEM_PROMPT = "You are Whales, faithful AI assistant."
    QUERY = "What's in the image?"
    MODEL_NAME = "gpt-4o-mini"

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        return_n=1,
        max_iteration=5,
        max_tokens=4096,
        temperature=0.7,
    )

    llm = Image_to_Text(
        system_prompt=SYSTEM_PROMPT,
        config=config,
        tools=None,
    )
    results = await llm.run_async(query=QUERY, context=None, filepath=FILEPATH)
    logger.info("Query: %s", QUERY)
    for result in results:
        logger.info(">>>> %s\n", result)


def exec_i2t_wo_file():
    from llm_agent_toolkit import ChatCompletionConfig
    from llm_agent_toolkit.core.open_ai import Image_to_Text

    SYSTEM_PROMPT = "You are Whales, faithful AI assistant."
    QUERY = "What's in the image?"
    MODEL_NAME = "gpt-4o-mini"

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        return_n=1,
        max_iteration=5,
        max_tokens=4096,
        temperature=0.7,
    )

    llm = Image_to_Text(
        system_prompt=SYSTEM_PROMPT,
        config=config,
        tools=None,
    )
    results = llm.run(query=QUERY, context=None)
    logger.info("Query: %s", QUERY)
    for result in results:
        logger.info(">>>> %s\n", result)


async def aexec_i2t_wo_file():
    from llm_agent_toolkit import ChatCompletionConfig
    from llm_agent_toolkit.core.open_ai import Image_to_Text

    SYSTEM_PROMPT = "You are Whales, faithful AI assistant."
    QUERY = "What's in the image?"
    MODEL_NAME = "gpt-4o-mini"

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        return_n=1,
        max_iteration=5,
        max_tokens=4096,
        temperature=0.7,
    )

    llm = Image_to_Text(
        system_prompt=SYSTEM_PROMPT,
        config=config,
        tools=None,
    )
    results = await llm.run_async(query=QUERY, context=None)
    logger.info("Query: %s", QUERY)
    for result in results:
        logger.info(">>>> %s\n", result)


def uppercase(text: str) -> str:
    """Turn input text to uppercase.

    Args:
        text (str): Input string to be turned to uppercase.

    Returns:
        str: Result string.
    """
    return text.upper()


def exec_i2t_w_tool():
    from llm_agent_toolkit import ChatCompletionConfig
    from llm_agent_toolkit.core.open_ai import Image_to_Text
    from llm_agent_toolkit.tool import LazyTool

    SYSTEM_PROMPT = "You are Whales, faithful AI assistant."
    QUERY = "Whats in the image? Return your response with uppercase."
    MODEL_NAME = "gpt-4o-mini"

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        return_n=1,
        max_iteration=5,
        max_tokens=128_000,
        temperature=0.7,
    )

    string_tool = LazyTool(uppercase, is_coroutine_function=False)

    llm = Image_to_Text(
        system_prompt=SYSTEM_PROMPT,
        config=config,
        tools=[string_tool],
    )
    results = llm.run(query=QUERY, context=None, filepath=FILEPATH)
    logger.info("Query: %s", QUERY)
    for result in results:
        logger.info(">>>> %s\n", result)


if __name__ == "__main__":
    load_dotenv()
    exec_i2t_w_tool()
