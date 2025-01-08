"""This file only shows that the listed functions are working.
It does not means the results are correct.
Please do not take this as tests.
"""

import asyncio
import json
import logging

from dotenv import load_dotenv
from pydantic import BaseModel

logging.basicConfig(
    filename="./snippet/output/example-deepseek.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
MODEL_NAME = "deepseek-chat"


def execute_t2t_wo_tool() -> None:
    from llm_agent_toolkit import ChatCompletionConfig
    from llm_agent_toolkit.core.deep_seek import Text_to_Text

    SYSTEM_PROMPT = "You are Whales, faithful AI assistant."
    QUERY = "What can you do for me?"

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        return_n=1,
        max_iteration=1,
        max_tokens=4096,
        max_output_tokens=512,
        temperature=0.7,
    )
    llm = Text_to_Text(system_prompt=SYSTEM_PROMPT, config=config, tools=None)
    results = llm.run(query=QUERY, context=None)
    logger.info("Query: %s", QUERY)
    for result in results:
        logger.info(">>>> %s\n", result)


async def aexecute_t2t_wo_tool() -> None:
    from llm_agent_toolkit import ChatCompletionConfig
    from llm_agent_toolkit.core.deep_seek import Text_to_Text

    SYSTEM_PROMPT = "You are Whales, faithful AI assistant."
    QUERY = "What can you do for me?"

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        return_n=1,
        max_iteration=1,
        max_tokens=4096,
        max_output_tokens=512,
        temperature=0.7,
    )
    llm = Text_to_Text(system_prompt=SYSTEM_PROMPT, config=config, tools=None)
    results = await llm.run_async(query=QUERY, context=None)
    logger.info("Query: %s", QUERY)
    for result in results:
        logger.info(">>>> %s\n", result)


class QnA(BaseModel):
    question: str
    answer: str


SPROMPT = f"""
You are a helpful assistant.

Response Schema:
{
    json.dumps(QnA.model_json_schema())
}

Note:
Alway response in JSON format without additional comments or explanation.
"""


def execute_t2tso() -> None:
    from llm_agent_toolkit import ChatCompletionConfig, ResponseMode
    from llm_agent_toolkit.core.deep_seek import Text_to_Text_SO

    SYSTEM_PROMPT = SPROMPT
    QUERY = "What can you do for me?"

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        return_n=1,
        max_iteration=1,
        max_tokens=4096,
        max_output_tokens=512,
        temperature=1.0,
    )
    llm = Text_to_Text_SO(
        system_prompt=SYSTEM_PROMPT,
        config=config,
    )
    results = llm.run(query=QUERY, context=None, mode=ResponseMode.JSON)
    result = results[0]["content"]
    jresult = json.loads(result)
    logger.info("Query: %s", QUERY)
    logger.info("Question: %s", jresult["question"])
    logger.info("Answer: %s", jresult["answer"])


async def aexecute_t2tso() -> None:
    from llm_agent_toolkit import ChatCompletionConfig, ResponseMode
    from llm_agent_toolkit.core.deep_seek import Text_to_Text_SO

    SYSTEM_PROMPT = SPROMPT
    QUERY = "What can you do for me?"

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        return_n=1,
        max_iteration=1,
        max_tokens=4096,
        max_output_tokens=512,
        temperature=1.0,
    )
    llm = Text_to_Text_SO(
        system_prompt=SYSTEM_PROMPT,
        config=config,
    )
    results = llm.run(query=QUERY, context=None, mode=ResponseMode.JSON)
    result = results[0]["content"]
    jresult = json.loads(result)
    logger.info("Query: %s", QUERY)
    logger.info("Question: %s", jresult["question"])
    logger.info("Answer: %s", jresult["answer"])


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


def execute_t2t_w_tool() -> None:
    from llm_agent_toolkit import ChatCompletionConfig
    from llm_agent_toolkit.core.deep_seek import Text_to_Text
    from llm_agent_toolkit.tool import LazyTool

    SYSTEM_PROMPT = "You are Whales, faithful AI math assistant."
    QUERY = "10 + 5 / 5 = ?"

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        return_n=1,
        max_iteration=7,
        max_tokens=4096,
        temperature=0.0,
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


async def aexecute_t2t_w_tool() -> None:
    from llm_agent_toolkit import ChatCompletionConfig
    from llm_agent_toolkit.core.deep_seek import Text_to_Text
    from llm_agent_toolkit.tool import LazyTool

    SYSTEM_PROMPT = "You are Whales, faithful AI math assistant."
    QUERY = "10 + 5 / 5 = ?"

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        return_n=1,
        max_iteration=7,
        max_tokens=4096,
        temperature=0.0,
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


def synchronous_tasks() -> None:
    logger.info("======= Synchronous tasks =======")
    execute_t2t_wo_tool()
    execute_t2t_w_tool()
    execute_t2tso()


async def asynchronous_tasks() -> None:
    logger.info("======= Asynchronous tasks =======")
    tasks = [aexecute_t2t_wo_tool(), aexecute_t2t_w_tool(), aexecute_t2tso()]
    await asyncio.gather(*tasks)


def try_deepseek_examples() -> None:
    synchronous_tasks()
    asyncio.run(asynchronous_tasks())


if __name__ == "__main__":
    load_dotenv()
    try_deepseek_examples()
