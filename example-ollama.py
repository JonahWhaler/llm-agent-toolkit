"""This file only shows that the listed functions are working.
It does not means the results are correct.
Please do not take this as tests.
"""

import asyncio
import logging


logging.basicConfig(
    filename="./snippet/output/example-ollama.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


CONNECTION_STRING = "http://localhost:11434"
FILEPATH = "./dev/classroom.jpg"
STANDARD_CHAT_COMPLETION_CONFIG = {
    "return_n": 1,
    "max_iteration": 5,
    "max_tokens": 4096,
    "max_output_tokens": 2048,
    "temperature": 0.7,
}


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
    from llm_agent_toolkit.core.local import Text_to_Text

    SYSTEM_PROMPT = "You are Whales, faithful AI assistant."
    QUERY = "What can you do for me?"
    MODEL_NAME = "llama3.2:3b"

    config = ChatCompletionConfig(
        name=MODEL_NAME, **STANDARD_CHAT_COMPLETION_CONFIG
    )  # max_iteration takes no effect when no tool is used
    llm = Text_to_Text(
        connection_string=CONNECTION_STRING,
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
    from llm_agent_toolkit.core.local import Text_to_Text

    SYSTEM_PROMPT = "You are Whales, faithful AI assistant."
    QUERY = "What can you do for me?"
    MODEL_NAME = "llama3.2:3b"

    config = ChatCompletionConfig(
        name=MODEL_NAME, **STANDARD_CHAT_COMPLETION_CONFIG
    )  # max_iteration takes no effect when no tool is used
    llm = Text_to_Text(
        connection_string=CONNECTION_STRING,
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
    from llm_agent_toolkit.core.local import Text_to_Text
    from llm_agent_toolkit.tool import LazyTool

    SYSTEM_PROMPT = "You are Whales, faithful AI assistant."
    QUERY = "10 + 5 / 5 = ?"
    MODEL_NAME = "qwen2.5:7b"

    config = ChatCompletionConfig(name=MODEL_NAME, **STANDARD_CHAT_COMPLETION_CONFIG)
    add_tool = LazyTool(adder, is_coroutine_function=False)
    div_tool = LazyTool(divider, is_coroutine_function=True)

    llm = Text_to_Text(
        connection_string=CONNECTION_STRING,
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
    from llm_agent_toolkit.core.local import Text_to_Text
    from llm_agent_toolkit.tool import LazyTool

    SYSTEM_PROMPT = "You are Whales, faithful AI assistant."
    QUERY = "10 + 5 / 5 = ?"
    MODEL_NAME = "qwen2.5:7b"

    config = ChatCompletionConfig(name=MODEL_NAME, **STANDARD_CHAT_COMPLETION_CONFIG)
    add_tool = LazyTool(adder, is_coroutine_function=False)
    div_tool = LazyTool(divider, is_coroutine_function=True)

    llm = Text_to_Text(
        connection_string=CONNECTION_STRING,
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
    from llm_agent_toolkit.core.local import Image_to_Text

    SYSTEM_PROMPT = "You are Whales, faithful AI assistant."
    QUERY = "What's in the image?"
    MODEL_NAME = "llava:7b"

    config = ChatCompletionConfig(
        name=MODEL_NAME, **STANDARD_CHAT_COMPLETION_CONFIG
    )  # max_iteration takes no effect when no tool is used

    llm = Image_to_Text(
        connection_string=CONNECTION_STRING,
        system_prompt=SYSTEM_PROMPT,
        config=config,
    )
    results = llm.run(query=QUERY, context=None, filepath=FILEPATH)
    logger.info("Query: %s", QUERY)
    for result in results:
        logger.info(">>>> %s\n", result)


async def aexec_i2t():
    from llm_agent_toolkit import ChatCompletionConfig
    from llm_agent_toolkit.core.local import Image_to_Text

    SYSTEM_PROMPT = "You are Whales, faithful AI assistant."
    QUERY = "What's in the image?"
    MODEL_NAME = "llava:7b"

    config = ChatCompletionConfig(
        name=MODEL_NAME, **STANDARD_CHAT_COMPLETION_CONFIG
    )  # max_iteration takes no effect when no tool is used

    llm = Image_to_Text(
        connection_string=CONNECTION_STRING,
        system_prompt=SYSTEM_PROMPT,
        config=config,
    )
    results = await llm.run_async(query=QUERY, context=None, filepath=FILEPATH)
    logger.info("Query: %s", QUERY)
    for result in results:
        logger.info(">>>> %s\n", result)


def exec_i2t_wo_file():
    from llm_agent_toolkit import ChatCompletionConfig
    from llm_agent_toolkit.core.local import Image_to_Text

    SYSTEM_PROMPT = "You are Whales, faithful AI assistant."
    QUERY = "What's in the image?"
    MODEL_NAME = "llava:7b"

    config = ChatCompletionConfig(
        name=MODEL_NAME, **STANDARD_CHAT_COMPLETION_CONFIG
    )  # max_iteration takes no effect when no tool is used

    llm = Image_to_Text(
        connection_string=CONNECTION_STRING,
        system_prompt=SYSTEM_PROMPT,
        config=config,
    )
    results = llm.run(query=QUERY, context=None)
    logger.info("Query: %s", QUERY)
    for result in results:
        logger.info(">>>> %s\n", result)


async def aexec_i2t_wo_file():
    from llm_agent_toolkit import ChatCompletionConfig
    from llm_agent_toolkit.core.local import Image_to_Text

    SYSTEM_PROMPT = "You are Whales, faithful AI assistant."
    QUERY = "What's in the image?"
    MODEL_NAME = "llava:7b"

    config = ChatCompletionConfig(
        name=MODEL_NAME, **STANDARD_CHAT_COMPLETION_CONFIG
    )  # max_iteration takes no effect when no tool is used

    llm = Image_to_Text(
        connection_string=CONNECTION_STRING,
        system_prompt=SYSTEM_PROMPT,
        config=config,
    )
    results = await llm.run_async(query=QUERY, context=None)
    logger.info("Query: %s", QUERY)
    for result in results:
        logger.info(">>>> %s\n", result)


def exec_t2e():
    from llm_agent_toolkit.encoder.local import OllamaEncoder

    MODEL_NAME = "bge-m3:latest"
    PLAIN_TEXT = "This is awesome!"

    encoder = OllamaEncoder(connection_string=CONNECTION_STRING, model_name=MODEL_NAME)
    embedding = encoder.encode(text=PLAIN_TEXT)
    logger.info("Plain Text: %s", PLAIN_TEXT)
    logger.info(">>>> Dimension: %d", len(embedding))


def exec_t2e_v2():
    from llm_agent_toolkit.encoder.local import OllamaEncoder

    MODEL_NAME = "bge-m3:latest"
    PLAIN_TEXT = "This is awesome!"

    encoder = OllamaEncoder(connection_string=CONNECTION_STRING, model_name=MODEL_NAME)
    embedding, token_count = encoder.encode_v2(text=PLAIN_TEXT)
    logger.info("Plain Text: %s", PLAIN_TEXT)
    logger.info(">>>> Dimension: %d", len(embedding))
    logger.info(">>>> Token Count: %d", token_count)


async def aexec_t2e():
    from llm_agent_toolkit.encoder.local import OllamaEncoder

    MODEL_NAME = "bge-m3:latest"
    PLAIN_TEXT = "This is awesome!"

    encoder = OllamaEncoder(connection_string=CONNECTION_STRING, model_name=MODEL_NAME)
    embedding = await encoder.encode_async(text=PLAIN_TEXT)
    logger.info("Plain Text: %s", PLAIN_TEXT)
    logger.info(">>>> Dimension: %d", len(embedding))


async def aexec_t2e_v2():
    from llm_agent_toolkit.encoder.local import OllamaEncoder

    MODEL_NAME = "bge-m3:latest"
    PLAIN_TEXT = "This is awesome!"

    encoder = OllamaEncoder(connection_string=CONNECTION_STRING, model_name=MODEL_NAME)
    embedding, token_count = await encoder.encode_v2_async(text=PLAIN_TEXT)
    logger.info("Plain Text: %s", PLAIN_TEXT)
    logger.info(">>>> Dimension: %d", len(embedding))
    logger.info(">>>> Token Count: %d", token_count)


def synchronous_tasks():
    exec_t2t_wo_tool()
    exec_t2t_w_tool()
    exec_i2t()
    exec_i2t_wo_file()
    exec_t2e()
    exec_t2e_v2()


async def asynchronous_tasks():
    tasks = [
        aexec_t2t_wo_tool(),
        aexec_t2t_w_tool(),
        aexec_i2t(),
        aexec_i2t_wo_file(),
        aexec_t2e(),
        aexec_t2e_v2(),
    ]
    await asyncio.gather(*tasks)


def try_ollama_examples():
    synchronous_tasks()
    asyncio.run(asynchronous_tasks())


if __name__ == "__main__":
    try_ollama_examples()
