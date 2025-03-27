import json
import asyncio
import logging

from dotenv import load_dotenv
from pydantic import BaseModel
from llm_agent_toolkit.core.gemini import (
    GeminiCore,
    Text_to_Text,
    Image_to_Text,
    StructuredOutput,
)
from llm_agent_toolkit import ChatCompletionConfig, ResponseMode

logging.basicConfig(
    filename="./dev/log/gemini.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def case_one_sync(config: ChatCompletionConfig) -> None:
    SYSTEM_PROMPT = "Be a faithful AI chatbot."
    QUERY = "Provide novel insights on 'Migration to the Mars'."
    llm = Text_to_Text(system_prompt=SYSTEM_PROMPT, config=config)
    responses, usage = llm.run(query=QUERY, context=None)
    for response in responses:
        logger.info(">> %s", response["content"])

    logger.info("Token Usage: %s", usage)


async def case_one_async(config: ChatCompletionConfig) -> None:
    SYSTEM_PROMPT = "Be a faithful AI chatbot."
    QUERY = "Provide novel insights on 'Migration to the Mars'."
    llm = Text_to_Text(system_prompt=SYSTEM_PROMPT, config=config)
    responses, usage = await llm.run_async(query=QUERY, context=None)
    for response in responses:
        logger.info(">> %s", response["content"])

    logger.info("Token Usage: %s", usage)


def case_two_sync(config: ChatCompletionConfig) -> None:
    SYSTEM_PROMPT = "Be a faithful AI chatbot."
    QUERY = "Write a creative story with the character in the picture as the main character."
    FILEPATH = "./dev/image/sample.jpg"
    llm = Image_to_Text(system_prompt=SYSTEM_PROMPT, config=config)
    responses, usage = llm.run(query=QUERY, context=None, filepath=FILEPATH)
    for response in responses:
        logger.info(">> %s", response["content"])

    logger.info("Token Usage: %s", usage)


async def case_two_async(config: ChatCompletionConfig) -> None:
    SYSTEM_PROMPT = "Be a faithful AI chatbot."
    QUERY = "Write a creative story with the character in the picture as the main character."
    FILEPATH = "./dev/image/sample.jpg"
    llm = Image_to_Text(system_prompt=SYSTEM_PROMPT, config=config)
    responses, usage = await llm.run_async(query=QUERY, context=None, filepath=FILEPATH)
    for response in responses:
        logger.info(">> %s", response["content"])

    logger.info("Token Usage: %s", usage)


class ImageDescription(BaseModel):
    summary: str
    long_description: str
    keywords: list[str]


def case_three_sync(config: ChatCompletionConfig) -> None:
    SYSTEM_PROMPT = """
    You are a faithful image interpreter. 

    Instruction:
    1. Identify the multiple perspective to describe the given image.
    2. Go through each perspective.
    3. Provide a long description of the given image.
    4. Provide a summary to the long description.
    5. Provide a list of keywords/tags related to the given image.
    """
    QUERY = "Process the input file."
    FILEPATH = "./dev/image/sample.jpg"
    llm = StructuredOutput(SYSTEM_PROMPT, config)
    responses, usage = llm.run(
        query=QUERY,
        context=None,
        filepath=FILEPATH,
        mode=ResponseMode.SO,
        format=ImageDescription,
    )
    text_content = responses[-1]["content"]
    logger.info("Text: %s", text_content)

    json_content = json.loads(text_content)
    for k, v in json_content.items():
        logger.info("-> %s: %s", k, v)

    logger.info("Token Usage: %s", usage)


async def case_three_async(config: ChatCompletionConfig) -> None:
    SYSTEM_PROMPT = """
    You are a faithful image interpreter. 

    Instruction:
    1. Identify the multiple perspective to describe the given image.
    2. Go through each perspective.
    3. Provide a long description of the given image.
    4. Provide a summary to the long description.
    5. Provide a list of keywords/tags related to the given image.
    """
    QUERY = "Process the input file."
    FILEPATH = "./dev/image/sample.jpg"
    llm = StructuredOutput(SYSTEM_PROMPT, config)
    responses, usage = await llm.run_async(
        query=QUERY,
        context=None,
        filepath=FILEPATH,
        mode=ResponseMode.SO,
        format=ImageDescription,
    )
    text_content = responses[-1]["content"]
    logger.info("Text: %s", text_content)

    json_content = json.loads(text_content)
    for k, v in json_content.items():
        logger.info("-> %s: %s", k, v)

    logger.info("Token Usage: %s", usage)


def case_four_sync(config: ChatCompletionConfig) -> None:
    SYSTEM_PROMPT = """
    You are a faithful image interpreter and story writer. 

    Instruction:
    1. Identify the multiple perspective to describe the given image.
    2. Go through each perspective.
    3. Provide a long description of the given image.
    4. Provide a summary to the long description.
    5. Provide a list of keywords/tags related to the given image.
    6. Write a story around the description.

    Output Format: JSON
    ---
    {
        \"keywords\": [{{keyword-i}}],
        \"long_description\": {{Long Description}},
        \"summary\": {{Summary}},
        \"story\": {{Story}}
    }
    ---
    """
    QUERY = "Process the input file."
    FILEPATH = "./dev/image/sample.jpg"
    llm = StructuredOutput(SYSTEM_PROMPT, config)
    responses, usage = llm.run(
        query=QUERY,
        context=None,
        filepath=FILEPATH,
        mode=ResponseMode.JSON,
    )
    text_content = responses[-1]["content"]
    logger.info("Text: %s", text_content)

    json_content = json.loads(text_content)
    for k, v in json_content.items():
        logger.info("-> %s: %s", k, v)

    logger.info("Token Usage: %s", usage)


async def case_four_async(config: ChatCompletionConfig) -> None:
    SYSTEM_PROMPT = """
    You are a faithful image interpreter and story writer. 

    Instruction:
    1. Identify the multiple perspective to describe the given image.
    2. Go through each perspective.
    3. Provide a long description of the given image.
    4. Provide a summary to the long description.
    5. Provide a list of keywords/tags related to the given image.
    6. Write a story around the description.

    Output Format: JSON
    ---
    {
        \"keywords\": [{{keyword-i}}],
        \"long_description\": {{Long Description}},
        \"summary\": {{Summary}},
        \"story\": {{Story}}
    }
    ---
    """
    QUERY = "Process the input file."
    FILEPATH = "./dev/image/sample.jpg"
    llm = StructuredOutput(SYSTEM_PROMPT, config)
    responses, usage = await llm.run_async(
        query=QUERY,
        context=None,
        filepath=FILEPATH,
        mode=ResponseMode.JSON,
    )
    text_content = responses[-1]["content"]
    logger.info("Text: %s", text_content)

    json_content = json.loads(text_content)
    for k, v in json_content.items():
        logger.info("-> %s: %s", k, v)

    logger.info("Token Usage: %s", usage)


def synchronous_tasks() -> None:
    logger.info("======= Synchronous tasks =======")
    model_name = "gemini-2.0-flash"
    config = ChatCompletionConfig(
        name=model_name,
        temperature=0.3,
        max_tokens=8192,
        max_output_tokens=2048,
        max_iteration=1,
    )
    case_one_sync(config)
    case_two_sync(config)
    case_three_sync(config)
    case_four_sync(config)


async def asynchronous_tasks() -> None:
    logger.info("======= Asynchronous tasks =======")
    model_name = "gemini-2.0-flash"
    config = ChatCompletionConfig(
        name=model_name,
        temperature=0.3,
        max_tokens=8192,
        max_output_tokens=2048,
        max_iteration=1,
    )
    tasks = [
        case_one_async(config),
        case_two_async(config),
        case_three_async(config),
        case_four_async(config),
    ]
    await asyncio.gather(*tasks)


def try_gemini_examples() -> None:
    asyncio.run(asynchronous_tasks())
    synchronous_tasks()


if __name__ == "__main__":
    load_dotenv()
    GeminiCore.load_csv("./files/gemini.csv")
    try_gemini_examples()
