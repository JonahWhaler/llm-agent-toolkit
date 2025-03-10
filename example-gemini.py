# import os
# import time
import asyncio
import logging
import json
from dotenv import load_dotenv

from llm_agent_toolkit.core.gemini.t2t import T2T_GMN_Core
from llm_agent_toolkit.core.gemini.i2t import I2T_GMN_Core
from llm_agent_toolkit.core.gemini.so import GMN_StructuredOutput_Core

from llm_agent_toolkit import ChatCompletionConfig, ResponseMode
from pydantic import BaseModel

logging.basicConfig(
    filename="./dev/log/gemini.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def exec_t2t():
    # time.sleep(5)
    MODEL_NAME = "gemini-2.0-flash"
    SYSTEM_PROMPT = "Be a faithful AI chatbot."
    PROMPT = "Give me a truly novel insight about humans."

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        temperature=0.7,
        max_tokens=8192,
        max_output_tokens=2048,
    )
    llm = T2T_GMN_Core(system_prompt=SYSTEM_PROMPT, config=config)
    responses = llm.run(query=PROMPT, context=None)
    for response in responses:
        logger.info(">> %s", response["content"])


async def aexec_t2t():
    await asyncio.sleep(1)
    MODEL_NAME = "gemini-2.0-flash"
    SYSTEM_PROMPT = "Be a faithful AI chatbot."
    PROMPT = "Give me a truly novel insight about humans."

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        temperature=0.7,
        max_tokens=8192,
        max_output_tokens=2048,
    )
    llm = T2T_GMN_Core(system_prompt=SYSTEM_PROMPT, config=config)
    responses = await llm.run_async(query=PROMPT, context=None)
    for response in responses:
        logger.info(">> %s", response["content"])


def exec_i2t():
    # time.sleep(5)
    MODEL_NAME = "gemini-2.0-flash"
    SYSTEM_PROMPT = "Be a faithful AI chatbot."
    PROMPT = (
        "Give me a prompt to reverse generate this image. Strictly no NSFW content!!!"
    )
    FILEPATH = "./dev/image/wednesday-addams-00.jpg"

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        temperature=0.7,
        max_tokens=8192,
        max_output_tokens=2048,
    )
    llm = I2T_GMN_Core(system_prompt=SYSTEM_PROMPT, config=config)
    responses = llm.run(query=PROMPT, context=None, filepath=FILEPATH)
    for response in responses:
        logger.info(">> %s", response["content"])


async def aexec_i2t():
    await asyncio.sleep(1)
    MODEL_NAME = "gemini-2.0-flash"
    SYSTEM_PROMPT = "Be a faithful AI chatbot."
    PROMPT = (
        "Give me a prompt to reverse generate this image. Strictly no NSFW content!!!"
    )
    FILEPATH = "./dev/image/wednesday-addams-00.jpg"

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        temperature=0.7,
        max_tokens=8192,
        max_output_tokens=2048,
    )
    llm = I2T_GMN_Core(system_prompt=SYSTEM_PROMPT, config=config)
    responses = llm.run(query=PROMPT, context=None, filepath=FILEPATH)
    for response in responses:
        logger.info(">> %s", response["content"])


JSON_PROMPT = """
Generate N number of random game character according to JSON schema below.

JSON Schema:
[
    {
        \"name\": {{Name:str}},
        \"attack\": {{Attack:int}},
        \"defense\": {{Defense:int}},
        \"career\": {{Warrior|Archer|NPC}}
    }
]
"""


def exec_so_json():
    # await asyncio.sleep(5)
    MODEL_NAME = "gemini-2.0-flash"
    SYSTEM_PROMPT = JSON_PROMPT
    PROMPT = "Create 3 unique chracters."

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        temperature=0.7,
        max_tokens=8192,
        max_output_tokens=2048,
    )
    llm = GMN_StructuredOutput_Core(system_prompt=SYSTEM_PROMPT, config=config)
    responses = llm.run(query=PROMPT, context=None, mode=ResponseMode.JSON)
    for response in responses:
        content = response["content"]
        logger.info(">> %s", content)
        jobj = json.loads(content)
        logger.info(jobj)


async def aexec_so_json():
    await asyncio.sleep(1)
    MODEL_NAME = "gemini-2.0-flash"
    SYSTEM_PROMPT = JSON_PROMPT
    PROMPT = "Create 3 unique chracters."

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        temperature=0.7,
        max_tokens=8192,
        max_output_tokens=2048,
    )
    llm = GMN_StructuredOutput_Core(system_prompt=SYSTEM_PROMPT, config=config)
    responses = llm.run(query=PROMPT, context=None, mode=ResponseMode.JSON)
    for response in responses:
        content = response["content"]
        logger.info(">> %s", content)
        jobj = json.loads(content)
        logger.info(jobj)


class Character(BaseModel):
    name: str
    attack: int
    defense: int
    career: str


class CustomResponse(BaseModel):
    characters: list[Character]


def exec_so_so():
    # await asyncio.sleep(5)
    MODEL_NAME = "gemini-2.0-flash"
    SYSTEM_PROMPT = "Generate N number of random game character."
    PROMPT = "Create 3 unique chracters."

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        temperature=0.7,
        max_tokens=8192,
        max_output_tokens=2048,
    )
    llm = GMN_StructuredOutput_Core(system_prompt=SYSTEM_PROMPT, config=config)
    responses = llm.run(
        query=PROMPT, context=None, mode=ResponseMode.SO, format=CustomResponse
    )
    for response in responses:
        content = response["content"]
        logger.info(">> %s", content)
        jobj = json.loads(content)
        logger.info(jobj)


async def aexec_so_so():
    await asyncio.sleep(1)
    MODEL_NAME = "gemini-2.0-flash"
    SYSTEM_PROMPT = "Generate N number of random game character."
    PROMPT = "Create 3 unique chracters."

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        temperature=0.7,
        max_tokens=8192,
        max_output_tokens=2048,
    )
    llm = GMN_StructuredOutput_Core(system_prompt=SYSTEM_PROMPT, config=config)
    responses = await llm.run_async(
        query=PROMPT, context=None, mode=ResponseMode.SO, format=CustomResponse
    )
    for response in responses:
        content = response["content"]
        logger.info(">> %s", content)
        jobj = json.loads(content)
        logger.info(jobj)


ImageDescriptionJSON = """
Describe the image using JSON Schema below.

JSON Schema:
{
    \"title\": {{Title:str}},
    \"summary\": {{Summary:str}},
    \"narrative\": {{Narative:str}}
}
"""


def exec_so_json_ii():
    # await asyncio.sleep(5)
    MODEL_NAME = "gemini-2.0-flash"
    SYSTEM_PROMPT = "You are a faithful AI chatbot."
    PROMPT = ImageDescriptionJSON
    FILEPATH = "./dev/image/wednesday-addams-00.jpg"

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        temperature=0.7,
        max_tokens=8192,
        max_output_tokens=2048,
    )
    llm = GMN_StructuredOutput_Core(system_prompt=SYSTEM_PROMPT, config=config)
    responses = llm.run(
        query=PROMPT, context=None, mode=ResponseMode.JSON, filepath=FILEPATH
    )
    for response in responses:
        content = response["content"]
        logger.info(">> %s", content)
        jobj = json.loads(content)
        logger.info(jobj)


async def aexec_so_json_ii():
    await asyncio.sleep(1)
    MODEL_NAME = "gemini-2.0-flash"
    SYSTEM_PROMPT = "You are a faithful AI chatbot."
    PROMPT = ImageDescriptionJSON
    FILEPATH = "./dev/image/wednesday-addams-00.jpg"

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        temperature=0.7,
        max_tokens=8192,
        max_output_tokens=2048,
    )
    llm = GMN_StructuredOutput_Core(system_prompt=SYSTEM_PROMPT, config=config)
    responses = await llm.run_async(
        query=PROMPT, context=None, mode=ResponseMode.JSON, filepath=FILEPATH
    )
    for response in responses:
        content = response["content"]
        logger.info(">> %s", content)
        jobj = json.loads(content)
        logger.info(jobj)


class ImageDescriptionModel(BaseModel):
    title: str
    summary: str
    narrative: str


def exec_so_so_ii():
    # await asyncio.sleep(5)
    MODEL_NAME = "gemini-2.0-flash"
    SYSTEM_PROMPT = "You are a faithful AI chatbot."
    PROMPT = ImageDescriptionJSON
    FILEPATH = "./dev/image/wednesday-addams-00.jpg"

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        temperature=0.7,
        max_tokens=8192,
        max_output_tokens=2048,
    )
    llm = GMN_StructuredOutput_Core(system_prompt=SYSTEM_PROMPT, config=config)
    responses = llm.run(
        query=PROMPT,
        context=None,
        mode=ResponseMode.SO,
        format=ImageDescriptionModel,
        filepath=FILEPATH,
    )
    for response in responses:
        content = response["content"]
        logger.info(">> %s", content)
        jobj = json.loads(content)
        logger.info(jobj)


async def aexec_so_so_ii():
    await asyncio.sleep(1)
    MODEL_NAME = "gemini-2.0-flash"
    SYSTEM_PROMPT = "You are a faithful AI chatbot."
    PROMPT = ImageDescriptionJSON
    FILEPATH = "./dev/image/wednesday-addams-00.jpg"

    config = ChatCompletionConfig(
        name=MODEL_NAME,
        temperature=0.7,
        max_tokens=8192,
        max_output_tokens=2048,
    )
    llm = GMN_StructuredOutput_Core(system_prompt=SYSTEM_PROMPT, config=config)
    responses = llm.run(
        query=PROMPT,
        context=None,
        mode=ResponseMode.SO,
        format=ImageDescriptionModel,
        filepath=FILEPATH,
    )
    for response in responses:
        content = response["content"]
        logger.info(">> %s", content)
        jobj = json.loads(content)
        logger.info(jobj)


def synchronous_tasks() -> None:
    logger.info("======= Synchronous tasks =======")
    tasks = {
        "default": [exec_t2t, exec_i2t],
        "text-generation": [exec_so_json, exec_so_so],
        "image-interpretation": [exec_so_json_ii, exec_so_so_ii],
    }

    for title, jobs in tasks.items():
        logger.info("Title: %s", title)
        for job in jobs:
            job()


async def asynchronous_tasks() -> None:
    logger.info("======= Asynchronous tasks =======")
    tasks = {
        "default": [aexec_t2t(), aexec_i2t()],
        "text-generation": [aexec_so_json(), aexec_so_so()],
        "image-interpretation": [aexec_so_json_ii, aexec_so_so_ii()],
    }

    for title, jobs in tasks.items():
        logger.info("Title: %s", title)
        await asyncio.gather(*jobs)


def try_gemini_examples() -> None:
    asyncio.run(asynchronous_tasks())
    synchronous_tasks()


if __name__ == "__main__":
    load_dotenv()
    try_gemini_examples()
