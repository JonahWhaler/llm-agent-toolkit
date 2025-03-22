# import os
# import time
import asyncio
import logging

# import json
from dotenv import load_dotenv

# from llm_agent_toolkit.core.gemini.t2t import T2T_GMN_Core
from llm_agent_toolkit.core.gemini import GeminiCore
from llm_agent_toolkit.core.gemini.t2t_w_tool import T2T_GMN_Core_W_Tool
from llm_agent_toolkit.core.gemini.i2t_w_tool import I2T_GMN_Core_W_Tool
# from llm_agent_toolkit.core.gemini.i2t import I2T_GMN_Core
# from llm_agent_toolkit.core.gemini.so import GMN_StructuredOutput_Core

from llm_agent_toolkit import ChatCompletionConfig, Tool  # , ResponseMode

# from pydantic import BaseModel
import json
import requests
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

logging.basicConfig(
    filename="./dev/log/gemini.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def adder(number_a: int, number_b: int) -> int:
    """Add number_a with number_b.

    Args:
        number_a (int): The first number.
        number_b (int): The second number.

    Returns:
        int (int): Results
    """
    return number_a + number_b


async def divider(number_a: int, number_b: int) -> float:
    """Divide number_a by number_b.

    Args:
        number_a (int): The first number.
        number_b (int): The second number.

    Returns:
        float: Results

    Raises:
        ValueError: When number_b is 0
    """
    if number_b == 0:
        raise ValueError("Division by zero.")
    return number_a / number_b


def exec_t2t(model_name: str):
    from llm_agent_toolkit.tool import LazyTool

    # time.sleep(5)
    SYSTEM_PROMPT = "Be a faithful AI chatbot."
    # PROMPT = "Solve 10 + 5 / 5 = ?"
    # PROMPT = "Solve 10 / 5 + 5 = ?"
    PROMPT = "Solve 37 + 20 / 5 + 43 = ?"

    add_tool = LazyTool(adder, is_coroutine_function=False)
    div_tool = LazyTool(divider, is_coroutine_function=True)
    tools = [add_tool, div_tool]
    config = ChatCompletionConfig(
        name=model_name,
        temperature=0.7,
        max_tokens=8192,
        max_output_tokens=2048,
        max_iteration=5,
    )
    llm = T2T_GMN_Core_W_Tool(system_prompt=SYSTEM_PROMPT, config=config, tools=tools)  # type: ignore
    responses, usage = llm.run(query=PROMPT, context=None)
    for response in responses:
        logger.info(">> %s", response["content"])


async def aexec_t2t(model_name: str) -> None:
    from llm_agent_toolkit.tool import LazyTool

    # time.sleep(5)
    # MODEL_NAME = "gemini-2.0-pro-exp-02-05"
    SYSTEM_PROMPT = "Be a faithful AI chatbot."
    # PROMPT = "Solve 10 + 5 / 5 = ?"
    # PROMPT = "Solve 10 / 5 + 5 = ?"
    PROMPT = "Solve 37 + 20 / 5 + 43 = ?"

    add_tool = LazyTool(adder, is_coroutine_function=False)
    div_tool = LazyTool(divider, is_coroutine_function=True)
    tools = [add_tool, div_tool]
    config = ChatCompletionConfig(
        name=model_name,
        temperature=0.7,
        max_tokens=8192,
        max_output_tokens=2048,
        max_iteration=5,
    )
    llm = T2T_GMN_Core_W_Tool(system_prompt=SYSTEM_PROMPT, config=config, tools=tools)  # type: ignore
    responses, usage = await llm.run_async(query=PROMPT, context=None)
    for response in responses:
        logger.info(">> %s: %s", response["role"], response["content"])


def character_counter(text: str) -> int:
    """
    Calculate the number of character in the text without whitespace.
    """
    logger.warning("text: %s", text)
    text = text.strip()
    return len(text.replace(" ", ""))


class DuckDuckGoSearchTool(Tool):
    def __init__(
        self, safesearch: str = "off", region: str = "my-en", pause: float = 3.0
    ):
        Tool.__init__(self, DuckDuckGoSearchTool.function_info(), True)
        self.safesearch = safesearch
        self.region = region
        self.pause = pause

    @staticmethod
    def function_info():
        from llm_agent_toolkit import (
            FunctionInfo,
            FunctionParameters,
            FunctionProperty,
            FunctionPropertyType,
        )

        return FunctionInfo(
            name="DuckDuckGoSearchTool",
            description="Search the internet via DuckDuckGO API.",
            parameters=FunctionParameters(
                properties=[
                    FunctionProperty(
                        name="query",
                        type=FunctionPropertyType.STRING,
                        description="Keyword that describe or define the query",
                    )
                ],
                type="object",
                required=["query"],
            ),
        )

    @property
    def random_user_agent(self) -> str:
        import random

        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) Firefox/120.0",
        ]
        return random.choice(user_agents)

    @property
    def headers(self) -> dict:
        return {
            "User-Agent": self.random_user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }

    async def run_async(self, params: str) -> str:
        import aiohttp

        await asyncio.sleep(self.pause)
        # Validate parameters
        if not self.validate(params=params):
            return json.dumps({"error": "Invalid parameters for DuckDuckGoSearchAgent"})
        # Load parameters
        params_dict: dict = json.loads(params)
        query = params_dict.get("query", None)
        logger.info("Query: %s", query)
        top_n = 5

        top_search = []
        with DDGS() as ddgs:
            for r in ddgs.text(
                keywords=query,
                region=self.region,
                safesearch=self.safesearch,
                max_results=top_n,
            ):
                top_search.append(r)

        # async with aiohttp.ClientSession() as session:
        #     tasks = [self.fetch_async(session, r["href"]) for r in top_search]
        #     search_results = await asyncio.gather(*tasks)
        #     for r, sr in zip(top_search, search_results):
        #         if sr:
        #             r["html"] = sr
        web_search_result = "\n\n".join([json.dumps(r) for r in top_search])
        # logger.info("web_search_result: %s", web_search_result)
        return web_search_result

    def run(self, params: str) -> str:
        import time

        time.sleep(self.pause)
        # Validate parameters
        if not self.validate(params=params):
            return json.dumps({"error": "Invalid parameters for DuckDuckGoSearchAgent"})
        # Load parameters
        params_dict: dict = json.loads(params)
        query = params_dict.get("query", None)
        logger.info("Query: %s", query)
        top_n = 5

        top_search = []
        with DDGS() as ddgs:
            try:
                for r in ddgs.text(
                    keywords=query,
                    region=self.region,
                    safesearch=self.safesearch,
                    max_results=top_n,
                ):
                    top_search.append(r)
            except Exception as error:
                logger.error(error)

        # for r in top_search:
        #     page = self.fetch(url=r["href"])
        #     if page:
        #         r["html"] = page

        web_search_result = "\n\n".join([json.dumps(r) for r in top_search])
        return web_search_result

    async def fetch_async(self, session, url):
        try:
            await asyncio.sleep(self.pause)
            async with session.get(url, headers=self.headers) as response:
                data = await response.text()
                soup = BeautifulSoup(data, "html.parser")
                return self.remove_whitespaces(soup.find("body").text)  # type: ignore
        except Exception as _:
            return None

    def fetch(self, url: str):
        try:
            page = requests.get(url=url, headers=self.headers, timeout=2, stream=False)
            soup = BeautifulSoup(page.text, "html.parser")
            body = soup.find("body")
            if body:
                t = body.text
                t = self.remove_whitespaces(t)
                return t
            return None
        except Exception as _:
            return None

    @staticmethod
    def remove_whitespaces(document_content: str) -> str:
        import re

        # original_len = len(document_content)
        cleaned_text = re.sub(r"\s+", " ", document_content)
        cleaned_text = re.sub(r"\n{3,}", "\n", cleaned_text)
        # updated_len = len(cleaned_text)
        # logger.info("Reduce from %d to %d", original_len, updated_len)
        return cleaned_text


def exec_i2t(model_name: str):
    from llm_agent_toolkit.tool import LazyTool

    SYSTEM_PROMPT = "Be a faithful AI chatbot."
    PROMPT = "Suggest URLs to look for similar images of this image."
    FILEPATH = "./dev/image/wednesday-addams-00.jpg"

    character_counter_tool = LazyTool(character_counter, is_coroutine_function=False)
    tools = [character_counter_tool, DuckDuckGoSearchTool()]
    config = ChatCompletionConfig(
        name=model_name,
        temperature=0.7,
        max_tokens=8192,
        max_output_tokens=4096,
        max_iteration=5,
    )
    llm = I2T_GMN_Core_W_Tool(system_prompt=SYSTEM_PROMPT, config=config, tools=tools)  # type: ignore
    responses, token_usage = llm.run(query=PROMPT, context=None, filepath=FILEPATH)
    logger.info("Token usage: %s", token_usage)
    for response in responses:
        logger.info(">> %s", response["content"])


async def aexec_i2t(model_name: str):
    from llm_agent_toolkit.tool import LazyTool

    SYSTEM_PROMPT = "Be a faithful AI chatbot."
    PROMPT = "Suggest some URLs to look for similar images of this image."
    FILEPATH = "./dev/image/jonah-pixel-art.jpeg"

    character_counter_tool = LazyTool(character_counter, is_coroutine_function=False)
    tools = [character_counter_tool, DuckDuckGoSearchTool()]
    config = ChatCompletionConfig(
        name=model_name,
        temperature=0.3,
        max_tokens=8192,
        max_output_tokens=4096,
        max_iteration=5,
    )
    llm = I2T_GMN_Core_W_Tool(system_prompt=SYSTEM_PROMPT, config=config, tools=tools)  # type: ignore
    responses, token_usage = await llm.run_async(
        query=PROMPT, context=None, filepath=FILEPATH
    )
    logger.info("Token usage: %s", token_usage)
    for response in responses:
        logger.info(">> %s", response["content"])


# JSON_PROMPT = """
# Generate N number of random game character according to JSON schema below.

# JSON Schema:
# [
#     {
#         \"name\": {{Name:str}},
#         \"attack\": {{Attack:int}},
#         \"defense\": {{Defense:int}},
#         \"career\": {{Warrior|Archer|NPC}}
#     }
# ]
# """


# def exec_so_json():
#     # await asyncio.sleep(5)
#     MODEL_NAME = "gemini-2.0-flash"
#     SYSTEM_PROMPT = JSON_PROMPT
#     PROMPT = "Create 3 unique chracters."

#     config = ChatCompletionConfig(
#         name=MODEL_NAME,
#         temperature=0.7,
#         max_tokens=8192,
#         max_output_tokens=2048,
#     )
#     llm = GMN_StructuredOutput_Core(system_prompt=SYSTEM_PROMPT, config=config)
#     responses = llm.run(query=PROMPT, context=None, mode=ResponseMode.JSON)
#     for response in responses:
#         content = response["content"]
#         logger.info(">> %s", content)
#         jobj = json.loads(content)
#         logger.info(jobj)


# async def aexec_so_json():
#     await asyncio.sleep(1)
#     MODEL_NAME = "gemini-2.0-flash"
#     SYSTEM_PROMPT = JSON_PROMPT
#     PROMPT = "Create 3 unique chracters."

#     config = ChatCompletionConfig(
#         name=MODEL_NAME,
#         temperature=0.7,
#         max_tokens=8192,
#         max_output_tokens=2048,
#     )
#     llm = GMN_StructuredOutput_Core(system_prompt=SYSTEM_PROMPT, config=config)
#     responses = llm.run(query=PROMPT, context=None, mode=ResponseMode.JSON)
#     for response in responses:
#         content = response["content"]
#         logger.info(">> %s", content)
#         jobj = json.loads(content)
#         logger.info(jobj)


# class Character(BaseModel):
#     name: str
#     attack: int
#     defense: int
#     career: str


# class CustomResponse(BaseModel):
#     characters: list[Character]


# def exec_so_so():
#     # await asyncio.sleep(5)
#     MODEL_NAME = "gemini-2.0-flash"
#     SYSTEM_PROMPT = "Generate N number of random game character."
#     PROMPT = "Create 3 unique chracters."

#     config = ChatCompletionConfig(
#         name=MODEL_NAME,
#         temperature=0.7,
#         max_tokens=8192,
#         max_output_tokens=2048,
#     )
#     llm = GMN_StructuredOutput_Core(system_prompt=SYSTEM_PROMPT, config=config)
#     responses = llm.run(
#         query=PROMPT, context=None, mode=ResponseMode.SO, format=CustomResponse
#     )
#     for response in responses:
#         content = response["content"]
#         logger.info(">> %s", content)
#         jobj = json.loads(content)
#         logger.info(jobj)


# async def aexec_so_so():
#     await asyncio.sleep(1)
#     MODEL_NAME = "gemini-2.0-flash"
#     SYSTEM_PROMPT = "Generate N number of random game character."
#     PROMPT = "Create 3 unique chracters."

#     config = ChatCompletionConfig(
#         name=MODEL_NAME,
#         temperature=0.7,
#         max_tokens=8192,
#         max_output_tokens=2048,
#     )
#     llm = GMN_StructuredOutput_Core(system_prompt=SYSTEM_PROMPT, config=config)
#     responses = await llm.run_async(
#         query=PROMPT, context=None, mode=ResponseMode.SO, format=CustomResponse
#     )
#     for response in responses:
#         content = response["content"]
#         logger.info(">> %s", content)
#         jobj = json.loads(content)
#         logger.info(jobj)


# ImageDescriptionJSON = """
# Describe the image using JSON Schema below.

# JSON Schema:
# {
#     \"title\": {{Title:str}},
#     \"summary\": {{Summary:str}},
#     \"narrative\": {{Narative:str}}
# }
# """


# def exec_so_json_ii():
#     # await asyncio.sleep(5)
#     MODEL_NAME = "gemini-2.0-flash"
#     SYSTEM_PROMPT = "You are a faithful AI chatbot."
#     PROMPT = ImageDescriptionJSON
#     FILEPATH = "./dev/image/wednesday-addams-00.jpg"

#     config = ChatCompletionConfig(
#         name=MODEL_NAME,
#         temperature=0.7,
#         max_tokens=8192,
#         max_output_tokens=2048,
#     )
#     llm = GMN_StructuredOutput_Core(system_prompt=SYSTEM_PROMPT, config=config)
#     responses = llm.run(
#         query=PROMPT, context=None, mode=ResponseMode.JSON, filepath=FILEPATH
#     )
#     for response in responses:
#         content = response["content"]
#         logger.info(">> %s", content)
#         jobj = json.loads(content)
#         logger.info(jobj)


# async def aexec_so_json_ii():
#     await asyncio.sleep(1)
#     MODEL_NAME = "gemini-2.0-flash"
#     SYSTEM_PROMPT = "You are a faithful AI chatbot."
#     PROMPT = ImageDescriptionJSON
#     FILEPATH = "./dev/image/wednesday-addams-00.jpg"

#     config = ChatCompletionConfig(
#         name=MODEL_NAME,
#         temperature=0.7,
#         max_tokens=8192,
#         max_output_tokens=2048,
#     )
#     llm = GMN_StructuredOutput_Core(system_prompt=SYSTEM_PROMPT, config=config)
#     responses = await llm.run_async(
#         query=PROMPT, context=None, mode=ResponseMode.JSON, filepath=FILEPATH
#     )
#     for response in responses:
#         content = response["content"]
#         logger.info(">> %s", content)
#         jobj = json.loads(content)
#         logger.info(jobj)


# class ImageDescriptionModel(BaseModel):
#     title: str
#     summary: str
#     narrative: str


# def exec_so_so_ii():
#     # await asyncio.sleep(5)
#     MODEL_NAME = "gemini-2.0-flash"
#     SYSTEM_PROMPT = "You are a faithful AI chatbot."
#     PROMPT = ImageDescriptionJSON
#     FILEPATH = "./dev/image/wednesday-addams-00.jpg"

#     config = ChatCompletionConfig(
#         name=MODEL_NAME,
#         temperature=0.7,
#         max_tokens=8192,
#         max_output_tokens=2048,
#     )
#     llm = GMN_StructuredOutput_Core(system_prompt=SYSTEM_PROMPT, config=config)
#     responses = llm.run(
#         query=PROMPT,
#         context=None,
#         mode=ResponseMode.SO,
#         format=ImageDescriptionModel,
#         filepath=FILEPATH,
#     )
#     for response in responses:
#         content = response["content"]
#         logger.info(">> %s", content)
#         jobj = json.loads(content)
#         logger.info(jobj)


# async def aexec_so_so_ii():
#     await asyncio.sleep(1)
#     MODEL_NAME = "gemini-2.0-flash"
#     SYSTEM_PROMPT = "You are a faithful AI chatbot."
#     PROMPT = ImageDescriptionJSON
#     FILEPATH = "./dev/image/wednesday-addams-00.jpg"

#     config = ChatCompletionConfig(
#         name=MODEL_NAME,
#         temperature=0.7,
#         max_tokens=8192,
#         max_output_tokens=2048,
#     )
#     llm = GMN_StructuredOutput_Core(system_prompt=SYSTEM_PROMPT, config=config)
#     responses = llm.run(
#         query=PROMPT,
#         context=None,
#         mode=ResponseMode.SO,
#         format=ImageDescriptionModel,
#         filepath=FILEPATH,
#     )
#     for response in responses:
#         content = response["content"]
#         logger.info(">> %s", content)
#         jobj = json.loads(content)
#         logger.info(jobj)


def synchronous_tasks() -> None:
    logger.info("======= Synchronous tasks =======")
    model_name = "gemini-2.0-flash"
    # exec_t2t(model_name)
    exec_i2t(model_name)
    #     tasks = {
    #         "default": [exec_t2t, exec_i2t],
    #         "text-generation": [exec_so_json, exec_so_so],
    #         "image-interpretation": [exec_so_json_ii, exec_so_so_ii],
    #     }

    #     for title, jobs in tasks.items():
    #         logger.info("Title: %s", title)
    #         for job in jobs:
    #             job()


async def asynchronous_tasks() -> None:
    logger.info("======= Asynchronous tasks =======")
    model_name = "gemini-2.0-flash"
    # tasks = {
    #     "default": [aexec_t2t(), aexec_i2t()],
    #     "text-generation": [aexec_so_json(), aexec_so_so()],
    #     "image-interpretation": [aexec_so_json_ii, aexec_so_so_ii()],
    # }

    # for title, jobs in tasks.items():
    #     logger.info("Title: %s", title)
    #     await asyncio.gather(*jobs)
    tasks = [
        # aexec_t2t(model_name),
        aexec_i2t(model_name)
    ]
    await asyncio.gather(*tasks)


def try_gemini_examples() -> None:
    asyncio.run(asynchronous_tasks())
    # synchronous_tasks()


if __name__ == "__main__":
    load_dotenv()
    GeminiCore.load_csv("./files/gemini.csv")
    try_gemini_examples()
