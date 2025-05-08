"""This file only shows that the listed functions are working.
It does not means the results are correct.
Please do not take this as tests.
"""

import json
import asyncio
import logging
from dotenv import load_dotenv

from pydantic import BaseModel

from llm_agent_toolkit import ChatCompletionConfig, ResponseMode
from llm_agent_toolkit.core.open_ai import (
    Text_to_Text,
    Image_to_Text,
    StructuredOutput,
    Reasoning_Core,
    OpenAICore,
)
from llm_agent_toolkit.tool import LazyTool

logging.basicConfig(
    filename="./dev/log/example-openai.log",
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
            max_output_tokens=2048,
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
            max_output_tokens=2048,
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
            max_output_tokens=2048,
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
            max_output_tokens=2048,
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


def execute_t2t_json(model_name: str, prompt: str) -> None:
    """
    Code snippet of calling LLM and expect a structured output (JSON).
    """
    # max_iteration = 1
    llm = StructuredOutput(
        system_prompt=SPROMPT,
        config=ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=1,
            max_tokens=4096,
            max_output_tokens=2048,
            temperature=0.7,
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


async def async_t2t_json(model_name: str, prompt: str) -> None:
    """
    Code snippet of calling LLM and expect a structured output (JSON).
    """
    # max_iteration = 1
    llm = StructuredOutput(
        system_prompt=SPROMPT,
        config=ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=1,
            max_tokens=4096,
            max_output_tokens=2048,
            temperature=0.7,
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


def execute_t2t_structured_output(model_name: str, prompt: str) -> None:
    """
    Code snippet of calling LLM and expect a structured output.
    """

    class ResponseModel(BaseModel):
        question: str
        answer: str
        enhanced_prompt: str

    system_prompt = """
    You are a Whales, a faithful AI assistant.
    When needed, you help the user to enhanced the prompt before providing an answer.
    You provide response according to the provided structure.
    """
    # max_iteration = 1
    llm = StructuredOutput(
        system_prompt=system_prompt,
        config=ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=1,
            max_tokens=4096,
            max_output_tokens=2048,
            temperature=0.7,
        ),
    )
    results, token_usage = llm.run(
        query=prompt,
        context=None,
        mode=ResponseMode.SO,
        format=ResponseModel,
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


async def async_t2t_structured_output(model_name: str, prompt: str) -> None:
    """
    Code snippet of calling LLM and expect a structured output.
    """

    class ResponseModel(BaseModel):
        question: str
        answer: str
        enhanced_prompt: str

    system_prompt = """
    You are a Whales, a faithful AI assistant.
    When needed, you help the user to enhanced the prompt before providing an answer.
    You provide response according to the provided structure.
    """
    # max_iteration = 1
    llm = StructuredOutput(
        system_prompt=system_prompt,
        config=ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=1,
            max_tokens=4096,
            max_output_tokens=2048,
            temperature=0.7,
        ),
    )
    results, token_usage = await llm.run_async(
        query=prompt,
        context=None,
        mode=ResponseMode.SO,
        format=ResponseModel,
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


def execute_i2t_wo_tool(model_name: str, prompt: str, filepath: str) -> None:
    """
    Code snippet of calling LLM for image interpretation without tools.
    """
    system_prompt = """
    You are a Whales, a faithful AI assistant.
    You are good at interpreting images and solve math problems.
    """
    # max_iteration = 1
    llm = Image_to_Text(
        system_prompt=system_prompt,
        config=ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=1,
            max_tokens=4096,
            max_output_tokens=2048,
            temperature=0.7,
        ),
        tools=None,
    )
    results, token_usage = llm.run(query=prompt, context=None, filepath=filepath)
    logger.info("Token Usage: %s", token_usage)
    logger.info("Prompt:\t\t%s", prompt)
    for result in results:
        output_string = f"{result['role']:15s}:\t{result['content']}"
        logger.info(output_string)


async def async_i2t_wo_tool(model_name: str, prompt: str, filepath: str) -> None:
    """
    Code snippet of calling LLM for image interpretation without tools.
    """
    system_prompt = """
    You are a Whales, a faithful AI assistant.
    You are good at interpreting images and solve math problems.
    """
    # max_iteration = 1
    llm = Image_to_Text(
        system_prompt=system_prompt,
        config=ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=1,
            max_tokens=4096,
            max_output_tokens=2048,
            temperature=0.7,
        ),
        tools=None,
    )
    results, token_usage = await llm.run_async(
        query=prompt, context=None, filepath=filepath
    )
    logger.info("Token Usage: %s", token_usage)
    logger.info("Prompt:\t\t%s", prompt)
    for result in results:
        output_string = f"{result['role']:15s}:\t{result['content']}"
        logger.info(output_string)


def execute_i2t_w_tool(model_name: str, prompt: str, filepath: str) -> None:
    """
    Code snippet of calling LLM for image interpretation with tools.
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

    system_prompt = """
    You are a Whales, a faithful AI assistant.
    You are good at interpreting images and solve math problems.
    """
    llm = Image_to_Text(
        system_prompt=system_prompt,
        config=ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=7,
            max_tokens=4096,
            max_output_tokens=2048,
            temperature=0.7,
        ),
        tools=[add_tool, div_tool, mul_tool],
    )
    results, token_usage = llm.run(query=prompt, context=None, filepath=filepath)
    logger.info("Token Usage: %s", token_usage)
    logger.info("Prompt:\t\t%s", prompt)
    for result in results:
        output_string = f"{result['role']:15s}:\t{result['content']}"
        logger.info(output_string)


async def async_i2t_w_tool(model_name: str, prompt: str, filepath: str) -> None:
    """
    Code snippet of calling LLM for image interpretation with tools.
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

    system_prompt = """
    You are a Whales, a faithful AI assistant.
    You are good at interpreting images and solve math problems.
    """
    llm = Image_to_Text(
        system_prompt=system_prompt,
        config=ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=7,
            max_tokens=4096,
            max_output_tokens=2048,
            temperature=0.7,
        ),
        tools=[add_tool, div_tool, mul_tool],
    )
    results, token_usage = await llm.run_async(
        query=prompt, context=None, filepath=filepath
    )
    logger.info("Token Usage: %s", token_usage)
    logger.info("Prompt:\t\t%s", prompt)
    for result in results:
        output_string = f"{result['role']:15s}:\t{result['content']}"
        logger.info(output_string)


def execute_i2t_json(model_name: str, prompt: str, filepath: str) -> None:
    """
    Code snippet of calling LLM for image interpretation and expect a structured output (JSON).
    """
    system_prompt = """You are a Whales, a faithful AI assistant.
    You are good at interpreting images and return the text content in JSON format.

    JSON Schema:
    ---
    {
        'sentences': [
            'sentence 1',
            'sentence 2',
            'sentence 3'
        ]
    }
    """
    # max_iteration = 1
    llm = StructuredOutput(
        system_prompt=system_prompt,
        config=ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=1,
            max_tokens=4096,
            max_output_tokens=2048,
            temperature=0.7,
        ),
    )
    results, token_usage = llm.run(
        query=prompt,
        context=None,
        mode=ResponseMode.JSON,
        filepath=filepath,
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


async def async_i2t_json(model_name: str, prompt: str, filepath: str) -> None:
    """
    Code snippet of calling LLM for image interpretation and expect a structured output (JSON).
    """
    system_prompt = """You are a Whales, a faithful AI assistant.
    You are good at interpreting images and return the text content in JSON format.

    JSON Schema:
    ---
    {
        'sentences': [
            'sentence 1',
            'sentence 2',
            'sentence 3'
        ]
    }
    """
    # max_iteration = 1
    llm = StructuredOutput(
        system_prompt=system_prompt,
        config=ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=1,
            max_tokens=4096,
            max_output_tokens=2048,
            temperature=0.7,
        ),
    )
    results, token_usage = llm.run(
        query=prompt,
        context=None,
        mode=ResponseMode.JSON,
        filepath=filepath,
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


def execute_i2t_structured_output(model_name: str, prompt: str, filepath: str) -> None:
    """
    Code snippet of calling LLM for image interpretation and expect a structured output.
    """

    class ResponseModel(BaseModel):
        sentences: list[str]

    system_prompt = """You are a Whales, a faithful AI assistant.
    You are good at interpreting images.
    You provide response according to the provided structure.
    """
    # max_iteration = 1
    llm = StructuredOutput(
        system_prompt=system_prompt,
        config=ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=1,
            max_tokens=4096,
            max_output_tokens=2048,
            temperature=0.7,
        ),
    )
    results, token_usage = llm.run(
        query=prompt,
        context=None,
        mode=ResponseMode.SO,
        format=ResponseModel,
        filepath=filepath,
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


async def async_i2t_structured_output(
    model_name: str, prompt: str, filepath: str
) -> None:
    """
    Code snippet of calling LLM for image interpretation and expect a structured output.
    """

    class ResponseModel(BaseModel):
        sentences: list[str]

    system_prompt = """You are a Whales, a faithful AI assistant.
    You are good at interpreting images.
    You provide response according to the provided structure.
    """
    # max_iteration = 1
    llm = StructuredOutput(
        system_prompt=system_prompt,
        config=ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=1,
            max_tokens=4096,
            max_output_tokens=2048,
            temperature=0.7,
        ),
    )
    results, token_usage = await llm.run_async(
        query=prompt,
        context=None,
        mode=ResponseMode.SO,
        format=ResponseModel,
        filepath=filepath,
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


def execute_reasoner_core(model_name: str, prompt: str, effort: str) -> None:
    """
    Code snippet of calling reasoning LLM.
    """
    # max_iteration = 1
    llm = Reasoning_Core(
        system_prompt="You are deep thinker.",
        config=ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=1,
            max_tokens=8192,
            max_output_tokens=4096,
            temperature=1.0,
        ),
        reasoning_effort=effort,
    )
    results, token_usage = llm.run(query=prompt, context=None)
    logger.info("Token Usage: %s", token_usage)
    logger.info("Prompt:\t\t%s", prompt)
    for result in results:
        output_string = f"{result['role']:15s}:\t{result['content']}"
        logger.info(output_string)


async def async_reasoner_core(model_name: str, prompt: str, effort: str) -> None:
    """
    Code snippet of calling reasoning LLM.
    """
    # max_iteration = 1
    llm = Reasoning_Core(
        system_prompt="You are deep thinker.",
        config=ChatCompletionConfig(
            name=model_name,
            return_n=1,
            max_iteration=1,
            max_tokens=8192,
            max_output_tokens=4096,
            temperature=1.0,
        ),
        reasoning_effort=effort,
    )
    results, token_usage = llm.run(query=prompt, context=None)
    logger.info("Token Usage: %s", token_usage)
    logger.info("Prompt:\t\t%s", prompt)
    for result in results:
        output_string = f"{result['role']:15s}:\t{result['content']}"
        logger.info(output_string)


def synchronous_tasks():
    CHAT_COMPLETION_MODELS = ["gpt-4.1", "gpt-4o"]
    REASONING_MODELS = ["o3-mini", "o4-mini"]
    logger.info("======= Synchronous tasks =======")
    for model in CHAT_COMPLETION_MODELS:
        logger.info("Model: %s", model)
        execute_t2t_wo_tool(
            model,
            "What is expected when two object meet each other in infinite speed and force?",
        )
        execute_t2t_wo_tool(
            model,
            "What is expected when two object meet each other in infinite speed and force?",
        )
        execute_t2t_w_tool(model, "Solve 13 * 17 + 25 / 5 = ?")
        execute_t2t_json(
            model,
            "Is DeepSeek V3 a suitable model for content moderation of social media platform?",
        )
        execute_t2t_json(model, "omo~")
        execute_t2t_structured_output(
            model,
            "Is DeepSeek V3 a suitable model for content moderation of social media platform?",
        )
        execute_t2t_structured_output(
            model, "Today is a great day! I should do something!"
        )
        execute_i2t_wo_tool(model, "Solve this", r"./dev/image/math_question.jpg")
        execute_i2t_w_tool(model, "Solve this", r"./dev/image/math_question.jpg")
        execute_i2t_json(
            model, "Identify the text in the image", r"./dev/image/math_question.jpg"
        )
        execute_i2t_structured_output(
            model, "Identify the text in the image", r"./dev/image/math_question.jpg"
        )

    for model in REASONING_MODELS:
        logger.info("Model: %s", model)
        execute_reasoner_core(model, "Solve 13 * 17 + 25 / 5 = ?", "medium")
        execute_reasoner_core(
            model,
            "What is expected when two object meet each other in infinite speed and force?",
            "high",
        )


async def asynchronous_tasks():
    CHAT_COMPLETION_MODELS = ["gpt-4.1", "gpt-4o"]
    REASONING_MODELS = ["o3-mini", "o4-mini"]
    logger.info("======= Asynchronous tasks =======")
    tasks = []
    for model in CHAT_COMPLETION_MODELS:
        logger.info("Model: %s", model)
        tasks.append(
            async_t2t_wo_tool(
                model,
                "What is expected when two object meet each other in infinite speed and force?",
            )
        )
        tasks.append(async_t2t_w_tool(model, "Solve 13 * 17 + 25 / 5 = ?"))
        tasks.append(
            async_t2t_json(
                model,
                "Is DeepSeek V3 a suitable model for content moderation of social media platform?",
            )
        )
        tasks.append(async_t2t_json(model, "Ulala~"))
        tasks.append(
            async_t2t_structured_output(
                model,
                "Is DeepSeek V3 a suitable model for content moderation of social media platform?",
            )
        )
        tasks.append(
            async_t2t_structured_output(
                model,
                "Today is a great day! I should do something!",
            )
        )
        tasks.append(
            async_i2t_wo_tool(model, "Solve this", r"./dev/image/math_question.jpg")
        )
        tasks.append(
            async_i2t_w_tool(model, "Solve this", r"./dev/image/math_question.jpg")
        )
        tasks.append(
            async_i2t_json(
                model,
                "Identify the text in the image",
                r"./dev/image/math_question.jpg",
            )
        )
        tasks.append(
            async_i2t_structured_output(
                model,
                "Identify the text in the image",
                r"./dev/image/math_question.jpg",
            )
        )

    for model in REASONING_MODELS:
        tasks.append(async_reasoner_core(model, "Solve 13 * 17 + 25 / 5 = ?", "medium"))
        tasks.append(
            async_reasoner_core(
                model,
                "What is expected when two object meet each other in infinite speed and force?",
                "high",
            )
        )

    await asyncio.gather(*tasks)


def try_openai_examples():
    synchronous_tasks()
    asyncio.run(asynchronous_tasks())


if __name__ == "__main__":
    load_dotenv()
    OpenAICore.load_csv(r"./files/openai.csv")
    try_openai_examples()
