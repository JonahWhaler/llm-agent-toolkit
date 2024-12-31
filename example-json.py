# import asyncio

# import os
import logging
import json
from typing import Type, TypeVar
from pydantic import BaseModel

from dotenv import load_dotenv
from llm_agent_toolkit.core.local import (
    Text_to_Text_SO,
    Image_to_Text_SO,
    OllamaCore,
)
from llm_agent_toolkit.core.open_ai import (
    OpenAICore,
    OAI_StructuredOutput_Core,
)

from llm_agent_toolkit import ChatCompletionConfig, ResponseMode

logging.basicConfig(
    filename="./snippet/output/example-JSON.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
OUTPUT_DIRECTORY = "./snippet/output"


class QnA(BaseModel):
    question: str
    answer: str


def run_ollama(
    llm: Text_to_Text_SO,
    initial_prompt: str,
    response_mode: ResponseMode | None,
    response_format: Type[T] | None,
) -> None:
    try:
        if response_mode and response_format:
            response = llm.run(
                query=initial_prompt,
                context=None,
                mode=response_mode,
                format=response_format,
            )[0]
        elif response_mode:
            response = llm.run(
                query=initial_prompt,
                context=None,
                mode=response_mode,
            )[0]
        else:
            response = llm.run(query=initial_prompt, context=None)[0]

        with open(
            f"{OUTPUT_DIRECTORY}/standard-output.md", "a", encoding="utf-8"
        ) as writer:
            writer.write("\n\n======= OLM One-Shot =======\n")
            if response_mode:
                writer.write(f"{response_mode.value}\n")
            else:
                writer.write("Default\n")
            writer.write(f"Prompt: {initial_prompt}\n")
            writer.write(f'Response: {response["content"]}\n')
    except Exception as e:
        logger.error("Exception: %s", e)


def run_ollama_ii(
    llm: Image_to_Text_SO,
    initial_prompt: str,
    filepath: str | None,
    response_mode: ResponseMode | None,
    response_format: Type[T] | None,
) -> None:
    try:
        if response_mode and response_format:
            response = llm.run(
                query=initial_prompt,
                context=None,
                filepath=filepath,
                mode=response_mode,
                format=response_format,
            )[0]
        elif response_mode:
            response = llm.run(
                query=initial_prompt,
                context=None,
                filepath=filepath,
                mode=response_mode,
            )[0]
        else:
            response = llm.run(query=initial_prompt, context=None, filepath=filepath)[0]

        with open(
            f"{OUTPUT_DIRECTORY}/standard-output.md", "a", encoding="utf-8"
        ) as writer:
            writer.write("\n\n======= OLM II One-Shot =======\n")
            if response_mode:
                writer.write(f"{response_mode.value}\n")
            else:
                writer.write("Default\n")
            writer.write(f"Prompt: {initial_prompt}\n")
            writer.write(f'Response: {response["content"]}\n')
    except Exception as e:
        logger.error("Exception: %s", e)


def run_openai(
    llm: OAI_StructuredOutput_Core,
    initial_prompt: str,
    filepath: str | None,
    response_mode: ResponseMode | None,
    response_format: Type[T] | None,
) -> None:
    try:
        if response_mode and response_format:
            response = llm.run(
                query=initial_prompt,
                context=None,
                filepath=filepath,
                mode=response_mode,
                format=response_format,
            )[0]
        elif response_mode:
            response = llm.run(
                query=initial_prompt,
                context=None,
                filepath=filepath,
                mode=response_mode,
            )[0]
        else:
            response = llm.run(query=initial_prompt, context=None, filepath=filepath)[0]

        with open(
            f"{OUTPUT_DIRECTORY}/standard-output.md", "a", encoding="utf-8"
        ) as writer:
            writer.write("\n\n======= OAI One-Shot =======\n")
            if response_mode:
                writer.write(f"{response_mode.value}\n")
            else:
                writer.write("Default\n")
            writer.write(f"Prompt: {initial_prompt}\n")
            writer.write(f'Response: {response["content"]}\n')
    except Exception as e:
        logger.error("Exception: %s", e)


SPROMPT = f"""
You are a helpful assistant.

Response Schema:
{
    json.dumps(QnA.model_json_schema())
}

Note:
Alway response in JSON format without additional comments or explanation.
"""


if __name__ == "__main__":
    load_dotenv()
    SYS_PROMPT = "You are a faithful Assistant."
    PROMPT = "Write a blog post about physician-assisted suicide (euthanasia)."
    II_PROMPT = "Write a story based on the provided image."
    FILEPATH = "./dev/classroom.jpg"

    logger.info("Ollama")
    # Run Ollama's models with `Structured Output`
    OllamaCore.load_csv("./files/ollama.csv")
    CONNECTION_STRING = "http://localhost:11434"

    logger.info("Text Generation")
    # Text Generation
    cfg = ChatCompletionConfig(
        name="llama3.2:3b", temperature=0.3, max_tokens=2048, max_output_tokens=1024
    )
    ## Structured Ouput
    run_ollama(
        llm=Text_to_Text_SO(
            connection_string=CONNECTION_STRING,
            system_prompt=SYS_PROMPT,
            config=cfg,
        ),
        initial_prompt=PROMPT,
        response_mode=ResponseMode.SO,
        response_format=QnA,
    )
    ## JSON Mode
    run_ollama(
        llm=Text_to_Text_SO(
            connection_string=CONNECTION_STRING,
            system_prompt=SPROMPT,
            config=cfg,
        ),
        initial_prompt=PROMPT,
        response_mode=ResponseMode.JSON,
        response_format=None,
    )
    ## Prompt Only
    run_ollama(
        llm=Text_to_Text_SO(
            connection_string=CONNECTION_STRING,
            system_prompt=SPROMPT,
            config=cfg,
        ),
        initial_prompt=PROMPT,
        response_mode=None,
        response_format=None,
    )

    logger.info("Image Interpretation")
    # Image Interpretation
    # Make sure model supports image interpretation
    iicfg = ChatCompletionConfig(
        name="llava:7b",
        temperature=0.3,
        max_tokens=32_000,
        max_output_tokens=2048,
    )
    ## Structured Ouput
    run_ollama_ii(
        llm=Image_to_Text_SO(
            connection_string=CONNECTION_STRING,
            system_prompt=SYS_PROMPT,
            config=iicfg,
        ),
        initial_prompt=II_PROMPT,
        filepath=FILEPATH,
        response_mode=ResponseMode.SO,
        response_format=QnA,
    )
    ## JSON Mode
    run_ollama_ii(
        llm=Image_to_Text_SO(
            connection_string=CONNECTION_STRING,
            system_prompt=SPROMPT,
            config=iicfg,
        ),
        initial_prompt=II_PROMPT,
        filepath=FILEPATH,
        response_mode=ResponseMode.JSON,
        response_format=None,
    )
    ## Prompt Only
    run_ollama_ii(
        llm=Image_to_Text_SO(
            connection_string=CONNECTION_STRING,
            system_prompt=SPROMPT,
            config=iicfg,
        ),
        initial_prompt=II_PROMPT,
        filepath=FILEPATH,
        response_mode=None,
        response_format=None,
    )

    logger.info("OpenAI")
    # Run OpenAI's models with `Structured Output`
    OpenAICore.load_csv(input_path="./files/openai.csv")

    logger.info("Text Generation")
    # Text Generation
    cfg = ChatCompletionConfig(
        name="gpt-4o", temperature=0.3, max_tokens=2048, max_output_tokens=1024
    )
    ## Structured Ouput
    run_openai(
        llm=OAI_StructuredOutput_Core(
            system_prompt=SYS_PROMPT,
            config=cfg,
        ),
        initial_prompt=PROMPT,
        filepath=None,
        response_mode=ResponseMode.SO,
        response_format=QnA,
    )
    ## JSON Mode
    run_openai(
        llm=OAI_StructuredOutput_Core(
            system_prompt=SPROMPT,
            config=ChatCompletionConfig(
                name="gpt-4o", temperature=0.3, max_tokens=2048, max_output_tokens=1024
            ),
        ),
        initial_prompt=PROMPT,
        filepath=None,
        response_mode=ResponseMode.JSON,
        response_format=None,
    )
    ## Prompt Only
    run_openai(
        llm=OAI_StructuredOutput_Core(
            system_prompt=SPROMPT,
            config=cfg,
        ),
        initial_prompt=PROMPT,
        filepath=None,
        response_mode=None,
        response_format=None,
    )

    logger.info("Image Interpretation")
    # Image Interpreter
    iicfg = ChatCompletionConfig(
        name="gpt-4o",
        temperature=0.3,
    )
    ## Structured Outout
    run_openai(
        llm=OAI_StructuredOutput_Core(
            system_prompt=SYS_PROMPT,
            config=iicfg,
        ),
        initial_prompt=II_PROMPT,
        filepath=FILEPATH,
        response_mode=ResponseMode.SO,
        response_format=QnA,
    )
    ## JSON Mode
    run_openai(
        llm=OAI_StructuredOutput_Core(
            system_prompt=SPROMPT,
            config=iicfg,
        ),
        initial_prompt=II_PROMPT,
        filepath=FILEPATH,
        response_mode=ResponseMode.JSON,
        response_format=None,
    )
    ## Prompt Only
    run_openai(
        llm=OAI_StructuredOutput_Core(
            system_prompt=SPROMPT,
            config=iicfg,
        ),
        initial_prompt=II_PROMPT,
        filepath=FILEPATH,
        response_mode=None,
        response_format=None,
    )
