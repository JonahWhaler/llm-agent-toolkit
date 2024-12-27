import asyncio

# import os
import logging
import json
from typing import Any
from pydantic import BaseModel

# import openai
from dotenv import load_dotenv
from llm_agent_toolkit.core.local import Text_to_Text, Text_to_Text_SO, OllamaCore
from llm_agent_toolkit import ChatCompletionConfig

logging.basicConfig(
    filename="./snippet/output/example-JSON.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

OllamaCore.load_csv("./files/ollama.csv")

CONNECTION_STRING = "http://localhost:11434"
STANDARD_CHAT_COMPLETION_CONFIG = {
    "return_n": 1,
    "max_iteration": 5,
    "max_tokens": 4096,
    "max_output_tokens": 2048,
    "temperature": 0.7,
}


class QnA(BaseModel):
    question: str
    answer: str


def run_ollama(initial_prompt: str):
    cfg = ChatCompletionConfig(name="qwen2.5:7b")
    try:
        llm = Text_to_Text_SO(
            connection_string=CONNECTION_STRING,
            system_prompt="You are a faithful assistant",
            config=cfg,
        )
        response = llm.run(query=initial_prompt, context=None, format=QnA)[0]
        with open("./answers.md", "a", encoding="utf-8") as writer:
            writer.write("======= One-Shot =======\n")
            writer.write(response["content"])

    except Exception as e:
        logger.error("Exception: %s", e)


class Body(BaseModel):
    completed: bool
    task: str
    result: str
    next_task: str


def execute_chain(initial_prompt: str):
    cfg = ChatCompletionConfig(
        name="qwen2.5:7b", max_iteration=10, max_tokens=32_000, max_output_tokens=4096
    )
    thinker = Text_to_Text_SO(
        connection_string=CONNECTION_STRING,
        system_prompt=f"You are a faithful assistant. You break the given tasks into sub-tasks and tackle them one at a time.\
            You will be called iteratively up to {cfg.max_iteration} iterations, your generation of this iteration will be used as the input of the next iteration.\
                Set completed as True if you have completed the task/prompt of the user. \
                    Finally, take your time to process the request, don't have to hurry to answer at the first shot.\
                        Good Luck!",
        config=cfg,
    )
    iteration = 0
    prompt = f"Iteration 0: {initial_prompt}"
    progress: list[dict] = []
    while iteration < cfg.max_iteration:
        logger.info("Iteration [%d]...", iteration)
        response = thinker.run(query=prompt, context=progress, format=Body)[0]
        jbody = json.loads(response["content"])

        with open("./progress.md", "a", encoding="utf-8") as writer:
            writer.write(f'======= {iteration} =======\n{response["content"]}\n')

        if "error" in jbody:
            logger.info("Error in JSON Body.")
            break

        if len(progress) == 0:
            progress.append({"role": "user", "content": initial_prompt})

        progress.append({"role": "user", "content": prompt})
        progress.append({"role": "assistant", "content": jbody["result"]})

        if jbody["completed"]:
            break

        prompt = f'Iteration {iteration}: {jbody["next_task"]}'

        logger.info("Progress: %s", prompt)
        iteration += 1

    llm = Text_to_Text(
        connection_string=CONNECTION_STRING,
        system_prompt="You are a faithful assistant",
        config=cfg,
    )
    response = llm.run(query=initial_prompt, context=progress[1:])[0]
    with open("./answers.md", "a", encoding="utf-8") as writer:
        writer.write("======= Multi-Shot =======\n")
        writer.write(response["content"])


async def run_ollama_async():
    cfg = ChatCompletionConfig(name="qwen2.5:7b")
    try:
        llm = Text_to_Text_SO(
            connection_string=CONNECTION_STRING,
            system_prompt="You are a faithful assistant",
            config=cfg,
        )
        responses = await llm.run_async(
            query="Why is the sky blue?", context=None, format=QnA
        )
        response = responses[0]
        j = json.loads(response["content"])
        print(j)
    except Exception as e:
        logger.error("Exception: %s", e)


# SPROMPT = f"""
# You are a helpful assistant.

# Response Schema:
# {
#     json.dumps(QnA.model_json_schema())
# }

# Note:
# Alway response in JSON format without additional comments or explanation.
# """

# logger.info(SPROMPT)


# def run_openai_json():
#     logger.info(">> run-openai")
#     messages = [
#         {"role": "system", "content": SPROMPT},
#         {
#             "role": "user",
#             "content": "Why is the sky blue?",
#         },
#     ]
#     try:
#         client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
#         response = client.chat.completions.create(
#             model="gpt-4o-2024-08-06",
#             messages=messages,
#             tools=None,
#             response_format={"type": "json_object"},
#             stream=False,
#         )
#         logger.info(">> %s", response.choices)
#         # j = json.loads(response.message.content)
#         # logger.info("j = %s | %s", j, type(j).__name__)
#     except Exception as e:
#         logger.error("Exception: %s", e)


# def run_openai_so():
#     logger.info(">> run-openai")
#     messages: list[dict | MessageBlock] = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": "I eat Laksa as breakfast.",
#         },
#     ]
#     try:
#         client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
#         response = client.beta.chat.completions.parse(
#             model="gpt-4o-2024-08-06",
#             messages=messages,
#             response_format=QnA,
#         )
#         logger.info(">> %s", response)
#         logger.info(response.choices[0].message.content)
#         # j = json.loads(response.message.content)
#         # logger.info("j = %s | %s", j, type(j).__name__)
#     except Exception as e:
#         logger.error("Exception: %s", e)


if __name__ == "__main__":
    load_dotenv()
    # asyncio.run(run_ollama_async())
    prompt = "Write a blog post about physician-assisted suicide (euthanasia)."
    run_ollama(initial_prompt=prompt)
    execute_chain(initial_prompt=prompt)
    # run_openai_json()
    # run_openai_so()
