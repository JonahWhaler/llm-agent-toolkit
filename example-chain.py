import logging
import json

from typing import Type, TypeVar
from pydantic import BaseModel

from llm_agent_toolkit.core.local import (
    # Text_to_Text,
    Text_to_Text_SO,
    OllamaCore,
)
from llm_agent_toolkit import ChatCompletionConfig, ResponseMode
from llm_agent_toolkit._util import MessageBlock, CreatorRole

logging.basicConfig(
    filename="./snippet/output/example-chain.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class Step(BaseModel):
    state: str


class ResponseBody(BaseModel):
    steps: list[Step]
    final_answer: str


def run_cot_in_a_single_run(
    llm: Text_to_Text_SO, prompt: str, response_body: Type[T], output_path: str
) -> None:
    llm_generated_response = llm.run(
        query=prompt, context=None, mode=ResponseMode.SO, format=response_body
    )[0]
    response_content = llm_generated_response["content"]
    json_body = json.loads(response_content)
    with open(output_path, "a", encoding="utf-8") as md:
        md.write("\n======= COT =======\n\n")
        md.write(f"Prompt: {prompt}\n\n")
        md.write("Steps")
        for idx, step in enumerate(json_body["steps"], start=1):
            md.write(f"\t[{idx}]: {step['state']}\n\n")
        md.write(f"$: {json_body['final_answer']}\n")


class IterativeBody(BaseModel):
    task: str
    result: str
    completed: bool
    next_task: str


def run_cot_in_multiple_run(
    llm: Text_to_Text_SO, prompt: str, response_body: Type[T], output_path: str
) -> None:
    iteration = 0
    progress: list[MessageBlock] = []
    initial_prompt = prompt
    final_result: str | None = None
    logger.info("==== BEG ====")
    with open(output_path, "a", encoding="utf-8") as md:
        md.write("\n======= COT =======\n\n")
        md.write(f"Prompt: {initial_prompt}\n\n")
        md.write("Steps\n\n")
        while iteration < llm.config.max_iteration:
            logger.info("Iteration [%d]...", iteration)
            if progress:
                result = llm.run(
                    query=f"[{iteration}]/{llm.config.max_iteration}: {prompt}",
                    context=progress,  # type: ignore
                    mode=ResponseMode.SO,
                    format=IterativeBody,
                )[0]
            else:
                result = llm.run(
                    query=f"[{iteration}]/{llm.config.max_iteration}: {prompt}",
                    context=None,  # type: ignore
                    mode=ResponseMode.SO,
                    format=IterativeBody,
                )[0]

            string_content = result["content"]
            json_body = json.loads(string_content)

            if json_body["completed"]:
                final_result = json_body["result"]
                md.write(f"$: {final_result}\n\n")
                break
            else:
                progress.append({"role": CreatorRole.USER.value, "content": prompt})
                progress.append(
                    {
                        "role": CreatorRole.ASSISTANT.value,
                        "content": json_body["result"],
                    }
                )
                prompt = json_body["next_task"]
                md.write(f"\t[{iteration}]: {string_content}\n\n")
            iteration += 1
    logger.info("==== END ====")


# def execute_chain(initial_prompt: str):
#     cfg = ChatCompletionConfig(
#         name="qwen2.5:7b", max_iteration=10, max_tokens=32_000, max_output_tokens=4096
#     )
#     thinker = Text_to_Text_SO(
#         connection_string=CONNECTION_STRING,
#         system_prompt=f"You are a faithful assistant. You break the given tasks into sub-tasks and tackle them one at a time.\
#             You will be called iteratively up to {cfg.max_iteration} iterations, your generation of this iteration will be used as the input of the next iteration.\
#                 Set completed as True if you have completed the task/prompt of the user. \
#                     Finally, take your time to process the request, don't have to hurry to answer at the first shot.\
#                         Good Luck!",
#         config=cfg,
#     )
#     iteration = 0
#     prompt = f"Iteration 0: {initial_prompt}"
#     progress: list[dict] = []
#     while iteration < cfg.max_iteration:
#         logger.info("Iteration [%d]...", iteration)
#         response = thinker.run(query=prompt, context=progress, format=Body)[0]
#         jbody = json.loads(response["content"])

#         with open("./progress.md", "a", encoding="utf-8") as writer:
#             writer.write(f'======= {iteration} =======\n{response["content"]}\n')

#         if "error" in jbody:
#             logger.info("Error in JSON Body.")
#             break

#         if len(progress) == 0:
#             progress.append({"role": "user", "content": initial_prompt})

#         progress.append({"role": "user", "content": prompt})
#         progress.append({"role": "assistant", "content": jbody["result"]})

#         if jbody["completed"]:
#             break

#         prompt = f'Iteration {iteration}: {jbody["next_task"]}'

#         logger.info("Progress: %s", prompt)
#         iteration += 1

#     llm = Text_to_Text(
#         connection_string=CONNECTION_STRING,
#         system_prompt="You are a faithful assistant",
#         config=cfg,
#     )
#     response = llm.run(query=initial_prompt, context=progress[1:])[0]
#     with open("./answers.md", "a", encoding="utf-8") as writer:
#         writer.write("\n======= Multi-Shot =======\n")
#         writer.write(response["content"])


MULTIPASS_SYS_PROMPT = """
You are a faithful assistant. You break the given tasks into sub-tasks and tackle them one at a time.
You will be called iteratively to progress the user's request, your generation of this iteration will be used as the input of the next iteration.
Set completed as True if you have completed the task/prompt of the user.
Finally, take your time to process the request, don't have to hurry to answer at the first shot.
"""


if __name__ == "__main__":
    OllamaCore.load_csv("./files/ollama.csv")
    CONNECTION_STRING = "http://localhost:11434"
    MATH_PROMPT = "2X + 3Y = 13; 3X + 4Y = 18; Calculate X and Y."
    CODE_PROMPT = "Implement a function to find prime number below 1000 in Python3. Provide me the function with comprehensive docstring."
    OUTPUT_PATH = "./snippet/output/chain.md"
    llm = Text_to_Text_SO(
        connection_string=CONNECTION_STRING,
        system_prompt="You are a faithful assistant. You break the given tasks into sub-tasks and tackle them one at a time.",
        config=ChatCompletionConfig(
            name="qwen2.5:14b",
            return_n=1,
            max_tokens=4096,
            max_output_tokens=4096,
            temperature=0.3,
        ),
    )
    run_cot_in_a_single_run(
        llm=llm, prompt=MATH_PROMPT, response_body=ResponseBody, output_path=OUTPUT_PATH
    )

    run_cot_in_a_single_run(
        llm=llm, prompt=CODE_PROMPT, response_body=ResponseBody, output_path=OUTPUT_PATH
    )
    multipass_llm = Text_to_Text_SO(
        connection_string=CONNECTION_STRING,
        system_prompt=MULTIPASS_SYS_PROMPT,
        config=ChatCompletionConfig(
            name="qwen2.5:14b",
            return_n=1,
            max_iteration=10,
            max_tokens=32_000,
            max_output_tokens=4096,
            temperature=0.3,
        ),
    )
    run_cot_in_multiple_run(
        llm=multipass_llm,
        prompt=MATH_PROMPT,
        response_body=IterativeBody,
        output_path=OUTPUT_PATH,
    )
    run_cot_in_multiple_run(
        llm=multipass_llm,
        prompt=CODE_PROMPT,
        response_body=IterativeBody,
        output_path=OUTPUT_PATH,
    )
