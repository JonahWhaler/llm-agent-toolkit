import logging
import json
from typing import Type, TypeVar
from pydantic import BaseModel

from llm_agent_toolkit.core.local import (
    Text_to_Text,
    Text_to_Text_SO,
    OllamaCore,
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
CONNECTION_STRING = "http://localhost:11434"


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
        writer.write("\n======= Multi-Shot =======\n")
        writer.write(response["content"])


if __name__ == "__main__":
    OllamaCore.load_csv("./files/ollama.csv")
