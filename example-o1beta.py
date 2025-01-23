"""This file only shows that the listed functions are working.
It does not means the results are correct.
Please do not take this as tests.
"""

import logging
from dotenv import load_dotenv

logging.basicConfig(
    filename="./snippet/output/example-o1beta.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def exec_o1beta(model_name: str, prompt: str):
    from llm_agent_toolkit import ChatCompletionConfig
    from llm_agent_toolkit.core.open_ai import O1Beta_OAI_Core, OpenAICore

    OpenAICore.load_csv("./files/openai.csv")

    config = ChatCompletionConfig(
        name=model_name,
        return_n=1,
        max_iteration=1,
        max_tokens=16_000,
        max_output_tokens=8192,
        temperature=1.0,
    )  # only temperature 1.0 is supported
    llm = O1Beta_OAI_Core(
        system_prompt="",
        config=config,
    )
    results = llm.run(query=prompt, context=None)
    with open("./snippet/output/o1beta.md", "a", encoding="utf-8") as md:
        md.write(f"\n\n==== {model_name} ====\n\n")
        md.write(f"Prompt: {prompt}\n\n")
        for result in results:
            md.write(f"{result['content']}\n\n")


def exec_reasoner(model_name: str, prompt: str):
    from llm_agent_toolkit import ChatCompletionConfig
    from llm_agent_toolkit.core.deep_seek import O1Beta_DS_Core

    config = ChatCompletionConfig(
        name=model_name, max_iteration=1, max_tokens=16_000, max_output_tokens=8192
    )
    llm = O1Beta_DS_Core(system_prompt="", config=config)
    results = llm.run(query=prompt, context=None)
    with open(
        "./snippet/output/o1beta-deepseek-reasoner.md", "a", encoding="utf-8"
    ) as md:
        md.write(f"\n\n==== {model_name} ====\n\n")
        md.write(f"Prompt: {prompt}\n\n")
        for result in results:
            md.write(f"{result['content']}\n\n")


PROMPT = """
If I am not your father, you are not my father, 
my father is not your father, your father is not my father, 
my father will never be your father, your father will never be my father. 
Who am I? Who are you?
"""


if __name__ == "__main__":
    # import os
    # import openai
    # load_dotenv()
    # models = ["o1-preview", "o1-mini"]

    # for model in models:
    #     exec_o1beta(model, PROMPT)
    exec_reasoner("deepseek-reasoner", PROMPT)
    # for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
    # client = openai.OpenAI(
    #     api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"
    # )
    # print(client.models.list())
