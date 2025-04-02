"""This file only shows that the listed functions are working.
It does not means the results are correct.
Please do not take this as tests.
"""

import logging
from dotenv import load_dotenv

logging.basicConfig(
    filename="./dev/log/example-reasoning.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def exec_o1beta(model_name: str, prompt: str, effort: str):
    from llm_agent_toolkit import ChatCompletionConfig
    from llm_agent_toolkit.core.open_ai import Reasoning_Core, OpenAICore

    OpenAICore.load_csv("./files/openai.csv")

    config = ChatCompletionConfig(
        name=model_name,
        return_n=1,
        max_iteration=1,
        max_tokens=16_000,
        max_output_tokens=8192,
        temperature=1.0,
    )  # only temperature 1.0 is supported

    follow_up_question = "Reply in Chinese (simplified)."
    llm = Reasoning_Core(
        system_prompt="You are a faithful assistant.",
        config=config,
        reasoning_effort=effort,
    )
    # Round 1
    r1, token_usage = llm.run(query=prompt, context=None)
    logger.info("Token usage: %s", token_usage)

    # Round 2
    r2, token_usage = llm.run(
        query=prompt,
        context=[
            {"role": "user", "content": prompt},
            r1[-1],
            {"role": "user", "content": follow_up_question},
        ],
    )
    logger.info("Token usage: %s", token_usage)

    with open(
        f"./dev/document/openai-{model_name}-{effort}.md", "a", encoding="utf-8"
    ) as md:
        md.write(f"\n\n==== {model_name} ====\n\n")
        md.write(f"Prompt: {prompt}\n\n")

        md.write("Round 1\n\n")
        md.write(r1[-1]["content"] + "\n\n")

        md.write(f"Follow-up question: {follow_up_question}\n\n")

        md.write("Round 2\n\n")
        md.write(r2[-1]["content"] + "\n\n")


def exec_reasoner(model_name: str, prompt: str):
    logger.info("==== %s ====", model_name)
    from llm_agent_toolkit import ChatCompletionConfig
    from llm_agent_toolkit.core.deep_seek import Reasoner_Core

    config = ChatCompletionConfig(
        name=model_name, max_iteration=1, max_tokens=16_000, max_output_tokens=8192
    )
    llm = Reasoner_Core(system_prompt="You are a faithful assistant.", config=config)
    results, token_usage = llm.run(query=prompt, context=None)
    logger.info("Token usage: %s", token_usage)
    with open(f"./dev/document/{model_name}.md", "a", encoding="utf-8") as md:
        md.write(f"\n\n==== {model_name} ====\n\n")
        md.write(f"Prompt: {prompt}\n\n")
        for result in results:
            md.write(f"{result['content']}\n\n")


def exec_thinker(model_name: str, prompt: str):
    logger.info("==== %s ====", model_name)
    from llm_agent_toolkit import ChatCompletionConfig
    from llm_agent_toolkit.core.gemini import Thinking_Core

    config = ChatCompletionConfig(
        name=model_name, max_iteration=1, max_tokens=16_000, max_output_tokens=8192
    )
    llm = Thinking_Core(system_prompt="You are a faithful assistant.", config=config)
    results, token_usage = llm.run(query=prompt, context=None)
    logger.info("Token usage: %s", token_usage)
    with open(f"./dev/document/{model_name}.md", "a", encoding="utf-8") as md:
        md.write(f"\n\n==== {model_name} ====\n\n")
        md.write(f"Prompt: {prompt}\n\n")
        for result in results:
            md.write(f"{result['content']}\n\n")


RIDDLE_PROMPT = """
If I am not your father, you are not my father, 
my father is not your father, your father is not my father, 
my father will never be your father, your father will never be my father. 
Who am I? Who are you?
"""


if __name__ == "__main__":
    # import os
    # import openai
    load_dotenv()
    # MODEL = "o3-mini"
    # EFFORT = "low"
    # exec_o1beta(MODEL, RIDDLE_PROMPT, EFFORT)

    QUESTION_PROMPT = "Compare the linear thinking and non-linear thinking. How their differences affects modern reasoning LLM?"
    # exec_o1beta("o3-mini", QUESTION_PROMPT, "medium")

    # exec_reasoner("deepseek-reasoner", QUESTION_PROMPT)
    gemini_2_0_flash_thinking = "gemini-2.0-flash-thinking-exp-01-21"
    gemini_2_5_pro = "gemini-2.5-pro-exp-03-25"
    exec_thinker(gemini_2_5_pro, QUESTION_PROMPT)
    # models = ["o1-preview", "o1-mini"]

    # for model in models:
    #     exec_o1beta(model, PROMPT)
    # exec_reasoner("deepseek-reasoner", PROMPT)
    # for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
    # client = openai.OpenAI(
    #     api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"
    # )
    # print(client.models.list())
