"""
This file aims to compare the prompt_eval_count of different vision models.

Apparently `llama3.2-vision:latest` consumption on image was not reflected on prompt_eval_count
and none of the tested models uses OpenAI's calculation.
"""

import logging
from llm_agent_toolkit.core.local import OllamaCore
from llm_agent_toolkit import ChatCompletionConfig
from llm_agent_toolkit.core.local import Image_to_Text

logging.basicConfig(
    filename="./snippet/output/example-ollama-ii.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CONNECTION_STRING = "http://localhost:11434"
STANDARD_CHAT_COMPLETION_CONFIG = {
    "return_n": 1,
    "max_iteration": 5,
    "max_tokens": 4096,
    "max_output_tokens": 2048,
    "temperature": 0.7,
}  # max_iteration takes no effect when no tool is used


def execute(model_name: str, filepath: str | None) -> None:
    SYSTEM_PROMPT = "You are Whales, faithful AI assistant."
    QUERY = "What's in the image?"

    config = ChatCompletionConfig(name=model_name, **STANDARD_CHAT_COMPLETION_CONFIG)
    llm = Image_to_Text(
        connection_string=CONNECTION_STRING,
        system_prompt=SYSTEM_PROMPT,
        config=config,
    )
    results = llm.run(query=QUERY, context=None, filepath=filepath)
    logger.info("Query: %s", QUERY)
    for result in results:
        logger.info(">>>> %s\n", result)


if __name__ == "__main__":
    OllamaCore.load_csv("./files/ollama.csv")

    models = ["llama3.2-vision:latest", "llava:7b", "llava-llama3:latest"]
    FILEPATH = "./dev/jonah-pixel-art.jpeg"

    for model in models:
        logger.info("Model: %s", model)
        execute(model, None)
        execute(model, FILEPATH)
