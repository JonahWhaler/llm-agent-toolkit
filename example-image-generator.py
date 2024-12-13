"""This file only shows that the listed functions are working.
It does not means the results are correct.
Please do not take this as tests.
"""

import asyncio
import logging
from dotenv import load_dotenv

logging.basicConfig(
    filename="./snippet/output/example-image-generation.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


MODEL_NAME = "dall-e-2"
USERNAME = "Jonah Whaler"

PROMPT_1 = """
Create a minimalist logo featuring a whale near a beach with stronger waves, a ship in the background, 
and a tree with a pondering man seated beneath it (without a hat). 
Use purplish tones for the overall theme, incorporating gradients or monochromatic shades for a modern look. 
Add the word "Jonah" in a clean, minimalist font at the bottom center.
"""

PROMPT_2 = """
Create a minimalist logo featuring a young shepherd holding a sling, 
standing confidently against a simple background with subtle hints of a valley or hill. 
Use clean lines and a muted, earthy color palette with browns and greens to evoke simplicity and strength. 
Add the name "David" in a modern, understated font at the bottom.
"""
TMP_DIRECTORY = "./dev"


def generate():
    from llm_agent_toolkit.image_generator.open_ai import OpenAIImageGenerator
    from llm_agent_toolkit import ImageGenerationConfig

    config = ImageGenerationConfig(
        name=MODEL_NAME,
        return_n=2,
        max_iteration=1,
        size="256x256",
        quality="standard",
        response_format="b64_json",
    )
    generator = OpenAIImageGenerator(config)
    results = generator.generate(
        prompt=PROMPT_1, username=USERNAME, tmp_directory=TMP_DIRECTORY
    )
    for result in results:
        logger.info(">>>> %s", result)


async def agenerate():
    from llm_agent_toolkit.image_generator.open_ai import OpenAIImageGenerator
    from llm_agent_toolkit import ImageGenerationConfig

    config = ImageGenerationConfig(
        name=MODEL_NAME,
        return_n=2,
        max_iteration=1,
        size="256x256",
        quality="standard",
        response_format="b64_json",
    )
    generator = OpenAIImageGenerator(config)
    results = await generator.generate_async(
        prompt=PROMPT_2, username=USERNAME, tmp_directory=TMP_DIRECTORY
    )
    for result in results:
        logger.info(">>>> %s", result)


def synchronous_tasks():
    generate()


async def asynchronous_tasks():
    tasks = [agenerate()]
    await asyncio.gather(*tasks)


def try_image_generator_examples():
    synchronous_tasks()
    asyncio.run(asynchronous_tasks())


if __name__ == "__main__":
    load_dotenv()
    try_image_generator_examples()
