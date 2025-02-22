import asyncio
import logging

from llm_agent_toolkit.encoder.transformers_emb import TransformerEncoder

logging.basicConfig(
    filename="./dev/log/example-transformers.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PLAIN_TEXT = """
LLM Agent Toolkit provides minimal, modular interfaces for core components in LLM-based applications. Simplify workflows with stateless interaction, embedding encoders, memory management, tool integration, and data loaders, designed for compatibility and scalability. It prioritizes simplicity and modularity by proposing minimal wrappers designed to work across common tools, discouraging direct access to underlying technologies. Specific implementations and examples will be documented separately in a Cookbook (planned).
"""


def exec_encode():
    logger.info("exec_encode()")
    encoder = TransformerEncoder(MODEL_NAME)
    embeddings = encoder.encode(PLAIN_TEXT)
    logger.info("Embeddings Length: %d", len(embeddings))


def exec_encode_v2():
    logger.info("exec_encode_v2()")
    encoder = TransformerEncoder(MODEL_NAME)
    embeddings, token_count = encoder.encode_v2(PLAIN_TEXT)
    logger.info("Embeddings Length: %d, Token Count: %d", len(embeddings), token_count)


async def aexec_encode():
    logger.info("aexec_encode()")
    encoder = TransformerEncoder(MODEL_NAME)
    embeddings = await encoder.encode_async(PLAIN_TEXT)
    logger.info("Embeddings Length: %d", len(embeddings))


async def aexec_encode_v2():
    logger.info("aexec_encode_v2()")
    encoder = TransformerEncoder(MODEL_NAME)
    embeddings, token_count = await encoder.encode_v2_async(PLAIN_TEXT)
    logger.info("Embeddings Length: %d, Token Count: %d", len(embeddings), token_count)


def synchronous_tasks():
    exec_encode()
    exec_encode_v2()


async def asynchronous_tasks():
    tasks = [aexec_encode(), aexec_encode_v2()]
    await asyncio.gather(*tasks)


def try_openai_examples():
    synchronous_tasks()
    asyncio.run(asynchronous_tasks())


if __name__ == "__main__":
    try_openai_examples()
