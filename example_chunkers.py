import logging
from llm_agent_toolkit.chunkers import SemanticChunker
from llm_agent_toolkit.encoder import OpenAIEncoder

logging.basicConfig(
    filename="./dev/log/chunkers.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load(input_path: str) -> str:
    with open(input_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    logger.info("START")
    content = load("./dev/document/chinese_doc.txt")
    encoder = OpenAIEncoder(model_name="text-embedding-3-small", dimension=512)
    chunker = SemanticChunker(
        encoder=encoder, config={"K": 3, "update_rate": 1.0, "MAX_ITERATION": 1000}
    )
    chunks = chunker.split(content)
    for chunk in chunks:
        logger.info(">>%s", chunk)
