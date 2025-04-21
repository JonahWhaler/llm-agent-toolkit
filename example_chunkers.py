import logging
from dotenv import load_dotenv

from llm_agent_toolkit import chunkers
from llm_agent_toolkit.encoder import OpenAIEncoder, GeminiEncoder

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


def test_static_chunkers(text: str) -> None:
    test_subjects = [
        {
            "name":
            "FixedCharacterChunker",
            "object":
            chunkers.FixedCharacterChunker(
                chunkers.FixedCharacterChunkerConfig(
                    chunk_length=128, stride_rate=1.0
                )
            )
        }, {
            "name":
            "FixedGroupChunker",
            "object":
            chunkers.FixedGroupChunker(
                chunkers.FixedGroupChunkerConfig(
                    G=len(text) // 128, level="character"
                )
            )
        }, {
            "name":
            "FixedGroupChunker",
            "object":
            chunkers.FixedGroupChunker(
                chunkers.FixedGroupChunkerConfig(
                    G=len(text) // 128, level="word"
                )
            )
        }, {
            "name": "SectionChunker",
            "object": chunkers.SectionChunker()
        }, {
            "name": "SentenceChunker",
            "object": chunkers.SentenceChunker()
        }
    ]
    for test_subject in test_subjects:
        chunker = test_subject["object"]
        name = test_subject["name"]
        logger.info(f"Testing {name}...")
        chunks = chunker.split(text)
        logger.info(f"Chunked {len(chunks)} chunks.")
        for i, chunk in enumerate(chunks):
            logger.info("C%d", i)
            logger.info("Chunk length: %d", len(chunk))
            logger.info("Content: %s", chunk)

            # logger.info("\n<%d>\n%s\n</%d>\n", i, chunk, i)
            if i > 5:
                break
        logger.info(f"Finished testing {name}.\n")


def test_stochastic_chunkers(text: str) -> None:
    encoder = GeminiEncoder(
        model_name="models/embedding-001",
        dimension=768,
        task_type="SEMANTIC_SIMILARITY"
    )
    test_subjects = [
        {
            "name":
            "SemanticChunker",
            "object":
            chunkers.SemanticChunker(
                encoder=encoder,
                config=chunkers.SemanticChunkerConfig(
                    n_chunk=len(text) // 512,
                    chunk_size=512,
                    update_rate=1.0,
                    min_coverage=0.9,
                    max_iteration=100,
                    randomness=0.5
                )
            )
        }, {
            "name":
            "HybridChunker",
            "object":
            chunkers.HybridChunker(
                encoder=encoder,
                config=chunkers.HybridChunkerConfig(
                    chunk_size=512,
                    update_rate=0.3,
                    min_coverage=0.9,
                    max_iteration=50,
                    randomness=0.25,
                    delta=0.0001,
                    patient=7
                )
            )
        }
    ]
    for test_subject in test_subjects:
        chunker = test_subject["object"]
        name = test_subject["name"]
        logger.info(f"Testing {name}...")
        chunks = chunker.split(text)
        logger.info(f"Chunked {len(chunks)} chunks.")
        for i, chunk in enumerate(chunks):
            logger.info("C%d", i)
            logger.info("Chunk length: %d", len(chunk))
            logger.info("Content: %s", chunk)

            # logger.info("\n<%d>\n%s\n</%d>\n", i, chunk, i)
            if i > 5:
                break
        logger.info(f"Finished testing {name}.\n")


if __name__ == "__main__":
    load_dotenv()
    logger.info("START")
    content = load("./dev/document/chinese_doc.txt")
    # test_static_chunkers(content)
    test_stochastic_chunkers(content)
