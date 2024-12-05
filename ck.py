import logging

from llm_agent_toolkit import chunkers
from llm_agent_toolkit.encoder.local import OllamaEncoder
from llm_agent_toolkit.loader import TextLoader

logging.basicConfig(
    filename="./snippet/output/chunkers.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CONTENT = """
半岛大水灾︱ 水灾未完全好转 北方大学延缓返校上课
time
18小时前
view
274点阅

（亚罗士打2日讯）水灾情况未完全好转，原定3日恢复到校上课的北方大学学生，将继续上网课至本月5日。

校方是于今日在面子书发布有关通告。

校方说，该校将于本月8日恢复返校上课。
"""


def run_semantic_chunker(
    model_name: str, filepath: str, tmp_directory: str, config: dict
):
    loader = TextLoader()
    content = loader.load(filepath)
    # content = CONTENT
    encoder = OllamaEncoder(
        connection_string="http://localhost:11434",
        model_name=model_name,
    )
    sc = chunkers.SemanticChunker(
        encoder=encoder,
        config=config,
    )
    chunks = sc.split(content)
    for i, chunk in enumerate(chunks, start=1):
        with open(
            f"{tmp_directory}/semantic-{model_name}-{i}.md", "w", encoding="utf-8"
        ) as writer:
            writer.write(f"{chunk}")


def run_sa_semantic_chunker(
    model_name: str, filepath: str, tmp_directory: str, config: dict
):
    loader = TextLoader()
    content = loader.load(filepath)
    # content = CONTENT
    encoder = OllamaEncoder(
        connection_string="http://localhost:11434",
        model_name=model_name,
    )
    sc = chunkers.SimulatedAnnealingSemanticChunker(
        encoder=encoder,
        config=config,
    )
    chunks = sc.split(content)
    for i, chunk in enumerate(chunks, start=1):
        with open(
            f"{tmp_directory}/sa-semantic-{model_name}-{i}.md", "w", encoding="utf-8"
        ) as writer:
            writer.write(f"{chunk}")


def how_to_encode(model_name: str):

    def calculate_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        similarity = dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0
        return similarity

    encoder = OllamaEncoder(
        connection_string="http://localhost:11434",
        model_name=model_name,
    )
    a = "（亚罗士打2日讯）水灾情况未完全好转，原定3日恢复到校上课的北方大学学生，将继续上网课至本月5日。"
    b = "校方是于今日在面子书发布有关通告。"
    print(f"Encoding {a}")
    vec1 = encoder.encode(a)
    print(f"Encoding {b}")
    vec2 = encoder.encode(b)
    cs = calculate_cosine_similarity(vec1, vec2)
    print(f"{a} vs {b} = {cs}")


def run_fcc():
    logger.info("run_fcc")
    ck = chunkers.FixedCharacterChunker(config={"chunk_size": 50, "stride_rate": 0.8})
    chunks = ck.split(CONTENT)
    for i, chunk in enumerate(chunks, start=1):
        logger.info("<%d><chunk>%s</chunk><len>%d</len></%d>", i, chunk, len(chunk), i)


def run_fgc():
    logger.info("run_fgc")
    ck = chunkers.FixedGroupChunker(
        config={"K": 4, "resolution": "skip", "level": "word"}
    )
    chunks = ck.split(CONTENT)
    for i, chunk in enumerate(chunks, start=1):
        logger.info(
            "\n\n<%d>\n\t<chunk>\n\t\t%s\n\t</chunk>\n\t<len>\n\t\t%d\n\t</len>\n</%d>",
            i,
            chunk,
            len(chunk),
            i,
        )


if __name__ == "__main__":
    # MODEL_NAME = "snowflake-arctic-embed"  # CTX_LENGTH=512
    FILEPATH = r"./dev/openai-text-generation.md"
    TMP = r"./dev"
    MODELS = ["bge-m3:latest"]  # "snowflake-arctic-embed", "mxbai-embed-large:latest",
    CONFIG_1 = {"K": 5, "MAX_ITERATION": 100, "update_rate": 0.4, "min_coverage": 0.9}
    CONFIG_2 = {
        "K": 5,
        "MAX_ITERATION": 20,
        "update_rate": 0.4,
        "min_coverage": 0.9,
        "temperature": 1.0,
        "cooling_rate": 0.0,
        "constants": (0, 0.5, 2.0, 0.5),
    }
    for mdl_name in MODELS:
        print(f"\n{mdl_name}\n")
        # run_sa_semantic_chunker(mdl_name, FILEPATH, TMP, CONFIG_2)
        run_semantic_chunker(mdl_name, FILEPATH, TMP, CONFIG_1)
    # how_to_encode(MODEL_NAME)
    # run_fcc()
    # run_fgc()
