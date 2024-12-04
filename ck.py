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


def run_semantic_chunker(model_name: str, filepath: str, tmp_directory: str):

    loader = TextLoader()
    content = loader.load(filepath)
    # content = CONTENT
    encoder = OllamaEncoder(
        connection_string="http://localhost:11434",
        model_name=model_name,
    )
    sc = chunkers.SemanticChunker(
        encoder=encoder,
        config={"K": 10, "MAX_ITERATION": 200, "update_rate": 0.4, "min_coverage": 0.9},
    )
    chunks = sc.split(content)
    for i, chunk in enumerate(chunks, start=1):
        with open(
            f"{tmp_directory}/semantic-{model_name}-{i}.md", "w", encoding="utf-8"
        ) as writer:
            writer.write(f"[{i}]\n{chunk}")


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


if __name__ == "__main__":
    # MODEL_NAME = "snowflake-arctic-embed"  # CTX_LENGTH=512
    FILEPATH = r"./dev/whisper_of_the_camellia.txt"
    TMP = r"./dev"
    MODELS = ["snowflake-arctic-embed", "mxbai-embed-large:latest", "bge-m3:latest"]
    for model_name in MODELS:
        print(f"\n{model_name}\n")
        run_semantic_chunker(model_name, FILEPATH, TMP)
    # how_to_encode(MODEL_NAME)
