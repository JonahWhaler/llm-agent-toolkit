import logging
from llm_agent_toolkit import encoder

logging.basicConfig(
    filename="./snippet/output/encoder.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_local_encoder():
    logger.info("BEGIN run_local_encoder")

    for profile in encoder.TransformerEncoder.SUPPORTED_MODELS:
        model_name = profile["name"]
        local_encoder = encoder.TransformerEncoder(model_name=model_name)
        query = "What is this package all about?"
        query_vector = local_encoder.encode(query)
        logger.info("Model Name = %s, Dimension: %d", model_name, len(query_vector))

    logger.info("END run_local_encoder")


if __name__ == "__main__":
    run_local_encoder()
