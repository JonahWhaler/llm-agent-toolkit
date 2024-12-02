import os
import logging
import openai
from .._encoder import Encoder, EncoderProfile

logger = logging.getLogger(name=__name__)


class OpenAIEncoder(Encoder):
    SUPPORTED_MODELS = (
        EncoderProfile(name="text-embedding-3-small", dimension=512, ctx_length=8191),
        EncoderProfile(name="text-embedding-3-small", dimension=1536, ctx_length=8191),
        EncoderProfile(name="text-embedding-3-large", dimension=256, ctx_length=8191),
        EncoderProfile(name="text-embedding-3-large", dimension=512, ctx_length=8191),
        EncoderProfile(name="text-embedding-3-large", dimension=1024, ctx_length=8191),
        EncoderProfile(name="text-embedding-3-large", dimension=3072, ctx_length=8191),
    )

    def __init__(self, model_name: str, dimension: int):
        for profile in OpenAIEncoder.SUPPORTED_MODELS:
            if model_name == profile["name"] and dimension == profile["dimension"]:
                ctx_length = profile["ctx_length"]
                break
        else:
            raise ValueError("Either model name or dimension are not supported.")
        super().__init__(model_name, dimension, ctx_length)

    def encode(self, text: str) -> list[float]:
        try:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.embeddings.create(
                model=self.__model_name,
                dimensions=self.__dimension,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(msg=f"{self.__model_name}.encode failed. Error: {str(e)}")
            raise
