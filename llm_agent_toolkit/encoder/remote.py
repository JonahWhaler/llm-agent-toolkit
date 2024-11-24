import os
import openai
from .._encoder import Encoder, EncoderProfile


class OpenAIEncoder(Encoder):
    SUPPORTED_MODELS = (
        EncoderProfile(name="text-embedding-3-small", dimension=512),
        EncoderProfile(name="text-embedding-3-small", dimension=1536),
        EncoderProfile(name="text-embedding-3-large", dimension=256),
        EncoderProfile(name="text-embedding-3-large", dimension=512),
        EncoderProfile(name="text-embedding-3-large", dimension=1024),
        EncoderProfile(name="text-embedding-3-large", dimension=3072),
    )

    def __init__(self, model_name: str, dimension: int):
        self.__model_name = model_name
        self.__dimension = dimension

        for profile in OpenAIEncoder.SUPPORTED_MODELS:
            if model_name == profile["name"] and dimension == profile["dimension"]:
                break
        else:
            raise ValueError("Either model name or dimension are not supported.")

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
            raise

    @property
    def dimension(self) -> int:
        return self.__dimension
