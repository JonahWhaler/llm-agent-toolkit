import logging
import torch
from transformers import AutoTokenizer, AutoModel  # type: ignore
import ollama

from .._encoder import Encoder, EncoderProfile

logger = logging.getLogger(name=__name__)


class TransformerEncoder(Encoder):
    """
    `TransformerEncoder` is a concrete implementation of `Encoder`.
    This class allow user transform text into embedding through `transformers`.
    """

    # List of models this project had tested.
    SUPPORTED_MODELS = (
        EncoderProfile(
            name="sentence-transformers/all-MiniLM-L6-v2",
            dimension=384,
            ctx_length=256,
        ),  # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        EncoderProfile(
            name="sentence-transformers/distilbert-base-nli-stsb-mean-tokens",
            dimension=768,
            ctx_length=128,
        ),  # https://huggingface.co/sentence-transformers/distilbert-base-nli-stsb-mean-tokens/tree/main
        EncoderProfile(
            name="sentence-transformers/all-mpnet-base-v2",
            dimension=768,
            ctx_length=128,
        ),  # https://huggingface.co/sentence-transformers/all-mpnet-base-v2
        EncoderProfile(
            name="sentence-transformers/paraphrase-MiniLM-L6-v2",
            dimension=384,
            ctx_length=128,
        ),  # https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2
        EncoderProfile(
            name="sentence-transformers/paraphrase-albert-small-v2",
            dimension=768,
            ctx_length=100,
        ),  # https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2
        EncoderProfile(
            name="sentence-transformers/bert-base-nli-mean-tokens",
            dimension=768,
            ctx_length=128,
        ),  # https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens
        EncoderProfile(
            name="sentence-transformers/roberta-base-nli-stsb-mean-tokens",
            dimension=768,
            ctx_length=128,
        ),  # https://huggingface.co/sentence-transformers/roberta-base-nli-stsb-mean-tokens
        # EncoderProfile(
        #     name="sentence-transformers/distilroberta-base-paraphrase-v1", dimension=768
        # ),
    )

    def __init__(
        self,
        model_name: str,
        dimension: int | None = None,
        ctx_length: int | None = None,
    ):
        """
        Initialize an encoder model.

        Parameters:
            - model_name (str): Name of the embedding model
            - dimension (int | None): Output dimension of the generated embedding. This will be ignored if the selected model is covered.
            - ctx_length (int | None): Number of word/token the embedding model can handle. This will be ignored if the selected model is covered.

        Raises:
            - (TypeError): If dimension or ctx_length is not type int.

        Warnings:
        - If the selected model has not been tested in this project.
        """
        for profile in TransformerEncoder.SUPPORTED_MODELS:
            if model_name == profile["name"]:
                dimension = profile["dimension"]
                ctx_length = profile["ctx_length"]
                break
        else:
            logger.warning(
                msg=f"{model_name} has not been tested in this project. Please ensure `dimension` and `ctx_length` are provided correctly."
            )
            if not isinstance(dimension, int):
                raise TypeError("Invalid argument. Expect dimension to be type int.")

            if not isinstance(ctx_length, int):
                raise TypeError("Invalid argument. Expect ctx_length to be type int.")
        super().__init__(model_name, dimension, ctx_length)

    def encode(self, text: str) -> list[float]:
        """Transform string to embedding."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.__model_name, use_fast=True)
            model = AutoModel.from_pretrained(self.__model_name)
            return self.__to_embedding(model, tokenizer, text)
        except Exception as e:
            logger.error(msg=f"{self.__model_name}.encode failed. Error: {str(e)}")
            raise

    @staticmethod
    def __to_embedding(model, tokenizer, text):
        """Transform string to embedding (detailed steps)."""
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            add_special_tokens=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
            return (
                embeddings[0].cpu().numpy()
            )  # Return as a NumPy array for easy handling


class OllamaEncoder(Encoder):
    """
    `OllamaEncoder` is a concrete implementation of `Encoder`.
    This class allow user transform text into embedding through `ollama`.
    """

    # List of models this project had tested.
    SUPPORTED_MODELS = (
        EncoderProfile(
            name="bge-m3", dimension=1024, ctx_length=8192
        ),  # ollama pull bge-m3
        EncoderProfile(
            name="mxbai-embed-large", dimension=1024, ctx_length=512
        ),  # ollama pull mxbai-embed-large
        EncoderProfile(
            name="snowflake-arctic-embed", dimension=1024, ctx_length=512
        ),  # ollama pull snowflake-arctic-embed
    )

    def __init__(
        self,
        connection_string: str,
        model_name: str,
        dimension: int | None = None,
        ctx_length: int | None = None,
    ):
        """
        Initialize an encoder model.

        Parameters:
            - connection_string (str): IP and PORT needed to access Ollama's API
            - model_name (str): Name of the embedding model
            - dimension (int | None): Output dimension of the generated embedding. This will be ignored if the selected model is covered.
            - ctx_length (int | None): Number of word/token the embedding model can handle. This will be ignored if the selected model is covered.

        Raises:
            - (TypeError): If dimension or ctx_length is not type int.

        Warnings:
        - If the selected model has not been tested in this project.
        """
        self.__connection_string = connection_string
        for profile in OllamaEncoder.SUPPORTED_MODELS:
            if profile["name"] == model_name:
                ctx_length = profile["ctx_length"]
                dimension = profile["dimension"]
                break
        else:
            logger.warning(
                msg=f"{model_name} has not been tested in this project. Please ensure `dimension` and `ctx_length` are provided correctly."
            )
            if not isinstance(dimension, int):
                raise TypeError("Invalid argument. Expect dimension to be type int.")

            if not isinstance(ctx_length, int):
                raise TypeError("Invalid argument. Expect ctx_length to be type int.")
        super().__init__(model_name, dimension, ctx_length)

    @property
    def CONN_STRING(self) -> str:
        """IP and PORT needed to access Ollama's API"""
        return self.__connection_string

    def encode(self, text: str) -> list[float]:
        """Transform string to embedding."""
        try:
            client = ollama.Client(host=self.CONN_STRING)
            response = client.embeddings(model=self.__model_name, prompt=text)  # type: ignore
            embedding = response.get("embedding")
            return [float(x) for x in embedding]
        except Exception as e:
            logger.error(msg=f"{self.__model_name}.encode failed. Error: {str(e)}")
            raise
