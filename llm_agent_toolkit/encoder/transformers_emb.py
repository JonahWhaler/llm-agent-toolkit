import logging
import torch
from transformers import AutoTokenizer, AutoModel  # type: ignore

from .._encoder import Encoder, EncoderProfile

logger = logging.getLogger(__name__)


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

    def encode(self, text: str, **kwargs) -> list[float]:
        """Transform string to embedding.

        Args:
            text (str): Content to be embedded.
            kwargs (dict): Ignored

        Returns:
            list[float]: Embedding

        Notes:
        * truncate=True
        * padding=False
        * add_special_tokens=True
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=False,
                add_special_tokens=True,
            )
            model = AutoModel.from_pretrained(self.model_name)
            return self.__to_embedding(model, tokens)
        except Exception as e:
            logger.error(msg=f"{self.model_name}.encode failed. Error: {str(e)}")
            raise

    def encode_v2(self, text: str, **kwargs) -> tuple[list[float], int]:
        """Transform string to embedding.

        Args:
            text (str): Content to be embedded.
            kwargs (dict): Ignored.

        Returns:
            tuple: Embedding, Token Count

        Notes:
        * truncate=True
        * padding=False
        * add_special_tokens=True
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=False,
                add_special_tokens=True,
            )
            token_count = len(tokens["input_ids"][0])  # type: ignore
            model = AutoModel.from_pretrained(self.model_name)
            return self.__to_embedding(model, tokens), token_count
        except Exception as e:
            logger.error(msg=f"{self.model_name}.encode failed. Error: {str(e)}")
            raise

    async def encode_async(self, text: str, **kwargs) -> list[float]:
        """Transform string to embedding.

        Args:
            text (str): Content to be embedded.
            kwargs (dict): Ignored

        Returns:
            list[float]: Embedding

        Notes:
        * truncate=True
        * padding=False
        * add_special_tokens=True
        * Not asynchronous! Implemented for the sake of interface consistency.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=False,
                add_special_tokens=True,
            )
            model = AutoModel.from_pretrained(self.model_name)
            return self.__to_embedding(model, tokens)
        except Exception as e:
            logger.error(msg=f"{self.model_name}.encode failed. Error: {str(e)}")
            raise

    async def encode_v2_async(self, text: str, **kwargs) -> tuple[list[float], int]:
        """Transform string to embedding.

        Args:
            text (str): Content to be embedded.
            kwargs (dict): Ignored.

        Returns:
            tuple: Embedding, Token Count

        Notes:
        * truncate=True
        * padding=False
        * add_special_tokens=True
        * Not asynchronous! Implemented for the sake of interface consistency.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=False,
                add_special_tokens=True,
            )
            token_count = len(tokens["input_ids"][0])  # type: ignore
            model = AutoModel.from_pretrained(self.model_name)
            return self.__to_embedding(model, tokens), token_count
        except Exception as e:
            logger.error(msg=f"{self.model_name}.encode failed. Error: {str(e)}")
            raise

    @staticmethod
    def __to_embedding(model, tokens):
        """Transform tokens to embedding (detailed steps)."""
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
            return (
                embeddings[0].cpu().numpy()
            )  # Return as a NumPy array for easy handling
