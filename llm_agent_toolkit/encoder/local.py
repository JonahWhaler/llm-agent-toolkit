from tarfile import SUPPORTED_TYPES
from typing import TypedDict

import torch
from transformers import AutoTokenizer, AutoModel, logging  # type: ignore
import ollama

from .._encoder import Encoder, EncoderProfile


class TransformerEncoder(Encoder):
    SUPPORTED_MODELS = (
        EncoderProfile(
            name="sentence-transformers/all-MiniLM-L6-v2", dimension=384
        ),  # A popular model for embeddings
        EncoderProfile(
            name="sentence-transformers/distilbert-base-nli-stsb-mean-tokens",
            dimension=768,
        ),
        EncoderProfile(name="sentence-transformers/all-mpnet-base-v2", dimension=768),
        EncoderProfile(
            name="sentence-transformers/paraphrase-MiniLM-L6-v2", dimension=384
        ),
        EncoderProfile(
            name="sentence-transformers/paraphrase-albert-small-v2", dimension=768
        ),
        EncoderProfile(
            name="sentence-transformers/bert-base-nli-mean-tokens", dimension=768
        ),
        EncoderProfile(
            name="sentence-transformers/roberta-base-nli-stsb-mean-tokens",
            dimension=768,
        ),
        # EncoderProfile(
        #     name="sentence-transformers/distilroberta-base-paraphrase-v1", dimension=768
        # ),
    )

    def __init__(self, model_name: str):
        self.__model_name = model_name
        for profile in TransformerEncoder.SUPPORTED_MODELS:
            if model_name == profile["name"]:
                self.__dimension = profile["dimension"]
                break
        else:
            raise ValueError(f"Selected encoder model are not supported: {model_name}")

    @property
    def dimension(self) -> int:
        return self.__dimension

    def encode(self, text: str) -> list[float]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.__model_name, use_fast=True)
            model = AutoModel.from_pretrained(self.__model_name)
            return self.__to_embedding(model, tokenizer, text)
        except Exception as e:
            raise

    @staticmethod
    def __to_embedding(model, tokenizer, text):
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
    SUPPORTED_MODELS = (
        EncoderProfile(name="bge-m3", dimension=1024),
        EncoderProfile(name="mxbai-embed-large", dimension=1024),
        EncoderProfile(name="snowflake-arctic-embed", dimension=1024),
    )

    def __init__(self, connection_string: str, model_name: str):
        self.__connection_string = connection_string
        self.__model_name = model_name
        self.__dimension = 1024

    @property
    def dimension(self) -> int:
        return self.__dimension

    @property
    def CONN_STRING(self) -> str:
        return self.__connection_string

    def encode(self, text: str) -> list[float]:
        try:
            client = ollama.Client(host=self.CONN_STRING)
            response = client.embeddings(model=self.__model_name, prompt=text)  # type: ignore
            embedding = response.get("embedding")
            return [float(x) for x in embedding]
        except Exception as e:
            print(str(e))
            raise
