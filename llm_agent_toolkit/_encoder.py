from abc import ABC, abstractmethod
from typing import TypedDict


class Encoder(ABC):
    def __init__(self, model_name: str, dimension: int, ctx_length: int):
        self.__model_name = model_name
        self.__dimension = dimension
        self.__ctx_length = ctx_length

    @property
    def model_name(self) -> str:
        """Name of the embedding model"""
        return self.__model_name

    @property
    def dimension(self) -> int:
        """Output dimension of the generated embedding."""
        return self.__dimension

    @property
    def ctx_length(self) -> int:
        """Number of word/token the embedding model can handle."""
        return self.__ctx_length

    @abstractmethod
    def encode(self, text: str) -> list[float]:
        raise NotImplementedError


class EncoderProfile(TypedDict):
    name: str
    dimension: int
    ctx_length: int
