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
    def encode(self, text: str, **kwargs) -> list[float]:
        """Transform string to embedding.

        Args:
            text (str): Content to be embedded.
            kwargs (dict): Optional additional arguments to customize encoding.
                            This is intended for use by subclasses to extend or modify
                            the behavior of the `encode` method, such as configuring
                            tokenization, truncation, padding, or any model-specific parameters.

        Returns:
            list[float]: Embedding
        """
        raise NotImplementedError

    @abstractmethod
    def encode_v2(self, text: str, **kwargs) -> tuple[list[float], int]:
        """Transform string to embedding.

        Args:
            text(str): Content to be embedded.
            kwargs (dict): Optional additional arguments to customize encoding.
                            This is intended for use by subclasses to extend or modify
                            the behavior of the `encode_v2` method, such as configuring
                            tokenization, truncation, padding, or any model-specific parameters.

        Returns:
            tuple: Embedding, Token Count
        """
        raise NotImplementedError

    @abstractmethod
    async def encode_async(self, text: str, **kwargs) -> list[float]:
        """Asynchronously transform string to embedding.

        Args:
            text (str): Content to be embedded.
            kwargs (dict): Optional additional arguments to customize encoding.
                            This is intended for use by subclasses to extend or modify
                            the behavior of the `encode` method, such as configuring
                            tokenization, truncation, padding, or any model-specific parameters.

        Returns:
            list[float]: Embedding
        """
        raise NotImplementedError

    @abstractmethod
    async def encode_v2_async(self, text: str, **kwargs) -> tuple[list[float], int]:
        """Asynchronously transform string to embedding.

        Args:
            text(str): Content to be embedded.
            kwargs (dict): Optional additional arguments to customize encoding.
                            This is intended for use by subclasses to extend or modify
                            the behavior of the `encode_v2` method, such as configuring
                            tokenization, truncation, padding, or any model-specific parameters.

        Returns:
            tuple: Embedding, Token Count
        """
        raise NotImplementedError


class EncoderProfile(TypedDict):
    name: str
    dimension: int
    ctx_length: int
