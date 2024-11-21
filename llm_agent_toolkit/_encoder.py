from abc import ABC, abstractmethod
from typing import TypedDict


class Encoder(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[float]:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError


class EncoderProfile(TypedDict):
    name: str
    dimension: int
