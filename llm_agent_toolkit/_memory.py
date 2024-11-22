from typing import Deque, TypedDict
from collections import deque
from itertools import islice
from abc import ABC, abstractmethod

from ._util import MessageBlock
from ._encoder import Encoder


class ShortTermMemory:
    def __init__(self, max_entry: int = 100):
        self.__dq: Deque[MessageBlock | dict] = deque(maxlen=max_entry)

    def push(self, message: MessageBlock | dict):
        self.__dq.append(message)

    def last_n(self, n: int = 10) -> list[MessageBlock | dict]:
        q_len = len(self.__dq)
        return list(islice(self.__dq, q_len - n, q_len))

    def to_list(self) -> list[MessageBlock | dict]:
        return list(self.__dq)

    def clear(self) -> None:
        self.__dq.clear()


class SplitTextConfig(TypedDict):
    chunk_size: int
    stride_rate: float


class VectorMemory(ABC):
    def __init__(
        self, vdb, encoder: Encoder, split_text_config: SplitTextConfig, **kwargs
    ):
        self.__vdb = vdb
        self.__encoder = encoder
        self.__split_text_config = split_text_config

    @property
    def encoder(self):
        return self.__encoder

    @property
    def vdb(self):
        return self.__vdb

    @abstractmethod
    def add(self, document_string: str, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def query(self, query_string: str, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def clear(self):
        raise NotImplementedError

    def split_text(self, text: str) -> list[str]:
        chunk_size = self.__split_text_config["chunk_size"]
        stride_rate = self.__split_text_config["stride_rate"]
        stride = int(chunk_size * stride_rate)
        chunks: list[str] = []
        text_length = len(text)
        for start in range(0, text_length, stride):
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]
            chunks.append(chunk)
        return chunks
