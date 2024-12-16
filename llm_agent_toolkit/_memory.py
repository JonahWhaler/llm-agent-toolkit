from typing import Deque
from collections import deque
from itertools import islice
from abc import ABC, abstractmethod

from ._util import MessageBlock
from ._encoder import Encoder
from ._chunkers import Chunker


class ShortTermMemory:
    def __init__(self, max_entry: int = 100):
        self.__dq: Deque[MessageBlock | dict] = deque(maxlen=max_entry)

    def push(self, message: MessageBlock | dict):
        self.__dq.append(message)

    def last_n(self, n: int = 10) -> list[MessageBlock | dict]:
        q_len = len(self.__dq)
        if n >= q_len:
            return self.to_list()
        return list(islice(self.__dq, q_len - n, q_len))

    def to_list(self) -> list[MessageBlock | dict]:
        return list(self.__dq)

    def clear(self) -> None:
        self.__dq.clear()


class VectorMemory(ABC):
    def __init__(self, vdb, encoder: Encoder, chunker: Chunker, **kwargs):
        self.__vdb = vdb
        self.__encoder = encoder
        self.__chunker = chunker

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
        chunks = self.__chunker.split(long_text=text)
        return chunks


class AsyncVectorMemory(ABC):
    def __init__(self, vdb, encoder: Encoder, chunker: Chunker, **kwargs):
        self.__vdb = vdb
        self.__encoder = encoder
        self.__chunker = chunker

    @property
    def encoder(self):
        return self.__encoder

    @property
    def vdb(self):
        return self.__vdb

    @abstractmethod
    async def add(self, document_string: str, **kwargs):
        raise NotImplementedError

    @abstractmethod
    async def query(self, query_string: str, **kwargs):
        raise NotImplementedError

    @abstractmethod
    async def clear(self):
        raise NotImplementedError

    def split_text(self, text: str) -> list[str]:
        chunks = self.__chunker.split(long_text=text)
        return chunks
