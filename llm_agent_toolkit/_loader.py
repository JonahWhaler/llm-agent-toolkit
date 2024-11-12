from abc import ABC, abstractmethod


class BaseLoader(ABC):
    @abstractmethod
    def load(self, input_path: str) -> str:
        raise NotImplementedError

    @abstractmethod
    async def load_async(self, input_path: str) -> str:
        raise NotImplementedError
