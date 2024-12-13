from abc import ABC, abstractmethod


class BaseLoader(ABC):
    """Although ABC is used here, but it leans towards interface more than an abstract class."""

    @abstractmethod
    def load(self, input_path: str) -> str:
        raise NotImplementedError

    @abstractmethod
    async def load_async(self, input_path: str) -> str:
        raise NotImplementedError
