from abc import abstractmethod, ABC

from ._util import TranscriptionConfig, ImageGenerationConfig, MessageBlock


class Transcriber(ABC):
    def __init__(self, config: TranscriptionConfig):
        self.__config = config

    @property
    def config(self) -> TranscriptionConfig:
        return self.__config

    @abstractmethod
    async def transcribe_async(
        self, prompt: str, filepath: str, tmp_directory: str, **kwargs
    ) -> list[MessageBlock | dict]:
        """Asynchronously run the LLM model to create a transcript from the audio in `filepath`.
        Use this method to explicitly express the intention to create transcript.

        Generated transcripts will be stored under `tmp_directory`.
        """
        raise NotImplementedError

    @abstractmethod
    def transcribe(
        self, prompt: str, filepath: str, tmp_directory: str, **kwargs
    ) -> list[MessageBlock | dict]:
        """Synchronously run the LLM model to create a transcript from the audio in `filepath`.
        Use this method to explicitly express the intention to create transcript.

        Generated transcripts will be stored under `tmp_directory`.
        """
        raise NotImplementedError


class ImageGenerator(ABC):
    def __init__(self, config: ImageGenerationConfig):
        self.__config = config

    @property
    def config(self) -> ImageGenerationConfig:
        return self.__config

    @property
    def model_name(self) -> str:
        return self.config.name

    @abstractmethod
    async def generate_async(
        self, prompt: str, username: str, tmp_directory: str, **kwargs
    ) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self, prompt: str, username: str, tmp_directory: str, **kwargs
    ) -> list[str]:
        raise NotImplementedError
