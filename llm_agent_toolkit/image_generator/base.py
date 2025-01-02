from abc import abstractmethod, ABC
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from .._util import ModelConfig


class ImageGenerationConfig(ModelConfig):
    size: str = "1024x1024"
    quality: str = "standard"
    response_format: str = "b64_json"

    @field_validator("quality")
    def quality_must_be_valid(cls, value):  # pylint: disable=no-self-argument
        new_value = value.strip()
        if not new_value:
            raise ValidationError("Expect quality to be a non-empty string")
        if new_value not in ["standard", "hd"]:
            raise ValueError("quality must be one of standard, hd")
        return new_value

    @field_validator("response_format")
    def response_format_must_be_valid(cls, value):  # pylint: disable=no-self-argument
        new_value = value.strip()
        if not new_value:
            raise ValidationError("Expect response_format to be a non-empty string")
        if new_value not in ["url", "b64_json"]:
            raise ValueError("response_format must be one of url, b64_json")
        return new_value

    @model_validator(mode="after")
    def size_must_be_valid(cls, values):  # pylint: disable=no-self-argument
        if values.name == "dall-e-2":
            if values.size not in ["1024x1024", "512x512", "256x256"]:
                raise ValueError("size must be one of 1024x1024, 512x512, 256x256")
        if values.name == "dall-e-3":
            if values.size not in ["1024x1024", "1792x1024", "1024x1792"]:
                raise ValueError("size must be one of 1024x1024, 1792x1024, 1024x1792")
        return values


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
