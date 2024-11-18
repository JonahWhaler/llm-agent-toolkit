from enum import Enum
from pydantic import BaseModel, field_validator, ValidationError, model_validator


class ModelConfig(BaseModel):
    model_name: str
    return_n: int = 1
    max_iteration: int = 10

    @field_validator("model_name")
    def model_name_must_be_valid(cls, value):  # pylint: disable=no-self-argument
        new_value = value.strip()
        if not new_value:
            raise ValidationError("Expect model_name to be a non-empty string")
        return new_value

    @field_validator("return_n")
    def return_n_must_be_positive(cls, v):  # pylint: disable=no-self-argument
        if v <= 0:
            raise ValueError("return_n must be positive")
        return v

    @field_validator("max_iteration")
    def max_iteration_must_be_positive(cls, v):  # pylint: disable=no-self-argument
        if v <= 0:
            raise ValueError("max_iteration must be positive")
        return v


class ChatCompletionConfig(ModelConfig):
    max_tokens: int = 4096
    temperature: float = 0.7

    @field_validator("max_tokens")
    def max_tokens_must_be_positive(cls, v):  # pylint: disable=no-self-argument
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v

    @field_validator("temperature")
    def temperature_must_be_between_0_and_2(cls, v):  # pylint: disable=no-self-argument
        if v < 0 or v > 2:
            raise ValueError("temperature must be between 0 and 2")
        return v


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
        if values.model_name == "dall-e-2":
            if values.size not in ["1024x1024", "512x512", "256x256"]:
                raise ValueError("size must be one of 1024x1024, 512x512, 256x256")
        if values.model_name == "dall-e-3":
            if values.size not in ["1024x1024", "1792x1024", "1024x1792"]:
                raise ValueError("size must be one of 1024x1024, 1792x1024, 1024x1792")
        return values


class CreatorRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
