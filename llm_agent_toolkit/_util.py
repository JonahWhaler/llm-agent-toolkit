from enum import Enum
from typing import TypedDict
from pydantic import BaseModel, field_validator, ValidationError


class ModelConfig(BaseModel):
    name: str
    return_n: int = 1
    max_iteration: int = 10

    @field_validator("name")
    def name_must_be_valid(cls, value):  # pylint: disable=no-self-argument
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
    """
    Attributes:
        max_tokens (int): The maximum number of tokens a run is allowed to spend.
        max_output_tokens (int): The maximum number of tokens a generation is allowed to generate.
        temperature (float): Controls the randomness of the generated text.
    """

    max_tokens: int = 4096
    max_output_tokens: int = 2048
    temperature: float = 0.7

    @field_validator("max_tokens")
    def max_tokens_must_be_positive(cls, v):  # pylint: disable=no-self-argument
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v

    @field_validator("max_output_tokens")
    def max_output_tokens_must_be_positive(cls, v):  # pylint: disable=no-self-argument
        if v <= 0:
            raise ValueError("max_output_tokens must be positive")
        return v

    @field_validator("temperature")
    def temperature_must_be_between_0_and_2(cls, v):  # pylint: disable=no-self-argument
        if v < 0 or v > 2:
            raise ValueError("temperature must be between 0 and 2")
        return v


class CreatorRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class RequiredMessageField(TypedDict, total=True):
    role: str
    content: str


class MessageBlock(RequiredMessageField, total=False):
    name: str  # function name when role is `function`


class ResponseMode(str, Enum):
    SO = "structured_output"
    JSON = "json_object"
    DEFAULT = "default"
