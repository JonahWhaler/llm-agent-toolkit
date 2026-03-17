from enum import Enum
from typing import TypedDict
from pydantic import BaseModel, field_validator, ValidationError, model_validator


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


class ReasoningEffort(Enum):
    NONE = "none"
    MINIMAL = "minimal"  # Gemini ONLY. Default option for Gemini's 3.x models
    LOW = "low"  # Default option for OpenAI's models
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"  # Gemini ONLY


class ReasoningBudget(BaseModel):
    enable_reasoning: bool = False
    reasoning_effort: ReasoningEffort = ReasoningEffort.NONE
    max_reasonging_tokens: int = 0
    include_reasoning_in_output: bool = True


class ChatCompletionConfig(ModelConfig):
    """
    Attributes:
        max_tokens (int): The maximum number of tokens a run is allowed to spend.
        max_output_tokens (int): The maximum number of tokens a generation is allowed to generate.
        temperature (float): Controls the randomness of the generated text.
        reasoning_budget (ReasoningBudget): Configuration for models with reasoning/thinking capabilities.
    """

    max_tokens: int = 4096
    max_output_tokens: int = 2048
    temperature: float = 0.7
    reasoning_budget: ReasoningBudget = ReasoningBudget()

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

    @model_validator(mode="after")
    def validation_logic(self) -> "ChatCompletionConfig":
        if self.max_output_tokens > self.max_tokens:
            raise ValueError(
                "max_output_tokens must be less than or equal to max_tokens"
            )

        if self.reasoning_budget.enable_reasoning:
            if (
                self.reasoning_budget.reasoning_effort != ReasoningEffort.NONE
                or self.reasoning_budget.max_reasonging_tokens == 0
            ):
                raise ValueError("Reasoning is enabled but budget is not set.")

            if self.reasoning_budget.max_reasonging_tokens > self.max_output_tokens:
                raise ValueError(
                    "max_reasonging_tokens must be less than or equal to max_output_tokens"
                )

        if (
            self.name.startswith("gemini-2.5")
            and self.reasoning_budget.enable_reasoning
            and self.reasoning_budget.max_reasonging_tokens <= 0
        ):
            # In Gemini's context, it's actually thinking instead of reasoning
            raise ValueError(
                "gemini-2.5 models's reasoning budget is controlled by setting the number of tokens that can be used for reasoning."
            )

        if (
            self.name.startswith("gemini-3")
            and self.reasoning_budget.enable_reasoning
            and self.reasoning_budget.reasoning_effort is ReasoningEffort.NONE
        ):
            raise ValueError(
                "gemini-3x models's reasoning budget is controlled by setting the reasoning effort."
            )

        return self


class CreatorRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"
    MODEL = "model"


class RequiredMessageField(TypedDict, total=True):
    role: str
    content: str


class MessageBlock(RequiredMessageField, total=False):
    name: str  # function name when role is `function`


class ResponseMode(str, Enum):
    SO = "structured_output"
    JSON = "json_object"
    DEFAULT = "default"


class UsagePurpose(str, Enum):
    CHAT = "chat"
    EMBEDDING = "embedding"


class TokenUsage(BaseModel):
    purpose: UsagePurpose = UsagePurpose.CHAT
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )
