from abc import abstractmethod, ABC
from dataclasses import dataclass


@dataclass
class ToolOutputItem:
    identifier: str
    value: str
    timestamp: str
    is_answer: bool


@dataclass
class ToolError:
    type: str
    message: str


@dataclass
class ToolOutput:
    tool_name: str
    result: list[ToolOutputItem] | None = None
    error: ToolError | None = None


class BaseTool(ABC):
    def __init__(
        self, tool_name: str, description: str,
        priority: int = 1, next_func: str | None = None
    ):
        self._name = tool_name
        self._description = description
        self._priority = priority
        self._next_func = next_func
        
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def priority(self) -> int:
        return self._priority
    
    @property
    def next_tool_name(self) -> str | None:
        return self._next_func

    @abstractmethod
    def validate(self, params: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def __call__(self, params: str) -> ToolOutput:
        raise NotImplementedError
