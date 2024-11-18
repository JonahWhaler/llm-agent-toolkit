from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, TypedDict, Union
import json


class FunctionPropertyType(Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    FUNCTION = "function"


class FunctionPropertyDict(TypedDict, total=False):
    name: str
    type: str
    description: str
    constraint: dict[str, Union[int, float, str, bool, list]]


class FunctionParametersDict(TypedDict):
    type: str
    properties: list[FunctionPropertyDict]
    required: list[str]


class FunctionInfoDict(TypedDict):
    """
    This is expected to aligned with OpenAI's `Function`

    from openai.types.chat.completion_create_params import Function

    Usage:
        function = Function(**function_info)
    """

    name: str
    description: str
    parameters: dict[str, object]


class ToolMetadata(TypedDict):
    type: str
    function: dict[str, object]


@dataclass
class FunctionParameterConstraint:
    """
    List of constraints for a function parameter.
    """

    maxLength: Optional[int] = None
    minLength: Optional[int] = None
    pattern: Optional[str] = None  # Not Supported
    format: Optional[str] = None  # Not Supported
    maximum: Optional[float] = None
    minimum: Optional[float] = None
    exclusiveMaximum: Optional[float] = None
    exclusiveMinimum: Optional[float] = None
    multipleOf: Optional[float] = None
    minItems: Optional[int] = None
    maxItems: Optional[int] = None
    uniqueItems: Optional[bool] = None
    # items: Optional[list[FunctionParameterConstraint]] = None
    enum: Optional[list[Any]] = None

    def to_dict(self) -> dict[str, Union[int, float, str, bool, list]]:
        d = {
            "minLength": self.minLength,
            "maxLength": self.maxLength,
            "pattern": self.pattern,
            "format": self.format,
            "maximum": self.maximum,
            "minimum": self.minimum,
            "exclusiveMaximum": self.exclusiveMaximum,
            "exclusiveMinimum": self.exclusiveMinimum,
            "multipleOf": self.multipleOf,
            "minItems": self.minItems,
            "maxItems": self.maxItems,
            "uniqueItems": self.uniqueItems,
            "enum": self.enum,
        }
        filtered_dict = {k: v for k, v in d.items() if v is not None}
        return filtered_dict


@dataclass
class FunctionProperty:
    name: str
    type: FunctionPropertyType
    description: str
    constraint: FunctionParameterConstraint | None = None

    def to_dict(self) -> dict[str, Union[str, int, float, bool, list]]:
        d: dict[str, Union[str, int, float, bool, list]] = {
            # "name": self.name,
            "type": self.type.value,
            "description": self.description,
        }
        if self.constraint is not None:
            d = {**d, **self.constraint.to_dict()}

        return d


@dataclass
class FunctionParameters:
    properties: list[FunctionProperty]
    type: str = "object"
    required: list[str] | None = None

    def to_dict(self) -> dict:
        properties = {}
        for p in self.properties:
            properties[p.name] = p.to_dict()
        if self.required is not None:
            return {
                "type": "object",
                "properties": properties,
                "required": self.required,
            }
        return {
            "type": "object",
            "properties": properties,
            "required": [],
        }


@dataclass
class FunctionInfo:
    name: str
    description: str
    parameters: FunctionParameters

    def to_dict(self) -> FunctionInfoDict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters.to_dict(),
        }


class Tool(ABC):
    """
    Abstract base class for creating tools compatible with OpenAI's tool calling interface.

    The `Tool` class serves as a blueprint for implementing tools that can be invoked
    through OpenAI's API. It encapsulates the function's metadata, input schema, and
    provides mechanisms for input validation and execution.

    **Attributes:**
    * info (dict): A dictionary containing metadata about the tool, following the OpenAI format.

    * is_coroutine_function (bool): A flag indicating whether the tool is asynchronous or not.

    **Methods:**
    * run(params: str) -> str:
        Executes the tool with the provided JSON-encoded parameters.

    * run_async(params: str) -> str:
        Asynchronously executes the tool with the provided JSON-encoded parameters.

    * show_info():
        Prints the tool's metadata in a formatted JSON structure.

    * validate(**params) -> tuple[bool, Optional[str]]:
        Validates the provided parameters against the tool's input schema.
        Returns a tuple where the first element is a boolean indicating validity,
        and the second element is an error message if validation fails.

    **Abstract Methods:**
    * run(params: str) -> str:
        Must be implemented by subclasses to define the tool's execution logic.

    * run_async(params: str) -> str:
        Must be implemented by subclasses to define the tool's asynchronous execution logic.

    **Initialization:**

    The constructor accepts a `FunctionInfo` object that contains all necessary
    metadata and schema information for the tool. It also performs post-initialization
    checks to ensure consistency and compliance of the provided function information.

    **Raises:**
    * ValueError: If there are inconsistencies in the function information,
                such as missing mandatory fields.
    """

    def __init__(self, func_info: FunctionInfo, is_coroutine_function: bool = False):
        self.__func_info = func_info
        self.__is_coroutine_function = is_coroutine_function
        self.__post_init()

    @abstractmethod
    def run(self, params: str) -> str:
        raise NotImplementedError

    @abstractmethod
    async def run_async(self, params: str) -> str:
        raise NotImplementedError

    @property
    def info(self) -> ToolMetadata:
        return {
            "type": "function",
            "function": {**self.__func_info.to_dict()},
        }

    @property
    def is_coroutine_function(self) -> bool:
        return self.__is_coroutine_function

    def show_info(self):
        print(json.dumps(self.info, indent=4))

    def __post_init(self):
        """
        Perform post-initialization checks to ensure the function information is consistent
        and compliant with the expected schema.

        - Verifies that all mandatory fields specified in the input schema are present
          in the properties.
        - Raises a `ValueError` if there are inconsistencies.

        Returns:
            None

        Raises:
            ValueError: If mandatory fields are missing from the function properties.
        """
        mandatory_fields = self.__func_info.parameters.required
        if mandatory_fields is not None:
            fields = set(p.name for p in self.__func_info.parameters.properties)
            inconsistent_fields = set(mandatory_fields) - fields
            if len(inconsistent_fields) > 0:
                raise ValueError(
                    f"Inconsistent mandatory fields: {', '.join(inconsistent_fields)}"
                )

    def validate_mandatory_fields(self, user_fields: list[str]):
        """
        Check if all mandatory fields are present in the user-provided fields.

        **Args:**
        * user_fields (list[str]): List of field names provided by the user.

        **Returns:**
        * (tuple):
            * (bool): True if all mandatory fields are present, False otherwise.
            * (Optional[str]): Error message if validation fails, else None.
        """
        mandatory_fields = self.__func_info.parameters.required
        if mandatory_fields is None:
            return True, None

        tracker = []
        for mandatory_field in mandatory_fields:
            tracker.append(mandatory_field in user_fields)

        if not all(tracker):
            missing_fields = []
            for idx, cond in enumerate(tracker):
                if not cond:
                    missing_fields.append(mandatory_fields[idx])
            return False, f"Missing mandatory fields: {', '.join(missing_fields)}"

        return True, None

    def validate(self, **params) -> tuple[bool, str | None]:
        """
        Validate input against the function schema.

        **Args:**
        * params (dict): Abitrary keyword arguments representing the input.

        **Returns:**
        * (tuple):
            * (bool): True if input is valid, False otherwise.
            * (Optional[str]): Error message if validation fails, else None.

        **Steps:**
        1. Ensure all mandatory fields are present.
        2. Detect unexpected fields.
        3. Validate the types and constraints of each field.

        **Notes:**
        * This method performs basic validation. For complex validation rules,
              override this method in the subclass.
        * It is assumed that the caller has already converted the input values
              to the correct types as specified in the schema.
        """
        # Ensure all mandatory fields are present
        user_fields = list(params.keys())
        has_mandatory_fields, error_msg = self.validate_mandatory_fields(user_fields)
        if not has_mandatory_fields:
            return False, error_msg

        # Detect Unexpected fields
        expected_fields = set(p.name for p in self.__func_info.parameters.properties)
        unexpected_fields = set(user_fields) - expected_fields
        if len(unexpected_fields) > 0:
            return False, f"Unexpected fields: {', '.join(unexpected_fields)}"

        # Validate Values
        user_values = list(params.values())
        properties = self.__func_info.parameters.properties
        for name, value in zip(user_fields, user_values):
            p_index = [idx for idx, p in enumerate(properties) if p.name == name][0]
            _property = properties[p_index]

            # Type Checking is essential to ensure the subsequent validation steps are conducted on the correct type
            if _property.type == FunctionPropertyType.STRING:
                if isinstance(value, str) is False:
                    return (
                        False,
                        f"Invalid type for {name}, expected string, got {type(value)}",
                    )
                if _property.constraint is not None:
                    if _property.constraint.minLength is not None:
                        if _property.constraint.minLength > len(value):
                            return (
                                False,
                                f"Invalid length for {name}, expected at least {_property.constraint.minLength}, got {len(value)}",
                            )
                    if _property.constraint.maxLength is not None:
                        if _property.constraint.maxLength < len(value):
                            return (
                                False,
                                f"Invalid length for {name}, expected at most {_property.constraint.maxLength}, got {len(value)}",
                            )
                    if _property.constraint.enum is not None:
                        if value not in _property.constraint.enum:
                            return (
                                False,
                                f"Invalid value for {name}, expected one of {_property.constraint.enum}, got {value}",
                            )
            elif _property.type == FunctionPropertyType.NUMBER:
                if isinstance(value, (int, float)) is False:
                    return (
                        False,
                        f"Invalid type for {name}, expected number, got {type(value)}",
                    )
                if _property.constraint is not None:
                    if _property.constraint.minimum is not None:
                        if _property.constraint.minimum > value:
                            return (
                                False,
                                f"Invalid value for {name}, expected at least {_property.constraint.minimum}, got {value}",
                            )
                    if _property.constraint.maximum is not None:
                        if _property.constraint.maximum < value:
                            return (
                                False,
                                f"Invalid value for {name}, expected at most {_property.constraint.maximum}, got {value}",
                            )
                    if _property.constraint.exclusiveMinimum is not None:
                        if _property.constraint.exclusiveMinimum >= value:
                            return (
                                False,
                                f"Invalid value for {name}, expected exclusive minimum {_property.constraint.exclusiveMinimum}, got {value}",
                            )
                    if _property.constraint.exclusiveMaximum is not None:
                        if _property.constraint.exclusiveMaximum <= value:
                            return (
                                False,
                                f"Invalid value for {name}, expected exclusive maximum {_property.constraint.exclusiveMaximum}, got {value}",
                            )
                    if _property.constraint.multipleOf is not None:
                        tmp = value % _property.constraint.multipleOf
                        if tmp != 0:
                            return (
                                False,
                                f"Invalid value for {name}, expected multiple of {_property.constraint.multipleOf}, got {value}",
                            )
            elif _property.type == FunctionPropertyType.BOOLEAN:
                if isinstance(value, bool) is False:
                    return (
                        False,
                        f"Invalid type for {name}, expected boolean, got {type(value)}",
                    )
            elif _property.type == FunctionPropertyType.OBJECT:
                if isinstance(value, dict) is False:
                    return (
                        False,
                        f"Invalid type for {name}, expected object, got {type(value)}",
                    )
            else:
                return (
                    False,
                    f"Invalid type for {name}, expected one of [string, number, boolean, object], got {_property.type}",
                )
        return True, None


if __name__ == "__main__":
    pass
