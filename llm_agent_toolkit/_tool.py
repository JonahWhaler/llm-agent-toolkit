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
    """"""

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

    def __dict__(self):
        d = dict()
        d["name"] = self.name
        d["type"] = self.type.value
        d["description"] = self.description
        if self.constraint is not None:
            for k, v in self.constraint.__dict__().items():
                d[k] = v
        return d


@dataclass
class FunctionSchema:
    properties: list[FunctionProperty]
    type: str = "object"
    required: list[str] | None = None

    def __dict__(self):
        d = dict()
        d["properties"] = [p.__dict__() for p in self.properties]
        if self.type is not None:
            d["type"] = self.type
        if self.required is not None:
            d["required"] = self.required
        return d


@dataclass
class FunctionInfo:
    name: str
    description: str
    input_schema: FunctionSchema

    def __dict__(self) -> dict[str, Any]:
        d = dict(name=self.name, description=self.description)
        d["input_schema"] = self.input_schema.__dict__()
        return d


class Tool(ABC):
    """
    Abstract base class for creating tools compatible with OpenAI's tool calling interface.

    The `Tool` class serves as a blueprint for implementing tools that can be invoked
    through OpenAI's API. It encapsulates the function's metadata, input schema, and
    provides mechanisms for input validation and execution.

    **Attributes:**
    * info (dict): A dictionary containing metadata about the tool, including its
                     name, description, and input schema.

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
    def info(self) -> dict:
        return self.__func_info.__dict__()
    
    @property
    def is_coroutine_function(self) -> bool:
        return self.__is_coroutine_function

    def show_info(self):
        print(json.dumps(self.__func_info.__dict__(), indent=4))

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
        mandatory_fields = self.__func_info.input_schema.required
        if mandatory_fields is not None:
            fields = set(p.name for p in self.__func_info.input_schema.properties)
            inconsistent_fields = set(mandatory_fields) - fields
            if len(inconsistent_fields) > 0:
                raise ValueError(
                    "Inconsistent mandatory fields: {}".format(
                        ", ".join(inconsistent_fields)
                    )
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
        mandatory_fields = self.__func_info.input_schema.required
        tracker = list()
        for mandatory_field in mandatory_fields:
            tracker.append(mandatory_field in user_fields)

        if not all(tracker):
            missing_fields = []
            for idx, cond in enumerate(tracker):
                if not cond:
                    missing_fields.append(mandatory_fields[idx])
            return False, "Missing mandatory fields: {}".format(
                ", ".join(missing_fields)
            )

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
        expected_fields = set(p.name for p in self.__func_info.input_schema.properties)
        unexpected_fields = set(user_fields) - expected_fields
        if len(unexpected_fields) > 0:
            return False, "Unexpected fields: {}".format(", ".join(unexpected_fields))

        # Validate Values
        user_values = list(params.values())
        properties = self.__func_info.input_schema.properties
        for name, value in zip(user_fields, user_values):
            p_index = [idx for idx, p in enumerate(properties) if p.name == name][0]
            _property = properties[p_index]

            # Type Checking is essential to ensure the subsequent validation steps are conducted on the correct type
            if _property.type == FunctionPropertyType.STRING:
                if isinstance(value, str) is False:
                    return False, "Invalid type for {}, expected string, got {}".format(
                        name, type(value)
                    )
                if _property.constraint is not None:
                    if _property.constraint.minLength is not None:
                        if _property.constraint.minLength > len(value):
                            return (
                                False,
                                "Invalid length for {}, expected at least {}, got {}".format(
                                    name, _property.constraint.minLength, len(value)
                                ),
                            )
                    if _property.constraint.maxLength is not None:
                        if _property.constraint.maxLength < len(value):
                            return (
                                False,
                                "Invalid length for {}, expected at most {}, got {}".format(
                                    name, _property.constraint.maxLength, len(value)
                                ),
                            )
                    if _property.constraint.enum is not None:
                        if value not in _property.constraint.enum:
                            return (
                                False,
                                "Invalid value for {}, expected one of {}, got {}".format(
                                    name, _property.constraint.enum, value
                                ),
                            )
            elif _property.type == FunctionPropertyType.NUMBER:
                if isinstance(value, (int, float)) is False:
                    return False, "Invalid type for {}, expected number, got {}".format(
                        name, type(value)
                    )
                if _property.constraint is not None:
                    if _property.constraint.minimum is not None:
                        if _property.constraint.minimum > value:
                            return (
                                False,
                                "Invalid value for {}, expected at least {}, got {}".format(
                                    name, _property.constraint.minimum, value
                                ),
                            )
                    if _property.constraint.maximum is not None:
                        if _property.constraint.maximum < value:
                            return (
                                False,
                                "Invalid value for {}, expected at most {}, got {}".format(
                                    name, _property.constraint.maximum, value
                                ),
                            )
                    if _property.constraint.exclusiveMinimum is not None:
                        if _property.constraint.exclusiveMinimum >= value:
                            return (
                                False,
                                "Invalid value for {}, expected exclusive minimum {}, got {}".format(
                                    name, _property.constraint.exclusiveMinimum, value
                                ),
                            )
                    if _property.constraint.exclusiveMaximum is not None:
                        if _property.constraint.exclusiveMaximum <= value:
                            return (
                                False,
                                "Invalid value for {}, expected exclusive maximum {}, got {}".format(
                                    name, _property.constraint.exclusiveMaximum, value
                                ),
                            )
                    if _property.constraint.multipleOf is not None:
                        tmp = value % _property.constraint.multipleOf
                        if tmp != 0:
                            return (
                                False,
                                "Invalid value for {}, expected multiple of {}, got {}".format(
                                    name, _property.constraint.multipleOf, value
                                ),
                            )
            elif _property.type == FunctionPropertyType.BOOLEAN:
                if isinstance(value, bool) is False:
                    return (
                        False,
                        "Invalid type for {}, expected boolean, got {}".format(
                            name, type(value)
                        ),
                    )
            elif _property.type == FunctionPropertyType.OBJECT:
                if isinstance(value, dict) is False:
                    return False, "Invalid type for {}, expected object, got {}".format(
                        name, type(value)
                    )
            else:
                return (
                    False,
                    "Invalid type for {}, expected one of [string, number, boolean, object], got {}".format(
                        name, _property.type
                    ),
                )
        return True, None


if __name__ == "__main__":
    pass
