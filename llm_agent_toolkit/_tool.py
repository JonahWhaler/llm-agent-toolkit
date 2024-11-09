from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
import json


class FunctionPropertyType(Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"


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

    def __dict__(self):
        d = dict()
        if self.minLength is not None:
            d["minLength"] = self.minLength
        if self.maxLength is not None:
            d["maxLength"] = self.maxLength
        if self.pattern is not None:
            d["pattern"] = self.pattern
        if self.format is not None:
            d["format"] = self.format
        if self.maximum is not None:
            d["maximum"] = self.maximum
        if self.minimum is not None:
            d["minimum"] = self.minimum
        if self.exclusiveMaximum is not None:
            d["exclusiveMaximum"] = self.exclusiveMaximum
        if self.exclusiveMinimum is not None:
            d["exclusiveMinimum"] = self.exclusiveMinimum
        if self.multipleOf is not None:
            d["multipleOf"] = self.multipleOf
        if self.minItems is not None:
            d["minItems"] = self.minItems
        if self.maxItems is not None:
            d["maxItems"] = self.maxItems
        if self.uniqueItems is not None:
            d["uniqueItems"] = self.uniqueItems
        if self.enum is not None:
            d["enum"] = self.enum
        return d


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
            property = properties[p_index]

            # Type Checking is essential to ensure the subsequent validation steps are conducted on the correct type
            if property.type == FunctionPropertyType.STRING:
                if isinstance(value, str) is False:
                    return False, "Invalid type for {}, expected string, got {}".format(
                        name, type(value)
                    )
                if property.constraint is not None:
                    if property.constraint.minLength is not None:
                        if property.constraint.minLength > len(value):
                            return (
                                False,
                                "Invalid length for {}, expected at least {}, got {}".format(
                                    name, property.constraint.minLength, len(value)
                                ),
                            )
                    if property.constraint.maxLength is not None:
                        if property.constraint.maxLength < len(value):
                            return (
                                False,
                                "Invalid length for {}, expected at most {}, got {}".format(
                                    name, property.constraint.maxLength, len(value)
                                ),
                            )
                    if property.constraint.enum is not None:
                        if value not in property.constraint.enum:
                            return (
                                False,
                                "Invalid value for {}, expected one of {}, got {}".format(
                                    name, property.constraint.enum, value
                                ),
                            )
            elif property.type == FunctionPropertyType.NUMBER:
                if isinstance(value, (int, float)) is False:
                    return False, "Invalid type for {}, expected number, got {}".format(
                        name, type(value)
                    )
                if property.constraint is not None:
                    if property.constraint.minimum is not None:
                        if property.constraint.minimum > value:
                            return (
                                False,
                                "Invalid value for {}, expected at least {}, got {}".format(
                                    name, property.constraint.minimum, value
                                ),
                            )
                    if property.constraint.maximum is not None:
                        if property.constraint.maximum < value:
                            return (
                                False,
                                "Invalid value for {}, expected at most {}, got {}".format(
                                    name, property.constraint.maximum, value
                                ),
                            )
                    if property.constraint.exclusiveMinimum is not None:
                        if property.constraint.exclusiveMinimum >= value:
                            return (
                                False,
                                "Invalid value for {}, expected exclusive minimum {}, got {}".format(
                                    name, property.constraint.exclusiveMinimum, value
                                ),
                            )
                    if property.constraint.exclusiveMaximum is not None:
                        if property.constraint.exclusiveMaximum <= value:
                            return (
                                False,
                                "Invalid value for {}, expected exclusive maximum {}, got {}".format(
                                    name, property.constraint.exclusiveMaximum, value
                                ),
                            )
                    if property.constraint.multipleOf is not None:
                        tmp = value % property.constraint.multipleOf
                        if tmp != 0:
                            return (
                                False,
                                "Invalid value for {}, expected multiple of {}, got {}".format(
                                    name, property.constraint.multipleOf, value
                                ),
                            )
            elif property.type == FunctionPropertyType.BOOLEAN:
                if isinstance(value, bool) is False:
                    return (
                        False,
                        "Invalid type for {}, expected boolean, got {}".format(
                            name, type(value)
                        ),
                    )
            elif property.type == FunctionPropertyType.OBJECT:
                if isinstance(value, dict) is False:
                    return False, "Invalid type for {}, expected object, got {}".format(
                        name, type(value)
                    )
            else:
                return (
                    False,
                    "Invalid type for {}, expected one of [string, number, boolean, object], got {}".format(
                        name, property.type
                    ),
                )
        return True, None


if __name__ == "__main__":
    pass
