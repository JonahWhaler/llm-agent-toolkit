from typing import Any, Callable, Coroutine, Union
import json
import asyncio
import inspect

from .._tool import (
    FunctionInfo,
    FunctionProperty,
    FunctionPropertyType,
    FunctionParameters,
    Tool,
)


class LazyTool(Tool):
    """
    Concrete implementation of the `Tool` class that automatically generates tool metadata
    based on a provided Python function.

    The `LazyTool` class introspects a given function to extract its signature and annotations,
    thereby generating the necessary `FunctionInfo` and `FunctionSchema` required by the
    `Tool` base class. This allows for seamless integration of regular Python functions
    into OpenAI's tool calling interface without manual specification of metadata.

    **Important Considerations:**

    * **Reliance on Function Signature and Type Annotations:**
        - `LazyTool` heavily relies on the provided function's signature and type annotations for its arguments and return value.
        - Accurate and precise type annotations are crucial to ensure that the input schema is correctly generated and that validations are performed accurately.
        - Missing or incorrect type annotations can lead to improper validation or runtime errors.

    * **Docstring Parsing:**
        - Currently, `LazyTool` does not parse the function's docstring beyond using it as a general description.
        - Parsing the docstring to extract detailed parameter descriptions and additional metadata is not implemented.
        - Enhancing `LazyTool` to parse and utilize docstrings for more comprehensive metadata generation is planned for future work.

    **Attributes:**
    * `__function` (Callable | Coroutine):
        The Python function that this tool wraps and executes.

    * `info` (dict):
        Inherited from the `Tool` base class. Contains metadata about the tool, including
        its name, description, and input schema.

    * `is_async` (bool):
        Inherited from the `Tool` base class. Indicates whether the wrapped function is
        asynchronous or not.

    **Methods:**
    * `run(params: str) -> str`:
        Executes the wrapped function with the provided JSON-encoded parameters.
        Validates the input before execution and returns the function's output or an error message.

    * `run_async(params: str) -> str`:
        Asynchronously executes the wrapped function with the provided JSON-encoded parameters.
        Validates the input before execution and returns the function's output or an error message.

    * `guess_type(annotation: Any) -> FunctionPropertyType`:
        Infers the `FunctionPropertyType` based on the type annotation of a function parameter.
        Supports basic types such as `str`, `int`, `float`, and `bool`. Defaults to `OBJECT` for
        unrecognized types.

    * `pre_init(function: Callable) -> FunctionInfo`:
        Generates a `FunctionInfo` object by analyzing the provided function's signature and
        docstring. Extracts parameter names, types, and descriptions to construct the input schema.

    **Initialization:**

    The constructor accepts a `Callable` (function) and performs the following steps:
    1. Introspects the function's signature and annotations to determine parameter types and requirements.
    2. Generates a `FunctionInfo` object encapsulating the function's metadata and input schema.
    3. Initializes the base `Tool` class with the generated `FunctionInfo`.

    **Raises:**
    * `ValueError`:
        If there are inconsistencies in the function's signature, such as missing mandatory fields
        or unsupported parameter types.
    """

    def __init__(
        self,
        function: Union[Callable[..., Any], Coroutine[Any, Any, str]],
        is_coroutine_function: bool = False,
    ):
        """
        Initialize the `LazyTool` with a given function.

        This constructor introspects the provided function to automatically generate the
        necessary `FunctionInfo` and `FunctionSchema`, then initializes the base `Tool`
        class with this information.

        Args:
            __function (Callable | Coroutine):
                The Python function to be wrapped by this tool. The function's signature
                and type annotations are used to generate the tool's input schema.

            is_coroutine_function (bool, optional):
                Whether the function is asynchronous or not. Defaults to False.

        Raises:
            ValueError:
                If the function's signature contains inconsistencies or unsupported types.

        Notes:
            - This constructor checks if the provided function is a coroutine function as specified by the `is_coroutine_function` flag.
        """
        assert inspect.iscoroutinefunction(function) == is_coroutine_function

        func_info = LazyTool.get_func_info(function)
        super().__init__(
            func_info=func_info, is_coroutine_function=is_coroutine_function
        )
        self.__function = function

    @classmethod
    def guess_type(cls, annotation: Any) -> FunctionPropertyType:
        """
        Infer the `FunctionPropertyType` based on a type annotation.

        Args:
            annotation (Any):
                The type annotation of a function parameter as a string (e.g., "str", "int").

        Returns:
            FunctionPropertyType:
                The corresponding `FunctionPropertyType` based on the annotation.
                Defaults to `OBJECT` if the annotation is unrecognized.
        """
        if annotation == "str":
            return FunctionPropertyType.STRING
        elif annotation == "int":
            return FunctionPropertyType.NUMBER
        elif annotation == "float":
            return FunctionPropertyType.NUMBER
        elif annotation == "bool":
            return FunctionPropertyType.BOOLEAN
        else:
            return FunctionPropertyType.OBJECT

    @classmethod
    def get_func_info(cls, function: Union[Callable, Coroutine]) -> FunctionInfo:
        """
        Generate `FunctionInfo` by analyzing the provided function.

        This method introspects the function's signature to extract parameter names and types.
        It constructs the input schema based on the function's parameters.

        Args:
            function (Callable):
                The Python function to analyze.

        Returns:
            FunctionInfo:
                An object containing the function's name, description, and input schema.
        """
        signature = inspect.signature(function)  # type: ignore
        parameters = signature.parameters
        ppt = []
        required = []
        for name, param in parameters.items():
            fp = FunctionProperty(
                name=name,
                type=LazyTool.guess_type(param.annotation.__name__),
                description=name,
            )
            ppt.append(fp)
            if param.default == inspect.Parameter.empty:
                required.append(name)

        input_schema = FunctionParameters(
            type="object", required=required, properties=ppt
        )
        doc = function.__doc__
        return FunctionInfo(
            name=function.__name__,
            description=function.__name__ if doc is None else doc.strip("\n "),
            parameters=input_schema,
        )

    def run(self, params: str) -> str:
        """
        Execute the wrapped function with the provided parameters.

        This method performs the following steps:
        1. Parses the JSON-encoded `params` string into a dictionary.
        2. Validates the parameters against the tool's input schema.
        3. If validation passes, calls the wrapped function with the parameters.
        4. Returns the function's output or an error message if validation fails.

        Args:
            params (str):
                A JSON-encoded string representing the parameters to pass to the function.

        Returns:
            str:
                The result of the function execution or an error message if validation fails.

        Notes:
            - Pass the wrapped function to `asyncrio.run` if the function is asynchronous.
        """
        j_params = json.loads(params)
        valid_input, error_msg = self.validate(**j_params)
        if not valid_input and error_msg:
            return error_msg
        if self.is_coroutine_function:
            return asyncio.run(self.__function(**j_params))  # type: ignore
        return self.__function(**j_params)  # type: ignore

    async def run_async(self, params: str) -> str:
        """
        Execute the wrapped function with the provided parameters.

        This method performs the following steps:
        1. Parses the JSON-encoded `params` string into a dictionary.
        2. Validates the parameters against the tool's input schema.
        3. If validation passes, calls the wrapped function with the parameters.
        4. Returns the function's output or an error message if validation fails.

        Args:
            params (str):
                A JSON-encoded string representing the parameters to pass to the function.

        Returns:
            str:
                The result of the function execution or an error message if validation fails.

        Notes:
            - Does not convert the wrapped function to `asynchronous` execution if the wrapped function is synchronous.
        """
        j_params = json.loads(params)
        valid_input, error_msg = self.validate(**j_params)
        if not valid_input and error_msg:
            return error_msg

        if self.is_coroutine_function:
            return await self.__function(**j_params)  # type: ignore
        return self.__function(**j_params)  # type: ignore
