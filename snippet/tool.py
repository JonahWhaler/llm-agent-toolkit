import logging
import asyncio

from llm_agent_toolkit import (
    Tool,
    FunctionInfo,
    FunctionParameters,
    FunctionProperty,
    FunctionPropertyType,
)
from llm_agent_toolkit.tool import LazyTool


logging.basicConfig(
    filename="./snippet/output/tool.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_lazy_tool_example_1():
    import json

    def connect_to_db(host: str, port: int, username: str, password: str):
        """
        Connects to a database and returns a connection object.

        Args:
            host (str): The hostname or IP address of the database server.
            port (int): The port number on which the database server is listening.
            username (str): The username to use when connecting to the database.
            password (str): The password to use when connecting to the database.

        Returns:
            A connection object to the database.
        """
        assert password == password
        logger.info("Connected to %s:%d as %s", host, port, username)
        return {
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "connected": True,
        }

    lazy_tool = LazyTool(function=connect_to_db, is_coroutine_function=False)

    logger.info("\n%s", json.dumps(lazy_tool.info, indent=4))

    result = lazy_tool.run(
        params=json.dumps(
            {
                "host": "localhost",
                "port": 5432,
                "username": "postgres",
                "password": "postgres",
            }
        )
    )

    logger.info("\n%s", json.dumps(result, indent=4))
    logger.info("END create_lazy_tool_example_1")


def create_lazy_tool_example_2():
    import json

    async def connect_to_db(host: str, port: int, username: str, password: str):
        """
        Connects to a database and returns a connection object.

        Args:
            host (str): The hostname or IP address of the database server.
            port (int): The port number on which the database server is listening.
            username (str): The username to use when connecting to the database.
            password (str): The password to use when connecting to the database.

        Returns:
            A connection object to the database.
        """
        assert password == password
        await asyncio.sleep(2)
        logger.info("Connected to %s:%d as %s", host, port, username)
        return {
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "connected": True,
        }

    lazy_tool = LazyTool(function=connect_to_db, is_coroutine_function=True)

    logger.info("\n%s", json.dumps(lazy_tool.info, indent=4))

    result = lazy_tool.run(
        params=json.dumps(
            {
                "host": "localhost",
                "port": 5432,
                "username": "postgres",
                "password": "postgres",
            }
        )
    )

    logger.info("\n%s", json.dumps(result, indent=4))
    logger.info("END create_lazy_tool_example_2")


async def create_lazy_tool_example_3():
    import json

    async def connect_to_db(host: str, port: int, username: str, password: str):
        """
        Connects to a database and returns a connection object.

        Args:
            host (str): The hostname or IP address of the database server.
            port (int): The port number on which the database server is listening.
            username (str): The username to use when connecting to the database.
            password (str): The password to use when connecting to the database.

        Returns:
            A connection object to the database.
        """
        assert password == password
        await asyncio.sleep(2)
        logger.info("Connected to %s:%d as %s", host, port, username)
        return {
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "connected": True,
        }

    lazy_tool = LazyTool(function=connect_to_db, is_coroutine_function=True)

    logger.info("\n%s", json.dumps(lazy_tool.info, indent=4))

    result = await lazy_tool.run_async(
        params=json.dumps(
            {
                "host": "localhost",
                "port": 5432,
                "username": "postgres",
                "password": "postgres",
            }
        )
    )

    logger.info("\n%s", json.dumps(result, indent=4))
    logger.info("END create_lazy_tool_example_3")


async def create_lazy_tool_example_4():
    import json

    def connect_to_db(host: str, port: int, username: str, password: str):
        """
        Connects to a database and returns a connection object.

        Args:
            host (str): The hostname or IP address of the database server.
            port (int): The port number on which the database server is listening.
            username (str): The username to use when connecting to the database.
            password (str): The password to use when connecting to the database.

        Returns:
            A connection object to the database.
        """
        assert password == password
        logger.info("Connected to %s:%d as %s", host, port, username)
        return {
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "connected": True,
        }

    lazy_tool = LazyTool(function=connect_to_db, is_coroutine_function=False)

    logger.info("\n%s", json.dumps(lazy_tool.info, indent=4))

    result = await lazy_tool.run_async(
        params=json.dumps(
            {
                "host": "localhost",
                "port": 5432,
                "username": "postgres",
                "password": "postgres",
            }
        )
    )

    logger.info("\n%s", json.dumps(result, indent=4))
    logger.info("END create_lazy_tool_example_4")


def create_custom_tool_example_1():
    import json

    class CustomTool(Tool):
        def __init__(
            self, func_info: FunctionInfo, is_coroutine_function: bool = False
        ):
            super().__init__(
                func_info=func_info, is_coroutine_function=is_coroutine_function
            )

        def run(self, params: str) -> str:
            j_params = json.loads(params)
            c = int(j_params["a"]) + int(j_params["b"])
            return str(c)

        async def run_async(self, params: str) -> str:
            j_params = json.loads(params)
            await asyncio.sleep(1)
            c = int(j_params["a"]) + int(j_params["b"])
            return str(c)

    fi = FunctionInfo(
        name="Two Number Sum",
        description="Returns the sum of two numbers",
        parameters=FunctionParameters(
            required=["a", "b"],
            type="object",
            properties=[
                FunctionProperty(
                    name="a",
                    type=FunctionPropertyType.NUMBER,
                    description="The first number",
                ),
                FunctionProperty(
                    name="b",
                    type=FunctionPropertyType.NUMBER,
                    description="The second number",
                ),
            ],
        ),
    )

    custom_tool = CustomTool(func_info=fi, is_coroutine_function=False)

    logger.info("\n%s", json.dumps(custom_tool.info, indent=4))

    A, B = 1, 2
    result = custom_tool.run(params=json.dumps({"a": A, "b": B}))

    logger.info("\n%d + %d = %s", A, B, json.dumps(result, indent=4))
    logger.info("END create_custom_tool_example_1")


async def create_custom_tool_example_2():
    import json

    class CustomTool(Tool):
        def __init__(
            self, func_info: FunctionInfo, is_coroutine_function: bool = False
        ):
            super().__init__(
                func_info=func_info, is_coroutine_function=is_coroutine_function
            )

        def run(self, params: str) -> str:
            j_params = json.loads(params)
            c = int(j_params["a"]) + int(j_params["b"])
            return str(c)

        async def run_async(self, params: str) -> str:
            j_params = json.loads(params)
            await asyncio.sleep(1)
            c = int(j_params["a"]) + int(j_params["b"])
            return str(c)

    fi = FunctionInfo(
        name="Two Number Sum",
        description="Returns the sum of two numbers",
        parameters=FunctionParameters(
            required=["a", "b"],
            type="object",
            properties=[
                FunctionProperty(
                    name="a",
                    type=FunctionPropertyType.NUMBER,
                    description="The first number",
                ),
                FunctionProperty(
                    name="b",
                    type=FunctionPropertyType.NUMBER,
                    description="The second number",
                ),
            ],
        ),
    )

    custom_tool = CustomTool(func_info=fi, is_coroutine_function=True)

    logger.info("\n%s", json.dumps(custom_tool.info, indent=4))

    A, B = 1, 2
    result = await custom_tool.run_async(params=json.dumps({"a": A, "b": B}))

    logger.info("\n%d + %d = %s", A, B, json.dumps(result, indent=4))
    logger.info("END create_custom_tool_example_2")


if __name__ == "__main__":
    logger.info("BEGIN LazyTool Example")
    create_lazy_tool_example_1()  # Synchronous
    create_lazy_tool_example_2()  # Asynchronous
    asyncio.run(
        create_lazy_tool_example_3()
    )  # Asynchronously execute asynchronous function
    asyncio.run(
        create_lazy_tool_example_4()
    )  # Asynchronously execute synchronous function
    logger.info("END LazyTool Example")

    logger.info("BEGIN CustomTool Example")
    create_custom_tool_example_1()
    asyncio.run(create_custom_tool_example_2())
    logger.info("END CustomTool Example")
