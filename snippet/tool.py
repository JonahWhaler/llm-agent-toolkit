import logging
import asyncio

# from llm_agent_toolkit import tool, Tool
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
