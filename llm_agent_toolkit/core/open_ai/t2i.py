import os
import base64
import openai
from ..._core import Core
from ..._util import (
    CreatorRole,
    ImageGenerationConfig,
    MessageBlock,
)


class T2I_OAI_Core(Core):
    """
    `T2I_OAI_Core` is a concrete implementation of the `Core` abstract class.
    It facilitates synchronous and asynchronous communication with OpenAI's API to generate images with the given query.

    Methods:
    - run(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> list[MessageBlock | dict]:
        Synchronously generate images based on the given query.
        `Context` is not supported.
    - run_async(query: str, context: list[MessageBlock | dict] | None, **kwargs) -> list[MessageBlock | dict]:
        Asynchronously generate images based on the given query.
        `Context` is not supported.

    Notes:
    - Supported image format: png, jpeg, gif, webp
    - Tools are not supported in current version.
    - Context is not supported in current version.
    """

    def __init__(
        self,
        system_prompt: str,
        config: ImageGenerationConfig,
    ):
        assert isinstance(config, ImageGenerationConfig)
        super().__init__(system_prompt, config, None)
        assert config.response_format == "b64_json"

    async def run_async(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        """
        Asynchronously generate images based on the given query.

        Args:
            query (str): The query to generate images for.
            context (list): A list of context messages or dictionaries.
            **kwargs: Additional keyword arguments. tmp_directory and user_name are optional.

        Returns:
            list: A list of generated images (filepath).

        Notes:
        - `Context` is not supported
        - Generated images are saved in the specified temporary directory
        - The file name is in the `{username}_{index}.png` format.
        """
        username: str = kwargs.get("user_name", "User")
        tmp_directory: str = kwargs.get("tmp_directory", "./")
        params = self.config.__dict__
        params["model"] = self.model_name
        params["user"] = username
        params["prompt"] = query
        for kw in ["name", "return_n", "max_iteration"]:
            del params[kw]

        output: list[MessageBlock] = []
        try:
            client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
            images_response = await client.images.generate(**params)
            for idx, image in enumerate(images_response.data):
                img_model = image.model_dump()
                img_b64 = img_model["b64_json"]
                img_decoding = base64.b64decode(img_b64)
                export_path = f"{tmp_directory}/{username}_{idx}.png"
                with open(export_path, "wb") as f:
                    f.write(img_decoding)

                output.append(
                    MessageBlock(role=CreatorRole.ASSISTANT.value, content=export_path)
                )

            return [*output]
        except Exception as e:
            # print(f"run_async: {e}")
            raise

    def run(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        """
        Synchronously generate images based on the given query.

        Args:
            query (str): The query to generate images for.
            context (list): A list of context messages or dictionaries.
            **kwargs: Additional keyword arguments. tmp_directory and user_name are optional.

        Returns:
            list: A list of generated images (filepath).

        Notes:
        - `Context` is not supported
        - Generated images are saved in the specified temporary directory
        - The file name is in the `{username}_{index}.png` format.
        """
        username: str = kwargs.get("user_name", "User")
        tmp_directory: str = kwargs.get("tmp_directory", "./")
        params = self.config.__dict__
        params["model"] = self.model_name
        params["user"] = username
        params["prompt"] = query
        for kw in ["name", "return_n", "max_iteration"]:
            del params[kw]

        output: list[MessageBlock] = []
        try:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            images_response = client.images.generate(**params)
            for idx, image in enumerate(images_response.data):
                img_model = image.model_dump()
                img_b64 = img_model["b64_json"]
                img_decoding = base64.b64decode(img_b64)
                export_path = f"{tmp_directory}/{username}_{idx}.png"
                with open(export_path, "wb") as f:
                    f.write(img_decoding)
                output.append(
                    MessageBlock(role=CreatorRole.ASSISTANT.value, content=export_path)
                )

            return [*output]
        except Exception as e:
            # print(f"run_async: {e}")
            raise
