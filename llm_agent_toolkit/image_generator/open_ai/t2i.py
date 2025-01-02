import os
import logging
import base64
import time

import openai

from ..base import ImageGenerator, ImageGenerationConfig

logger = logging.getLogger(__name__)


class OpenAIImageGenerator(ImageGenerator):
    def __init__(self, config: ImageGenerationConfig):
        ImageGenerator.__init__(self, config)
        if not self.__available():
            raise ValueError("%s is not available in OpenAI's model listing.")

    def __available(self) -> bool:
        try:
            client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])
            for model in client.models.list():
                if self.model_name == model.id:
                    return True
            return False
        except Exception as e:
            logger.error("Exception: %s", e)
            raise

    async def generate_async(
        self, prompt: str, username: str, tmp_directory: str, **kwargs
    ) -> list[str]:
        params = self.config.__dict__
        params["model"] = self.model_name
        params["user"] = username
        params["prompt"] = prompt
        params["n"] = self.config.return_n
        for kw in ["name", "return_n", "max_iteration"]:
            del params[kw]

        output: list[str] = []
        try:
            client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
            images_response = await client.images.generate(**params)
            for idx, image in enumerate(images_response.data, start=1):
                export_path = f"{tmp_directory}/{username}_{int(time.time())}_{idx}.png"
                self.save_file(image, export_path)
                output.append(export_path)

            return output
        except Exception as e:
            logger.error("Exception: %s", e)
            raise

    def generate(
        self, prompt: str, username: str, tmp_directory: str, **kwargs
    ) -> list[str]:
        params = self.config.__dict__
        params["model"] = self.model_name
        params["user"] = username
        params["prompt"] = prompt
        params["n"] = self.config.return_n
        for kw in ["name", "return_n", "max_iteration"]:
            del params[kw]

        output: list[str] = []
        try:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            images_response = client.images.generate(**params)
            for idx, image in enumerate(images_response.data, start=1):
                export_path = f"{tmp_directory}/{username}_{int(time.time())}_{idx}.png"
                self.save_file(image, export_path)
                output.append(export_path)

            return output
        except Exception as e:
            logger.error("Exception: %s", e)
            raise

    @staticmethod
    def save_file(image, export_path: str) -> None:
        image_model = image.model_dump()
        img_b64 = image_model["b64_json"]
        img_decoding = base64.b64decode(img_b64)
        with open(export_path, "wb") as f:
            f.write(img_decoding)
