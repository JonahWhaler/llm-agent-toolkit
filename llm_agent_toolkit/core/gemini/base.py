import os
import logging
from math import ceil

from PIL import Image
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiCore:
    def __init__(self, model_name: str):
        self.__model_name = model_name
        if not GeminiCore.__available(model_name):
            raise ValueError(
                "%s is not available in Gemini's model listing.", model_name
            )

    @staticmethod
    def __available(model_name: str) -> bool:
        try:
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            response = client.models.list()
            padded_name = f"models/{model_name}"
            for m in response.page:
                if padded_name == m.name:
                    return True
            return False
        except Exception as e:
            logger.error("Exception: %s", str(e))
        return False

    @staticmethod
    def build_profile(model_name: str) -> dict[str, bool | int | str]:
        profile: dict[str, bool | int | str] = {
            "name": model_name,
            "context_length": 128_000,
            "max_output_tokens": 8192,
            "text_generation": True,
            "tool": False,
        }
        return profile

    @staticmethod
    def calculate_image_tokens(width: int, height: int) -> int:
        token_per_tile = 258
        if width <= 384 and height <= 384:
            return token_per_tile

        smaller_dim_size = width if width < height else height

        tile_size = smaller_dim_size / 1.5
        tile_size = max(256, tile_size)
        tile_size = min(768, tile_size)

        tiles_width = ceil(width / tile_size)
        tiles_height = ceil(height / tile_size)

        number_of_tile = tiles_width * tiles_height
        return token_per_tile * number_of_tile

    @staticmethod
    def calculate_token_count(
        msgs: list[types.Content], imgs: list[str] | None = None
    ) -> int:
        """Calculate the token count for the given messages.
        Efficient but not accurate. Child classes should implement a more accurate version.

        Args:
            msgs (list[MessageBlock | dict[str, Any]]): A list of messages.
            imgs (list[str] | None): A list of image path.

        Returns:
            int: The token count.
        """
        CONVERSION_FACTOR = 0.5  # Magic Number!!! Don't trust me!!!
        character_count: int = 0
        for msg in msgs:
            parts = msg.parts
            if parts is None:
                continue

            for p in parts:
                p_text = getattr(p, "text", None)
                if p_text:
                    character_count += len(p_text)
        text_tokens = ceil(character_count * CONVERSION_FACTOR)

        image_tokens = 0
        if imgs:
            for img in imgs:
                with Image.open(img) as image:
                    width, height = image.size
                    image_tokens += GeminiCore.calculate_image_tokens(width, height)
        logger.info("Text: %d toks\nImage: %d toks", text_tokens, image_tokens)
        estimated_tokens = text_tokens + image_tokens
        return estimated_tokens
