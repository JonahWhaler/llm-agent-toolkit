import os
import logging
from math import ceil

from PIL import Image
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiCore:
    csv_path: str | None = None

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

    @classmethod
    def load_csv(cls, input_path: str):
        COLUMNS_STRING = "name,context_length,max_output_tokens,text_generation,tool,text_input,image_input,audio_input,text_output,image_output,audio_output,structured_output,remarks"
        EXPECTED_COLUMNS = set(COLUMNS_STRING.split(","))
        # Begin validation
        with open(input_path, "r", encoding="utf-8") as csv:
            header = csv.readline()
            header = header.strip()
            columns = header.split(",")
            # Expect no columns is missing
            diff = EXPECTED_COLUMNS.difference(set(columns))
            if diff:
                raise ValueError(f"Missing columns in {input_path}: {', '.join(diff)}")
            # Expect all columns are in exact order
            if header != COLUMNS_STRING:
                raise ValueError(
                    f"Invalid header in {input_path}: \n{header}\n{COLUMNS_STRING}"
                )

            for line in csv:
                values = line.strip().split(",")
                name: str = values[0]
                for column, value in zip(columns, values):
                    if column in ["name", "remarks"]:
                        assert isinstance(
                            value, str
                        ), f"{name}.{column} must be a string."
                    elif column in ["context_length", "max_output_tokens"] and value:
                        try:
                            _ = int(value)
                        except ValueError:
                            print(f"{name}.{column} must be an integer.")
                            raise
                    elif value:
                        assert value.lower() in [
                            "true",
                            "false",
                        ], f"{name}.{column} must be a boolean."
        # End validation
        GeminiCore.csv_path = input_path

    @staticmethod
    def build_profile(model_name: str) -> dict[str, bool | int | str]:
        profile: dict[str, bool | int | str] = {"name": model_name}
        if GeminiCore.csv_path:
            with open(GeminiCore.csv_path, "r", encoding="utf-8") as csv:
                header = csv.readline()
                columns = header.strip().split(",")
                while True:
                    line = csv.readline()
                    if not line:
                        break
                    values = line.strip().split(",")
                    if values[0] == model_name:
                        for column, value in zip(columns[1:], values[1:]):
                            if column == "context_length":
                                profile[column] = int(value)
                            elif column == "max_output_tokens":
                                profile[column] = 2048 if value == "" else int(value)
                            elif column == "remarks":
                                profile[column] = value
                            elif value == "TRUE":
                                profile[column] = True
                            else:
                                profile[column] = False
                        break

        # If GeminiCore.csv_path is not set or some fields are missing
        # Assign default values
        if "context_length" not in profile:
            # Most supported context length
            profile["context_length"] = 2048
        if "tool" not in profile:
            # Assume supported
            profile["tool"] = True
        if "text_generation" not in profile:
            # Assume supported
            profile["text_generation"] = True

        return profile

    @staticmethod
    def calculate_image_tokens(width: int, height: int) -> int:
        """
        Estimation calculation based on link below:
        https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding#image-requirements

        Modification:
        1. Add `token_per_tile` to the product of `token_per_tile` and `number_of_tile`
        """
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
        return token_per_tile + token_per_tile * number_of_tile

    @staticmethod
    def calculate_token_count(
        model_name: str,
        system_prompt: str,
        msgs: list[types.Content],
        imgs: list[str] | None = None,
    ) -> int:
        """Calculate the token count for the given messages.

        Args:
            msgs (list[MessageBlock | dict[str, Any]]): A list of messages.
            imgs (list[str] | None): A list of image path.

        Returns:
            int: The token count.

        Notes:
        * https://ai.google.dev/gemini-api/docs/tokens?lang=python
        * Why not use count_tokens to estimate token count needed to process images?
            * As Bytes: No effect
            * As ImageFile: 259 per image, does not scale according to the image size.
        """
        text_contents = [system_prompt]
        for msg in msgs:
            parts = msg.parts
            if parts is None:
                continue
            for p in parts:
                p_text = getattr(p, "text", None)
                if p_text:
                    text_contents.append(p_text)

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        count_token_response = client.models.count_tokens(
            model=model_name,
            contents=text_contents,  # type: ignore
        )
        text_tokens = count_token_response.total_tokens
        if text_tokens is None:
            text_tokens = 0

        image_tokens = 0
        if imgs:
            for img in imgs:
                with Image.open(img) as image:
                    width, height = image.size
                    image_tokens += GeminiCore.calculate_image_tokens(width, height)

        estimated_tokens = text_tokens + image_tokens
        logger.info(
            "Usage Estimation: %d toks\nText: %d toks\nImage: %d toks",
            estimated_tokens,
            text_tokens,
            image_tokens,
        )
        return estimated_tokens
