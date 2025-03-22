import os
import logging
from math import ceil
from typing import Any, Optional
from PIL import Image
from google import genai
from google.genai import types

from ..._util import CreatorRole, TokenUsage
from ..._core import MessageBlock

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
            logger.error("Exception: %s", str(e), exc_info=True, stack_info=True)
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
                            logger.warning(f"{name}.{column} must be an integer.")
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
        text_token_count = count_token_response.total_tokens
        if text_token_count is None:
            text_token_count = 0

        image_token_count = 0
        if imgs:
            for img in imgs:
                with Image.open(img) as image:
                    width, height = image.size
                    image_token_count += GeminiCore.calculate_image_tokens(
                        width, height
                    )

        estimated_tokens = text_token_count + image_token_count
        logger.debug(
            "Token Estimation:\nText: %d\nImage: %d",
            text_token_count,
            image_token_count,
        )
        return estimated_tokens

    @staticmethod
    def update_usage(
        usage: types.GenerateContentResponseUsageMetadata | None,
        token_usage: TokenUsage | None = None,
    ) -> TokenUsage:
        """Transforms GenerateContentResponseUsageMetadata to TokenUsage. This is a adapter function.

        Notes:
        * When finish_reason=<FinishReason.MALFORMED_FUNCTION_CALL: 'MALFORMED_FUNCTION_CALL'>, candidates_token_count is None
        """
        if usage is None:
            raise RuntimeError("Response usage is None.")

        ptc = usage.prompt_token_count
        ctc = usage.candidates_token_count
        if ptc is None:
            ptc = 0

        if ctc is None:
            ctc = 0

        if token_usage is None:
            token_usage = TokenUsage(input_tokens=ptc, output_tokens=ctc)
        else:
            token_usage.input_tokens += ptc
            token_usage.output_tokens += ctc
        logger.debug("Token Usage: %s", token_usage)
        return token_usage

    @staticmethod
    def preprocessing(
        query: str,
        context: Optional[list[MessageBlock | dict]],
        filepath: Optional[str] = None,
    ) -> list[types.Content]:
        """Adapter function to transform MessageBlock to types.Content."""
        output: list[types.Content] = []
        if context is not None:
            for ctx in context:
                _role = ctx["role"]
                if _role == "system":
                    # This can happend when user force an system message into the context
                    _role = "model"
                output.append(
                    types.Content(
                        role=_role,
                        parts=[types.Part.from_text(text=ctx["content"])],
                    )
                )

        if filepath:
            ext = os.path.splitext(filepath)[-1][1:]
            ext = "jpeg" if ext == "jpg" else ext
            mime_type = f"image/{ext}"  # Assume image file only.
            with open(filepath, "rb") as f:
                data_bytes = f.read()
            output.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=data_bytes, mime_type=mime_type),
                        types.Part.from_text(text=query),
                    ],
                )
            )
        else:
            output.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=query)],
                )
            )
        return output

    @staticmethod
    def postprocessing(msgs: list[types.Content]) -> list[MessageBlock | dict]:
        """Adapter function to transform types.Content to MessageBlock."""
        output_list: list[MessageBlock | dict] = []
        for msg in msgs:
            role = getattr(msg, "role", None)
            parts: list[types.Part] | None = getattr(msg, "parts", None)
            assert role is not None
            assert parts is not None
            content = parts[0].text
            if content:
                output_list.append(
                    MessageBlock(
                        role=CreatorRole.ASSISTANT.value if role == "model" else role,
                        content=content,
                    )
                )
            # Parts without the text attribute will be skipped.

        return output_list

    @staticmethod
    def warning_message(
        iteration: int,
        max_iteration: int,
        token_usage: TokenUsage,
        max_tokens: int,
        available_tokens: int,
    ) -> str:
        """
        Generate a warning message given various conditions.
        This funtion assume a warning message is needed.

        Args:
            iteration (int): current iteration count.
            max_iteration (int): maximum iteration allowed.
            token_usage (TokenUsage): token usage record.
            max_tokens (int): maximum token allowed.
            available_tokens (int): available tokens.

        Returns:
            str: A warning message.
        """
        warning_message = "Warning: "
        if iteration >= max_iteration:
            warning_message += "Iteration limit reached."
        elif token_usage.total_tokens >= max_tokens:
            warning_message += f"Maximum token count reached. \
                {token_usage.total_tokens} > {max_tokens}"
        elif available_tokens <= 0:
            warning_message += "No tokens available."
        else:
            warning_message += "Unknown reason."
        return warning_message

    @staticmethod
    def get_function_call(
        response: types.GenerateContentResponse,
    ) -> Optional[dict[str, Any]]:
        try:
            candidates: Optional[list[types.Candidate]] = response.candidates
            if not candidates:
                return None

            content: Optional[types.Candidate] = getattr(candidates[0], "content", None)
            if not content:
                return None

            parts: Optional[list[types.Part]] = getattr(content, "parts", None)
            if not parts:
                return None

            function_call: Optional[types.FunctionCall] = getattr(
                parts[0],
                "function_call",
                None,
            )
            if not function_call:
                return None

            return {
                "id": function_call.id,
                "name": function_call.name,
                "arguments": function_call.args,
            }
        except Exception as e:
            # logger.warning("Function call not found: %s", str(e))
            return None

    @staticmethod
    def get_response_text(response: types.GenerateContentResponse) -> str | None:
        try:
            candidates: Optional[list[types.Candidate]] = response.candidates
            if not candidates:
                return None

            content: Optional[types.Candidate] = getattr(candidates[0], "content", None)
            if content is None:
                return None

            parts: Optional[list[types.Part]] = getattr(content, "parts", None)
            if parts is None:
                return None

            response_text = getattr(parts[0], "text", None)
            if response_text is None:
                return response.text

            return response_text
        except Exception as e:
            # logger.warning("Response text not found: %s", str(e))
            return None

    @staticmethod
    def get_finish_reason(
        response: types.GenerateContentResponse,
    ) -> Optional[types.FinishReason]:
        try:
            candidates: Optional[list[types.Candidate]] = response.candidates
            if not candidates:
                return None

            finish_reason: Optional[types.FinishReason] = getattr(
                candidates[0], "finish_reason", None
            )
            if finish_reason is None:
                return None

            return finish_reason
        except Exception as e:
            # logger.warning("Response text not found: %s", str(e))
            return None


TOOL_PROMPT = """
Utilize tools to solve the problems. 
Results from tools will be kept in the context. 
Calling the tools repeatedly is highly discouraged.
"""
