import os
import logging
import time
import asyncio
from random import random

import openai
from openai import RateLimitError


logger = logging.getLogger(__name__)


class OpenAITTS:
    """
    OpenAI Text-to-Speech (TTS) class.
    This class provides methods to generate speech from text using OpenAI's TTS API.
    It supports various models, voices, and response formats.

    Attributes:
        SUPPORTED_MODELS (tuple): Supported TTS models.
        RESPONSE_FORMATS (tuple): Supported response formats.
        SUPPORTED_VOICES (tuple): Supported voices.
        MAX_INPUT_LENGTH (int): Maximum input length for text.
        MAX_ATTEMPT (int): Maximum number of attempts for generating speech.
        DELAY_FACTOR (float): Factor to increase delay between attempts.
        MAX_DELAY (float): Maximum delay between attempts.

    **Methods**:
        * __update_delay(delay) -> float: Updates the delay for retrying requests.
        * export(output_path, content) -> None: Exports the generated speech to a file.
        * generate(text, output_path) -> None: Generates speech from text and saves it to a file.
        * async_generate(text, output_path) -> None: Asynchronously generates speech from text and saves it to a file.
    """

    SUPPORTED_MODELS = ("tts-1", "tts-1-hd", "gpt-4o-mini-tts")
    RESPONSE_FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")
    SUPPORTED_VOICES = (
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "fable",
        "onyx",
        "nova",
        "sage",
        "shimmer",
        "verse",
    )

    MAX_INPUT_LENGTH: int = 4096
    MAX_ATTEMPT: int = 5
    DELAY_FACTOR: float = 1.5
    MAX_DELAY: float = 60.0

    def __init__(
        self,
        model: str = "gpt-4o-mini-tts",
        voice: str = "alloy",
        speed: float = 1.0,
        response_format: str = "mp3",
    ):
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model must be one of {self.SUPPORTED_MODELS}, got {model}"
            )

        if voice not in self.SUPPORTED_VOICES:
            raise ValueError(
                f"Voice must be one of {self.SUPPORTED_VOICES}, got {voice}"
            )

        if speed < 0.25 or speed > 4.0:
            raise ValueError("Speed must be between 0.25 and 4.0")

        if response_format not in self.RESPONSE_FORMATS:
            raise ValueError(
                f"Output format must be one of {self.RESPONSE_FORMATS}, got {response_format}"
            )

        self.model = model
        self.voice = voice
        self.speed = speed
        self.response_format = response_format

    def __update_delay(self, delay: float) -> float:
        new_delay = delay * self.DELAY_FACTOR
        # Add some randomness to allow bulk requests to retry at a slightly different timing
        new_delay += random() * 5.0
        return min(new_delay, self.MAX_DELAY)

    @staticmethod
    def export(output_path: str, content: bytes) -> None:
        with open(output_path, "wb") as f:
            f.write(content)
        logger.info("Exported to %s", output_path)

    def generate(self, text: str, output_path: str) -> None:
        _text = text.strip()
        if len(_text) >= self.MAX_INPUT_LENGTH:
            raise ValueError(f"Text length exceeds {self.MAX_INPUT_LENGTH} characters")

        _, ext = os.path.splitext(output_path)
        if ext == "":
            raise ValueError("Output file must have an extension")

        ext = ext[1:]  # Remove the leading dot
        if ext != self.response_format:
            raise ValueError(
                f"Output file extension must be {self.response_format}, got {ext}"
            )

        attempt: int = 1
        delay: float = 5.0
        solved: bool = False
        while not solved and attempt < self.MAX_ATTEMPT:
            try:
                client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])
                response = client.audio.speech.create(
                    input=_text,
                    model=self.model,
                    voice=self.voice,
                    response_format=self.response_format,
                    speed=self.speed,
                )
                self.export(output_path, response.content)
                solved = True
            except RateLimitError as rle:
                logger.warning("RateLimitError: %s", rle)
                warn_msg = f"[{attempt}] Retrying in {delay} seconds..."
                logger.warning(warn_msg)
                time.sleep(delay)
                attempt += 1
                delay = self.__update_delay(delay)
                continue
            except Exception as e:
                logger.error("Exception: %s", e, exc_info=True, stack_info=True)
                raise

        if not solved:
            raise RuntimeError("Max re-attempt reached")

    async def async_generate(self, text: str, output_path: str) -> None:
        _text = text.strip()
        if len(_text) >= self.MAX_INPUT_LENGTH:
            raise ValueError(f"Text length exceeds {self.MAX_INPUT_LENGTH} characters")

        _, ext = os.path.splitext(output_path)
        if ext == "":
            raise ValueError("Output file must have an extension")

        ext = ext[1:]  # Remove the leading dot
        if ext != self.response_format:
            raise ValueError(
                f"Output file extension must be {self.response_format}, got {ext}"
            )

        attempt: int = 1
        delay: float = 5.0
        solved: bool = False
        while not solved and attempt < self.MAX_ATTEMPT:
            try:
                client = openai.AsyncClient(api_key=os.environ["OPENAI_API_KEY"])
                response = await client.audio.speech.create(
                    input=_text,
                    model=self.model,
                    voice=self.voice,
                    response_format=self.response_format,
                    speed=self.speed,
                )
                self.export(output_path, response.content)
                solved = True
            except RateLimitError as rle:
                logger.warning("RateLimitError: %s", rle)
                warn_msg = f"[{attempt}] Retrying in {delay} seconds..."
                logger.warning(warn_msg)
                await asyncio.sleep(delay)
                attempt += 1
                delay = self.__update_delay(delay)
                continue
            except Exception as e:
                logger.error("Exception: %s", e, exc_info=True, stack_info=True)
                raise

        if not solved:
            raise RuntimeError("Max re-attempt reached")
