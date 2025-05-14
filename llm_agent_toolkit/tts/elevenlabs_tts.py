import os
import logging
import time
import asyncio
from random import random
import httpx
import elevenlabs
from elevenlabs.client import ElevenLabs, AsyncElevenLabs
from elevenlabs import VoiceSettings


class ElevenLabsTTS:
    """
    ElevenLabs Text-to-Speech (TTS) class.

    This class provides methods to generate speech audio from text using the ElevenLabs API.
    It supports multiple output formats, sampling rates, bitrates, and voices, and includes
    retry logic for robust API interaction.

    Attributes:
        RESPONSE_FORMATS (dict): Supported output formats and their valid (sampling_rate, bitrate) tuples.
        SUPPORTED_VOICES (dict): Mapping of voice names to (voice_id, language_code).
        MAX_ATTEMPT (int): Maximum number of retry attempts for API calls.
        DELAY_FACTOR (float): Factor to increase delay between retries.
        MAX_DELAY (float): Maximum delay between retries (in seconds).

    Args:
        model (str): The ElevenLabs model to use.
        voice (str): The voice name to use.
        speed (float): Playback speed multiplier.
        response_format (str): Output audio format (e.g., "mp3").
        sampling_rate_hz (int): Audio sampling rate in Hz.
        bitrate (int | None): Audio bitrate in kbps (if applicable).

    **Methods**:
        * generate(text: str, output_path: str) -> None:
            Generate speech from text and save to the specified file.
        * export(output_path: str, content: bytes) -> None:
            Save audio content to a file.
        * __update_delay(delay: float) -> float:
            Update the delay for retrying requests.
        * export(output_path: str, content: bytes) -> None:
            Save audio content to a file.

    **Notes**:
        - Make sure to set the `ELEVENLABS_API_KEY` in your environment variables.
    """

    SUPPORTED_MODELS = {
        "eleven_multilingual_v2": 10_000,
        "eleven_flash_v2_5": 40_000,
        "eleven_flash_v2": 30_000,
        "eleven_turbo_v2_5": 40_000,
        "eleven_turbo_v2": 30_000,
    }
    RESPONSE_FORMATS: dict[str, list[tuple[int, int | None]]] = {
        "mp3": [
            (22050, 32),
            (44100, 32),
            (44100, 64),
            (44100, 96),
            (44100, 128),
            (44100, 192),
        ],
        "pcm": [(8000,), (16000,), (22050,), (24000,), (44100,), (48000,)],
        "opus": [(48000, 32), (48000, 64), (48000, 96), (48000, 128), (48000, 192)],
        "ulaw": [(8000,)],
        "alaw": [(8000,)],
    }
    SUPPORTED_VOICES = {
        "James Gao": ("4VZIsMPtgggwNg7OXbPY", "zh"),
        "Amy": ("bhJUNIXWQQ94l8eI2VUf", "zh"),
        "Mark - Natural Conversations": ("UgBBYS2sOqTuMpoF3BR0", "en"),
        "Jessica Anne Bogart - Conversations": ("g6xIsTj2HwM6VR4iXFCw", "en"),
    }

    MAX_ATTEMPT: int = 5
    DELAY_FACTOR: float = 1.5
    MAX_DELAY: float = 60.0

    def __init__(
        self,
        model: str = "elevenlabs",
        voice: str = "alloy",
        speed: float = 1.0,
        response_format: str = "mp3",
        sampling_rate_hz: int = 44100,
        bitrate: int | None = None,
    ):
        supported_models = list(self.SUPPORTED_MODELS.keys())
        if model not in supported_models:
            raise ValueError(f"Model must be one of {supported_models}, got {model}")

        supported_voices = list(self.SUPPORTED_VOICES.keys())
        if voice not in supported_voices:
            raise ValueError(f"Voice must be one of {supported_voices}, got {voice}")

        if speed < 0.7 or speed > 1.2:
            raise ValueError("Speed must be between 0.25 and 4.0")

        supported_formats = list(self.RESPONSE_FORMATS.keys())
        if response_format not in supported_formats:
            raise ValueError(
                f"Output format must be one of {supported_formats}, got {response_format}"
            )

        for format_name in supported_formats:
            if response_format != format_name:
                continue

            options = self.RESPONSE_FORMATS[format_name]
            valid: bool = False
            for option in options:
                sr, *br = option
                if sr != sampling_rate_hz:
                    continue

                if len(br) == 0:
                    valid = True
                else:
                    if br[0] == bitrate:
                        valid = True
                break

            if not valid:
                raise ValueError(
                    f"Output format {response_format} with sampling rate {sampling_rate_hz} and bitrate {bitrate} is not supported"
                )
        # elevenlabs.text_to_speech.types.TextToSpeechConvertRequestOutputFormat
        self.model = model
        self.voice = voice
        self.speed = speed
        self.response_format = response_format
        self.output_format = f"{response_format}_{sampling_rate_hz}"
        if bitrate:
            self.output_format += f"_{bitrate}"

    @staticmethod
    def export(output_path: str, content: bytes) -> None:
        with open(output_path, "wb") as f:
            f.write(content)

    def __update_delay(self, delay: float) -> float:
        new_delay = delay * self.DELAY_FACTOR
        # Add some randomness to allow bulk requests to retry at a slightly different timing
        new_delay += random() * 5.0
        return min(new_delay, self.MAX_DELAY)

    def generate(self, text: str, output_path: str) -> None:
        _text = text.strip()
        if len(_text) >= self.SUPPORTED_MODELS[self.model]:
            raise ValueError(
                f"Text length exceeds the maximum limit of {self.SUPPORTED_MODELS[self.model]} characters"
            )

        _, ext = os.path.splitext(output_path)
        if ext != self.response_format:
            raise ValueError(
                f"Output file extension must be {self.response_format}, got {ext}"
            )

        httpx_timeout: float = max(60.0, len(_text) / 1000 * 60.0)
        voice_id, _ = self.SUPPORTED_VOICES[self.voice]

        attempt: int = 1
        delay: float = 5.0
        solved: bool = False
        while not solved and attempt <= self.MAX_ATTEMPT:
            try:
                client = ElevenLabs(
                    api_key=os.environ["ELEVENLABS_API_KEY"],
                    httpx_client=httpx.Client(timeout=httpx_timeout),
                )

                audio = client.text_to_speech.convert(
                    text=_text,
                    voice_id=voice_id,
                    model_id=self.model,
                    output_format=self.output_format,
                    voice_settings=VoiceSettings(speed=self.speed),
                )

                self.export(output_path, audio)
                solved = True
            except httpx.TimeoutException as te:
                logging.error("[%d] Timeout error: %s.", attempt, str(te))
                attempt += 1
                httpx_timeout *= 1.5
            except elevenlabs.errors.BadRequestError as bre:
                logging.error("[%d] Bad request error: %s.", attempt, str(bre))
                attempt += 1
                delay = self.__update_delay(delay)
                time.sleep(delay)
            except Exception as e:
                logging.error(
                    "[%d] Error: %s.", attempt, str(e), exc_info=True, stack_info=True
                )
                raise

        if not solved:
            raise RuntimeError("Max re-attempt reached")

    async def generate_async(self, text: str, output_path: str) -> None:
        _text = text.strip()
        if len(_text) >= self.SUPPORTED_MODELS[self.model]:
            raise ValueError(
                f"Text length exceeds the maximum limit of {self.SUPPORTED_MODELS[self.model]} characters"
            )

        _, ext = os.path.splitext(output_path)
        if ext != self.response_format:
            raise ValueError(
                f"Output file extension must be {self.response_format}, got {ext}"
            )

        httpx_timeout: float = max(60.0, len(_text) / 1000 * 60.0)
        voice_id, _ = self.SUPPORTED_VOICES[self.voice]

        attempt: int = 1
        delay: float = 5.0
        solved: bool = False
        while not solved and attempt <= self.MAX_ATTEMPT:
            try:
                client = AsyncElevenLabs(
                    api_key=os.environ["ELEVENLABS_API_KEY"],
                    httpx_client=httpx.AsyncClient(timeout=httpx_timeout),
                )

                audio = await client.text_to_speech.convert(
                    text=_text,
                    voice_id=voice_id,
                    model_id=self.model,
                    output_format=self.output_format,
                    voice_settings=VoiceSettings(speed=self.speed),
                )

                self.export(output_path, audio)
                solved = True
            except httpx.TimeoutException as te:
                logging.error("[%d] Timeout error: %s.", attempt, str(te))
                attempt += 1
                httpx_timeout *= 1.5
            except elevenlabs.errors.BadRequestError as bre:
                logging.error("[%d] Bad request error: %s.", attempt, str(bre))
                attempt += 1
                delay = self.__update_delay(delay)
                await asyncio.sleep(delay)
            except Exception as e:
                logging.error(
                    "[%d] Error: %s.", attempt, str(e), exc_info=True, stack_info=True
                )
                raise

        if not solved:
            raise RuntimeError("Max re-attempt reached")
