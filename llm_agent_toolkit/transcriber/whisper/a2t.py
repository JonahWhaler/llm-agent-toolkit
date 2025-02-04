import os
import logging
import json
from typing import Any
import whisper

from ..base import AudioParameter, Transcriber, TranscriptionConfig, AudioHelper
from ..._util import MessageBlock, CreatorRole

logger = logging.getLogger(__name__)


class LocalWhisperTranscriber(Transcriber):
    """
    Transcriber.
    """

    def __init__(
        self,
        config: TranscriptionConfig,
        directory: str,
        audio_parameter: AudioParameter | None = None,
    ):
        Transcriber.__init__(self, config)
        self.__dir = directory
        if audio_parameter is None:
            self.__audio_parameter = AudioParameter()
        else:
            self.__audio_parameter = audio_parameter
        if not self.__available():
            raise ValueError(
                "%s is not available in the model listing.", self.model_name
            )

    def __available(self) -> bool:
        try:
            if self.model_name not in whisper.available_models():
                return False
            # Load model to ~/.cache/whisper
            whisper.load_model(
                name=self.model_name, download_root=self.directory, in_memory=False
            )
            return True
        except Exception as e:
            logger.error("Exception: %s", e)
            raise
        return False

    @property
    def model_name(self) -> str:
        return self.config.name

    @property
    def directory(self) -> str:
        return self.__dir

    async def transcribe_async(
        self, prompt: str, filepath: str, tmp_directory: str, **kwargs
    ) -> list[MessageBlock | dict[str, Any]]:
        """Calling the async function is meaningless because it's runniing on local CPU."""
        return self.transcribe(prompt, filepath, tmp_directory, **kwargs)

    def transcribe(
        self, prompt: str, filepath: str, tmp_directory: str, **kwargs
    ) -> list[MessageBlock | dict[str, Any]]:
        if filepath is None or tmp_directory is None:
            raise ValueError("filepath and tmp_directory are required")

        ext = os.path.splitext(filepath)[-1]
        params: dict = self.__audio_parameter.model_dump()
        params["output_format"] = ext[1:]
        try:
            model = whisper.load_model(
                name=self.model_name, download_root=self.directory, in_memory=True
            )
            chunks = AudioHelper.generate_chunks(
                input_path=filepath, tmp_directory=tmp_directory, **params
            )
            pages = []
            file_object: dict[str, str | list] = {
                "filename": os.path.basename(filepath)
            }
            for idx, chunk_path in enumerate(chunks, start=1):
                result: dict = model.transcribe(
                    audio=chunk_path,
                    temperature=self.config.temperature,
                    word_timestamps=False,
                    condition_on_previous_text=False,
                    initial_prompt=prompt,
                )
                page: dict[str, str | int | list] = {"page_index": idx}

                if self.config.response_format == "json":
                    segments = result["segments"]
                    minimal_segments = []
                    for segment in segments:
                        minimal_segments.append(
                            {
                                "start": round(segment["start"], 2),
                                "end": round(segment["end"], 2),
                                "text": segment["text"].strip(),
                            }
                        )
                    page["segments"] = minimal_segments
                else:  # if self.config.response_format == "text":
                    transcript: str = result["text"]
                    page["text"] = transcript.strip()
                pages.append(page)
            file_object["transcript"] = pages
            output_string = json.dumps(file_object, ensure_ascii=False)
            return [{"role": CreatorRole.ASSISTANT.value, "content": output_string}]
        except Exception as e:
            logger.error("Exception: %s", e)
            raise
