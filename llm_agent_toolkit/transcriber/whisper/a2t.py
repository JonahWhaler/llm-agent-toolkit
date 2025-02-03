import os
import io
import logging
from typing import Any
import whisper

from ..base import Transcriber, TranscriptionConfig, AudioHelper
from ..._util import MessageBlock, CreatorRole

logger = logging.getLogger(__name__)


class LocalWhisperTranscriber(Transcriber):
    """
    Transcriber.
    """

    def __init__(self, config: TranscriptionConfig, directory: str):
        Transcriber.__init__(self, config)
        if not self.__available():
            raise ValueError(
                "%s is not available in the model listing.", self.model_name
            )
        self.__dir = directory

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
        try:
            output: list[MessageBlock] = []
            chunks = AudioHelper.generate_chunks(
                input_path=filepath, tmp_directory=tmp_directory, output_format=ext[1:]
            )
            model = whisper.load_model(
                name=self.model_name, download_root=self.directory, in_memory=True
            )
            for idx, chunk_path in enumerate(chunks):
                result = model.transcribe(
                    audio=chunk_path,
                    temperature=self.config.temperature,
                    word_timestamps=False,
                    condition_on_previous_text=False,
                )
                transcript: str = getattr(result, "text", "")
                transcript = transcript.strip()
                # BEGIN DEBUG
                filename_wo_ext = os.path.basename(chunk_path).split(".")[0]
                export_path = f"{tmp_directory}/{filename_wo_ext}.md"
                with open(export_path, "w", encoding="utf-8") as writer:
                    writer.write(transcript)
                # END DEBUG
                output.append(
                    {
                        "role": CreatorRole.ASSISTANT.value,
                        "content": f"[{idx+1}]:{transcript}",
                    }
                )
            return [*output]
        except Exception as e:
            logger.error("Exception: %s", e)
            raise
