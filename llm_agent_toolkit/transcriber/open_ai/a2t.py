import os
import io
import logging
from typing import Any
import openai

from ..base import Transcriber, TranscriptionConfig, AudioHelper
from ..._util import MessageBlock, CreatorRole

logger = logging.getLogger(__name__)


class OpenAITranscriber(Transcriber):
    """
    `OpenAITranscriber` is a concrete implementation of the `Transcriber` abstract class.
    It facilitates synchronous and asynchronous communication with OpenAI's API to transcribe audio files.

    Methods:
    - transcribe(query: str, context: list[MessageBlock | dict[str, Any]] | None, filepath: str, tmp_directory: str, **kwargs) -> list[MessageBlock | dict]:
        Synchronously create transcript from the given audio file.
    - transcribe_async(query: str, context: list[MessageBlock | dict[str, Any]] | None, filepath: str, tmp_directory: str, **kwargs) -> list[MessageBlock | dict]:
        Asynchronously create transcript from the given audio file.

    Notes:
    - Only accept audio file in OGG and MP3 format!!!
    - Large audio files will be split into multiple chunks, overlapping is supported.
    - Generated file are stored with md format.
    """

    def __init__(self, config: TranscriptionConfig):
        Transcriber.__init__(self, config)
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

    @property
    def model_name(self) -> str:
        return self.config.name

    async def transcribe_async(
        self, prompt: str, filepath: str, tmp_directory: str, **kwargs
    ) -> list[MessageBlock | dict[str, Any]]:
        """Asynchronously run the LLM model to create a transcript from the audio in `filepath`.
        Use this method to explicitly express the intention to create transcript.

        Generated transcripts will be stored under `tmp_directory`.
        """
        if filepath is None or tmp_directory is None:
            raise ValueError("filepath and tmp_directory are required")

        ext = os.path.splitext(filepath)[-1]
        params = self.config.__dict__
        params["model"] = self.model_name
        for kw in ["name", "return_n", "max_iteration"]:
            del params[kw]
        try:
            output: list[MessageBlock] = []
            chunks = AudioHelper.generate_chunks(
                input_path=filepath, tmp_directory=tmp_directory, output_format=ext[1:]
            )
            client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
            for idx, chunk_path in enumerate(chunks):
                with open(chunk_path, "rb") as f:
                    audio_data = f.read()
                    buffer = io.BytesIO(audio_data)
                    buffer.name = filepath
                    buffer.seek(0)

                params["file"] = buffer
                params["prompt"] = f"PROMPT={prompt}\nPage={idx+1}"
                transcript = await client.audio.transcriptions.create(**params)
                # BEGIN DEBUG
                filename_wo_ext = os.path.basename(chunk_path).split(".")[0]
                export_path = f"{tmp_directory}/{filename_wo_ext}.md"
                with open(export_path, "w", encoding="utf-8") as writer:
                    writer.write(transcript.strip())
                # END DEBUG
                output.append(
                    MessageBlock(
                        role=CreatorRole.ASSISTANT.value,
                        content=f"[{idx+1}]\n{transcript.strip()}",
                    )
                )
            return [*output]
        except Exception as e:
            logger.error("Exception: %s", e)
            raise

    def transcribe(
        self, prompt: str, filepath: str, tmp_directory: str, **kwargs
    ) -> list[MessageBlock | dict[str, Any]]:
        """Synchronously run the LLM model to create a transcript from the audio in `filepath`.
        Use this method to explicitly express the intention to create transcript.

        Generated transcripts will be stored under `tmp_directory`.
        """
        if filepath is None or tmp_directory is None:
            raise ValueError("filepath and tmp_directory are required")

        ext = os.path.splitext(filepath)[-1]
        params = self.config.__dict__
        params["model"] = self.model_name
        for kw in ["name", "return_n", "max_iteration"]:
            del params[kw]
        try:
            output: list[MessageBlock] = []
            chunks = AudioHelper.generate_chunks(
                input_path=filepath, tmp_directory=tmp_directory, output_format=ext[1:]
            )
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            for idx, chunk_path in enumerate(chunks):
                with open(chunk_path, "rb") as f:
                    audio_data = f.read()
                    buffer = io.BytesIO(audio_data)
                    buffer.name = filepath
                    buffer.seek(0)

                params["file"] = buffer
                params["prompt"] = f"PROMPT={prompt}\nPage={idx+1}"
                transcript = client.audio.transcriptions.create(**params)
                # BEGIN DEBUG
                filename_wo_ext = os.path.basename(chunk_path).split(".")[0]
                export_path = f"{tmp_directory}/{filename_wo_ext}.md"
                with open(export_path, "w", encoding="utf-8") as writer:
                    writer.write(transcript.strip())
                # END DEBUG
                output.append(
                    MessageBlock(
                        role=CreatorRole.ASSISTANT.value,
                        content=f"[{idx+1}]\n{transcript.strip()}",
                    )
                )
            return [*output]
        except Exception as e:
            logger.error("Exception: %s", e)
            raise
