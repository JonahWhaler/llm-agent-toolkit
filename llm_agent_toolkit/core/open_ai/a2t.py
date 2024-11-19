import os
import io
import math
import openai

from ..._core import A2T_Core
from ..._audio import AudioHelper
from ..._util import (
    CreatorRole,
    TranscriptionConfig,
    MessageBlock,
)


class A2T_OAI_Core(A2T_Core):
    """
    Notes:
    - Only accept audio file in OGG format!!!
    - Large audio files will be split into multiple chunks, overlapping is not supported.
    """

    def __init__(
        self,
        system_prompt: str,
        config: TranscriptionConfig,
        tools: list | None = None,
    ):
        super().__init__(system_prompt, config, None)

    async def run_async(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        filepath: str | None = kwargs.get("filepath", None)
        tmp_directory = kwargs.get("tmp_directory", None)
        if filepath is None or tmp_directory is None:
            raise ValueError("filepath and tmp_directory are required")
        ext = os.path.splitext(filepath)[-1]
        try:
            output = []
            chunks = AudioHelper.generate_chunks(
                input_path=filepath, tmp_directory=tmp_directory, output_format=ext
            )
            for idx, chunk_path in enumerate(chunks):
                with open(chunk_path, "rb") as f:
                    audio_data = f.read()
                    buffer = io.BytesIO(audio_data)
                    buffer.name = filepath
                    buffer.seek(0)
                client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
                params = self.config.__dict__
                params["file"] = buffer
                params["prompt"] = (
                    f"SYSTEM={self.system_prompt}\nQUERY={query}\nPage={idx+1}"
                )
                for kw in ["model_name", "return_n", "max_iteration"]:
                    del params[kw]
                transcript = await client.audio.transcriptions.create(**params)
                output.append(
                    {
                        "role": CreatorRole.ASSISTANT.value,
                        "content": f"Page={idx+1}\n{transcript.strip()}",
                    }
                )
            return [*output]
        except Exception as e:
            print(f"run_async: {e}")
            raise

    def run(
        self, query: str, context: list[MessageBlock | dict] | None, **kwargs
    ) -> list[MessageBlock | dict]:
        filepath: str | None = kwargs.get("filepath", None)
        tmp_directory = kwargs.get("tmp_directory", None)
        if filepath is None or tmp_directory is None:
            raise ValueError("filepath and tmp_directory are required")
        ext = os.path.splitext(filepath)[-1]
        model_name = self.config.model_name
        params = self.config.__dict__
        params["model"] = model_name
        for kw in ["model_name", "return_n", "max_iteration"]:
            del params[kw]

        try:
            output = []
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            chunks = AudioHelper.generate_chunks(
                input_path=filepath, tmp_directory=tmp_directory, output_format=ext
            )
            for idx, chunk_path in enumerate(chunks):
                with open(chunk_path, "rb") as f:
                    audio_data = f.read()
                    buffer = io.BytesIO(audio_data)
                    buffer.name = filepath
                    buffer.seek(0)

                params["file"] = buffer
                params["prompt"] = (
                    f"SYSTEM={self.system_prompt}\nQUERY={query}\nPage={idx+1}"
                )
                transcript = client.audio.transcriptions.create(**params)
                filename_wo_ext = os.path.basename(chunk_path).split(".")[0]
                export_path = f"{tmp_directory}/{filename_wo_ext}.md"
                with open(export_path, "w") as writer:
                    writer.write(transcript.strip())

                output.append(
                    {
                        "role": CreatorRole.ASSISTANT.value,
                        "content": f"Page={idx+1}\n{transcript.strip()}",
                    }
                )
            return [*output]
        except Exception as e:
            print(f"run_async: {e}")
            raise
