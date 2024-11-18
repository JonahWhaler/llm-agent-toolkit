import os
import io
import math
import openai
import ffmpeg  # https://pypi.org/project/ffmpeg-python/

from ..._core import A2T_Core
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
        try:
            output = []
            chunks = self.to_chunks(input_path=filepath, tmp_directory=tmp_directory)
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
        try:
            output = []
            chunks = self.to_chunks(input_path=filepath, tmp_directory=tmp_directory)
            for idx, chunk_path in enumerate(chunks):
                with open(chunk_path, "rb") as f:
                    audio_data = f.read()
                    buffer = io.BytesIO(audio_data)
                    buffer.name = filepath
                    buffer.seek(0)
                client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                params = self.config.__dict__
                params["file"] = buffer
                params["prompt"] = (
                    f"SYSTEM={self.system_prompt}\nQUERY={query}\nPage={idx+1}"
                )
                for kw in ["model_name", "return_n", "max_iteration"]:
                    del params[kw]
                transcript = client.audio.transcriptions.create(**params)
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

    # @staticmethod
    def to_chunks(self, input_path: str, tmp_directory: str, **kwargs) -> list[str]:
        max_size_mb = kwargs.get("max_size_mb", 20)
        audio_bitrate = kwargs.get("audio_bitrate", "128k")

        slices: list[str] = []
        try:
            # Get input file information
            probe = ffmpeg.probe(input_path)
            duration = float(probe["format"]["duration"])

            # Convert audio_bitrate to bits per second
            bitrate_bps = (
                int(audio_bitrate[:-1]) * 1024
            )  # Convert 'xxxk' to bits/second

            # Calculate expected output size in bytes
            expected_size_bytes = (bitrate_bps * duration) / 8

            # Calculate the number of slices based on expected output size
            num_slices = math.ceil(expected_size_bytes / (max_size_mb * 1024 * 1024))

            # Calculate the duration of each slice
            slice_duration = duration / num_slices

            # Convert and slice the audio
            for i in range(num_slices):
                start_time = i * slice_duration
                output_file = os.path.join(tmp_directory, f"slice{i + 1}.ogg")
                try:
                    # Convert and slice
                    stream = ffmpeg.input(input_path, ss=start_time, t=slice_duration)
                    stream = ffmpeg.output(
                        stream,
                        output_file,
                        acodec="libvorbis",
                        audio_bitrate=audio_bitrate,
                    )
                    ffmpeg.run(stream, overwrite_output=True)

                    # Print information about the exported file
                    output_probe = ffmpeg.probe(output_file)
                    output_size = int(output_probe["format"]["size"]) / (
                        1024 * 1024
                    )  # Size in MB
                    print(f"Exported {output_file}")
                    print(f"Size: {output_size:.2f} MB")

                    # Print progress
                    progress = (i + 1) / num_slices * 100
                    print(f"Progress: {progress:.2f}%")

                    slices.append(output_file)
                except ffmpeg.Error as e:
                    print(f"Error processing slice {i + 1}:")
                    if e.stderr is not None:
                        print(e.stderr.decode())
                    else:
                        print(str(e))
            return slices
        except ffmpeg.Error as e:
            print("Error during file processing:")
            if e.stderr is not None:
                print(e.stderr.decode())
            else:
                print(str(e))

        return slices
