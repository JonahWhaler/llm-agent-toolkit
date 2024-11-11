from llm_agent_toolkit._core import Core
from llm_agent_toolkit._util import (
    OpenAIRole, OpenAIMessage, ContextMessage, ChatCompletionConfig,
)
import os
import openai
import io
import ffmpeg   # https://pypi.org/project/ffmpeg-python/
import math


class A2T_OAI_Core(Core):
    """
    Notes:
    - Only accept audio file in OGG format!!!
    - Large audio files will be split into multiple chunks, overlapping is not supported.
    """

    def __init__(
            self, system_prompt: str, model_name: str, config: ChatCompletionConfig = ChatCompletionConfig(),
            tools: list | None = None
    ):
        super().__init__(
            system_prompt, model_name, config, tools
        )

    async def run_async(
            self,
            query: str,
            context: list[ContextMessage | dict] | None,
            **kwargs
    ) -> list[OpenAIMessage | dict]:
        filepath: str | None = kwargs.get("filepath", None)
        tmp_directory = kwargs.get("tmp_directory", "../")
        if filepath is None or os.path.exists(filepath) is False:
            raise Exception("File does not exist")
        if os.path.exists(tmp_directory) is False:
            raise Exception("Temporary directory does not exist")
        templated_prompt = f"SYSTEM={self.system_prompt}\nQUERY={query}"
        try:
            output = []
            for chunk_path in self.to_chunks(input_file=filepath, tmp_directory=tmp_directory):
                with open(chunk_path, "rb") as f:
                    audio_data = f.read()
                    buffer = io.BytesIO(audio_data)
                    buffer.name = filepath
                    buffer.seek(0)
                client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
                transcript = await client.audio.transcriptions.create(
                    file=buffer, model=self.model_name, response_format='text',
                    temperature=self.config.temperature, prompt=templated_prompt
                )
                output.append(
                    OpenAIMessage(role=OpenAIRole.ASSISTANT, content=transcript)
                )

            return output
        except Exception as e:
            print(f"run_async: {e}")
            raise

    def run(
            self,
            query: str,
            context: list[ContextMessage | dict] | None,
            **kwargs
    ) -> list[OpenAIMessage | dict]:
        filepath: str | None = kwargs.get("filepath", None)
        tmp_directory = kwargs.get("tmp_directory", "../")
        if filepath is None or os.path.exists(filepath) is False:
            raise Exception("File does not exist")
        if os.path.exists(tmp_directory) is False:
            raise Exception("Temporary directory does not exist")
        templated_prompt = f"SYSTEM={self.system_prompt}\nQUERY={query}"
        try:
            output = []
            for chunk_path in self.to_chunks(input_file=filepath, tmp_directory=tmp_directory):
                with open(chunk_path, "rb") as f:
                    audio_data = f.read()
                    buffer = io.BytesIO(audio_data)
                    buffer.name = filepath
                    buffer.seek(0)
                client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                transcript = client.audio.transcriptions.create(
                    file=buffer, model=self.model_name, response_format='text',
                    temperature=self.config.temperature, prompt=templated_prompt
                )
                output.append(
                    OpenAIMessage(role=OpenAIRole.ASSISTANT, content=transcript)
                )

            return output
        except Exception as e:
            print(f"run_async: {e}")
            raise

    @staticmethod
    def to_chunks(
            input_file: str, max_size_mb: int = 20,
            audio_bitrate: str = '128k', tmp_directory: str = "./"):
        slices = []
        try:
            # Get input file information
            probe = ffmpeg.probe(input_file)
            duration = float(probe['format']['duration'])

            # Convert audio_bitrate to bits per second
            bitrate_bps = int(audio_bitrate[:-1]) * 1024  # Convert 'xxxk' to bits/second

            # Calculate expected output size in bytes
            expected_size_bytes = (bitrate_bps * duration) / 8

            # Calculate the number of slices based on expected output size
            num_slices = math.ceil(expected_size_bytes / (max_size_mb * 1024 * 1024))

            # Calculate the duration of each slice
            slice_duration = duration / num_slices

            # Convert and slice the audio
            for i in range(num_slices):
                start_time = i * slice_duration
                output_file = os.path.join(tmp_directory, f'slice{i + 1}.ogg')
                try:
                    # Convert and slice
                    stream = ffmpeg.input(input_file, ss=start_time, t=slice_duration)
                    stream = ffmpeg.output(stream, output_file, acodec='libvorbis', audio_bitrate=audio_bitrate)
                    ffmpeg.run(stream, overwrite_output=True)

                    # Print information about the exported file
                    output_probe = ffmpeg.probe(output_file)
                    output_size = int(output_probe['format']['size']) / (1024 * 1024)  # Size in MB
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
