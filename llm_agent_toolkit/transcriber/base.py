import io
import os
from pathlib import Path
import math
from abc import abstractmethod, ABC
from typing import Any
from pydub import AudioSegment

# Ignore Skipping analyzing "ffmpeg": module is installed, but missing library stubs or py.typed marker
# https://pypi.org/project/ffmpeg-python/
import ffmpeg  # type: ignore

from pydantic import (
    BaseModel,
    FilePath,
    DirectoryPath,
    field_validator,
    ValidationError,
    model_validator,
)

from .._util import MessageBlock, ModelConfig


class ConvertAudioInput(BaseModel):
    filepath: FilePath
    buffer_name: str
    output_folder: DirectoryPath

    @field_validator("buffer_name")
    def validate_buffer_name(  # pylint: disable=no-self-argument
        cls, value: str
    ) -> str:
        new_value = value.strip()
        if not new_value:
            raise ValidationError("Expect buffer_name to be a non-empty string")
        return new_value


class AudioHelper:
    """
    A utility class for handling audio processing tasks, including format conversion,
    validation, and audio chunking.

    Methods:
        convert_to_ogg_if_necessary(filepath: str, buffer_name: str, output_folder: str) -> str | None:
            Converts the input audio file to OGG format if it is not already in OGG.
            Saves the converted file to the specified output folder.

        validate_input(file_path: str, buffer_name: str, output_folder: str) -> None:
            Validates the provided input parameters to ensure that the file paths and buffer
            names meet the requirements.

        to_chunks(input_path: str, tmp_directory: str, **kwargs) -> list[str]:
            Splits a given audio file into smaller chunks with optional overlap for smoother playback.
            Supports OGG and MP3 output formats.

        generate_chunks(input_path: str, tmp_directory: str, **kwargs):
            A generator version of the to_chunks method that yields each chunk's path as it is created,
            allowing more efficient processing for large files.

    """

    @classmethod
    def convert_to_ogg_if_necessary(
        cls,
        filepath: str,
        buffer_name: str,
        output_folder: str,
    ) -> str | None:
        """
        If the audio file is not in OGG format, it will be converted to OGG.

        Args:
            filepath (str): The path to the audio file.
            buffer_name (str): The name of the buffer.
            output_folder (str): The folder path to save the converted audio file.

        Returns:
            str | None: The path of the converted audio file.
            None: If the audio file is already in OGG format.
        """
        AudioHelper.validate_input(filepath, buffer_name, output_folder)

        ext = os.path.splitext(filepath)[-1]
        if ext in [".ogg", ".oga"]:
            return None

        with open(filepath, "rb") as reader:
            audio_data = reader.read()
            buffer = io.BytesIO(audio_data)

            audio = AudioSegment.from_file(buffer)
            ogg_stream = io.BytesIO()
            audio.export(ogg_stream, format="ogg")

            ogg_stream.seek(0)
            audio_bytes: bytes = ogg_stream.getvalue()
            buffer = io.BytesIO(audio_bytes)

            buffer.name = f"{buffer_name}"
            buffer.seek(0)

        output_path = f"{output_folder}/{buffer_name}.ogg"
        with open(output_path, "wb") as writer:
            writer.write(buffer.getvalue())

        return output_path

    @classmethod
    def validate_input(
        cls,
        file_path: str,
        buffer_name: str,
        output_folder: str,
    ) -> None:
        """Validate input filepath, buffer name, and output folder."""

        _ = ConvertAudioInput(
            filepath=Path(file_path),
            buffer_name=buffer_name,
            output_folder=Path(output_folder),
        )

    @classmethod
    def to_chunks(cls, input_path: str, tmp_directory: str, **kwargs) -> list[str]:
        max_size_mb = kwargs.get("max_size_mb", 20)
        audio_bitrate = kwargs.get(
            "audio_bitrate", "160k"
        )  # Higher bitrate for better quality
        sample_rate = kwargs.get("sample_rate", 48000)  # Maintain original sample rate
        channels = kwargs.get("channels", 2)  # Stereo output
        output_format = kwargs.get(
            "output_format", "ogg"
        )  # Default to ogg, but allow mp3 as well
        overlap_duration = kwargs.get(
            "overlap_duration", 0
        )  # Overlap duration in seconds

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
                start_time = max(
                    0, i * slice_duration - (overlap_duration if i > 0 else 0)
                )
                actual_duration = slice_duration + (
                    overlap_duration if i < num_slices - 1 else 0
                )
                output_file = os.path.join(
                    tmp_directory, f"slice{i + 1}.{output_format}"
                )
                try:
                    # Convert and slice
                    stream = ffmpeg.input(input_path, ss=start_time, t=actual_duration)
                    stream = ffmpeg.filter(stream, "loudnorm")  # Normalize loudness

                    if output_format == "ogg":
                        stream = ffmpeg.output(
                            stream,
                            output_file,
                            acodec="libvorbis",  # OGG format using Vorbis codec
                            audio_bitrate=audio_bitrate,
                            ar=sample_rate,  # Preserve original sample rate (48000 Hz)
                            ac=channels,  # Maintain stereo output
                            compression_level=2,  # Lower compression for better quality
                        )
                    elif output_format == "mp3":
                        stream = ffmpeg.output(
                            stream,
                            output_file,
                            acodec="libmp3lame",  # MP3 format using LAME codec
                            audio_bitrate=audio_bitrate,
                            ar=sample_rate,  # Preserve original sample rate (48000 Hz)
                            ac=channels,  # Maintain stereo output
                        )
                    else:
                        raise ValueError(f"Unsupported output format: {output_format}")

                    ffmpeg.run(stream, overwrite_output=True, quiet=True)

                    # Print information about the exported file
                    # output_probe = ffmpeg.probe(output_file)
                    # output_size = int(output_probe["format"]["size"]) / (
                    #     1024 * 1024
                    # )  # Size in MB
                    # print(f"Exported {output_file}")
                    # print(f"Size: {output_size:.2f} MB")

                    # Print progress
                    # progress = (i + 1) / num_slices * 100
                    # print(f"Progress: {progress:.2f}%")

                    slices.append(output_file)
                except ffmpeg.Error as e:
                    print(f"Error processing slice {i + 1}:")
                    if e.stderr is not None:
                        print(e.stderr.decode())
                    else:
                        print(str(e))
                    raise

            return slices
        except ffmpeg.Error as e:
            print("Error during file processing:")
            if e.stderr is not None:
                print(e.stderr.decode())
            else:
                print(str(e))
            raise

        return slices

    @classmethod
    def generate_chunks(cls, input_path: str, tmp_directory: str, **kwargs):
        """
        Generator version of the to_chunks method that yields audio chunks one at a time.

        Args:
            input_path (str): Path to the input audio file.
            tmp_directory (str): Directory to store the temporary chunks.
            **kwargs: Additional optional parameters for customization:
                - max_size_mb (int): Maximum size of each chunk in megabytes.
                - audio_bitrate (str): Bitrate for the output audio.
                - sample_rate (int): Sample rate for the output audio.
                - channels (int): Number of audio channels.
                - output_format (str): Format for output files ('ogg' or 'mp3').
                - overlap_duration (int): Duration of overlap between chunks in seconds.

        Yields:
            str: Path to each generated audio chunk.
        """
        max_size_mb = kwargs.get("max_size_mb", 20)
        audio_bitrate = kwargs.get(
            "audio_bitrate", "160k"
        )  # Higher bitrate for better quality
        sample_rate = kwargs.get("sample_rate", 48000)  # Maintain original sample rate
        channels = kwargs.get("channels", 2)  # Stereo output
        output_format = kwargs.get(
            "output_format", "ogg"
        )  # Default to ogg, but allow mp3 as well
        overlap_duration = kwargs.get(
            "overlap_duration", 0
        )  # Overlap duration in seconds

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
                start_time = max(
                    0, i * slice_duration - (overlap_duration if i > 0 else 0)
                )
                actual_duration = slice_duration + (
                    overlap_duration if i < num_slices - 1 else 0
                )
                output_file = os.path.join(
                    tmp_directory, f"slice{i + 1}.{output_format}"
                )
                try:
                    # Convert and slice
                    stream = ffmpeg.input(input_path, ss=start_time, t=actual_duration)
                    stream = ffmpeg.filter(stream, "loudnorm")  # Normalize loudness

                    if output_format == "ogg":
                        stream = ffmpeg.output(
                            stream,
                            output_file,
                            acodec="libvorbis",  # OGG format using Vorbis codec
                            audio_bitrate=audio_bitrate,
                            ar=sample_rate,  # Preserve original sample rate (48000 Hz)
                            ac=channels,  # Maintain stereo output
                            compression_level=2,  # Lower compression for better quality
                        )
                    elif output_format == "mp3":
                        stream = ffmpeg.output(
                            stream,
                            output_file,
                            acodec="libmp3lame",  # MP3 format using LAME codec
                            audio_bitrate=audio_bitrate,
                            ar=sample_rate,  # Preserve original sample rate (48000 Hz)
                            ac=channels,  # Maintain stereo output
                        )
                    else:
                        raise ValueError(f"Unsupported output format: {output_format}")

                    ffmpeg.run(stream, overwrite_output=True, quiet=True)

                    # Print information about the exported file
                    # output_probe = ffmpeg.probe(output_file)
                    # output_size = int(output_probe["format"]["size"]) / (
                    #     1024 * 1024
                    # )  # Size in MB
                    # print(f"Exported {output_file}")
                    # print(f"Size: {output_size:.2f} MB")

                    # # Print progress
                    # progress = (i + 1) / num_slices * 100
                    # print(f"Progress: {progress:.2f}%")

                    # Yield the output file path
                    yield output_file
                except ffmpeg.Error as e:
                    print(f"Error processing slice {i + 1}:")
                    if e.stderr is not None:
                        print(e.stderr.decode())
                    else:
                        print(str(e))
                    raise

        except ffmpeg.Error as e:
            print("Error during file processing:")
            if e.stderr is not None:
                print(e.stderr.decode())
            else:
                print(str(e))
            raise


class TranscriptionConfig(ModelConfig):
    temperature: float = 0.7
    response_format: str = "text"
    timestamp_granularities: list[str] = []

    @field_validator("temperature")
    def temperature_must_be_between_0_and_2(cls, v):  # pylint: disable=no-self-argument
        if v < 0 or v > 2:
            raise ValueError("temperature must be between 0 and 2")
        return v

    @field_validator("response_format")
    def response_format_must_be_valid(cls, value):  # pylint: disable=no-self-argument
        new_value = value.strip()
        if not new_value:
            raise ValidationError("Expect response_format to be a non-empty string")
        if new_value not in ["text", "verbose_json"]:
            raise ValueError("response_format must be one of text, verbose_json")
        return new_value

    @field_validator("name")
    def name_must_be_valid(cls, value):  # pylint: disable=no-self-argument
        new_value = value.strip()
        if not new_value:
            raise ValidationError("Expect model_name to be a non-empty string")
        if new_value not in ["whisper-1"]:
            raise ValueError("model_name must be one of whisper-1")
        return new_value

    @model_validator(mode="after")
    def timestamp_granularities_must_be_valid(
        cls, values
    ):  # pylint: disable=no-self-argument
        if values.response_format == "text":
            return values
        if len(values.timestamp_granularities) == 0:
            raise ValueError(
                "timestamp_granularities must be specified when response_format is verbose_json"
            )
        for granularity in values.timestamp_granularities:
            if granularity not in ["segment", "word"]:
                raise ValueError(
                    "timestamp_granularities must be segment or word or both."
                )
        return values


class Transcriber(ABC):
    def __init__(self, config: TranscriptionConfig):
        self.__config = config

    @property
    def config(self) -> TranscriptionConfig:
        return self.__config

    @abstractmethod
    async def transcribe_async(
        self, prompt: str, filepath: str, tmp_directory: str, **kwargs
    ) -> list[MessageBlock | dict[str, Any]]:
        """Asynchronously run the LLM model to create a transcript from the audio in `filepath`.
        Use this method to explicitly express the intention to create transcript.

        Generated transcripts will be stored under `tmp_directory`.
        """
        raise NotImplementedError

    @abstractmethod
    def transcribe(
        self, prompt: str, filepath: str, tmp_directory: str, **kwargs
    ) -> list[MessageBlock | dict[str, Any]]:
        """Synchronously run the LLM model to create a transcript from the audio in `filepath`.
        Use this method to explicitly express the intention to create transcript.

        Generated transcripts will be stored under `tmp_directory`.
        """
        raise NotImplementedError
