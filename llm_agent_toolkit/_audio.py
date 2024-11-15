import io
import os
from pathlib import Path
from pydub import AudioSegment
from pydantic import (
    BaseModel,
    FilePath,
    DirectoryPath,
    field_validator,
    ValidationError,
)


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
        if ext in ["ogg", "oga"]:
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
