import os
import openai
from dataclasses import dataclass
from enum import Enum


class EmbeddingModel:
    def __init__(
            self, embedding_model: str = "text-embedding-3-small",
    ):
        self.__embedding_model = embedding_model

    def text_to_embedding(self, text: str):
        try:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.embeddings.create(
                input=text, model=self.__embedding_model)
            return response.data[0].embedding
        except Exception as e:
            print(f"text_to_embedding: {e}")
            raise


@dataclass
class ChatCompletionConfig:
    max_tokens: int = 4096
    temperature: float = 0.7
    n: int = 1


@dataclass
class ImageGenerationConfig:
    size: str = "1024x1024"
    n: int = 1
    response_format: str = "b64_json"


class OpenAIRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class OpenAIFunction:
    call_id: str
    function_name: str
    args: str

    def __dict__(self):
        return {
            "call_id": self.call_id,
            "function_name": self.function_name,
            "args": self.args
        }


@dataclass
class OpenAIMessage:
    role: OpenAIRole
    content: str
    functions: list[OpenAIFunction] | None = None

    def __dict__(self):
        if self.role == OpenAIRole.FUNCTION:
            return {
                "role": self.role.value,
                "content": self.content,
                "functions": [
                    f.__dict__() for f in self.functions
                ]
            }
        return {
            "role": self.role.value,
            "content": self.content
        }


@dataclass
class ContextMessage:
    role: OpenAIRole
    content: str
    name: str | None = None

    def __dict__(self):
        if self.role == OpenAIRole.FUNCTION:
            return {
                "role": self.role.value,
                "content": self.content,
                "name": self.name
            }
        return {
            "role": self.role.value,
            "content": self.content
        }


class AudioHelper:
    import io

    @classmethod
    def convert_to_ogg_if_necessary(
            cls, filepath: str, buffer_name: str, mimetype: str, output_path: str | None = None
    ) -> io.BytesIO:
        import io
        from pydub import AudioSegment

        ext = mimetype.split('/')[1]

        with open(filepath, "rb") as reader:
            audio_data = reader.read()
            buffer = io.BytesIO(audio_data)
            if ext not in ["ogg", "oga"]:
                audio = AudioSegment.from_file(buffer)
                ogg_stream = io.BytesIO()
                audio.export(ogg_stream, format="ogg")

                ogg_stream.seek(0)
                buffer = ogg_stream.getvalue()
                buffer = io.BytesIO(buffer)

            buffer.name = f"{buffer_name}.ogg"
            buffer.seek(0)

            if output_path is not None:
                with open(output_path, 'wb') as writer:
                    writer.write(buffer.getvalue())
            return buffer
