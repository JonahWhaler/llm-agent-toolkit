"""This file only shows that the listed functions are working.
It does not means the results are correct.
Please do not take this as tests.
"""

import asyncio
import logging
from dotenv import load_dotenv

from llm_agent_toolkit.transcriber.whisper import LocalWhisperTranscriber
from llm_agent_toolkit.transcriber import TranscriptionConfig, AudioParameter

logging.basicConfig(
    filename="./dev/log/local-whisper.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


TMP_DIRECTORY = "./dev/output"
MODEL_DIRECTORY = "./dev/models"

AUDIO_PATH = "./dev/audio/short.mp3"
CHUNK_SIZE = 5  # MB
PROMPT = "一座祭坛、支搭帐篷、挖一口井"
MODEL_NAME = "tiny"


def store(content: str, filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def transcribe(
    audio_file_path: str,
    output_directory: str,
    prompt: str,
    model_name: str,
    response_format: str,
) -> None:
    config = TranscriptionConfig(
        name=model_name, temperature=0.2, response_format=response_format
    )
    ap = AudioParameter(max_size_mb=CHUNK_SIZE)
    llm = LocalWhisperTranscriber(config, directory=MODEL_DIRECTORY, audio_parameter=ap)
    transcripts = llm.transcribe(
        prompt=prompt, filepath=audio_file_path, tmp_directory=output_directory
    )
    export_path = f"{output_directory}/audio-local-{response_format}.md"
    markdown_content = [f"{transcript['content']}\n" for transcript in transcripts]
    store("\n".join(markdown_content), export_path)


async def atranscribe(
    audio_file_path: str,
    output_directory: str,
    prompt: str,
    model_name: str,
    response_format: str,
) -> None:
    config = TranscriptionConfig(
        name=model_name, temperature=0.2, response_format=response_format
    )
    ap = AudioParameter(max_size_mb=CHUNK_SIZE)
    llm = LocalWhisperTranscriber(config, directory=MODEL_DIRECTORY, audio_parameter=ap)
    transcripts = await llm.transcribe_async(
        prompt=prompt, filepath=audio_file_path, tmp_directory=output_directory
    )
    export_path = f"{output_directory}/audio-local-{response_format}.md"
    markdown_content = [f"{transcript['content']}\n" for transcript in transcripts]
    store("\n".join(markdown_content), export_path)


def synchronous_tasks():
    transcribe(AUDIO_PATH, TMP_DIRECTORY, PROMPT, MODEL_NAME, "text")
    transcribe(AUDIO_PATH, TMP_DIRECTORY, PROMPT, MODEL_NAME, "json")


async def asynchronous_tasks():
    tasks = [
        atranscribe(AUDIO_PATH, TMP_DIRECTORY, PROMPT, MODEL_NAME, "text"),
        atranscribe(AUDIO_PATH, TMP_DIRECTORY, PROMPT, MODEL_NAME, "json"),
    ]
    await asyncio.gather(*tasks)


def try_transcriber_examples():
    synchronous_tasks()
    # asyncio.run(asynchronous_tasks())


if __name__ == "__main__":
    load_dotenv()
    try_transcriber_examples()
