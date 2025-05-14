"""
Explore text-to-speech (TTS) API.
"""

import logging
from dotenv import load_dotenv
from llm_agent_toolkit.tts import OpenAITTS
from llm_agent_toolkit.chunkers import SectionChunker

logging.basicConfig(
    filename="./dev/log/text-to-speech.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data


if __name__ == "__main__":
    load_dotenv()
    INPUT_PATH = "sample.md"
    content = load(INPUT_PATH)

    oai_tts = OpenAITTS(
        model="gpt-4o-mini-tts",
        voice="nova",
        speed=1.0,
        response_format="mp3",
    )

    splitter = SectionChunker()
    chunks = splitter.split(content)

    for i, chunk in enumerate(chunks):
        content = chunk.strip()
        logger.info("Generating audio for chunk %d", i)
        logger.info("Chunk content: %s", content)
        oai_tts.generate(text=content, output_path=f"openai_audio-{i}.mp3")
