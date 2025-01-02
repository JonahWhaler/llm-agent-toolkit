# """This file only shows that the listed functions are working.
# It does not means the results are correct.
# Please do not take this as tests.
# """

# import asyncio
# import logging
# from dotenv import load_dotenv

# logging.basicConfig(
#     filename="./snippet/output/example-transcriber.log",
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )
# logger = logging.getLogger(__name__)


# AUDIO_PATH = "./dev/sample.mp3"
# PROMPT = "This is a message based on biblical foundation."
# TMP_DIRECTORY = "./dev"


# def transcribe():
#     from llm_agent_toolkit.transcriber.open_ai import OpenAITranscriber
#     from llm_agent_toolkit.transcriber import TranscriptionConfig

#     config = TranscriptionConfig(
#         name="whisper-1",
#         return_n=1,
#         max_iteration=1,
#         temperature=0.7,
#         response_format="text",
#     )
#     llm = OpenAITranscriber(config)
#     transcripts = llm.transcribe(
#         prompt=PROMPT, filepath=AUDIO_PATH, tmp_directory=TMP_DIRECTORY
#     )
#     export_path = f"{TMP_DIRECTORY}/audio.md"
#     with open(export_path, "w", encoding="utf-8") as markdown:
#         for transcript in transcripts:
#             markdown.write(f"{transcript['content']}\n")


# async def atranscribe():
#     from llm_agent_toolkit.transcriber.open_ai import OpenAITranscriber
#     from llm_agent_toolkit.transcriber import TranscriptionConfig

#     config = TranscriptionConfig(
#         name="whisper-1",
#         return_n=1,
#         max_iteration=1,
#         temperature=0.7,
#         response_format="text",
#     )
#     llm = OpenAITranscriber(config)
#     transcripts = await llm.transcribe_async(
#         prompt=PROMPT, filepath=AUDIO_PATH, tmp_directory=TMP_DIRECTORY
#     )
#     export_path = f"{TMP_DIRECTORY}/audio.md"
#     with open(export_path, "w", encoding="utf-8") as markdown:
#         for transcript in transcripts:
#             markdown.write(f"{transcript['content']}\n")


# def synchronous_tasks():
#     transcribe()


# async def asynchronous_tasks():
#     tasks = [atranscribe()]
#     await asyncio.gather(*tasks)


# def try_transcriber_examples():
#     synchronous_tasks()
#     asyncio.run(asynchronous_tasks())


# if __name__ == "__main__":
#     load_dotenv()
#     try_transcriber_examples()
