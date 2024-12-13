# import os
# from .._loader import BaseLoader
# from .._core import Core, A2T_Core


# class A2T2TLoader(BaseLoader):

#     SUPPORTED_EXTENSIONS = (".mp3", ".ogg", ".oga")

#     def __init__(
#         self,
#         transcript_polisher: Core,
#         transcriber: A2T_Core,
#         writer: Core,
#         tmp_directory: str,
#     ):
#         self.__transcript_polisher = transcript_polisher
#         self.__transcriber = transcriber
#         self.__writer = writer
#         self.__tmp_directory = tmp_directory

#     def load(self, filepath: str) -> str:
#         A2T2TLoader.raise_if_invalid(filepath)

#         # Transform audio to text
#         print("Step 1: Transform audio to text")
#         markdown_content = []

#         transcripts = self.__transcriber.run(
#             query="",
#             context=None,
#             filepath=filepath,
#             tmp_directory=self.__tmp_directory,
#         )

#         for transcript in transcripts:
#             transcript_string = transcript.get("content", "")

#             markdown_content.append(transcript_string)

#         full_transcript = "".join(markdown_content)

#         # Polish the transcript
#         print("Step 2: Polish the transcript")
#         corrected_transcripts = []
#         for page_index, text_chunk in enumerate(
#             self.text_to_chunks(full_transcript, chunk_size=8192, stride_rate=0.8),
#             start=1,
#         ):
#             results = self.__transcript_polisher.run(
#                 query=f"Please work on this text chunk:\n{text_chunk}",
#                 context=None,  # type: ignore
#             )
#             result = results[-1]  # Tap into the last result
#             corrected_transcripts.append(result.get("content", ""))

#         # Turn the segmented transcripts into a single transcript
#         print("Step 3: Turn the segmented transcripts into a single transcript")
#         full_corrected_transcript = "".join(corrected_transcripts)
#         with open(f"{self.__tmp_directory}/full_corrected_transcript.txt", "w") as f:
#             f.write(full_corrected_transcript)

#         assert len(full_corrected_transcript) < 128 * 1000

#         results = self.__writer.run(
#             query=full_corrected_transcript,
#             context=None,  # type: ignore
#         )
#         result = results[-1]  # Tap into the last result
#         return result.get("content", "Not Available")

#     async def load_async(self, filepath: str) -> str:
#         A2T2TLoader.raise_if_invalid(filepath)

#         # Transform audio to text
#         print("Step 1: Transform audio to text")
#         markdown_content = []
#         transcripts = await self.__transcriber.run_async(
#             query="",
#             context=None,
#             filepath=filepath,
#             tmp_directory=self.__tmp_directory,
#         )

#         for transcript in transcripts:
#             transcript_string = transcript.get("content", "")

#             markdown_content.append(transcript_string)

#         full_transcript = "".join(markdown_content)

#         # Polish the transcript
#         print("Step 2: Polish the transcript")
#         corrected_transcripts = []
#         for text_chunk in self.text_to_chunks(
#             full_transcript, chunk_size=8192, stride_rate=0.8
#         ):
#             results = await self.__transcript_polisher.run_async(
#                 query=f"Please work on this text chunk:\n{text_chunk}",
#                 context=None,  # type: ignore
#             )
#             result = results[-1]  # Tap into the last result
#             corrected_transcripts.append(result.get("content", ""))

#         # Turn the segmented transcripts into a single transcript
#         print("Step 3: Turn the segmented transcripts into a single transcript")
#         full_corrected_transcript = "".join(corrected_transcripts)
#         assert len(full_corrected_transcript) < 128 * 1000

#         results = await self.__writer.run_async(
#             query="Please execute your role as defined in the system prompt.",
#             context=None,  # type: ignore
#         )
#         result = results[-1]  # Tap into the last result
#         return result.get("content", "Not Available")

#     @staticmethod
#     def text_to_chunks(text: str, chunk_size: int = 4096, stride_rate: float = 0.8):
#         stride = int(chunk_size * stride_rate)
#         chunks: list[str] = []
#         for start in range(0, len(text), stride):
#             end = min(start + chunk_size, len(text))
#             chunk = text[start:end]
#             chunks.append(chunk)
#         return chunks

#     @staticmethod
#     def raise_if_invalid(input_path: str):
#         if not all(
#             [
#                 input_path is not None,
#                 isinstance(input_path, str),
#                 input_path.strip() != "",
#             ]
#         ):
#             raise ValueError("Invalid input path: Path must be a non-empty string.")

#         _, ext = os.path.splitext(input_path)
#         if ext.lower() not in A2T2TLoader.SUPPORTED_EXTENSIONS:
#             supported = ", ".join(A2T2TLoader.SUPPORTED_EXTENSIONS)
#             raise ValueError(
#                 f"Unsupported file format: '{ext}'. Supported formats are: {supported}."
#             )

#         if not os.path.exists(input_path):
#             raise FileNotFoundError(f"File not found: '{input_path}'.")
