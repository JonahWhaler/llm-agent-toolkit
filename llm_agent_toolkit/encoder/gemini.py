import os
import logging
import time
import asyncio
from random import random
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types
from .._encoder import Encoder, EncoderProfile

logger = logging.getLogger(name=__name__)


class GeminiEncoder(Encoder):
    """
    Notes:
    - https://ai.google.dev/gemini-api/docs/embeddings#python
    - https://ai.google.dev/gemini-api/docs/models#gemini-embedding
    """

    SUPPORTED_MODELS = (
        EncoderProfile(name="models/embedding-001", dimension=768, ctx_length=2048),
        EncoderProfile(
            name="models/text-embedding-004", dimension=768, ctx_length=2048
        ),
        EncoderProfile(
            name="gemini-embedding-exp-03-07", dimension=768, ctx_length=8192
        ),
    )

    SUPPORTED_TASK_TYPES = (
        "SEMANTIC_SIMILARITY",
        "CLASSIFICATION",
        "CLUSTERING",
        "CODE_RETRIEVAL_QUERY",
        "RETRIEVAL_DOCUMENT",
        "RETRIEVAL_QUERY",
        "QUESTION_ANSWERING",
        "FACT_VERIFICATION",
    )

    MAX_ATTEMPT: int = 5
    DELAY_FACTOR: float = 1.5
    MAX_DELAY: float = 60.0

    def __init__(self, model_name: str, dimension: int, task_type: str):
        for profile in GeminiEncoder.SUPPORTED_MODELS:
            if model_name == profile["name"] and dimension == profile["dimension"]:
                ctx_length = profile["ctx_length"]
                break
        else:
            raise ValueError("Either model name or dimension are not supported.")
        super().__init__(model_name, dimension, ctx_length)

        if task_type not in GeminiEncoder.SUPPORTED_TASK_TYPES:
            raise ValueError(
                "task_type must be one of SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING, CODE_RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, QUESTION_ANSWERING, FACT_VERIFICATION"
            )
        self.task_type = task_type

    @property
    def config(self):
        return types.EmbedContentConfig(task_type=self.task_type)

    @staticmethod
    def estimate_token_count(text: str) -> int:
        """
        Estimate the number of tokens in the text.
        This is not exact, but it's close enough for our purposes.
        """
        if text.isascii():
            return len(text) // 4
        return len(text) // 2

    def __update_delay(self, delay: float) -> float:
        new_delay = delay * self.DELAY_FACTOR
        # Add some randomness to allow bulk requests to retry at a slightly different timing
        new_delay += random() * 5.0
        return min(new_delay, self.MAX_DELAY)

    def encode(self, text: str, **kwargs) -> list[float]:
        attempt: int = 0
        delay: float = 5.0
        while attempt < self.MAX_ATTEMPT:
            try:
                client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
                response = client.models.embed_content(
                    model=self.model_name,
                    contents=text,
                    config=self.config,
                )
                if (
                    response.embeddings
                    and response.embeddings[0]
                    and response.embeddings[0].values
                ):
                    if attempt > 0:
                        logger.debug("[%d] Request resolved!", attempt)
                    return response.embeddings[0].values
                raise ValueError(
                    f"Failed to extract embeddings from response. Response: {response}"
                )
            except RuntimeError:
                raise
            except Exception as e:
                error_object = e.__dict__
                error_code = error_object["code"]
                if error_code not in [429, 500, 503]:
                    raise

                error_message = error_object["message"]
                logger.warning(
                    "%s\n[%d] Retrying in %.2f seconds", error_message, attempt, delay
                )
                time.sleep(delay)
                attempt += 1
                delay = self.__update_delay(delay)
                continue

        raise RuntimeError("Max re-attempt reached")

    @staticmethod
    async def acall(
        model_name: str, config: types.EmbedContentConfig, text: str
    ) -> types.EmbedContentResponse:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        with ThreadPoolExecutor() as executor:
            future = executor.submit(
                client.models.embed_content,
                model=model_name,
                contents=text,
                config=config,
            )
            # Makes the future awaitable
            response = await asyncio.wrap_future(future)
            return response

    async def encode_async(self, text: str, **kwargs) -> list[float]:
        attempt: int = 0
        delay: float = 5.0
        while attempt < self.MAX_ATTEMPT:
            try:
                response = await self.acall(
                    model_name=self.model_name, config=self.config, text=text
                )
                if (
                    response.embeddings
                    and response.embeddings[0]
                    and response.embeddings[0].values
                ):
                    if attempt > 0:
                        logger.info("[%d] Request resolved!", attempt)
                    return response.embeddings[0].values
                raise RuntimeError(
                    f"Failed to extract embeddings from response. Response: {response}"
                )
            except RuntimeError:
                raise
            except Exception as e:
                error_object = e.__dict__
                error_code = error_object["code"]
                if error_code not in [429, 500, 503]:
                    raise

                error_message = error_object["message"]
                logger.warning(
                    "%s\n[%d] Retrying in %.2f seconds", error_message, attempt, delay
                )
                await asyncio.sleep(delay)
                attempt += 1
                delay = self.__update_delay(delay)
                continue

        raise RuntimeError("Max re-attempt reached")

    def encode_v2(self, text: str, **kwargs) -> tuple[list[float], int]:
        try:
            return (self.encode(text), self.estimate_token_count(text))
        except Exception as e:
            logger.error(
                msg=f"{self.model_name}.encode failed. Error: {str(e)}",
                exc_info=True,
                stack_info=True,
            )
            raise

    async def encode_v2_async(self, text: str, **kwargs) -> tuple[list[float], int]:
        try:
            return (await self.encode_async(text), self.estimate_token_count(text))
        except Exception as e:
            logger.error(
                msg=f"{self.model_name}.encode failed. Error: {str(e)}",
                exc_info=True,
                stack_info=True,
            )
            raise
