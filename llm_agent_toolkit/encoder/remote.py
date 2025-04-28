import os
import logging
import time
import asyncio
from typing import Optional
import openai
from openai import RateLimitError
from .._encoder import Encoder, EncoderProfile

logger = logging.getLogger(name=__name__)


class OpenAIEncoder(Encoder):
    SUPPORTED_MODELS = (
        EncoderProfile(
            name="text-embedding-3-small", dimension=512, ctx_length=8191
        ),
        EncoderProfile(
            name="text-embedding-3-small", dimension=1536, ctx_length=8191
        ),
        EncoderProfile(
            name="text-embedding-3-large", dimension=256, ctx_length=8191
        ),
        EncoderProfile(
            name="text-embedding-3-large", dimension=512, ctx_length=8191
        ),
        EncoderProfile(
            name="text-embedding-3-large", dimension=1024, ctx_length=8191
        ),
        EncoderProfile(
            name="text-embedding-3-large", dimension=3072, ctx_length=8191
        ),
    )

    def __init__(self, model_name: str, dimension: int):
        for profile in OpenAIEncoder.SUPPORTED_MODELS:
            if model_name == profile["name"] and dimension == profile[
                "dimension"]:
                ctx_length = profile["ctx_length"]
                break
        else:
            raise ValueError(
                "Either model name or dimension are not supported."
            )
        super().__init__(model_name, dimension, ctx_length)

    def encode(self, text: str, **kwargs) -> list[float]:
        try:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.embeddings.create(
                model=self.model_name,
                dimensions=self.dimension,
                input=text,
            )
            return response.data[0].embedding
        except RateLimitError as rle:
            logger.error(
                msg=
                f"{self.model_name}.encode failed. Rate limit error: {str(rle)}",
                exc_info=True,
                stack_info=True,
            )

            delay: Optional[float] = kwargs.get("delay", None)
            attempt: Optional[int] = kwargs.get("attempt", None)

            if delay is None:
                delay = 5.0

            if attempt is None:
                attempt = 1

            if attempt > 5:
                logger.warning("Max attempts reached. Raising error.")
                raise

            logger.info(f"Retrying in {delay} seconds...")
            time.sleep(delay)
            return self.encode(text, delay=delay * 1.5, attempt=attempt + 1)
        except Exception as e:
            logger.error(
                msg=f"{self.model_name}.encode failed. Error: {str(e)}",
                exc_info=True,
                stack_info=True,
            )
            raise

    def encode_v2(self, text: str, **kwargs) -> tuple[list[float], int]:
        try:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.embeddings.create(
                model=self.model_name,
                dimensions=self.dimension,
                input=text,
            )
            return (response.data[0].embedding, response.usage.total_tokens)

        except RateLimitError as rle:
            logger.error(
                msg=
                f"{self.model_name}.encode failed. Rate limit error: {str(rle)}",
                exc_info=True,
                stack_info=True,
            )

            delay: Optional[float] = kwargs.get("delay", None)
            attempt: Optional[int] = kwargs.get("attempt", None)

            if delay is None:
                delay = 5.0

            if attempt is None:
                attempt = 1

            if attempt > 5:
                logger.warning("Max attempts reached. Raising error.")
                raise

            logger.info(f"Retrying in {delay} seconds...")
            time.sleep(delay)
            return self.encode_v2(text, delay=delay * 1.5, attempt=attempt + 1)
        except Exception as e:
            logger.error(
                msg=f"{self.model_name}.encode failed. Error: {str(e)}",
                exc_info=True,
                stack_info=True,
            )
            raise

    async def encode_async(self, text: str, **kwargs) -> list[float]:
        try:
            client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = await client.embeddings.create(
                model=self.model_name,
                dimensions=self.dimension,
                input=text,
            )
            return response.data[0].embedding

        except RateLimitError as rle:
            logger.error(
                msg=
                f"{self.model_name}.encode failed. Rate limit error: {str(rle)}",
                exc_info=True,
                stack_info=True,
            )

            delay: Optional[float] = kwargs.get("delay", None)
            attempt: Optional[int] = kwargs.get("attempt", None)

            if delay is None:
                delay = 5.0

            if attempt is None:
                attempt = 1

            if attempt > 5:
                logger.warning("Max attempts reached. Raising error.")
                raise

            logger.info(f"Retrying in {delay} seconds...")
            await asyncio.sleep(delay)
            return await self.encode_async(
                text, delay=delay * 1.5, attempt=attempt + 1
            )
        except Exception as e:
            logger.error(
                msg=f"{self.model_name}.encode failed. Error: {str(e)}",
                exc_info=True,
                stack_info=True,
            )
            raise

    async def encode_v2_async(self, text: str,
                              **kwargs) -> tuple[list[float], int]:
        try:
            client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = await client.embeddings.create(
                model=self.model_name,
                dimensions=self.dimension,
                input=text,
            )
            return (response.data[0].embedding, response.usage.total_tokens)
        except RateLimitError as rle:
            logger.error(
                msg=
                f"{self.model_name}.encode failed. Rate limit error: {str(rle)}",
                exc_info=True,
                stack_info=True,
            )

            delay: Optional[float] = kwargs.get("delay", None)
            attempt: Optional[int] = kwargs.get("attempt", None)

            if delay is None:
                delay = 5.0

            if attempt is None:
                attempt = 1

            if attempt > 5:
                logger.warning("Max attempts reached. Raising error.")
                raise

            logger.info(f"Retrying in {delay} seconds...")
            await asyncio.sleep(delay)
            return await self.encode_v2_async(
                text, delay=delay * 1.5, attempt=attempt + 1
            )
        except Exception as e:
            logger.error(
                msg=f"{self.model_name}.encode failed. Error: {str(e)}",
                exc_info=True,
                stack_info=True,
            )
            raise
