import asyncio
import logging

import chromadb

from llm_agent_toolkit.memory import (
    ChromaMemory,
    AsyncChromaMemory,
    FaissIFL2DB,
    FaissHNSWDB,
    FaissMemory,
)
from llm_agent_toolkit.encoder.local import OllamaEncoder
from llm_agent_toolkit.chunkers import FixedGroupChunker


logging.basicConfig(
    filename="./snippet/output/example-memory.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


MODEL_NAME = "mxbai-embed-large:latest"
TEXT_LIST = ["Orange", "Apple", "Banana", "Python", "Java", "JavaScript"]

CONNECTION_STRING = "http://localhost:11434"
TMP_DIRECTORY = "./snippet/output"


def run_ChromaMemory():
    encoder = OllamaEncoder(connection_string=CONNECTION_STRING, model_name=MODEL_NAME)
    vdb = chromadb.Client(
        settings=chromadb.Settings(
            is_persistent=True,
            persist_directory=f"{TMP_DIRECTORY}/chat",
        )
    )
    chunker = FixedGroupChunker(
        config={
            "K": 1,
        },
    )
    memory = ChromaMemory(
        vdb=vdb,
        encoder=encoder,
        chunker=chunker,
        namespace="default",
        overwrite=True,
    )
    for i, text in enumerate(TEXT_LIST, start=1):
        memory.add(document_string=text, identifier=f"test|{i}")

    results = memory.query(query_string="Programming", return_n=3)
    logger.info("Results: %s", results)


async def run_AsyncChromaMemory():
    encoder = OllamaEncoder(connection_string=CONNECTION_STRING, model_name=MODEL_NAME)
    vdb = chromadb.Client(
        settings=chromadb.Settings(
            is_persistent=True,
            persist_directory=f"{TMP_DIRECTORY}/chat",
        )
    )
    chunker = FixedGroupChunker(
        config={
            "K": 1,
        }
    )
    memory = AsyncChromaMemory(
        vdb=vdb,
        encoder=encoder,
        chunker=chunker,
        namespace="default",
        overwrite=True,
    )
    for i, text in enumerate(TEXT_LIST, start=1):
        await memory.add(document_string=text, identifier=f"test|{i}")

    results = await memory.query(query_string="Fruit", return_n=3)
    logger.info("Results: %s", results)


def run_FaissMemory_IFL2():
    encoder = OllamaEncoder(connection_string=CONNECTION_STRING, model_name=MODEL_NAME)
    db = FaissIFL2DB(
        namespace="test", dimension=encoder.dimension, db_folder=TMP_DIRECTORY
    )
    chunker = FixedGroupChunker(
        config={
            "K": 1,
        }
    )
    vm = FaissMemory(vdb=db, encoder=encoder, chunker=chunker)
    vm.clear()

    for i, text in enumerate(TEXT_LIST, start=1):
        vm.add(document_string=text, identifier=f"test|{i}", metadata={"stage": "test"})

    results = vm.query(query_string="Fruit", return_n=3)
    logger.info("Results: %s", results)


def run_FaissMemory_HNSW():
    encoder = OllamaEncoder(connection_string=CONNECTION_STRING, model_name=MODEL_NAME)
    db = FaissHNSWDB(
        namespace="test", dimension=encoder.dimension, db_folder=TMP_DIRECTORY
    )
    chunker = FixedGroupChunker(
        config={
            "K": 1,
        }
    )
    vm = FaissMemory(vdb=db, encoder=encoder, chunker=chunker, overwrite=True)
    # vm.clear()

    for i, text in enumerate(TEXT_LIST, start=1):
        vm.add(document_string=text, identifier=f"test|{i}", metadata={"stage": "test"})

    results = vm.query(query_string="Fruit", return_n=3)
    logger.info("Results: %s", results)


if __name__ == "__main__":
    # run_ChromaMemory()
    # run_FaissMemory_IFL2()
    run_FaissMemory_HNSW()
    # asyncio.run(run_AsyncChromaMemory())
