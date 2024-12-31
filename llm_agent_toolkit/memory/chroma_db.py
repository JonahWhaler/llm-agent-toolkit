import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid
from copy import deepcopy

import chromadb

from .._encoder import Encoder
from .._memory import VectorMemory, AsyncVectorMemory
from .._chunkers import Chunker

logger = logging.getLogger(__name__)


class ChromaMemory(VectorMemory):
    """
    Notes:
    - Never swap encoder
    """

    def __init__(
        self,
        vdb: chromadb.ClientAPI,  # type: ignore
        encoder: Encoder,
        chunker: Chunker,
        **kwargs,
    ):
        super().__init__(vdb, encoder, chunker, **kwargs)
        self.__namespace = kwargs.get("namespace", "default")
        overwrite: bool = kwargs.get("overwrite", False)

        assert isinstance(self.vdb, chromadb.ClientAPI)  # type: ignore

        if overwrite:
            try:
                self.vdb.delete_collection(name=self.__namespace)
                # delete_collection raises InvalidCollectionException
                # if attempt to delete non-exists collection
            except Exception as e:
                logger.info(
                    "%s is not found in the vector database. Ignore %s.",
                    self.__namespace,
                    e,
                )
            finally:
                self.vdb.create_collection(
                    name=self.__namespace, metadata={"hnsw:space": "cosine"}
                )
        else:
            # Create collection if not already present
            self.vdb.get_or_create_collection(
                name=self.__namespace, metadata={"hnsw:space": "cosine"}
            )

    def add(self, document_string: str, **kwargs):
        assert isinstance(self.vdb, chromadb.ClientAPI)  # type: ignore
        collection = self.vdb.get_or_create_collection(name=self.__namespace)
        identifier = kwargs.get("identifier", str(uuid.uuid4()))
        metadata = kwargs.get("metadata", {})
        document_chunks = self.split_text(document_string)

        ids = []
        metas = []
        for i in range(len(document_chunks)):
            meta = deepcopy(metadata)
            meta["page"] = i
            meta["parent"] = identifier
            metas.append(meta)
            ids.append(f"{identifier}-{i}")

        collection.add(
            documents=document_chunks,
            metadatas=metas,
            ids=ids,
            embeddings=[self.encoder.encode(chunk) for chunk in document_chunks],
        )

    def query(self, query_string: str, **kwargs):
        assert isinstance(self.vdb, chromadb.ClientAPI)  # type: ignore
        return_n = kwargs.get("return_n", 5)
        advance_filter = kwargs.get("advance_filter", None)
        output_types = kwargs.get(
            "output_types", ["documents", "metadatas", "distances"]
        )
        params = {
            "n_results": return_n,
            "where": advance_filter,
            "include": output_types,
        }
        collection = self.vdb.get_or_create_collection(name=self.__namespace)
        query_embedding = self.encoder.encode(query_string)
        results = collection.query(
            query_embedding,
            **params,
        )
        result: dict[str, list | None] = {"ids": results["ids"][0]}
        for meta in output_types:
            if meta in results and results[meta]:
                result[meta] = results[meta][0]

        return {"query": query_string, "result": result}

    def clear(self):
        self.vdb.delete_collection(name=self.__namespace)


def _add_(
    collection: chromadb.Collection,
    ids: list[str],
    embeddings: list,
    documents: list[str],
    metadatas: list | None,
):
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )


def _query_(
    collection: chromadb.Collection,
    query_embedding: list,
    n_results: int,
    where: dict | None,
    include: list,
):
    return collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where=where,
        include=include,
    )


def _delete_collection_(client: chromadb.ClientAPI, collection_name: str):  # type: ignore
    return client.delete_collection(name=collection_name)


class AsyncChromaMemory(AsyncVectorMemory):
    def __init__(
        self,
        vdb: chromadb.ClientAPI,  # type: ignore
        encoder: Encoder,
        chunker: Chunker,
        max_workers: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(vdb, encoder, chunker, **kwargs)
        self.__namespace = kwargs.get("namespace", "default")
        self.__overwrite: bool = kwargs.get("overwrite", False)
        self.__executor = ThreadPoolExecutor(max_workers=max_workers)
        self.init()

    def init(self) -> None:
        assert isinstance(self.vdb, chromadb.ClientAPI)  # type: ignore
        if self.__overwrite:
            try:
                self.vdb.delete_collection(name=self.__namespace)
                # delete_collection raises InvalidCollectionException
                # if attempt to delete non-exists collection
            except Exception as e:
                logger.info(
                    "%s is not found in the vector database. Ignore %s.",
                    self.__namespace,
                    e,
                )
            finally:
                self.vdb.create_collection(
                    name=self.__namespace, metadata={"hnsw:space": "cosine"}
                )
        else:
            # Create collection if not already present
            self.vdb.get_or_create_collection(
                name=self.__namespace, metadata={"hnsw:space": "cosine"}
            )

    async def add(self, document_string: str, **kwargs) -> None:
        assert isinstance(self.vdb, chromadb.ClientAPI)  # type: ignore
        collection = self.vdb.get_or_create_collection(name=self.__namespace)
        identifier = kwargs.get("identifier", str(uuid.uuid4()))
        metadata = kwargs.get("metadata", {})
        document_chunks = self.split_text(document_string)

        ids = []
        metas = []
        for i in range(len(document_chunks)):
            meta = deepcopy(metadata)
            meta["page"] = i
            meta["parent"] = identifier
            metas.append(meta)
            ids.append(f"{identifier}-{i}")

        emb_tasks = [self.encoder.encode_async(chunk) for chunk in document_chunks]
        embeddings = await asyncio.gather(*emb_tasks)

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self.__executor,
                _add_,
                collection,
                ids,
                embeddings,
                document_chunks,
                metas,
            )
        except Exception as e:
            logger.error("Failed to add document: %s", e)
            raise

    async def query(self, query_string, **kwargs):
        assert isinstance(self.vdb, chromadb.ClientAPI)  # type: ignore
        return_n = kwargs.get("return_n", 5)
        advance_filter = kwargs.get("advance_filter", None)
        output_types = kwargs.get(
            "output_types", ["documents", "metadatas", "distances"]
        )
        collection = self.vdb.get_or_create_collection(name=self.__namespace)
        query_embedding = await self.encoder.encode_async(query_string)

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            self.__executor,
            _query_,
            collection,
            [query_embedding],
            return_n,
            advance_filter,
            output_types,
        )

        result: dict[str, list | None] = {"ids": results["ids"][0]}
        for meta in output_types:
            if meta in results and results[meta]:
                result[meta] = results[meta][0]

        return {"query": query_string, "result": result}

    async def clear(self):
        assert isinstance(self.vdb, chromadb.AsyncClientAPI)  # type: ignore
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self.__executor, _delete_collection_, self.vdb, self.__namespace
        )  # type: ignore
