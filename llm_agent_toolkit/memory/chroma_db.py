import uuid
from copy import deepcopy

import chromadb

from .._encoder import Encoder
from .._memory import VectorMemory, SplitTextConfig


class ChromaMemory(VectorMemory):
    """
    Notes:
    - Never swap encoder
    """

    def __init__(
        self,
        vdb: chromadb.ClientAPI,
        encoder: Encoder,
        split_text_config: SplitTextConfig,
        **kwargs,
    ):
        super().__init__(vdb, encoder, split_text_config, **kwargs)
        self.__namespace = kwargs.get("namespace", "default")
        overwrite: bool = kwargs.get("overwrite", False)
        if self.vdb.get_collection(name=self.__namespace) and overwrite == True:
            self.vdb.delete_collection(name=self.__namespace)
            self.vdb.create_collection(
                name=self.__namespace, metadata={"hnsw:space": "cosine"}
            )
        else:
            self.vdb.create_collection(
                name=self.__namespace, metadata={"hnsw:space": "cosine"}
            )

    def add(self, document_string: str, **kwargs):
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
        return {
            "query": query_string,
            "result": {
                "ids": results["ids"][0],
                "document": results["documents"][0],
                "distance": results["distances"][0],
                "metadata": results["metadatas"][0],
            },
        }

    def clear(self):
        self.vdb.delete_collection(name=self.__namespace)
