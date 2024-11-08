import json
import chromadb
from typing import Optional
from llm_agent_toolkit._base import BaseTool
from llm_agent_toolkit._util import EmbeddingModel

FILE_EXPLORER_AGENT_PROMPT = """
Description: Searching and analyzing content from the local vector database of documents and files.

Parameters:

`query` (string): The query to be executed on local vector database. Constraint: 1 <= len(query) <= 200
`namespace` (string): The namespace to search in.
`top_n` (int): The number of results to return. Default is 5. Constraint: 1 <= top_n <= 20

Returns:

`response` (dict): {"result": [({ID}, {DOC}, {LAST_UPDATED}, {IS_ANSWER})]}
`next_func` (string): The name of the next function to call.

**Keywords:** document, file, database, vector, embedding, storage, local, internal, archive, repository, collection, folder, directory, record, content, passage, section, page, chapter, text, stored, saved, existing, indexed

**Relevant Query Types:**
- "Find documents about..."
- "Search our files for..."
- "Look up internal information about..."
- "Get stored content related to..."
- "Find passages mentioning..."
"""


class FileExplorerAgent(BaseTool):
    def __init__(
            self, priority: int = 1, next_func: Optional[str] = None,
            vector_database: Optional[chromadb.ClientAPI] = None,
            encoder: EmbeddingModel = EmbeddingModel(), threshold: float = 0.5
    ):
        super().__init__(
            tool_name="FileExplorerAgent",
            description=FILE_EXPLORER_AGENT_PROMPT,
            priority=priority,
            next_func=next_func
        )
        self.__vdb = vector_database
        self.encoder = encoder
        self.__threshold = threshold

    def validate(self, params: str) -> bool:
        params = json.loads(params)
        namespace: str = params.get("namespace", None)
        query: str = params.get("query", None)
        top_n = params.get("top_n", 5)
        if query is None or namespace is None:
            return False
        query = query.strip()
        condition = [len(query) > 0, len(query) <=
                     200, top_n >= 1, top_n <= 20]
        if not all(condition):
            return False
        return True

    async def __call__(self, params: str) -> dict:
        if not self.validate(params):
            return {"error": "Invalid parameters for FileExplorerAgent"}
        params = json.loads(params)
        namespace = params.get("namespace", None)
        query = params.get("query", None)
        top_n = params.get("top_n", 20)
        v_collection = self.__vdb.get_or_create_collection(
            name=namespace, metadata={"hnsw:space": "cosine"}
        )
        output = dict()
        query_embedding = self.encoder.text_to_embedding(query)
        results = v_collection.query(query_embedding, n_results=top_n,
                                     where={
                                         "$and": [
                                             {"deleted": False},
                                             {"isMedia": True}
                                         ]
                                     },
                                     include=['metadatas', 'documents', 'distances'])
        ids = results['ids'][0]
        docs = results['documents'][0]
        dists = results['distances'][0]
        metas = results['metadatas'][0]
        if len(ids) >= 5:
            max_threshold = FileExplorerAgent.calculate_percentile(data=dists, percentile=50)
            min_threshold = FileExplorerAgent.calculate_percentile(data=dists, percentile=20)
        else:
            max_threshold = 2
            min_threshold = 0
        relevant_docs: list[tuple[str, str, str, bool]] = []
        for identifier, doc, dist, meta in zip(ids, docs, dists, metas):
            d = (identifier, doc, meta["lastUpdated"], False)
            if min_threshold <= dist <= max_threshold:
                # Whether to polish the doc with metadata
                relevant_docs.append(d)
            elif dist < self.__threshold:
                relevant_docs.append(d)
        if len(relevant_docs) != 0:
            relevant_docs.sort(key=lambda x: x[2])
        output["result"] = relevant_docs
        return output

    @classmethod
    def calculate_percentile(cls, data: list, percentile: float) -> float:
        # Validation
        if not data:
            raise ValueError("Input list cannot be empty.")

        n = len(data)
        if n == 1:
            return data[0]

        if percentile < 0 or percentile > 100:
            raise ValueError("Percentile must be between 0 and 100.")

        sorted_data = sorted(data)
        index = (n - 1) * percentile / 100

        lower_index = int(index)
        upper_index = lower_index + 1

        lower_value = sorted_data[lower_index]
        upper_value = sorted_data[upper_index]

        interpolated_value = lower_value + (index - lower_index) * (
                upper_value - lower_value
        )

        return interpolated_value
