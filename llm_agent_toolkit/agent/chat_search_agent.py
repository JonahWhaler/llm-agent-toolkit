import json
import chromadb
from typing import Optional
from llm_agent_toolkit._base import BaseTool
from llm_agent_toolkit._util import EmbeddingModel

CHAT_REPLAYER_AGENT_PROMPT = """
Description: Searching and analyzing recent conversation history and chat interactions.

Parameters:

`query` (string): User's command/query. Constraint: 1 <= len(query) <= 200
`top_n` (int): The number of results to return. Default is 5. Constraint: 1 <= top_n <= 10

Returns:

`response` (dict): {"result": [({ID}, {DOC}, {LAST_UPDATED}, {IS_ANSWER})]}
`next_func` (string): The name of the next function to call.

**Primary Functions:**
- Search through recent chat history
- Identify relevant previous discussions
- Track conversation context and references
- Find earlier mentions and decisions

**Keywords:** chat, conversation, history, dialogue, discussion, message, interaction, previous, recent, earlier, past, context, thread, exchange, communication, response, reply, mention, reference, memory, recall, log, transcript

**Relevant Query Types:**
- "What did we discuss about..."
- "Find our previous conversation about..."
- "When did we last talk about..."
- "What was decided regarding..."
- "Show me earlier mentions of..."

**Notes:**
- This is particularly useful when the query is highly rely on the context of the conversation.
"""


class ChatReplayerAgent(BaseTool):
    def __init__(
        self, priority=1, next_func: str | None = None,
        vector_database: Optional[chromadb.ClientAPI] = None,
        encoder: EmbeddingModel = EmbeddingModel(), threshold: float = 0.5
    ):
        super().__init__(
            tool_name="ChatReplayerAgent",
            description=CHAT_REPLAYER_AGENT_PROMPT,
            priority=priority,
            next_func=next_func,
        )
        self.__vdb = vector_database
        self.encoder = encoder
        self.__threshold = threshold

    async def __call__(self, params: str) -> dict:
        params = json.loads(params)
        namespace = params.get("namespace", None)
        query = params.get("query", None)
        top_n = params.get("top_n", 20)

        collection = self.__vdb.get_or_create_collection(
            name=namespace, metadata={"hnsw:space": "cosine"}
        )
        query_embedding = self.encoder.text_to_embedding(query)

        recent_ids = self.get_recent_ids(collection=collection, top_n=top_n)
        relevant_ids = self.get_relevant_ids(
            query_embedding=query_embedding, collection=collection, top_n=top_n
        )
        target_ids = list(relevant_ids | recent_ids)
        if len(target_ids) == 0:
            return {"result": []}

        output = []
        result = collection.get(ids=target_ids, include=[
                                'documents', 'metadatas'])
        ids = result['ids']
        metas = result['metadatas']
        docs = result['documents']
        for identifier, meta, doc in zip(ids, metas, docs):
            chat = (identifier, doc, meta["lastUpdated"], meta["isAnswer"])
            output.append(chat)

        output = sorted(output, key=lambda x: x[2])
        return {"result": output}

    def validate(self, params: str) -> bool:
        params = json.loads(params)
        query = params.get("query", None)
        namespace = params.get("namespace", None)
        top_n = params.get("top_n", 5)
        if query is None or namespace is None:
            return False
        query = query.strip()
        conditions = [len(query) > 0, len(query) <=
                      200, top_n >= 1, top_n <= 10]
        if not all(conditions):
            return False
        return True

    @staticmethod
    def get_recent_ids(collection, top_n: int = 5):
        # Identify the ids of the most recent chats
        result = collection.get(
            where={
                "$and": [
                    {"deleted": False},
                    {"isMedia": False}
                ]
            },
            include=['metadatas', ]
        )
        ids = result['ids']
        metas = result['metadatas']
        chat_history = []
        for identifier, meta in zip(ids, metas):
            chat = (identifier, meta["lastUpdated"])
            chat_history.append(chat)
        if len(chat_history) == 0:
            return ()
        chat_history.sort(key=lambda x: x[1])
        recent_chat_history = chat_history[-top_n:]
        return set(map(lambda x: x[0], recent_chat_history))

    @staticmethod
    def get_relevant_ids(query_embedding, collection, top_n: int = 5, threshold: float = 0.5):
        result = collection.query(
            query_embedding, n_results=top_n,
            where={
                "$and": [
                    {"deleted": False},
                    {"isMedia": False}
                ]
            },
            include=['metadatas', 'distances']
        )
        ids = result['ids'][0]
        dists = result['distances'][0]
        metas = result['metadatas'][0]
        if len(ids) >= 5:
            max_threshold = ChatReplayerAgent.calculate_percentile(dists, 50)
            min_threshold = ChatReplayerAgent.calculate_percentile(dists, 20)
        else:
            max_threshold = 2
            min_threshold = 0
        chat_history = []
        for identifier, dist, meta in zip(ids, dists, metas):
            chat = (identifier, meta["lastUpdated"])
            if min_threshold <= dist <= max_threshold:
                chat_history.append(chat)
            elif dist < threshold:
                chat_history.append(chat)
        if len(chat_history) == 0:
            return chat_history
        chat_history.sort(key=lambda x: x[1])
        return set(map(lambda x: x[0], chat_history))

    @staticmethod
    def calculate_percentile(data: list, percentile: float):
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
