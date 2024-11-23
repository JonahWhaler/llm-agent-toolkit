import os
import uuid
from copy import deepcopy
import logging
from enum import Enum
from abc import ABC, abstractmethod

import faiss  # type: ignore
import numpy as np

from ._storage import SQLite3_Storage
from .._encoder import Encoder
from .._memory import VectorMemory, SplitTextConfig

logger = logging.getLogger(__name__)


class LoadStrategy(Enum):
    LAZY: str = "Lazy"
    EAGER: str = "Eager"


class FaissDB(ABC):
    @abstractmethod
    def add(
        self,
        documents: list[str],
        metadatas: list[dict],
        ids: list[str],
        embeddings: list[list[float]],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        query_embedding,
        n_results: int = 5,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def remove(self, ids: list[str], **kwargs) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def reconstruct(self, encoder: Encoder) -> None:
        raise NotImplementedError


class FaissIFL2DB(FaissDB):
    """
    Attributes:
    - db_path: str: The path to the sqlite database.
    - dimension: int: The dimension of the embeddings.
    - namespace: str: The namespace of the database.
    - index: faiss.IndexFlatL2: The faiss index.
    - sqlite: SQLite3_Storage: The sqlite storage.

    **ACID Compliance:**
    - SQLite: Yes
    - Faiss: No

    **Thread Safety:**
    - SQLite: Yes
    - Faiss: No
    """

    def __init__(
        self,
        namespace: str,
        dimension: int,
        db_folder: str | None,
        load_strategy: LoadStrategy = LoadStrategy.LAZY,
    ):
        self.__db_path = db_folder
        self.__dimension = dimension
        self.__namespace = namespace
        assert self.__db_path is not None
        assert os.path.exists(self.__db_path)

        if os.path.exists(f"{self.__db_path}/{namespace}.index") and os.path.exists(
            f"{self.__db_path}/{namespace}.db"
        ):
            logger.info("Loading...")
            index = faiss.read_index(f"{self.__db_path}/{namespace}.index")
        else:
            logger.info("Creating...")
            index = faiss.IndexFlatL2(self.__dimension)
            faiss.write_index(index, f"{self.__db_path}/{namespace}.index")

        sqlite = SQLite3_Storage(
            db_path=f"{self.__db_path}/{namespace}.db",
            table_name=namespace,
            overwrite=False,
        )

        self.__load_strategy = load_strategy

        if load_strategy == LoadStrategy.EAGER:
            self.__index = index
            self.__sqlite = sqlite
        else:
            self.__index = None
            self.__sqlite = None  # type: ignore

    @property
    def index_path(self) -> str:
        return f"{self.__db_path}/{self.__namespace}.index"

    @property
    def db_path(self) -> str:
        return f"{self.__db_path}/{self.__namespace}.db"

    @property
    def load_strategy(self) -> LoadStrategy:
        return self.__load_strategy

    def add(
        self,
        documents: list[str],
        metadatas: list[dict],
        ids: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """
        Add documents to the database.

        Args:
            documents (list[str]): A list of documents.
            metadatas (list[dict]): A list of metadatas.
            ids (list[str]): A list of ids.
            embeddings (list[list[float]]): A list of embeddings.

        Raises:
            Exception: If failed to load the sqlite database.
            Exception: If failed to laod the index.

        Returns:
            None

        Notes:
            - TODO: Support `where` parsing and query
            - When one of the add operation fails, the database will do it's very best to recover.
        """
        try:
            index = (
                self.__index
                if self.__index is not None
                else faiss.read_index(self.index_path)
            )
            sqlite = (
                self.__sqlite
                if self.__sqlite is not None
                else SQLite3_Storage(
                    db_path=self.db_path, table_name=self.__namespace, overwrite=False
                )
            )
        except Exception as loading_error:
            logger.error(msg=f"Loading Error: {loading_error}")
            raise
        track = []
        for i, d, m, e in zip(ids, documents, metadatas, embeddings):
            inserted = False
            counter = len(sqlite.keys())
            try:
                # Add to SQLite
                sqlite.set(
                    key=str(counter),
                    value={
                        "id": i,
                        "document": d,
                        "metadata": m,
                    },
                )
                # Add to Faiss
                np_embedding = np.expand_dims(np.array(e, dtype=np.float32), axis=0)
                index.add(np_embedding)
                inserted = True
                track.append(counter)
            except AssertionError as a_e:
                # Inconsistent counter
                logger.error(msg=str(a_e))
                raise
            except Exception as e:
                # Backtracking...
                logger.error(msg=str(e))
                for counter in track:
                    v = sqlite.get(key=str(counter))
                    if v is not None:
                        # Inserted to SQLite but not to Faiss
                        sqlite.drop(key=str(counter))
                    # if v is None:
                    # Insertion to SQLite failed
                raise
            finally:
                if inserted:
                    faiss.write_index(index, self.index_path)

    def query(
        self,
        query_embedding,
        n_results: int = 5,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> list[dict]:
        """
        Select the most relevant documents from the database.

        Args:
        query_embedding (list[float]): Embedding of the query.
        n_results (int, optional): Number of results to return. Defaults to 5.
        where (dict, optional): Filter to apply to the results. Defaults to None.
        include (list[str], optional): List of fields to include in the results. Defaults to None.

        Returns:
        list[dict]: List of documents, distances and other metadata (optional).

        Raises:
        AssertionError: If the length of query_embedding is not equal to self.__dimension.
        Exception: If the database cannot be loaded.
        Exception: If the query operation failed.

        Notes:
        - TODO: Support `where`
        """
        assert len(query_embedding) == self.__dimension
        try:
            index = (
                self.__index
                if self.__index is not None
                else faiss.read_index(self.index_path)
            )
            sqlite = (
                self.__sqlite
                if self.__sqlite is not None
                else SQLite3_Storage(
                    db_path=self.db_path, table_name=self.__namespace, overwrite=False
                )
            )
        except Exception as loading_error:
            logger.error(msg=f"Loading Error: {loading_error}")
            raise

        try:
            np_query_embedding = np.array(query_embedding, dtype=np.float32)
            np_query_embedding = np.expand_dims(np_query_embedding, axis=0)
            assert np_query_embedding.shape == (1, self.__dimension)
            distances, indices = index.search(np_query_embedding, k=n_results)
            distances = distances[0]
            indices = indices[0]
            # Post Processing
            output = []
            for index, distance in zip(indices, distances):
                value_dict = sqlite.get(key=str(index))
                assert value_dict is not None, "Inconsistency Detected"
                # Skip the deleted
                if "is_deleted" in value_dict:
                    if value_dict["is_deleted"] is True:
                        continue
                _output_object = {
                    "id": value_dict["id"],
                    "distance": distance,
                    "document": value_dict["document"],
                }
                # Include selected fields
                if include is not None:
                    for key in include:
                        _output_object[key] = value_dict[key]
                output.append(_output_object)

            return output
        except Exception as e:
            logger.error(msg=str(e))
            raise

    def remove(self, ids: list[str], **kwargs) -> list[str]:
        """
        Soft delete documents from the backend. Marks the selected records as "is_deleted".

        Args:
        ids (list[str]): List of document ids to remove.

        Returns:
        list[str]: List of document ids removed.

        Raises:
        - Exception: If failed to load the sqlite database.

        Notes:
        - Does not raise exception if the document is not found
        """
        try:
            sqlite = (
                self.__sqlite
                if self.__sqlite is not None
                else SQLite3_Storage(
                    db_path=self.db_path, table_name=self.__namespace, overwrite=False
                )
            )
        except Exception as loading_error:
            logger.error(msg=f"Loading Error: {loading_error}")
            raise
        removed_ids = []
        for key in ids:
            value_dict = sqlite.get(key=key)
            if value_dict is None:
                continue
            value_dict["is_deleted"] = True
            sqlite.set(key=key, value=value_dict)
            removed_ids.append(key)
        return removed_ids

    def reconstruct(self, encoder: Encoder) -> None:
        try:
            sqlite = (
                self.__sqlite
                if self.__sqlite is not None
                else SQLite3_Storage(
                    db_path=self.db_path, table_name=self.__namespace, overwrite=False
                )
            )
        except Exception as loading_error:
            logger.error(msg=f"Loading Error: {loading_error}")
            raise
        # Build a tmp sqlite db
        tmp_sqlite_path = f"{self.__db_path}/tmp.db"
        tmp_db = SQLite3_Storage(
            db_path=tmp_sqlite_path, table_name=self.__namespace, overwrite=True
        )
        # Build a tmp faiss index
        tmp_index = faiss.IndexFlatL2(self.__dimension)
        counter = 0  # Start from 0
        # Iterate over the original db
        for key in sqlite.keys():
            value_dict = sqlite.get(key=key)
            # Skip the deleted
            if "is_deleted" in value_dict and value_dict["is_deleted"] is True:
                continue
            # Add to tmp db
            tmp_db.set(key=str(counter), value=value_dict)
            # Add to tmp index
            _embedding = np.array(encoder.encode(value_dict["document"]))
            _embedding = np.expand_dims(_embedding, axis=0)
            # pylint: disable = no-value-for-parameter
            tmp_index.add(_embedding.astype(np.float32))
            counter += 1
        os.replace(tmp_sqlite_path, self.db_path)
        if self.load_strategy == LoadStrategy.EAGER:
            self.__sqlite = SQLite3_Storage(
                db_path=self.db_path, table_name=self.__namespace, overwrite=False
            )
            self.__index = tmp_index
        faiss.write_index(tmp_index, self.index_path)


class FaissHNSWDB(FaissDB):
    """
    Attributes:
    - db_path: str: The path to the sqlite database.
    - dimension: int: The dimension of the embeddings.
    - namespace: str: The namespace of the database.
    - index: faiss.IndexHNSW: The faiss index.
    - sqlite: SQLite3_Storage: The sqlite storage.

    **ACID Compliance:**
    - SQLite: Yes
    - Faiss: No

    **Thread Safety:**
    - SQLite: Yes
    - Faiss: No
    """

    M: int = 32
    EF_CONSTRUCTION: int = 40
    EF_SEARCH: int = 16

    def __init__(
        self,
        namespace: str,
        dimension: int,
        db_folder: str | None,
        load_strategy: LoadStrategy = LoadStrategy.LAZY,
    ):
        self.__db_path = db_folder
        self.__dimension = dimension
        self.__namespace = namespace
        assert self.__db_path is not None
        assert os.path.exists(self.__db_path)

        assert self.__db_path is not None
        assert os.path.exists(self.__db_path)

        if os.path.exists(f"{self.__db_path}/{namespace}.index") and os.path.exists(
            f"{self.__db_path}/{namespace}.db"
        ):
            logger.info("Loading...")
            index = faiss.read_index(f"{self.__db_path}/{namespace}.index")
        else:
            logger.info("Creating...")
            index = faiss.IndexHNSWFlat(self.__dimension, FaissHNSWDB.M)
            index.hnsw.efConstruction = FaissHNSWDB.EF_CONSTRUCTION
            index.hnsw.efSearch = FaissHNSWDB.EF_SEARCH
            faiss.write_index(index, f"{self.__db_path}/{namespace}.index")

        sqlite = SQLite3_Storage(
            db_path=f"{self.__db_path}/{namespace}.db",
            table_name=namespace,
            overwrite=False,
        )

        self.__load_strategy = load_strategy

        if load_strategy == LoadStrategy.EAGER:
            self.__index = index
            self.__sqlite = sqlite
        else:
            self.__index = None
            self.__sqlite = None  # type: ignore

    @property
    def index_path(self) -> str:
        return f"{self.__db_path}/{self.__namespace}.index"

    @property
    def db_path(self) -> str:
        return f"{self.__db_path}/{self.__namespace}.db"

    @property
    def load_strategy(self) -> LoadStrategy:
        return self.__load_strategy

    def add(
        self,
        documents: list[str],
        metadatas: list[dict],
        ids: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """
        Add documents to the database.

        Args:
            documents (list[str]): A list of documents.
            metadatas (list[dict]): A list of metadatas.
            ids (list[str]): A list of ids.
            embeddings (list[list[float]]): A list of embeddings.

        Raises:
            Exception: If failed to load the sqlite database.
            Exception: If failed to laod the index.

        Returns:
            None

        Notes:
            - TODO: Support `where` parsing and query
            - When one of the add operation fails, the database will do it's very best to recover.
        """
        try:
            index = (
                self.__index
                if self.__index is not None
                else faiss.read_index(self.index_path)
            )
            sqlite = (
                self.__sqlite
                if self.__sqlite is not None
                else SQLite3_Storage(
                    db_path=self.db_path, table_name=self.__namespace, overwrite=False
                )
            )
        except Exception as loading_error:
            logger.error(msg=f"Loading Error: {loading_error}")
            raise
        track = []
        for i, d, m, e in zip(ids, documents, metadatas, embeddings):
            inserted = False
            counter = len(sqlite.keys())
            try:
                # Add to SQLite
                sqlite.set(
                    key=str(counter),
                    value={
                        "id": i,
                        "document": d,
                        "metadata": m,
                    },
                )
                # Add to Faiss
                index.add(np.expand_dims(np.array(e, dtype=np.float32), axis=0))
                inserted = True
                track.append(counter)
            except AssertionError as a_e:
                # Inconsistent counter
                logger.error(msg=str(a_e))
                raise
            except Exception as e:
                # Backtracking...
                logger.error(msg=str(e))
                for counter in track:
                    v = sqlite.get(key=str(counter))
                    if v is not None:
                        # Inserted to SQLite but not to Faiss
                        sqlite.drop(key=str(counter))
                    # if v is None:
                    # Insertion to SQLite failed
                raise
            finally:
                if inserted:
                    faiss.write_index(index, self.index_path)

    def query(
        self,
        query_embedding,
        n_results: int = 5,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> list[dict]:
        """
        Select the most relevant documents from the database.

        Args:
        query_embedding (list[float]): Embedding of the query.
        n_results (int, optional): Number of results to return. Defaults to 5.
        where (dict, optional): Filter to apply to the results. Defaults to None.
        include (list[str], optional): List of fields to include in the results. Defaults to None.

        Returns:
        list[dict]: List of documents, distances and other metadata (optional).

        Raises:
        AssertionError: If the length of query_embedding is not equal to self.__dimension.
        Exception: If the database cannot be loaded.
        Exception: If the query operation failed.

        Notes:
        - TODO: Support `where`
        """
        assert len(query_embedding) == self.__dimension
        try:
            index = (
                self.__index
                if self.__index is not None
                else faiss.read_index(self.index_path)
            )
            sqlite = (
                self.__sqlite
                if self.__sqlite is not None
                else SQLite3_Storage(
                    db_path=self.db_path, table_name=self.__namespace, overwrite=False
                )
            )
        except Exception as loading_error:
            logger.error(msg=f"Loading Error: {loading_error}")
            raise

        try:
            np_query_embedding = np.expand_dims(
                np.array(query_embedding, dtype=np.float32), axis=0
            )
            assert np_query_embedding.shape == (1, self.__dimension)
            distances, indices = index.search(np_query_embedding, k=n_results)
            distances = distances[0]
            indices = indices[0]
            # Post Processing
            output = []
            for index, distance in zip(indices, distances):
                value_dict = sqlite.get(key=str(index))
                assert value_dict is not None, "Inconsistency Detected"
                # Skip the deleted
                if "is_deleted" in value_dict:
                    if value_dict["is_deleted"] is True:
                        continue
                _output_object = {
                    "id": value_dict["id"],
                    "distance": distance,
                    "document": value_dict["document"],
                }
                # Include selected fields
                if include is not None:
                    for key in include:
                        _output_object[key] = value_dict[key]
                output.append(_output_object)

            return output
        except Exception as e:
            logger.error(msg=str(e))
            raise

    def remove(self, ids: list[str], **kwargs) -> list[str]:
        """
        Soft delete documents from the backend. Marks the selected records as "is_deleted".

        Args:
        ids (list[str]): List of document ids to remove.

        Returns:
        list[str]: List of document ids removed.

        Raises:
        - Exception: If failed to load the sqlite database.

        Notes:
        - Does not raise exception if the document is not found
        """
        try:
            sqlite = (
                self.__sqlite
                if self.__sqlite is not None
                else SQLite3_Storage(
                    db_path=self.db_path, table_name=self.__namespace, overwrite=False
                )
            )
        except Exception as loading_error:
            logger.error(msg=f"Loading Error: {loading_error}")
            raise
        removed_ids = []
        for key in ids:
            value_dict = sqlite.get(key=key)
            if value_dict is None:
                continue
            value_dict["is_deleted"] = True
            sqlite.set(key=key, value=value_dict)
            removed_ids.append(key)
        return removed_ids

    def reconstruct(self, encoder: Encoder) -> None:
        try:
            sqlite = (
                self.__sqlite
                if self.__sqlite is not None
                else SQLite3_Storage(
                    db_path=self.db_path, table_name=self.__namespace, overwrite=False
                )
            )
        except Exception as loading_error:
            logger.error(msg=f"Loading Error: {loading_error}")
            raise
        # Build a tmp sqlite db
        tmp_sqlite_path = f"{self.__db_path}/tmp.db"
        tmp_db = SQLite3_Storage(
            db_path=tmp_sqlite_path, table_name=self.__namespace, overwrite=True
        )
        # Build a tmp faiss index
        tmp_index = faiss.IndexHNSWFlat(self.__dimension, FaissHNSWDB.M)
        tmp_index.hnsw.efConstruction = FaissHNSWDB.EF_CONSTRUCTION
        tmp_index.hnsw.efSearch = FaissHNSWDB.EF_SEARCH
        counter = 0  # Start from 0
        # Iterate over the original db
        for key in sqlite.keys():
            value_dict = sqlite.get(key=key)
            # Skip the deleted
            if "is_deleted" in value_dict and value_dict["is_deleted"] is True:
                continue
            # Add to tmp db
            tmp_db.set(key=str(counter), value=value_dict)
            # Add to tmp index
            _embedding = np.array(
                encoder.encode(value_dict["document"]), dtype=np.float32
            )
            _embedding = np.expand_dims(_embedding, axis=0)
            # pylint: disable = no-value-for-parameter
            tmp_index.add(_embedding.astype(np.float32))
            counter += 1
        os.replace(tmp_sqlite_path, self.db_path)
        if self.load_strategy == LoadStrategy.EAGER:
            self.__sqlite = SQLite3_Storage(
                db_path=self.db_path, table_name=self.__namespace, overwrite=False
            )
            self.__index = tmp_index
        faiss.write_index(tmp_index, self.index_path)


class FaissMemory(VectorMemory):
    def __init__(
        self,
        vdb: FaissDB,
        encoder: Encoder,
        split_text_config: SplitTextConfig,
        **kwargs,
    ):
        super().__init__(vdb, encoder, split_text_config, **kwargs)

    def add(self, document_string: str, **kwargs):
        identifier = kwargs.get("identifier", str(uuid.uuid4()))
        metadata = kwargs.get("metadata", None)
        document_chunks = self.split_text(document_string)

        if metadata is not None:
            metas = []
            ids = []
            for i in range(len(document_chunks)):
                meta = deepcopy(metadata)
                meta["page"] = i
                meta["parent"] = identifier
                metas.append(meta)
                ids.append(f"{identifier}-{i}")
            self.vdb.add(
                documents=document_chunks,
                metadatas=metas,
                ids=ids,
                embeddings=self.encoder.encode(document_chunks),
            )
        else:
            self.vdb.add(
                documents=document_chunks,
                ids=[f"{identifier}-{i}" for i in range(len(document_chunks))],
                embeddings=self.encoder.encode(document_chunks),
            )
