import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

from chromadb import HttpClient
from chromadb.api import ClientAPI
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore


class VectorStore(ABC):

    @abstractmethod
    def add_documents(self, collection, documents):
        pass

    @abstractmethod
    def get_documents(self, collection, query):
        pass

    @abstractmethod
    def get_documents_by_vector(self, collection, vector):
        pass

    @abstractmethod
    def delete_documents(self, collection, document_ids):
        pass

    @abstractmethod
    def get_document_by_id(self, collection, document_id):
        pass


# 로컬 테스트용 InmemoryVectorStore
class MemoryVectorStore(VectorStore):
    def __init__(self, embedding: Embeddings):
        self.embedding = embedding
        self.stores = {}

    def _get_or_create_store(self, collection_name):
        if collection_name not in self.stores:
            self.stores[collection_name] = InMemoryVectorStore(embedding=self.embedding)
        return self.stores[collection_name]

    def add_documents(self, collection_name, documents: Any) -> Sequence[str]:
        store = self._get_or_create_store(collection_name)
        return store.add_documents(documents)

    def get_documents(self, collection_name, query: str) -> Any:
        store = self._get_or_create_store(collection_name)
        return store.similarity_search(query)

    def get_documents_by_vector(
        self, collection_name, vector: List[float]
    ) -> List[Dict[str, Any]]:
        store = self._get_or_create_store(collection_name)
        return store.similarity_search_by_vector(vector)

    def delete_documents(self, collection_name, document_ids: Sequence[str]):
        store = self._get_or_create_store(collection_name)
        store.delete(document_ids)

    def get_document_by_id(self, collection_name, document_id: str) -> Any:
        store = self._get_or_create_store(collection_name)
        return store.get_by_ids([document_id])


CRHOMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = os.environ.get("CHROMA_PORT", 8001)


class ChromaVectorStore(VectorStore):
    def __init__(self, embedding: Embeddings):

        client: ClientAPI = HttpClient(
            host=CRHOMA_HOST,
            port=CHROMA_PORT,
            ssl=False,
        )

        self.embedding = embedding
        self.client = client  # from your previous code
        self.stores = {}

    def _get_or_create_store(self, collection_name):
        if collection_name not in self.stores:
            self.stores[collection_name] = Chroma(
                client=self.client,
                embedding_function=self.embedding,
                collection_name=collection_name,
            )
        return self.stores[collection_name]

    def add_documents(self, collection_name, documents: Any) -> Sequence[str]:
        store = self._get_or_create_store(collection_name)
        return store.add_documents(documents)

    def get_documents(self, collection_name, query: str) -> Any:
        store = self._get_or_create_store(collection_name)
        return store.similarity_search(query)

    def get_documents_by_vector(
        self, collection_name, vector: List[float]
    ) -> List[Dict[str, Any]]:
        store = self._get_or_create_store(collection_name)
        return store.similarity_search_by_vector(vector)

    def delete_documents(self, collection_name, document_ids: Sequence[str]):
        store = self._get_or_create_store(collection_name)
        store.delete(document_ids)

    def get_document_by_id(self, collection_name, document_id: str) -> Any:
        store = self._get_or_create_store(collection_name)
        return store.get_by_ids([document_id])
