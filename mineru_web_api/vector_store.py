import os
from abc import ABC, abstractmethod

from chromadb import HttpClient
from chromadb.api import ClientAPI
from embeddings import huggingface_embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStore(ABC):

    @abstractmethod
    def add_documents(self, collection, documents, *args, **kwargs) -> list[str]:
        pass

    @abstractmethod
    def get_documents(
        self, collection: str, query: str, *args, **kwargs
    ) -> list[Document]:
        pass

    @abstractmethod
    def get_documents_by_vector(
        self, collection: str, vector: list[float], *args, **kwargs
    ) -> list[Document]:
        pass

    @abstractmethod
    def delete_documents(
        self, collection: str, document_ids: list[str], *args, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def get_document_by_id(
        self, collection: str, document_id: str, *args, **kwargs
    ) -> Document | None:
        pass


CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = os.environ.get("CHROMA_PORT", 8001)


class ChromaVectorStore(VectorStore):
    def __init__(self, embedding: Embeddings, client: ClientAPI = None):

        if client is None:
            client: ClientAPI = HttpClient(
                host=CHROMA_HOST,
                port=CHROMA_PORT,
                ssl=False,
            )

        self.embedding = embedding
        self.client = client  # from your previous code
        self.stores: dict[str, Chroma] = {}

    def _get_or_create_store(self, collection_name):
        if collection_name not in self.stores:
            self.stores[collection_name] = Chroma(
                client=self.client,
                embedding_function=self.embedding,
                collection_name=collection_name,
            )
        return self.stores[collection_name]

    def add_documents(self, collection_name, documents, *args, **kwargs) -> list[str]:
        store = self._get_or_create_store(collection_name)
        return store.add_documents(documents, *args, **kwargs)

    def get_documents(
        self, collection_name, query: str, *args, **kwargs
    ) -> list[Document]:
        store = self._get_or_create_store(collection_name)
        return store.similarity_search(query, *args, **kwargs)

    def get_documents_by_vector(
        self, collection_name, vector: list[float], *args, **kwargs
    ) -> list[Document]:
        store = self._get_or_create_store(collection_name)
        return store.similarity_search_by_vector(vector, *args, **kwargs)

    def delete_documents(
        self, collection_name, document_ids: list[str], *args, **kwargs
    ):
        store = self._get_or_create_store(collection_name)
        store.delete(document_ids, *args, **kwargs)

    def get_document_by_id(self, collection_name, document_id: str) -> Document | None:
        store = self._get_or_create_store(collection_name)
        docs = store.get_by_ids([document_id])
        if docs:
            return docs[0]
        return None


vector_store = ChromaVectorStore(
    embedding=huggingface_embeddings,
)
