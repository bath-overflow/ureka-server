from langchain_core.retrievers import BaseRetriever

from server.repositories.vector_store import vector_store


class DocumentRetriever(BaseRetriever):
    def __init__(self, collection: str):
        self.collection = collection

    def get_relevant_documents(self, query: str) -> list:
        return vector_store.get_documents(self.collection, query)

    def add_documents(self, documents: list) -> None:
        vector_store.add_documents(self.collection, documents)

    def delete_documents(self, document_ids: list) -> None:
        vector_store.delete_documents(self.collection, document_ids)

    def get_document_by_id(self, document_id: str) -> list:
        return vector_store.get_document_by_id(self.collection, document_id)

    def get_documents_by_vector(self, vector: list) -> list:
        return vector_store.get_documents_by_vector(self.collection, vector)
