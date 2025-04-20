import pytest
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from server.repositories.vector_store import (
    MemoryVectorStore,
    VectorStore,
)


@pytest.fixture
def vector_store() -> VectorStore:
    """
    Fixture to create a new instance of MemoryVectorStore for each test.
    """
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    return MemoryVectorStore(embedding)


def test_add_documents(vector_store):
    collection_name = "test_collection"
    documents = [
        Document(page_content="This is a test document.", metadata={"id": "1"}),
        Document(page_content="This is another test document.", metadata={"id": "2"}),
        Document(page_content="This is a third test document.", metadata={"id": "3"}),
    ]
    ids = vector_store.add_documents(collection_name, documents)

    # Verify that documents were added using get_document_by_id
    for _ in documents:
        result = vector_store.get_document_by_id(collection_name, ids[0])
        assert len(result) == 1


def test_get_documents(vector_store):
    collection_name = "test_collection"
    documents = [
        Document(page_content="This is a test document.", metadata={"id": "1"}),
        Document(page_content="This is another test document.", metadata={"id": "2"}),
        Document(page_content="This is a third test document.", metadata={"id": "3"}),
    ]
    vector_store.add_documents(collection_name, documents)

    query = "test document"
    retrieved_docs = vector_store.get_documents(collection_name, query)
    assert len(retrieved_docs) == 3


def test_delete_documents(vector_store):
    collection_name = "test_collection"
    documents = [
        Document(page_content="This is a test document.", metadata={"id": "1"}),
        Document(page_content="This is another test document.", metadata={"id": "2"}),
        Document(page_content="This is a third test document.", metadata={"id": "3"}),
    ]
    ids = vector_store.add_documents(collection_name, documents)

    vector_store.delete_documents(collection_name, [ids[0]])

    # Check that the deleted doc is gone and others remain
    deleted_doc = vector_store.get_document_by_id(collection_name, ids[0])
    assert not deleted_doc  # Should be empty or None

    remaining_doc = vector_store.get_document_by_id(collection_name, ids[1])
    assert remaining_doc[0].metadata["id"] == "2"


def test_search_by_vector(vector_store):
    collection_name = "test_collection"
    documents = [
        Document(page_content="This is a test document.", metadata={"id": "1"}),
        Document(page_content="This is another test document.", metadata={"id": "2"}),
        Document(page_content="This is a third test document.", metadata={"id": "3"}),
    ]
    vector_store.add_documents(collection_name, documents)

    query_vector = vector_store.embedding.embed_query("test document")
    retrieved_docs = vector_store.get_documents_by_vector(collection_name, query_vector)
    assert len(retrieved_docs) >= 1
    assert any(doc.metadata["id"] in {"1", "2", "3"} for doc in retrieved_docs)
