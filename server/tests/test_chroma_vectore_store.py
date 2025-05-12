from langchain_core.documents import Document


def test_add_and_retrieve_documents(vector_store):
    collection = "test_collection"
    docs = [
        Document(page_content="First doc", metadata={"id": "1"}),
        Document(page_content="Second doc", metadata={"id": "2"}),
        Document(page_content="Third doc", metadata={"id": "3"}),
    ]
    vector_store.add_documents(collection, docs)
    results = vector_store.get_documents(collection, "doc")
    assert len(results) >= 3


def test_delete_document(vector_store):
    collection = "test_collection"
    docs = [
        Document(page_content="Only doc to delete", metadata={"id": "delete_me"}),
    ]
    ids = vector_store.add_documents(collection, docs)
    vector_store.delete_documents(collection, ids)
    result = vector_store.get_document_by_id(collection, "delete_me")
    assert result == []
