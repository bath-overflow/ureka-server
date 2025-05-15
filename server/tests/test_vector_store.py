def test_add_documents(vector_store, isolated_collection, test_documents):
    collection_name = isolated_collection
    ids = vector_store.add_documents(collection_name, test_documents)

    # Verify that documents were added using get_document_by_id
    for doc_id in ids:
        doc = vector_store.get_document_by_id(collection_name, doc_id)
        assert doc is not None


def test_get_documents(vector_store, isolated_collection, test_documents):
    collection_name = isolated_collection

    vector_store.add_documents(collection_name, test_documents)
    results = vector_store.get_documents(collection_name, "test document")
    assert len(results) == len(test_documents)


def test_delete_documents(vector_store, isolated_collection, test_documents):
    collection_name = isolated_collection
    ids = vector_store.add_documents(collection_name, test_documents)

    vector_store.delete_documents(collection_name, [ids[0]])

    # Check that the deleted doc is gone and others remain
    deleted_doc = vector_store.get_document_by_id(collection_name, ids[0])
    assert deleted_doc is None

    remaining_doc = vector_store.get_document_by_id(collection_name, ids[1])
    assert remaining_doc.metadata["idx"] == test_documents[1].metadata["idx"]


def test_search_by_vector(vector_store, isolated_collection, test_documents):
    collection_name = isolated_collection
    vector_store.add_documents(collection_name, test_documents)

    query_vector = vector_store.embedding.embed_query("test document")
    retrieved_docs = vector_store.get_documents_by_vector(collection_name, query_vector)
    assert len(retrieved_docs) >= 1

    indices = [doc.metadata["idx"] for doc in test_documents]
    assert any(doc.metadata["idx"] in indices for doc in retrieved_docs)
