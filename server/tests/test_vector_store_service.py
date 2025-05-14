from unittest.mock import patch

import pytest

from server.services.vector_store import add_file_to_vector_store, pdf_to_documents


def test_pdf_to_documents(sample_pdf):
    # Test converting PDF content to documents
    source_name = "test.pdf"
    documents = pdf_to_documents(sample_pdf, source_name)

    # Check number of pages
    assert len(documents) == 2

    # Check content of documents
    assert documents[0].page_content == "This is a test PDF document\nPage 1 content\n"
    assert documents[1].page_content == "This is page 2\nMore content here\n"

    # Check metadata
    assert documents[0].metadata["source"] == source_name
    assert documents[0].metadata["page"] == 1
    assert documents[0].metadata["total_pages"] == 2

    assert documents[1].metadata["source"] == source_name
    assert documents[1].metadata["page"] == 2
    assert documents[1].metadata["total_pages"] == 2


def test_add_file_to_vector_store_pdf_validation():
    # Test that non-PDF files are rejected (.txt for example)
    with pytest.raises(ValueError, match="Only PDF files are currently supported"):
        add_file_to_vector_store("test_collection", b"content", "document.txt")


def test_add_file_to_vector_store(vector_store, isolated_collection, sample_pdf):
    # Test actual addition to vector store

    # Use the vector_store fixture in add_file_to_vector_store
    with patch("server.services.vector_store.vector_store", vector_store):
        doc_ids = add_file_to_vector_store(isolated_collection, sample_pdf, "test.pdf")

        # Check if added
        assert len(doc_ids) > 0

        # Check if documents are retrievable
        retrieved_docs = vector_store.get_documents(isolated_collection, "test")

        assert len(retrieved_docs) > 0
        assert any(
            "test.pdf" in str(doc.metadata.get("source")) for doc in retrieved_docs
        )

        # Retrieve the first document by ID
        doc = vector_store.get_document_by_id(isolated_collection, doc_ids[0])
        assert doc is not None
        assert "test.pdf" in str(doc.metadata["source"])
