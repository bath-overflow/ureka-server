import io

from langchain_core.documents import Document
from pypdf import PdfReader

from server.repositories.vector_store import vector_store


def pdf_to_documents(
    pdf_content: bytes,
    source_name: str,
) -> list[Document]:
    """
    Convert a PDF file to a list of LangChain Document objects.

    Args:
        pdf_content: The binary content of the PDF file
        source_name: Name to use as the source in document metadata
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks

    Returns:
        List of Document objects
    """
    # Create PDF reader from bytes
    pdf_reader = PdfReader(io.BytesIO(pdf_content))

    # Extract text from each page with metadata
    documents: list[Document] = []
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text.strip():  # Skip empty pages
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_name,
                        "page": i + 1,
                        "total_pages": len(pdf_reader.pages),
                    },
                )
            )

    return documents


def add_file_to_vector_store(
    collection_name: str,
    file_content: bytes,
    file_name: str,
) -> list[str]:
    """
    Process a PDF file and add its contents to the vector store.

    Args:
        collection_name: Name of the collection to add documents to
        file_content: The binary content of the file
        file_name: Name of the file (should be a PDF)
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks

    Returns:
        List of document IDs added to the vector store
    """
    # Validate file is a PDF
    if not file_name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are currently supported")

    # Convert PDF to documents
    documents = pdf_to_documents(
        file_content,
        source_name=file_name,
    )

    # Add documents to vector store
    doc_ids = vector_store.add_documents(collection_name, documents)

    return doc_ids
