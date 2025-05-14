# conftest.py
import io
import uuid

import pytest
from chromadb import Client
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from server.repositories.vector_store import ChromaVectorStore, VectorStore
from server.utils.db import Base, get_db


# -----------------------------
# Chroma 관련 fixture
# -----------------------------
@pytest.fixture(scope="session")
def chroma_client() -> Client:
    return Client()


@pytest.fixture(scope="session")
def embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
def vector_store(chroma_client, embeddings) -> VectorStore:
    return ChromaVectorStore(embedding=embeddings, client=chroma_client)


@pytest.fixture
def isolated_collection(chroma_client):
    name = f"test_{uuid.uuid4().hex[:8]}"
    yield name
    chroma_client.delete_collection(name)


@pytest.fixture
def sample_pdf():
    # Create a minimal real PDF in memory for more authentic testing
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    # First page
    c.drawString(100, 700, "This is a test PDF document")
    c.drawString(100, 650, "Page 1 content")
    c.showPage()

    # Second page
    c.drawString(100, 700, "This is page 2")
    c.drawString(100, 650, "More content here")
    c.showPage()

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# -----------------------------
# SQLAlchemy (SQLite for test)
# -----------------------------
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def db_session():
    import server.models.document_model
    import server.models.project_model

    _ = server.models.document_model
    _ = server.models.project_model

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True, scope="function")
def override_get_db(db_session):
    from server.main import app  # ✅ 지연 import (핵심)

    app.dependency_overrides[get_db] = lambda: db_session


@pytest.fixture
def test_documents():
    return [
        Document(page_content="This is a test document.", metadata={"idx": "1"}),
        Document(page_content="This is another test document.", metadata={"idx": "2"}),
        Document(page_content="This is a third test document.", metadata={"idx": "3"}),
    ]
