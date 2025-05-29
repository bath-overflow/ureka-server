from datetime import timedelta
from io import BytesIO

from fastapi import HTTPException, UploadFile
from langchain_core.retrievers import BaseRetriever
from sqlalchemy.orm import Session

from server.repositories.document_store import (
    create_document,
    delete_document_by_filename,
    get_document_by_filename,
    get_documents_by_project,
)
from server.repositories.vector_store import vector_store
from server.services.vector_store import add_file_to_vector_store
from server.utils.db import MINIO_URL, minio_client


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


async def upload_and_register_document(
    db: Session, project_id: str, file: UploadFile, bucket_name: str = "documents"
):
    # 1. MinIO에 업로드
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    file_content = await file.read()
    file_size = len(file_content)

    minio_client.put_object(
        bucket_name=bucket_name,
        object_name=file.filename,
        data=BytesIO(file_content),
        length=file_size,
        content_type=file.content_type,
    )

    # 2. VectorStore에 문서 추가
    if file.filename.lower().endswith(".pdf"):
        # Use project_id as the collection name for vector store
        add_file_to_vector_store(
            collection_name=project_id,
            file_content=file_content,
            file_name=file.filename,
        )

    file_url = f"http://{MINIO_URL}/{bucket_name}/{file.filename}"

    # 3. DB에 문서 저장
    return create_document(
        db=db,
        project_id=project_id,
        filename=file.filename,
        file_url=file_url,
        size=file_size,
    )


def list_documents(db: Session, project_id: str):
    documents = get_documents_by_project(db, project_id)
    return {"documents": documents}


def get_document_metadata(db: Session, project_id: str, filename: str):
    document = get_document_by_filename(db, project_id, filename)
    if not document:
        raise HTTPException(status_code=404, detail="Resource not found")

    # presigned URL 생성
    presigned_url = minio_client.presigned_get_object(
        bucket_name="documents", object_name=filename, expires=timedelta(hours=1)
    )

    document.file_url = presigned_url
    return document


def delete_document(db: Session, project_id: str, filename: str):
    document = get_document_by_filename(db, project_id, filename)
    if not document:
        raise HTTPException(status_code=404, detail="Resource not found")

    # MinIO에서 삭제 (object 존재 여부는 확인하지 않음)
    try:
        minio_client.remove_object("documents", filename)
    except Exception as e:
        print(f"Error deleting object from MinIO: {e}")
        pass

    # DB에서 삭제
    delete_document_by_filename(db, project_id, filename)
    return {"message": "Document deleted successfully", "id": document.id}

def get_relevant_docs_text(collection_name: str, query: str) -> str:
    retriever = DocumentRetriever(collection_name)
    docs = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in docs])
