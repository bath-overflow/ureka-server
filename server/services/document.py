import asyncio
import re
from datetime import timedelta
from io import BytesIO

import httpx
from fastapi import HTTPException, UploadFile
from langchain_core.retrievers import BaseRetriever
from minio.error import S3Error
from sqlalchemy.orm import Session

from server.models.document_model import Document
from server.repositories.document_store import (
    create_document,
    delete_document_by_filename,
    get_document_by_filename,
    get_documents_by_project,
)
from server.repositories.vector_store import vector_store
from server.utils.config import PARSE_SERVER_BASE_URL
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


async def _parse_split_and_save(
    project_id: str, filename: str, file_bytes: bytes, content_type: str | None
):
    """
    Send file for parsing, splitting, and vectorization to the GPU server.
    """
    files = {"file": (filename, file_bytes, content_type)}
    data = {"project_id": project_id}

    timeout_settings = httpx.Timeout(300.0, connect=60.0)  # 5 min total, 1 min connect

    async with httpx.AsyncClient(timeout=timeout_settings) as client:
        try:
            print(f"BG: Sending {filename} to GPU server for project {project_id}")
            response = await client.post(
                f"{PARSE_SERVER_BASE_URL}/file_parse", files=files, data=data
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            print(
                f"BG: HTTPStatusError for {filename} from GPU server: "
                f"{e.response.status_code} - {e.response.text}"
            )
        except httpx.RequestError as e:
            print(
                f"BG: RequestError for {filename}, could not connect to GPU server: "
                f"{str(e)}"
            )
        except Exception as e:
            print(
                f"BG: Unexpected error for {filename} during GPU processing: {str(e)}"
            )


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

    # 2. Pass document to GPU server for processing
    # GPU does the following:
    #   - Convert file to markdown content
    #   - Split markdown content into documents
    #   - Add documents to VectorStore
    asyncio.create_task(
        _parse_split_and_save(
            project_id=project_id,
            filename=file.filename,
            file_bytes=file_content,
            content_type=file.content_type,
        )
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


def get_document_metadata(
    db: Session, project_id: str, filename: str
) -> Document | None:
    """
    Returns metadata for the original document, including a presigned URL for
    downloading.

    Args:
        db (Session): Database session.
        project_id (str): Project ID to which the document belongs.
        filename (str): The ORIGINAL document's filename. (e.g. "report.pdf")
    """
    # Look up in DB
    document = get_document_by_filename(db, project_id, filename)
    if not document:
        raise HTTPException(status_code=404, detail="Resource not found")

    # Replace the file URL with presigned URL
    presigned_url = minio_client.presigned_get_object(
        bucket_name="documents",
        object_name=document.filename,
        expires=timedelta(hours=1),
    )

    document.file_url = presigned_url
    return document


def get_markdown_document_metadata(db: Session, project_id: str, filename: str) -> dict:
    """
    Returns metadata and content for the markdown version of the document.
    Note that the image URLs in the markdown content are replaced with presigned URLs.

    Args:
        db (Session): Database session.
        project_id (str): Project ID to which the document belongs.
        filename (str): The ORIGINAL document's filename.
            (non-markdown version - e.g. "report.pdf")

    Returns:
        dict: Metadata including filename, upload date, size, and markdown content with
            image URLs replaced. Upload date is the original document's upload date.
    """
    # Look up in DB
    document = get_document_by_filename(db, project_id, filename)
    if not document:
        raise HTTPException(status_code=404, detail="Resource not found")

    # Get the markdown file name
    original_db_filename = document.filename
    filename_without_extension = original_db_filename.rsplit(".", 1)[0]
    markdown_doc_filename = f"{filename_without_extension}.md"

    minio_object_name = f"{project_id}/{markdown_doc_filename}"

    try:
        # Check existence and get metadata including size
        object_stat = minio_client.stat_object("markdowns", minio_object_name)
        markdown_size = object_stat.size
    except S3Error as e:
        if e.code == "NoSuchKey":
            # If the markdown file does not exist, raise a 404 error
            raise HTTPException(
                status_code=404,
                detail=f"Markdown file for {original_db_filename} not found",
            )
        # Handle other S3 errors
        raise HTTPException(
            status_code=500, detail=f"Error accessing markdown file: {str(e)}"
        )
    except Exception as e:
        # Handle other unexpected errors
        raise HTTPException(
            status_code=500, detail=f"Error accessing markdown file: {str(e)}"
        )

    md_data_response = None
    try:
        md_data_response = minio_client.get_object("markdowns", minio_object_name)
        markdown_content_bytes = md_data_response.read()
        markdown_content_str = markdown_content_bytes.decode("utf-8")
    except Exception as e:
        print(f"Error reading markdown object {minio_object_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not read markdown file for {original_db_filename}",
        )
    finally:
        if md_data_response:
            md_data_response.close()
            md_data_response.release_conn()

    def image_url_replacer(match: re.Match[str]) -> str:
        image_alt_text = match.group(1)
        image_path = match.group(2)

        # Images are in "images" bucket, object name is the filename directly.
        try:
            # Check if image exists in MinIO before generating URL
            minio_client.stat_object("images", image_path)

            presigned_image_url = minio_client.presigned_get_object(
                "images", image_path, expires=timedelta(hours=1)
            )
            return f"![{image_alt_text}]({presigned_image_url})"
        # Return original tag if presigning fails
        except S3Error as e:
            if e.code == "NoSuchKey":
                print(f"Image '{image_path}' not found in MinIO.")
            return match.group(0)
        except Exception as e:
            print(f"Error generating presigned URL for image '{image_path}': {e}")
            return match.group(0)

    # Regex to find markdown image tags: ![alt_text](image_path)
    # This regex captures alt text in group 1 and image path in group 2.
    # It assumes image paths do not contain ')'
    processed_markdown_content = re.sub(
        r"!\[(.*?)\]\(([^)]+)\)", image_url_replacer, markdown_content_str
    )

    return {
        "filename": markdown_doc_filename,
        "upload_date": document.upload_date,  # Original document's upload date
        "size": markdown_size,
        "file_content": processed_markdown_content,
    }


def delete_document(db: Session, project_id: str, filename: str):
    document = get_document_by_filename(db, project_id, filename)
    if not document:
        raise HTTPException(status_code=404, detail="Resource not found")

    # Delete the original document from MinIO
    try:
        minio_client.remove_object("documents", filename)
    except S3Error as e:
        if e.code == "NoSuchKey":
            print(f"Document {filename} not found in MinIO. Proceeding.")

    original_db_filename = document.filename
    filename_without_extension = original_db_filename.rsplit(".", 1)[0]
    markdown_doc_filename_base = f"{filename_without_extension}.md"
    minio_object_name = f"{project_id}/{markdown_doc_filename_base}"

    found_markdown_file = False

    md_data_response = None
    markdown_content_str = ""
    # Find image filenames referenced in the markdown file
    try:
        md_data_response = minio_client.get_object("markdowns", minio_object_name)
        markdown_content_bytes = md_data_response.read()
        markdown_content_str = markdown_content_bytes.decode("utf-8")
        found_markdown_file = True
    except S3Error as e:
        if e.code == "NoSuchKey":
            print(
                f"Markdown file {minio_object_name} not found. "
                "Cannot extract image list for deletion."
            )
        else:
            print(f"S3Error reading markdown file {minio_object_name}: {e}")
    except Exception as e:
        print(f"Error reading markdown file {minio_object_name}: {e}")
    finally:
        if md_data_response:
            md_data_response.close()
            md_data_response.release_conn()

    if not found_markdown_file:
        print(f"Markdown file {minio_object_name} not found. No images to delete.")
        # Proceed to delete the document from DB
        delete_document_by_filename(db, project_id, filename)
        return {"message": "Document deleted successfully", "id": document.id}

    image_filenames_to_delete = []
    # The regex finds paths within ![](...)
    image_filenames = re.findall(r"!\[.*?\]\(([^)]+)\)", markdown_content_str)
    for filename in image_filenames:
        # Filter out full URLs (should not happen but just in case)
        if not (filename.startswith("http://") or filename.startswith("https://")):
            image_filenames_to_delete.append(filename)

    # Delete the markdown file from MinIO
    try:
        minio_client.remove_object("markdowns", minio_object_name)
        print(f"Attempted deletion of markdown file: {minio_object_name}")
    except S3Error as e:
        if e.code == "NoSuchKey":
            print(f"Markdown file {minio_object_name} not found for deletion.")
        else:
            print(f"S3Error deleting markdown file {minio_object_name}: {e}")
    except Exception as e:
        print(f"Unexpected error deleting markdown file {minio_object_name}: {e}")

    # Delete associated images from MinIO
    if not image_filenames_to_delete:
        print("No images found in markdown content for deletion.")
        delete_document_by_filename(db, project_id, filename)
        return {"message": "Document deleted successfully", "id": document.id}

    for filename in image_filenames_to_delete:
        try:
            # Images are stored in the "images" bucket with the object name
            # being the filename directly
            minio_client.remove_object("images", filename)
        except S3Error as e:
            if e.code == "NoSuchKey":
                print(f"Image {filename} not found in bucket 'images' for deletion.")
            else:
                print(f"S3Error deleting image {filename} from MinIO: {e}")
        except Exception as e:
            print(f"Unexpected error deleting image {filename}: {e}")

    # DB에서 삭제
    delete_document_by_filename(db, project_id, filename)
    return {"message": "Document deleted successfully", "id": document.id}
