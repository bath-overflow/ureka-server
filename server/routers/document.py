from fastapi import APIRouter, Depends, File, HTTPException, Path, UploadFile
from sqlalchemy.orm import Session

from server.models.document_model import DocumentResponse
from server.services.document import (
    delete_document,
    get_document_metadata,
    get_markdown_content,
    list_documents,
    upload_and_register_document,
)
from server.utils.db import get_db

router = APIRouter(prefix="/projects/{projectId}/resources")


@router.post("/", status_code=201)
async def upload_document(
    projectId: str = Path(..., description="Project ID"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a resource to a project
    """
    document = await upload_and_register_document(db, projectId, file)
    return {"filename": document.filename}


@router.get(
    "/",
)
def get_documents(
    projectId: str = Path(..., description="Project ID"), db: Session = Depends(get_db)
):
    """
    Get uploaded resources for a project
    """
    result = list_documents(db, projectId)
    return [
        {
            "filename": doc.filename,
            "uploadDate": doc.upload_date.isoformat(),
            "size": doc.size,
        }
        for doc in result["documents"]
    ]


@router.get("/{filename}", response_model=DocumentResponse)
def get_document(
    projectId: str = Path(..., description="Project ID"),
    filename: str = Path(..., description="File name"),
    db: Session = Depends(get_db),
):
    """
    Get details of a specific resource
    """
    doc = get_document_metadata(db, projectId, filename)

    markdown_content_value: str | None = None
    try:
        markdown_content_value = get_markdown_content(db, projectId, filename)
    except HTTPException as e:
        # If 404 occurs, we set content to None and proceed.
        # Other errors (e.g., 500 from MinIO access) will propagate.
        if e.status_code != 404:
            raise e

    return {
        "filename": doc.filename,
        "uploadDate": doc.upload_date.isoformat(),
        "file_url": doc.file_url,
        "size": doc.size,
        "markdown_content": markdown_content_value,
    }


@router.delete("/{filename}", status_code=204)
def delete_document_route(
    projectId: str = Path(..., description="Project ID"),
    filename: str = Path(..., description="File name"),
    db: Session = Depends(get_db),
):
    """
    Delete a specific resource from a project
    """
    delete_document(db, projectId, filename)
    return None
