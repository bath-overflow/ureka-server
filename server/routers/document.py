# api/documents.py

from typing import List

from fastapi import APIRouter, Depends, File, Path, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from server.models.document import DocumentResponse
from server.services.document import (
    delete_document,
    get_document_metadata,
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


@router.get("/", response_model=List[DocumentResponse])
def get_documents(
    projectId: str = Path(..., description="Project ID"), db: Session = Depends(get_db)
):
    """
    Get uploaded resources for a project
    """
    result = list_documents(db, projectId)
    return [{"filename": doc.filename} for doc in result["documents"]]


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
    return {
        "filename": doc.filename,
        "uploadDate": doc.upload_date.isoformat(),
        "file_url": doc.file_url,
        "size": doc.size,
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
    return JSONResponse(status_code=204, content=None)
