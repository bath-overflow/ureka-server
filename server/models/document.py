from typing import List

from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Integer, String, func

from server.utils.db import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    file_url = Column(String, nullable=False)
    size = Column(Integer)
    upload_date = Column(DateTime, default=func.now())


class DocumentCreate(BaseModel):
    """
    Document creation model.
    """

    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Document Title",
                "content": "Document content goes here.",
            }
        }


class DocumentResponse(BaseModel):
    """
    Document response model.
    """

    id: int = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    created_at: str = Field(..., description="Document creation date")
    updated_at: str = Field(..., description="Document update date")

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "title": "Document Title",
                "content": "Document content goes here.",
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-01T00:00:00Z",
            }
        }


class DocumentListResponse(BaseModel):
    """
    Document list response model.
    """

    documents: List[DocumentResponse] = Field(..., description="List of documents")
    total: int = Field(..., description="Total number of documents")

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "id": 1,
                        "title": "Document Title 1",
                        "content": "Document content goes here.",
                        "created_at": "2023-01-01T00:00:00Z",
                        "updated_at": "2023-01-01T00:00:00Z",
                    },
                    {
                        "id": 2,
                        "title": "Document Title 2",
                        "content": "Document content goes here.",
                        "created_at": "2023-01-01T00:00:00Z",
                        "updated_at": "2023-01-01T00:00:00Z",
                    },
                ],
                "total": 2,
            }
        }


class DocumentDeleteResponse(BaseModel):
    """
    Document delete response model.
    """

    message: str = Field(..., description="Delete message")
    id: int = Field(..., description="Document ID")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Document deleted successfully",
                "id": 1,
            }
        }
