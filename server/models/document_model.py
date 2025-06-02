from fastapi import UploadFile
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

    file: UploadFile = Field(..., description="Document file")

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

    filename: str = Field(..., description="Document filename")
    file_url: str = Field(..., description="Document file URL")
    size: int = Field(..., description="Document size")
    upload_date: str = Field(
        ..., alias="uploadDate", description="Document upload date"
    )


class MarkdownContentResponse(BaseModel):
    """
    Response model for markdown content.
    """

    filename: str = Field(..., description="Filename of the document")
    size: int = Field(..., description="Size of the document file in bytes")
    file_content: str = Field(..., description="Markdown content of the document")
    upload_date: str = Field(
        ..., alias="uploadDate", description="Upload date of the document"
    )
