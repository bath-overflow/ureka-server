# domains/document/repository.py

from typing import List, Optional

from sqlalchemy.orm import Session

from server.models.document_model import Document


def create_document(
    db: Session, project_id: str, filename: str, file_url: str, size: int
) -> Document:
    document = Document(
        project_id=project_id, filename=filename, file_url=file_url, size=size
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


def get_documents_by_project(db: Session, project_id: str) -> List[Document]:
    return (
        db.query(Document)
        .filter(Document.project_id == project_id)
        .order_by(Document.upload_date.desc())
        .all()
    )


def get_document_by_filename(
    db: Session, project_id: str, filename: str
) -> Optional[Document]:
    return (
        db.query(Document)
        .filter(Document.project_id == project_id, Document.filename == filename)
        .first()
    )


def delete_document_by_filename(db: Session, project_id: str, filename: str) -> bool:
    doc = get_document_by_filename(db, project_id, filename)
    if doc:
        db.delete(doc)
        db.commit()
        return True
    return False
