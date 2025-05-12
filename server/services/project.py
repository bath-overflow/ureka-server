import uuid

from fastapi import HTTPException
from sqlalchemy.orm import Session

from server.models.project_model import Project, ProjectCreate
from server.repositories.project_store import (
    create_project,
    delete_project,
    get_all_projects,
    get_project_by_id,
    update_project,
)


def create_new_project(db: Session, data: ProjectCreate) -> Project:
    new_project = Project(
        id=str(uuid.uuid4()), title=data.title, description=data.description
    )
    return create_project(db, new_project)


def list_projects(db: Session) -> list[Project]:
    return get_all_projects(db)


def get_project_or_404(db: Session, project_id: str) -> Project:
    project = get_project_by_id(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def update_project_info(
    db: Session, project_id: str, title: str, description: str
) -> Project:
    project = get_project_or_404(db, project_id)
    return update_project(db, project, title, description)


def delete_project_by_id(db: Session, project_id: str):
    project = get_project_or_404(db, project_id)
    delete_project(db, project)
