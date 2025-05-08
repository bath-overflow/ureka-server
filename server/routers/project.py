from fastapi import APIRouter, Depends, Path
from sqlalchemy.orm import Session

from server.models.project import (
    ProjectCreate,
    ProjectResponse,
    ProjectUpdate,
)
from server.services.project import (
    create_new_project,
    delete_project_by_id,
    get_project_or_404,
    list_projects,
    update_project_info,
)
from server.utils.db import get_db

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("/", response_model=list[ProjectResponse])
def get_projects(db: Session = Depends(get_db)):
    return list_projects(db)


@router.post("/", response_model=ProjectResponse, status_code=201)
def create_project(data: ProjectCreate, db: Session = Depends(get_db)):
    return create_new_project(db, data)


@router.get("/{projectId}", response_model=ProjectResponse)
def get_project(projectId: str = Path(...), db: Session = Depends(get_db)):
    return get_project_or_404(db, projectId)


@router.put("/{projectId}", response_model=ProjectResponse)
def update_project(
    projectId: str = Path(...), data: ProjectUpdate = ..., db: Session = Depends(get_db)
):
    return update_project_info(db, projectId, data.title, data.description)


@router.delete("/{projectId}", status_code=204)
def delete_project(projectId: str = Path(...), db: Session = Depends(get_db)):
    delete_project_by_id(db, projectId)
