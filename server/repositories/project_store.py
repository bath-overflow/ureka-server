from sqlalchemy.orm import Session

from server.models.project import Project


def get_all_projects(db: Session) -> list[Project]:
    return db.query(Project).all()


def get_project_by_id(db: Session, project_id: str) -> Project | None:
    return db.query(Project).filter(Project.id == project_id).first()


def create_project(db: Session, project: Project) -> Project:
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def update_project(
    db: Session, project: Project, title: str, description: str
) -> Project:
    project.title = title
    project.description = description
    db.commit()
    db.refresh(project)
    return project


def delete_project(db: Session, project: Project):
    db.delete(project)
    db.commit()
