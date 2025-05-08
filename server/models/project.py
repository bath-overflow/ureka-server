from pydantic import BaseModel, Field
from sqlalchemy import Column, String

from server.utils.db import Base


# -------------------------
# ORM 모델
# -------------------------
class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(String, nullable=True)


# -------------------------
# Pydantic 스키마
# -------------------------
class ProjectCreate(BaseModel):
    title: str = Field(..., description="Project title")
    description: str = Field("", description="Project description")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Ureka Research Project",
                "description": "A project for learning assistant research",
            }
        }


class ProjectUpdate(ProjectCreate):
    pass


class ProjectResponse(ProjectCreate):
    id: str

    class Config:
        from_attributes = True


class ProjectListResponse(BaseModel):
    projects: list[ProjectResponse]
