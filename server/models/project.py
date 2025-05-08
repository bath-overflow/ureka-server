from datetime import datetime

from pydantic import BaseModel, Field


class Project(BaseModel):
    """
    Project model.
    """

    id: int = Field(..., description="Project ID")
    name: str = Field(..., description="Project name")
    description: str = Field(..., description="Project description")
    created_at: datetime = Field(..., description="Project creation date")
    updated_at: datetime = Field(..., description="Project update date")

    class Config:
        orm_mode = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "name": "Project Name",
                "description": "Project description goes here.",
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-01T00:00:00Z",
            }
        }
