from datetime import datetime

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """
    Chat message model.
    """

    id: str = Field(..., description="Chat message ID (UUID or ObjectId)")
    role: str = Field(..., description="Role of the user (user or assistant)")
    message: str = Field(..., description="Chat message")
    created_at: datetime = Field(..., description="Chat creation timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "role": "user",
                "message": "Hello, how can I help you?",
                "created_at": "2023-01-01T00:00:00Z",
            }
        }


class ChatHistory(BaseModel):
    """
    Chat history model.
    """

    id: str = Field(..., description="Chat history ID")
    # project_id: str = Field(..., description="Project ID")
    # user_id: str = Field(..., description="User ID")
    messages: list[ChatMessage] = Field(
        default_factory=list, description="List of chat messages"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "messages": [
                    {
                        "id": 1,
                        "role": "user",
                        "message": "Hello, how can I help you?",
                        "created_at": "2023-01-01T00:00:00Z",
                    }
                ],
            }
        }


class ChatHistoryResponse(BaseModel):
    """
    Chat history response model.
    """

    id: str = Field(..., description="Chat history ID")
    messages: list[ChatMessage] = Field(
        default_factory=list, description="List of chat messages"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "messages": [
                    {
                        "id": 1,
                        "role": "user",
                        "message": "Hello, how can I help you?",
                        "created_at": "2023-01-01T00:00:00Z",
                    }
                ],
            }
        }
