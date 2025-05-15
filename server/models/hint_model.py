from pydantic import BaseModel, Field


class HintRequest(BaseModel):
    """
    Request body model for hint request.
    """

    message: str = Field(
        ..., description="The user message for which hint is requested"
    )

    class Config:
        json_schema_extra = {
            "example": {"message": "What is the difference between a list and a tuple?"}
        }


class HintResponse(BaseModel):
    """
    Response body model for hint response.
    """

    hint: str = Field(..., description="The generated hint based on the message")

    class Config:
        json_schema_extra = {
            "example": {"hint": "Lists are mutable, whereas tuples are immutable."}
        }
