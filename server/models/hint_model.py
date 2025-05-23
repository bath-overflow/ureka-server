from pydantic import BaseModel, Field


class HintRequest(BaseModel):
    """
    Request body model for hint request.
    """
    prev_question: str = Field(..., description="Previous question from the llm")

