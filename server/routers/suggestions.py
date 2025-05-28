from fastapi import APIRouter, Path
from server.services.suggestions import SuggestionService

suggestion_router = APIRouter()
suggestion_service = SuggestionService()

@suggestion_router.get("/chat/{chat_id}/suggestions")
async def get_suggestions(chat_id: str = Path(..., description="Chat session ID")):
    result = await suggestion_service.graph.ainvoke({
        "collection_name": chat_id,
    })

    return {
        "suggested_questions": result.get("suggested_questions", [])
    }
