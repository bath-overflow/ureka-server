from fastapi import APIRouter, Path
from server.services.recommend import RecommendService

recommend_router = APIRouter()
recommend_service = RecommendService()

@recommend_router.get("/chat/{chat_id}/recommend")
async def get_recommendations(chat_id: str = Path(..., description="Chat session ID")):
    result = await recommend_service.graph.ainvoke({
        "collection_name": chat_id,
    })

    return {
        "recommended_questions": result.get("recommended_questions", [])
    }
