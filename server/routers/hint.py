from fastapi import APIRouter, Path
from fastapi.responses import JSONResponse
from server.services.hint import HintService

hint_router = APIRouter()
hint_service = HintService()

@hint_router.post("/chat/{chat_id}/hint")
async def get_hint(
    chat_id: str = Path(..., description="Chat session ID")
):
    # Generate the hint response synchronously
    hint = await hint_service.generate_hint_response(chat_id)
    return {"hint": hint}
