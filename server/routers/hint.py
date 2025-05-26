from fastapi import APIRouter, Path, Body
from fastapi.responses import StreamingResponse
from server.services.hint import stream_hint_response

hint_router = APIRouter()

@hint_router.post("/chat/{chat_id}/hint", response_class=StreamingResponse)
async def get_hint(
    chat_id: str = Path(..., description="Chat session ID")
):

    # Stream the hint response
    return StreamingResponse(
        stream_hint_response(chat_id),
        media_type="text/plain"
    )