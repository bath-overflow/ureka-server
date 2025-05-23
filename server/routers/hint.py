from fastapi import APIRouter, Path, Body
from fastapi.responses import StreamingResponse
from server.models.hint_model import HintRequest
from server.services.hint import stream_hint_response

hint_router = APIRouter()

@hint_router.post("/chat/{chat_id}/hint", response_class=StreamingResponse)
async def get_hint(
    chat_id: str = Path(..., description="Chat session ID"),
    request: HintRequest = Body(..., description="Previous question"),
):
    # prev_question은 request.body 안에 있음
    prev_question = request.prev_question

    # Stream the hint response
    return StreamingResponse(
        stream_hint_response(chat_id, prev_question),
        media_type="text/plain"
    )