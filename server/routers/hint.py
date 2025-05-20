from fastapi import APIRouter

from server.models.hint_model import HintRequest, HintResponse
from server.services.hint import HintService

hint_router = APIRouter()
hint_service = HintService()

@hint_router.post("/chat/{chat_id}/hint", response_model=HintResponse)
async def get_hint(chat_id: str, request: HintRequest):

    # 힌트 생성
    hint = hint_service.generate_hint(chat_id, request.message)

    return HintResponse(hint=hint)