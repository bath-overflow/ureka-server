from fastapi import APIRouter

from server.models.hint import HintRequest, HintResponse
from server.services.chat import ChatService
from server.services.hint import HintService

hint_router = APIRouter()
chat_service = ChatService()
hint_service = HintService()


@hint_router.post("/chat/{chat_id}/hint", response_model=HintResponse)
async def get_hint(chat_id: str, request: HintRequest):
    # chat_history = chat_service.get_history(chat_id)

    # 힌트 생성
    hint = hint_service.generate_hint(chat_id, request.message)

    return HintResponse(hint=hint)