import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.models.chat import ChatHistoryResponse, ChatMessage
from server.services.chat import ChatEvent, ChatInfo, ChatService

chat_router = APIRouter()

chat_service = ChatService()


@chat_router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket, chat_id: str = None):
    """
    WebSocket 연결 및 메시지 처리
    """
    if chat_id is None:
        chat_id = uuid.uuid4().hex
    await chat_service.connect_user(chat_id, websocket)
    try:
        await chat_service.send_message_to_user(
            chat_id,
            ChatEvent.CONNECTED.value,
            ChatInfo.CONNECTED.value,
        )
        while True:
            data = await websocket.receive_text()

            # 메시지 저장
            message = ChatMessage.model_validate_strings(data)
            chat_service.save_message(chat_id, message)

            # Echo 메시지 전송 (예시)
            await chat_service.send_message_to_user(
                chat_id, ChatEvent.MESSAGE_RECEIVED, message.message
            )
    except WebSocketDisconnect:
        chat_service.disconnect_user(chat_id)


@chat_router.get("/chat/{chat_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(chat_id: str):
    """
    특정 채팅방의 메시지 히스토리 가져오기
    """
    chat_history = chat_service.get_history(chat_id)
    if chat_history:
        return chat_history
    else:
        return {"message": "No chat history found."}
