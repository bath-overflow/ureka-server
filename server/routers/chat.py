from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.models.chat import ChatHistoryResponse, ChatMessage
from server.services.chat import ChatService

chat_router = APIRouter()

chat_service = ChatService()


@chat_router.websocket("/ws/chat/{chat_id}")
async def chat_websocket(websocket: WebSocket, chat_id: str):
    """
    WebSocket 연결 및 메시지 처리
    """
    await chat_service.connect_user(chat_id, websocket)
    try:
        await chat_service.send_message_to_user(chat_id, f"User {chat_id} connected.")
        while True:
            data = await websocket.receive_text()

            # 메시지 저장
            message = ChatMessage(content=data)
            chat_service.save_message(chat_id, message)

            # Echo 메시지 전송 (예시)
            await chat_service.send_message_to_user(chat_id, f"Echo: {data}")
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
