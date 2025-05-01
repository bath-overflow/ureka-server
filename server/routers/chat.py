from enum import Enum

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

chat_router = APIRouter()


class ChatEvent(str, Enum):
    SEND_MESSAGE = "send_message"
    JOIN_CHAT = "join_chat"
    LEAVE_CHAT = "leave_chat"
    MESSAGE_RECEIVED = "message_received"
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    ERROR = "error"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, event: ChatEvent, message: str):
        for connection in self.active_connections:
            await connection.send_text(f"{event.value}: {message}")


manager = ConnectionManager()


@chat_router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await manager.broadcast(ChatEvent.CONNECTED, "A user has connected.")
        while True:
            data = await websocket.receive_text()
            # LLM 응답 생성 로직 추가 지금은 단순 echo
            await manager.broadcast(ChatEvent.MESSAGE_RECEIVED, data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(ChatEvent.DISCONNECTED, "A user has disconnected.")
