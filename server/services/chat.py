from enum import Enum
from typing import AsyncGenerator, Callable

from fastapi import WebSocket

from server.models.chat_model import ChatHistory, ChatMessage
from server.repositories.chat_store import (
    append_chat_message,
    create_chat_history,
    get_chat_history,
)


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


class ChatInfo(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    MESSAGE_RECEIVED = "message_received"
    END_OF_STREAM = "<EOS>"


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, chat_id: str, websocket: WebSocket):
        """
        특정 사용자 ID를 기준으로 WebSocket 연결을 저장
        """
        await websocket.accept()
        self.active_connections[chat_id] = websocket

    def disconnect(self, chat_id: str):
        """
        사용자 연결을 제거
        """
        if chat_id in self.active_connections:
            del self.active_connections[chat_id]

    async def send_message(self, chat_id: str, event: str, message: str):
        """
        특정 사용자에게 메시지 전송
        """
        if chat_id in self.active_connections:
            websocket = self.active_connections[chat_id]
            await websocket.send_text(f"{event}: {message}")

    async def broadcast(self, event: str, message: str):
        """
        모든 연결된 사용자에게 메시지 전송
        추후 심층 토론 기능에 활용
        """
        for _, websocket in self.active_connections.items():
            await websocket.send_text(f"{event}: {message}")


class ChatService:
    def __init__(self):
        # 사용자 ID를 기준으로 WebSocket 연결 및 히스토리 관리
        self.connection_manager = ConnectionManager()

    async def connect_user(self, chat_id: str, websocket: WebSocket):
        """
        사용자 연결 관리 및 채팅방 생성
        """
        await self.connection_manager.connect(chat_id, websocket)

    def save_message(self, chat_id: str, message: ChatMessage):
        """
        메시지를 DB에 저장하고 채팅 히스토리 업데이트
        """

        chat_history = get_chat_history(chat_id)
        if not chat_history:
            create_chat_history(chat_id)
        append_chat_message(chat_id, message)

    def get_history(self, chat_id: str) -> ChatHistory:
        """
        특정 채팅방의 메시지 히스토리 가져오기
        """
        chat_history = get_chat_history(chat_id)
        if not chat_history:
            return create_chat_history(chat_id)
        return chat_history

    def disconnect_user(self, chat_id: str):
        """
        사용자 연결 해제 및 정리
        """
        self.connection_manager.disconnect(chat_id)

    async def send_message_to_user(self, chat_id: str, event: str, message: str):
        """
        특정 사용자에게 메시지 전송
        """
        await self.connection_manager.send_message(chat_id, event, message)

    async def generate_and_stream_message_to_user(
        self,
        chat_id: str,
        user_message: str,
        stream_generator: Callable[[str, str], AsyncGenerator[str, None]],
    ) -> str | None:
        """
        Generate and stream AI response

        Args:
            chat_id: The chat ID
            user_message: The user's message to be used as AI input
            stream_generator: A function that takes chat_id and user_message and
            returns an async generator of response tokens

        Returns:
            The full AI response text, or None if failed
        """
        full_response = ""
        response_started = False

        async for token in stream_generator(chat_id, user_message):
            # Check token validity
            if token is None or not isinstance(token, str):
                continue

            response_started = True
            # Send each token individually
            await self.send_message_to_user(
                chat_id, ChatEvent.SEND_MESSAGE.value, token
            )
            full_response += token

        # Only signal end of stream if we've sent something
        if response_started:
            await self.send_message_to_user(
                chat_id, ChatEvent.SEND_MESSAGE.value, ChatInfo.END_OF_STREAM.value
            )
            return full_response
        else:
            raise ValueError("No valid tokens received from AI response stream.")


service = ChatService()
