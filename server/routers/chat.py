import json
import traceback
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.models.chat import ChatHistoryResponse, ChatMessage
from server.services.chat import ChatEvent, ChatInfo, ChatService
from server.services.langgraph_service import stream_chat_response

chat_router = APIRouter()

chat_service = ChatService()


@chat_router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket, chat_id: str = None):
    """
    WebSocket connection handler with streaming AI responses
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
            data_dict = json.loads(data)

            # Parse and save user message
            user_message = ChatMessage.model_validate(data_dict)
            chat_service.save_message(chat_id, user_message)

            # Notify client that we received the message
            await chat_service.send_message_to_user(
                chat_id, ChatEvent.MESSAGE_RECEIVED, "Processing your message..."
            )

            # Stream AI response
            full_response = ""

            async for token in stream_chat_response(chat_id, user_message.message):
                # Send each token individually
                # await websocket.send_text(f"{ChatEvent.SEND_MESSAGE}: {token}")
                await chat_service.send_message_to_user(
                    chat_id, ChatEvent.SEND_MESSAGE, token
                )
                full_response += token

            # Signal end of stream
            await chat_service.send_message_to_user(
                chat_id, ChatEvent.SEND_MESSAGE, "<EOS>"
            )

            # Save the complete AI response to chat history
            assistant_message = ChatMessage(
                role="assistant",
                message=full_response,
            )
            chat_service.save_message(chat_id, assistant_message)

    except WebSocketDisconnect:
        chat_service.disconnect_user(chat_id)
        print(f"Client {chat_id} disconnected.")
    except Exception as e:
        # Send error to client
        error_message = f"Error: {str(e)}"
        await chat_service.send_message_to_user(chat_id, ChatEvent.ERROR, error_message)
        # Then disconnect
        chat_service.disconnect_user(chat_id)
        traceback.print_exc()


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
