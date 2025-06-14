import json
import traceback
import uuid

from fastapi import (
    APIRouter,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import ValidationError

from server.models.chat_model import ChatHistoryResponse, ChatMessage
from server.services.chat import ChatEvent, ChatInfo
from server.services.chat import service as chat_service
from server.services.debate_graph_service import stream_debate_response
from server.services.langgraph_service import stream_chat_response

chat_router = APIRouter()


@chat_router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket, chat_id: str = None):
    """
    WebSocket connection handler with streaming AI responses
    """
    connection_established = False

    try:
        if chat_id is not None and chat_id.strip() == "":
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION, reason="Invalid chat_id"
            )
            return

        is_new_chat = chat_id is None
        if is_new_chat:
            chat_id = uuid.uuid4().hex

        await chat_service.connect_user(chat_id, websocket)
        connection_established = True

        await chat_service.send_message_to_user(
            chat_id,
            ChatEvent.CONNECTED.value,
            ChatInfo.CONNECTED.value,
        )
        if is_new_chat:
            await chat_service.send_message_to_user(
                chat_id,
                ChatEvent.CONNECTED.value,
                f"Your chat_id is: {chat_id}",
            )
        while True:
            data = await websocket.receive_text()
            # Parse incoming message
            try:
                data_dict = json.loads(data)
                user_message = ChatMessage.model_validate(data_dict)
            except json.JSONDecodeError:
                await chat_service.send_message_to_user(
                    chat_id,
                    ChatEvent.ERROR.value,
                    "Invalid message format: Expected JSON",
                )
                continue
            except ValidationError:
                await chat_service.send_message_to_user(
                    chat_id, ChatEvent.ERROR.value, "Invalid message structure"
                )
                continue

            if user_message.message.strip() == "":
                await chat_service.send_message_to_user(
                    chat_id, ChatEvent.ERROR.value, "Empty message not allowed"
                )
                continue
            if user_message.role != "user":
                await chat_service.send_message_to_user(
                    chat_id,
                    ChatEvent.ERROR.value,
                    "Invalid role: Only 'user' role is allowed",
                )
                continue

            try:
                chat_service.save_message(chat_id, user_message)
            except Exception as e:
                await chat_service.send_message_to_user(
                    chat_id,
                    ChatEvent.ERROR.value,
                    f"Failed to save message: {str(e)}",
                )
                continue

            # Notify client that we received the message
            await chat_service.send_message_to_user(
                chat_id,
                ChatEvent.MESSAGE_RECEIVED.value,
                "Processing your message...",
            )

            # Process and stream AI response
            try:
                full_response = await chat_service.generate_and_stream_message_to_user(
                    chat_id,
                    user_message.message,
                    stream_chat_response,
                )

                # Save the complete AI response to chat history
                if full_response:  # Only save if we have a valid response
                    assistant_message = ChatMessage(
                        role="assistant",
                        message=full_response,
                    )
                    chat_service.save_message(chat_id, assistant_message)

            except Exception as e:
                error_msg = f"Failed to generate response: {str(e)}"
                await chat_service.send_message_to_user(
                    chat_id, ChatEvent.ERROR.value, error_msg
                )
                print(f"AI response generation error for chat {chat_id}: {str(e)}")
                traceback.print_exc()

    except WebSocketDisconnect:
        if connection_established:
            chat_service.disconnect_user(chat_id)
            print(f"Client {chat_id} disconnected.")
    except Exception as e:
        # Unexpected error
        print(f"Unexpected error in WebSocket handler for chat {chat_id}: {str(e)}")
        traceback.print_exc()

        # Try to send error to client if connection is still valid
        if connection_established:
            try:
                error_message = f"Server error: {str(e)}"
                await chat_service.send_message_to_user(
                    chat_id, ChatEvent.ERROR.value, error_message
                )
            except Exception as e:
                print(f"Failed to send error message to client: {str(e)}")

            # Ensure cleanup
            chat_service.disconnect_user(chat_id)


@chat_router.websocket("/ws/debate/{chat_id}")
async def debate_websocket(websocket: WebSocket, chat_id: str):
    """
    WebSocket connection handler for debate mode
    that creates a new chat branch from an existing chat
    """
    if not chat_id or not chat_id.strip():
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION, reason="Invalid chat_id"
        )
        return

    # Create a new chat ID for the debate
    debate_chat_id = f"{chat_id}_debate_{uuid.uuid4().hex}"
    connection_established = False

    try:
        # Copy existing chat history to new debate chat
        existing_history = chat_service.get_history(chat_id)
        if existing_history:
            for message in existing_history.messages:
                chat_service.save_message(debate_chat_id, message)

        await chat_service.connect_user(debate_chat_id, websocket)
        connection_established = True

        await chat_service.send_message_to_user(
            debate_chat_id,
            ChatEvent.CONNECTED.value,
            ChatInfo.CONNECTED.value,
        )

        while True:
            user_message = ChatMessage(role="user", message="dummy message")

            # Process and stream AI response
            try:
                # Use debate_graph_service to generate and stream responses
                full_response = await chat_service.generate_and_stream_message_to_user(
                    debate_chat_id,
                    user_message.message,
                    stream_debate_response,
                )

                # Save the complete response to chat history
                if full_response:
                    # full_response format: "[ROLE] message"
                    parts = full_response.split(" ", 1)
                    parts[0] = parts[0].strip("[]")  # Remove brackets if present

                    if len(parts) > 1:
                        role = parts[0].lower()  # Convert "FRIEND" to "friend"
                        message = parts[1].strip()
                    else:
                        # Warning: Unexpected format, handle gracefully
                        role = "assistant"
                        message = full_response

                    debate_message = ChatMessage(
                        role=role,
                        message=message,
                    )
                    chat_service.save_message(debate_chat_id, debate_message)

            except Exception as e:
                error_msg = f"Failed to generate response: {str(e)}"
                await chat_service.send_message_to_user(
                    debate_chat_id, ChatEvent.ERROR.value, error_msg
                )
                print(
                    f"AI response generation error for debate chat {debate_chat_id}: {str(e)}"
                )
                traceback.print_exc()

            data = await websocket.receive_text()
            # Parse incoming message
            try:
                data_dict = json.loads(data)
                user_message = ChatMessage.model_validate(data_dict)
            except json.JSONDecodeError:
                await chat_service.send_message_to_user(
                    debate_chat_id,
                    ChatEvent.ERROR.value,
                    "Invalid message format: Expected JSON",
                )
                continue
            except ValidationError:
                await chat_service.send_message_to_user(
                    debate_chat_id,
                    ChatEvent.ERROR.value,
                    "Invalid message structure",
                )
                continue

            if user_message.message.strip() == "":
                await chat_service.send_message_to_user(
                    debate_chat_id,
                    ChatEvent.ERROR.value,
                    "Empty message not allowed",
                )
                continue
            if user_message.role != "user":
                await chat_service.send_message_to_user(
                    debate_chat_id,
                    ChatEvent.ERROR.value,
                    "Invalid role: Only 'user' role is allowed",
                )
                continue

            try:
                chat_service.save_message(debate_chat_id, user_message)
            except Exception as e:
                await chat_service.send_message_to_user(
                    debate_chat_id,
                    ChatEvent.ERROR.value,
                    f"Failed to save message: {str(e)}",
                )
                continue

            # Notify client that we received the message
            await chat_service.send_message_to_user(
                debate_chat_id,
                ChatEvent.MESSAGE_RECEIVED.value,
                "Processing your message...",
            )
    except WebSocketDisconnect:
        if connection_established:
            chat_service.disconnect_user(debate_chat_id)
            print(f"Client {debate_chat_id} disconnected from debate.")
    except Exception as e:
        # Unexpected error
        print(
            f"Unexpected error in debate WebSocket handler for chat {debate_chat_id}: {str(e)}"
        )
        traceback.print_exc()

        # Try to send error to client if connection is still valid
        if connection_established:
            try:
                error_message = f"Server error: {str(e)}"
                await chat_service.send_message_to_user(
                    debate_chat_id, ChatEvent.ERROR.value, error_message
                )
            except Exception as e:
                print(f"Failed to send error message to client: {str(e)}")

            # Ensure cleanup
            chat_service.disconnect_user(debate_chat_id)


@chat_router.get("/chat/{chat_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(chat_id: str):
    """
    특정 채팅방의 메시지 히스토리 가져오기
    """
    if not chat_id or not chat_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid chat_id"
        )

    try:
        chat_history = chat_service.get_history(chat_id)
        if chat_history:
            return chat_history
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chat history: {str(e)}",
        )
