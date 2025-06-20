import json
import random
import traceback
import uuid
from typing import List

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
from server.services.langgraph_service import (
    stream_chat_response,
    stream_simple_chat_response,
)

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

            # Process and stream AI response
            try:
                # TODO Implement Generate Debate response
                roles = ["friend", "moderator"]
                selected_role = random.choice(roles)

                # Simulate a debate response
                debate_response = f"[{selected_role.upper()}] I understand your point, but let me offer a different perspective..."

                # Send the response
                # TODO Stream message to client
                await chat_service.send_message_to_user(
                    debate_chat_id,
                    ChatEvent.SEND_MESSAGE.value,
                    debate_response,
                )

                # Save the response to chat history
                assistant_message = ChatMessage(
                    role=selected_role,
                    message=debate_response,
                )
                chat_service.save_message(debate_chat_id, assistant_message)

                # Signal end of stream
                await chat_service.send_message_to_user(
                    debate_chat_id,
                    ChatEvent.SEND_MESSAGE.value,
                    ChatInfo.END_OF_STREAM.value,
                )

            except Exception as e:
                error_msg = f"Failed to generate response: {str(e)}"
                await chat_service.send_message_to_user(
                    debate_chat_id, ChatEvent.ERROR.value, error_msg
                )
                print(
                    f"AI response generation error for debate chat {debate_chat_id}: {str(e)}"
                )
                traceback.print_exc()

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


@chat_router.post("/chat/{chat_id}")
async def chat_http(chat_id: str, message_data: ChatMessage):
    """
    HTTP POST endpoint for sending a chat message and getting AI response

    Args:
        chat_id: The chat session ID
        message_data: ChatMessage object with role and message content

    Returns:
        JSON response with the AI's reply
    """
    if not chat_id or not chat_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid chat_id"
        )

    if not message_data.message or not message_data.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Empty message not allowed"
        )

    if message_data.role != "user":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role: Only 'user' role is allowed",
        )

    try:
        # Save user message to chat history
        chat_service.save_message(chat_id, message_data)

        # Generate AI response using the same method as WebSocket
        full_response = await chat_service.generate_and_stream_message_to_user(
            chat_id,
            message_data.message,
            stream_chat_response,
        )

        # Save AI response to chat history
        if full_response:
            assistant_message = ChatMessage(
                role="assistant",
                message=full_response,
            )
            chat_service.save_message(chat_id, assistant_message)

            return {
                "ai_response": full_response,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate AI response",
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat message: {str(e)}",
        )


@chat_router.post("/simple-chat/{chat_id}")
async def simple_chat_http(chat_id: str, message_data: ChatMessage):
    """
    HTTP POST endpoint for sending a chat message and getting AI response

    Args:
        chat_id: The chat session ID
        message_data: ChatMessage object with role and message content

    Returns:
        JSON response with the AI's reply
    """
    if not chat_id or not chat_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid chat_id"
        )

    if not message_data.message or not message_data.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Empty message not allowed"
        )

    if message_data.role != "user":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role: Only 'user' role is allowed",
        )

    try:
        # Save user message to chat history
        chat_service.save_message(chat_id, message_data)

        # Generate AI response using the same method as WebSocket
        full_response = await chat_service.generate_and_stream_message_to_user(
            chat_id,
            message_data.message,
            stream_simple_chat_response,
        )

        # Save AI response to chat history
        if full_response:
            assistant_message = ChatMessage(
                role="assistant",
                message=full_response,
            )
            chat_service.save_message(chat_id, assistant_message)

            return {
                "ai_response": full_response,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate AI response",
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat message: {str(e)}",
        )


@chat_router.post("/set-chat-history/{chat_id}")
async def set_chat_history(chat_id: str, messages: List[ChatMessage]):
    """
    Set the complete chat history for a given chat ID

    Args:
        chat_id: The chat session ID
        messages: List of ChatMessage objects with role and message content

    Returns:
        JSON response confirming the chat history was set
    """
    if not chat_id or not chat_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid chat_id"
        )

    if not messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Messages list cannot be empty",
        )

    try:
        # Clear existing chat history by replacing it with a new empty one
        from server.repositories.chat_store import create_or_replace_chat_history

        # Create a new empty chat history, replacing any existing one
        create_or_replace_chat_history(chat_id)

        # Save each message to the new chat history
        for message in messages:
            if not message.message or not message.message.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Empty message not allowed in chat history",
                )

            chat_service.save_message(chat_id, message)

        return {
            "status": "success",
            "message": f"Chat history set successfully for chat_id: {chat_id}",
            "messages_count": len(messages),
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set chat history: {str(e)}",
        )


if __name__ == "__main__":
    # Check if set_chat_history works

    import asyncio

    from server.repositories.chat_store import create_chat_history

    chat_id = "test_chat_123"
    send_messages = [
        ChatMessage(role="user", message="Hello, this is a test message."),
        ChatMessage(role="assistant", message="Hello! How can I assist you today?"),
    ]

    try:
        create_chat_history(chat_id)
        for msg in send_messages:
            chat_service.save_message(chat_id, msg)
    except Exception as e:
        print(f"Failed to set chat history: {str(e)}")

    chat_history = get_chat_history(chat_id)
    if chat_history:
        saved_messages = chat_history.messages
        for m1, m2 in zip(send_messages, saved_messages):
            assert m1.role == m2.role, f"Role mismatch: {m1.role} != {m2.role}"
            assert (
                m1.message == m2.message
            ), f"Message mismatch: {m1.message} != {m2.message}"

    else:
        print(f"No chat history found for chat_id: {chat_id}")
        raise Exception(f"Chat history not found for chat_id: {chat_id}")

    # Now check if set_chat_history works
    new_messages = [
        ChatMessage(
            role="user", message="This is a new message after setting history."
        ),
        ChatMessage(
            role="assistant", message="This is the AI's response to the new message"
        ),
    ]

    # set_chat_history(chat_id, new_messages)
    asyncio.run(set_chat_history(chat_id, new_messages))

    # Verify the new messages were set correctly
    chat_history = get_chat_history(chat_id)
    if chat_history:
        saved_messages = chat_history.messages
        for m1, m2 in zip(new_messages, saved_messages[-len(new_messages) :]):
            assert m1.role == m2.role, f"Role mismatch: {m1.role} != {m2.role}"
            assert (
                m1.message == m2.message
            ), f"Message mismatch: {m1.message} != {m2.message}"
    else:
        print(f"No chat history found for chat_id: {chat_id}")
        raise Exception(f"Chat history not found for chat_id: {chat_id}")
