# from pprint import pprint
import traceback

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from server.services.langgraph_service import (
    ChatMessageRequest,
    ChatResponse,
    graph,
    load_and_process_pdf,
)

chat_router = APIRouter()


@chat_router.post("/{project_id}/resources")
async def upload_resources_for_chat(
    project_id: str,
    file: UploadFile,
):
    """
    Initializes a chat session by processing an uploaded PDF.
    The PDF content will be associated with the project_id.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    try:
        file_content = await file.read()

        # Save the file to local storage (for now)
        file_name = f"{project_id}_{file.filename}"
        with open(file_name, "wb") as f:
            f.write(file_content)

        # Use project_id as the collection name in the vector store
        load_and_process_pdf(
            collection_name=project_id, file_name=file_name or "uploaded.pdf"
        )
        return {"message": f"PDF processed successfully for thread {project_id}"}
    except Exception as e:
        print(f"Error processing PDF for thread {project_id}: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")


@chat_router.post("/{project_id}/message", response_model=ChatResponse)
async def handle_chat_message(
    project_id: str,
    request: ChatMessageRequest,
):
    """
    Handles a user message within a chat thread.
    """

    config = {"configurable": {"thread_id": project_id}}

    try:
        stream_input = {
            "user_query": request.message,
            "collection_name": project_id,
            "messages": [HumanMessage(content=request.message)],
        }

        final_state = None
        async for event in graph.astream(
            stream_input,
            config=config,
            stream_mode="values",  # Get the full state values at each step
        ):
            # Keep track of the latest state values
            final_state = event
            # Print the current state for debugging
            # print("State:")
            # pprint(event)
            # print("======================", flush=True)

        if final_state and final_state.get("messages"):
            last_message = final_state["messages"][-1]
            if isinstance(last_message, AIMessage):
                # Check if validation failed indicated by valid=False in state
                if final_state.get("valid") is False:
                    return ChatResponse(error="Validation failed.", state=final_state)
                else:
                    return ChatResponse(
                        response=last_message.content, state=final_state
                    )
            else:
                # This might happen if the graph ends unexpectedly after a tool call
                # or validation failure
                print(
                    f"Thread {project_id}: Graph ended with non-AI message:",
                    f"{last_message}",
                )
                return ChatResponse(
                    error="An internal error occurred while processing the response.",
                    state=final_state,
                )
        else:
            # Handle cases where the graph didn't produce a final message
            print(
                f"Thread {project_id}: Graph execution did not result in a final",
                "message state.",
            )
            return ChatResponse(
                error="Could not generate a response.", state=final_state
            )

    except Exception as e:
        print(f"Error handling message for thread {project_id}: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process message: {e}")


@chat_router.post("/{project_id}/message/stream")
async def handle_chat_message_stream(
    project_id: str,
    request: ChatMessageRequest,
):
    """
    Handles a user message within a chat thread and streams the response token by token.
    Uses Server-Sent Events (SSE) to stream the response.
    """

    async def generate_response():
        """Inner async generator to stream the response."""
        config = {"configurable": {"thread_id": project_id}}
        current_node = None
        final_response = ""

        try:
            stream_input = {
                "user_query": request.message,
                "collection_name": project_id,
                "messages": [HumanMessage(content=request.message)],
            }

            # Stream the response from the graph with token-by-token streaming
            async for chunk, metadata in graph.astream(
                stream_input,
                config=config,
                stream_mode="messages",  # Stream token by token
            ):
                current_node = metadata["langgraph_node"]

                # Only stream tokens from the generate_final_answer node
                if current_node == "generate_final_answer" and isinstance(
                    chunk, AIMessage
                ):
                    # For token-by-token streaming, the content is the new token
                    token = chunk.content
                    final_response += token
                    yield f"event: token\ndata: {token}\n\n"

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            print(f"Error handling message for thread {project_id}: {e}")
            traceback.print_exc()
            yield f"event: error\ndata: {error_msg}\n\n"

    # Return a streaming response
    return StreamingResponse(generate_response(), media_type="text/event-stream")
