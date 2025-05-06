import os
from typing import Annotated, Any, Dict, Literal, TypedDict

# from langgraph.checkpoint.memory import InMemorySaver
import aiosqlite
from langchain_community.document_loaders import PyPDFLoader  # Keep for BytesIO loading
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode
from pydantic import BaseModel

from server.repositories.vector_store import vector_store
from server.services.llm import llm


# --- PDF Loading and Processing ---
def load_and_process_pdf(collection_name: str, file_name: str = "uploaded.pdf"):
    """Loads PDF from bytes, splits text, creates documents,
    and adds to vector store."""
    print(f"Loading and processing PDF for collection: {collection_name}...")
    loader = PyPDFLoader(file_name)
    pages = loader.load()

    merged_content = " ".join(page.page_content for page in pages)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_text(merged_content)
    print(
        f"Split PDF into {len(all_splits)} sub-documents for collection \
            {collection_name}."
    )

    documents = [
        Document(page_content=split, metadata={"source": file_name, "split_idx": i})
        for i, split in enumerate(all_splits)
    ]

    # Use the server's vector_store instance and specify the collection name
    ids = vector_store.add_documents(
        collection_name=collection_name, documents=documents
    )
    print(
        f"Added {len(documents)} documents to vector store collection \
            '{collection_name}'. IDs: {ids}"
    )
    return ids  # Return the IDs of the added documents


# --- Load Prompts ---
def load_prompt(file_name: str) -> str:
    """Loads a prompt from a text file relative to this service file."""
    prompt_dir = os.path.join(
        os.path.dirname(__file__), "../prompts"
    )  # Assumes prompts are in server/prompts/
    file_path = os.path.join(prompt_dir, file_name)
    try:
        with open(file_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file '{file_path}' not found.")
        raise FileNotFoundError(f"Prompt file '{file_path}' not found.")


teacher_prompt = load_prompt("teacher_prompt.txt")
dean_prompt = load_prompt("dean_prompt.txt")
initial_query_prompt = load_prompt("genai_query_prompt.txt")


# --- State Definition ---
class State(TypedDict):
    initial_query: str | None
    initial_answer: str | None
    messages: Annotated[list[AnyMessage], add_messages]
    valid: bool | None
    collection_name: str | None


# --- Tool Definition ---
@tool
def retrieve(
    query: str, collection_name: Annotated[str, InjectedState("collection_name")]
):
    """Retrieves relavant documents uploaded by the user based on the query."""
    print(
        f"--- Retrieving documents for query: '{query}' in collection: \
            '{collection_name}' ---"
    )
    # Use the server's vector_store instance
    retrieved_docs = vector_store.get_documents(
        collection_name=collection_name, query=query
    )
    print(
        f"--- Retrieved {len(retrieved_docs)} documents from collection \
            '{collection_name}' ---"
    )
    return [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in retrieved_docs
    ]


# --- Node Definitions ---
# Nodes need access to collection_name,
# pass it via state or function arguments if necessary


def generate_initial_answer(state: State):
    """Generates the initial concise answer to the user's query."""
    initial_query = state.get("initial_query")
    if not initial_query:
        print("Error: initial_query not found in state for generate_initial_answer")
        raise ValueError("Initial query not found in state")

    if state.get("initial_answer") is not None:
        return state

    print("--- Generating Initial Answer ---")
    initial_msg_content = initial_query_prompt + f"\n\n{initial_query}"
    response = llm.invoke([HumanMessage(content=initial_msg_content)])
    answer = response.content
    print("--- Initial Answer Generated ---")
    return {"initial_answer": answer}


def query_or_respond(state: State):
    """Decides whether to retrieve documents or respond directly."""
    print("--- Teacher: Querying or Responding ---")
    collection_name = state.get("collection_name")
    if not collection_name:
        print("Error: collection_name not found in state for query_or_respond")
        raise ValueError("Collection name not found in state")

    llm_with_tools = llm.bind_tools([retrieve])

    template = teacher_prompt
    template += f"<query>{state.get('initial_query', 'N/A')}</query>\n"
    template += f"<answer>{state.get('initial_answer', 'N/A')}</answer>\n"

    history = ""
    for message in state.get("messages", []):
        if isinstance(message, HumanMessage):
            history += f"[Student]: {message.content}\n"
        elif isinstance(message, AIMessage) and not message.tool_calls:
            history += f"[Teacher]: {message.content}\n"

    message_content = template + history + "[Teacher]: "

    response = llm_with_tools.invoke([HumanMessage(message_content)])

    student_part_start_idx = response.content.find("[Student]")
    if student_part_start_idx != -1:
        response.content = response.content[:student_part_start_idx].strip()

    return {"messages": [response]}


# ToolNode executes the bound tool call from the AIMessage
tools_node = ToolNode([retrieve])


def generate(state: State):
    """Generates the final response incorporating retrieved documents."""
    print("--- Teacher: Generating Response with Context ---")
    collection_name = state.get(
        "collection_name"
    )  # Needed for context if prompts require it
    if not collection_name:
        print("Error: collection_name not found in state for generate")
        raise ValueError("Collection name not found in state")

    recent_tool_messages = []
    for message in reversed(state.get("messages", [])):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    docs_content = "\n\n".join(
        f"Source: {res['metadata'].get('source', 'N/A')}, \
            Split Index: {res['metadata'].get('split_idx', 'N/A')}\n\
            Content: {res['page_content']}\n"
        for msg in tool_messages
        for res in (msg.content if isinstance(msg.content, list) else [])
        if isinstance(res, dict)
    )
    docs_content = f"<reference>{docs_content}</reference>\n" if docs_content else ""

    template = teacher_prompt
    template += f"<query>{state.get('initial_query', 'N/A')}</query>\n"
    template += f"<answer>{state.get('initial_answer', 'N/A')}</answer>\n"

    history = ""
    messages_for_history = [
        m for m in state.get("messages", []) if not isinstance(m, ToolMessage)
    ]
    for message in messages_for_history:
        if isinstance(message, HumanMessage):
            history += f"[Student]: {message.content}\n"
        elif isinstance(message, AIMessage) and not message.tool_calls:
            history += f"[Teacher]: {message.content}\n"

    message_content = template + history + docs_content + "[Teacher]: "

    response = llm.invoke([HumanMessage(message_content)])

    student_part_start_idx = response.content.find("[Student]")
    if student_part_start_idx != -1:
        response.content = response.content[:student_part_start_idx].strip()

    return {"messages": [response]}


def validate_output(state: State):
    """Validates the Teacher's response using the Dean prompt."""
    print("--- Dean: Validating Teacher's Response ---")
    messages = state.get("messages", [])
    if not messages:
        print("--- Dean: No messages to validate ---")
        raise ValueError("No messages to validate")

    last_message = messages[-1]
    # Should not happen
    if not isinstance(last_message, AIMessage) or last_message.tool_calls:
        print(
            "--- Dean: Last message is not a final AI response, skipping validation ---"
        )
        raise ValueError("Last message is not a valid AI response")

    template = dean_prompt
    template += f"[Student]: {state.get('initial_query', 'N/A')}\n"

    history = ""
    for message in messages:
        if isinstance(message, HumanMessage):
            history += f"[Student]: {message.content}\n"
        elif isinstance(message, AIMessage) and not message.tool_calls:
            history += f"[Teacher]: {message.content}\n"

    message_content = template + history + "[Dean]: "

    response = llm.invoke([HumanMessage(message_content)])
    validation_result = response.content.strip().lower() == "true"
    print(f"--- Dean Validation Result: {validation_result} ---")

    if validation_result:
        return {"valid": True}
    else:
        print("--- Dean Rejected Response, Removing Last Message ---")
        return {
            "messages": [RemoveMessage(id=last_message.id)],
            "valid": False,
        }


# --- Graph Definition ---
graph_builder = StateGraph(State)

graph_builder.add_node("generate_initial_answer", generate_initial_answer)
graph_builder.add_node("query_or_respond", query_or_respond)
graph_builder.add_node("tools", tools_node)  # Use the ToolNode instance
graph_builder.add_node("generate", generate)
graph_builder.add_node("validate", validate_output)

graph_builder.set_entry_point("generate_initial_answer")


def route_after_query(state: State) -> Literal["tools", "validate"]:
    messages = state.get("messages", [])
    if not messages:
        print(
            "--- Routing Warning: No messages found in state for route_after_query ---"
        )
        return "validate"
    ai_message = messages[-1]
    # Check if the last message is an AIMessage and has tool_calls
    if isinstance(ai_message, AIMessage) and ai_message.tool_calls:
        print(f"--- Routing: Query -> Tools (Tool calls: {ai_message.tool_calls}) ---")
        return "tools"
    print("--- Routing: Query -> Validate ---")
    return "validate"


def route_after_validate(state: State):
    if state.get("valid", False):
        print("--- Routing: Validate -> END ---")
        return END
    else:
        print("--- Routing: Validate -> Query (Re-prompting) ---")
        return "query_or_respond"


graph_builder.add_edge("generate_initial_answer", "query_or_respond")
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", "validate")

graph_builder.add_conditional_edges(
    "query_or_respond", route_after_query, {"tools": "tools", "validate": "validate"}
)
graph_builder.add_conditional_edges(
    "validate", route_after_validate, {"query_or_respond": "query_or_respond", END: END}
)

# --- Checkpoint Saver ---
# Using InMemorySaver, state is lost on server restart.
# For persistence, replace with a persistent checkpointer
# (e.g., LangGraph's Redis or Postgres savers)
# memory = InMemorySaver()

conn = aiosqlite.connect("socratic_teacher.db", check_same_thread=False)
memory = AsyncSqliteSaver(conn)

# --- Compile Graph ---
graph = graph_builder.compile(checkpointer=memory)


# --- Pydantic Models for API ---
class ChatInitRequest(BaseModel):
    file_name: str | None = "uploaded.pdf"


class ChatMessageRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str | None = None
    error: str | None = None
    state: Dict[str, Any] | None = None  # Optionally return state for debugging
