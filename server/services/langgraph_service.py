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
initial_answer_prompt = load_prompt("genai_query_prompt.txt")


# --- State Definition ---
class State(TypedDict):
    user_query: str | None
    initial_answer: str | None
    messages: Annotated[list[AnyMessage], add_messages]
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
def pre_retrieve(state: State):
    """Given the query, decides whether to retrieve documents or use LLM knowledge."""
    print("--- Pre-retrieve: Analyzing query ---")
    user_query = state.get("user_query")
    if not user_query:
        print("Error: No user query found in state for pre_retrieve")
        raise ValueError("User query not found in state")

    collection_name = state.get("collection_name")
    if not collection_name:
        print("Error: collection_name not found in state for pre_retrieve")
        raise ValueError("Collection name not found in state")

    llm_with_tools = llm.bind_tools([retrieve])

    prompt = (
        "You are an assistant that decides whether external knowledge is needed to",
        "answer a question.\n",
        "Given the following question, decide if you need to search for additional",
        "context from documents.\n",
        "If document retrieval is needed, call the retrieve tool with a well-formed",
        "search query.\n\n",
        f"Question: {user_query}\n\n",
        "Think carefully - if this is a question about specific information that might",
        "be in the uploaded documents,",
        "use the retrieve tool. If it's general knowledge or doesn't require",
        "specific document content, answer simply 'PASS'.\n",
    )

    response = llm_with_tools.invoke([HumanMessage(content=prompt)])

    return {"messages": [response]}


def generate_initial_answer(state: State):
    """Generates the initial concise answer to the user's query."""
    user_query = state.get("user_query")
    if not user_query:
        print("Error: user_query not found in state for generate_initial_answer")
        raise ValueError("User query not found in state")

    # Check if there are tool messages (retrieval results)
    recent_tool_messages = []
    for message in reversed(state.get("messages", [])):
        if isinstance(message, ToolMessage):
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format retrieved documents if any
    docs_content = ""
    if tool_messages:
        docs_content = "\n\n".join(
            f"Source: {res['metadata'].get('source', 'N/A')}, "
            f"Split Index: {res['metadata'].get('split_idx', 'N/A')}\n"
            f"Content: {res['page_content']}\n"
            for msg in tool_messages
            for res in (msg.content if isinstance(msg.content, list) else [])
            if isinstance(res, dict)
        )
        docs_content = f"<reference>{docs_content}</reference>\n"

    # Generate initial answer
    msg_content = initial_answer_prompt + f"\n\n{user_query}\n\n{docs_content}"
    response = llm.invoke([HumanMessage(content=msg_content)])
    answer = response.content

    print("--- Initial Answer Generated ---")
    return {"initial_answer": answer}


# ToolNode executes the bound tool call from the AIMessage
tools_node = ToolNode([retrieve])


def generate_final_answer(state: State):
    """Generates the final response using teacher prompt, initial answer,
    and retrieval results."""
    print("--- Teacher: Generating Response with Context ---")
    user_query = state.get("user_query")
    initial_answer = state.get("initial_answer")

    if not user_query or not initial_answer:
        print("Error: missing user_query or initial_answer in generate_final_answer")
        raise ValueError("Missing required state elements")

    recent_tool_messages = []
    for message in reversed(state.get("messages", [])):
        if isinstance(message, ToolMessage):
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format retrieved documents
    docs_content = ""
    if tool_messages:
        docs_content = "\n\n".join(
            f"Source: {res['metadata'].get('source', 'N/A')}, "
            f"Split Index: {res['metadata'].get('split_idx', 'N/A')}\n"
            f"Content: {res['page_content']}\n"
            for msg in tool_messages
            for res in (msg.content if isinstance(msg.content, list) else [])
            if isinstance(res, dict)
        )
        docs_content = f"<reference>{docs_content}</reference>\n"

    template = teacher_prompt
    template += f"<query>{state.get('initial_query')}</query>\n"
    template += f"<answer>{state.get('initial_answer')}</answer>\n"
    template += docs_content

    history = ""
    messages_for_history = [
        m for m in state.get("messages", []) if not isinstance(m, ToolMessage)
    ]
    for message in messages_for_history:
        if isinstance(message, HumanMessage):
            history += f"[Student]: {message.content}\n"
        elif (
            isinstance(message, AIMessage)
            and not message.tool_calls
            and not message.content.strip().lower() == "pass"
        ):
            history += f"[Teacher]: {message.content}\n"

    message_content = template + history + "[Teacher]: "

    response = llm.invoke([HumanMessage(message_content)])

    student_part_start_idx = response.content.find("[Student]")
    if student_part_start_idx != -1:
        response.content = response.content[:student_part_start_idx].strip()

    return {"messages": [response]}


def cleanup_messages(state: State):
    """Cleans up the message history by removing tool calls, tool messages,
    and 'PASS' messages."""
    print("--- Cleaning up message history ---")

    messages = state.get("messages", [])
    message_to_remove_ids = []

    for message in messages:
        # Remove AI messages with tool calls
        if isinstance(message, AIMessage) and message.tool_calls:
            message_to_remove_ids.append(message.id)

        # Remove Tool Messages
        elif isinstance(message, ToolMessage):
            message_to_remove_ids.append(message.id)

        # Remove AI messages that just say "PASS"
        elif (
            isinstance(message, AIMessage) and message.content.strip().lower() == "pass"
        ):
            message_to_remove_ids.append(message.id)

    # Create RemoveMessage objects for each index to remove
    remove_messages = [RemoveMessage(id=id) for id in message_to_remove_ids]

    return {"messages": remove_messages}


# --- Routing Functions ---
def route_after_query(state: State) -> Literal["tools", "generate_initial_answer"]:
    messages = state.get("messages", [])
    if not messages:
        print(
            "--- Routing Warning: No messages found in state for route_after_query ---"
        )
        return "generate_initial_answer"

    ai_message = messages[-1]
    # Check if the last message is an AIMessage and has tool_calls
    if isinstance(ai_message, AIMessage) and ai_message.tool_calls:
        print(
            "--- Routing: "
            f"Pre-retrieve -> Tools (Tool calls: {ai_message.tool_calls}) ---"
        )
        return "tools"
    print("--- Routing: Pre-retrieve -> Generate Initial Answer (No tool calls) ---")
    return "generate_initial_answer"


# --- Graph Definition ---
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("pre_retrieve", pre_retrieve)
graph_builder.add_node("tools", tools_node)
graph_builder.add_node("generate_initial_answer", generate_initial_answer)
graph_builder.add_node("generate_final_answer", generate_final_answer)
graph_builder.add_node("cleanup_messages", cleanup_messages)

# Set entry point
graph_builder.set_entry_point("pre_retrieve")

# Add edges
graph_builder.add_conditional_edges(
    "pre_retrieve",
    route_after_query,
    {"tools": "tools", "generate_initial_answer": "generate_initial_answer"},
)
graph_builder.add_edge("tools", "generate_initial_answer")
graph_builder.add_edge("generate_initial_answer", "generate_final_answer")
graph_builder.add_edge("generate_final_answer", "cleanup_messages")
graph_builder.add_edge("cleanup_messages", END)

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
