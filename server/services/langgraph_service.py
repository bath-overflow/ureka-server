import traceback
from typing import Annotated, AsyncGenerator, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode

from server.repositories.vector_store import vector_store
from server.services.llm import llm
from server.services.prompt import PromptService


# --- State Definition ---
class State(TypedDict):
    user_query: Optional[str]
    initial_answer: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]
    collection_name: Optional[str]


class LangGraphService:
    def __init__(self):
        self.graph = self._build_graph()
        self.prompt_service = PromptService()

    def _build_graph(self) -> StateGraph:
        """Build and compile the LangGraph."""
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("load_chat_history", self._load_chat_history)
        graph_builder.add_node("pre_retrieve", self._pre_retrieve)
        graph_builder.add_node("tools", self._create_tool_node())
        graph_builder.add_node("generate_initial_answer", self._generate_initial_answer)
        graph_builder.add_node("generate_final_answer", self._generate_final_answer)
        graph_builder.add_node("cleanup_messages", self._cleanup_messages)

        # Set entry point
        graph_builder.set_entry_point("load_chat_history")

        # Add edges
        graph_builder.add_edge("load_chat_history", "pre_retrieve")
        graph_builder.add_conditional_edges(
            "pre_retrieve",
            self._route_after_query,
            {"tools": "tools", "generate_initial_answer": "generate_initial_answer"},
        )
        graph_builder.add_edge("tools", "generate_initial_answer")
        graph_builder.add_edge("generate_initial_answer", "generate_final_answer")
        graph_builder.add_edge("generate_final_answer", "cleanup_messages")
        graph_builder.add_edge("cleanup_messages", END)

        return graph_builder.compile()

    @staticmethod
    def _create_tool_node() -> ToolNode:
        """Create a tool node for document retrieval."""

        @tool
        def retrieve(
            query: str,
            collection_name: Annotated[str, InjectedState("collection_name")],
        ):
            """Retrieves relevant documents uploaded by the user based on the query."""
            print(
                "--- Retrieving documents for query: '{query}' in collection: "
                f"'{collection_name}' ---"
            )
            retrieved_docs = vector_store.get_documents(
                collection_name=collection_name, query=query
            )
            print(
                f"--- Retrieved {len(retrieved_docs)} documents from collection "
                f"'{collection_name}' ---"
            )
            return [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in retrieved_docs
            ]

        return ToolNode([retrieve])

    def _load_chat_history(self, state: State) -> Dict[str, List[AnyMessage]]:
        """Load chat history from ChatHistory at the start of the graph."""
        print("--- Loading chat history ---")
        collection_name = state.get("collection_name")
        if not collection_name:
            print("Error: No collection name found in state for load_chat_history")
            raise ValueError("Collection name not found in state")

        # Get chat history from ChatService
        from server.routers.chat import chat_service

        chat_history = chat_service.get_history(collection_name)
        if chat_history is None:
            print(f"Error: No chat history found for collection '{collection_name}'")
            raise ValueError(
                f"No chat history found for collection '{collection_name}'"
            )

        # Add chat history messages to the state
        messages = [
            (
                HumanMessage(content=chat_msg.message)
                if chat_msg.role == "user"
                else AIMessage(content=chat_msg.message)
            )
            for chat_msg in chat_history.messages
        ]

        print(f"--- Loaded {len(messages)} messages from chat history ---")
        return {"messages": messages}

    def _pre_retrieve(self, state: State) -> Dict[str, List[AnyMessage]]:
        """
        Given the query, decide whether to retrieve documents or use LLM knowledge.
        """
        print("--- Pre-retrieve: Analyzing query ---")
        user_query = state.get("user_query")
        if not user_query:
            print("Error: No user query found in state for pre_retrieve")
            raise ValueError("User query not found in state")

        collection_name = state.get("collection_name")
        if not collection_name:
            print("Error: collection_name not found in state for pre_retrieve")
            raise ValueError("Collection name not found in state")

        llm_with_tools = llm.bind_tools(
            [self._create_tool_node().tools_by_name["retrieve"]]
        )

        prompt = (
            "You are an assistant that decides whether external knowledge is needed to",
            "answer a question.\n",
            "Given the following question, decide if you need to search for additional",
            "context from documents.\n",
            "If document retrieval is needed, call the retrieve tool with a",
            "well-formed search query.\n\n",
            f"Question: {user_query}\n\n",
            "Think carefully - if this is a question about specific information that",
            "might be in the uploaded documents,",
            "use the retrieve tool. If it's general knowledge or doesn't require",
            "specific document content, answer simply 'PASS'.\n",
        )

        response = llm_with_tools.invoke([HumanMessage(content=prompt)])
        return {"messages": [response]}

    def _generate_initial_answer(self, state: State) -> Dict[str, str]:
        """Generate the initial concise answer to the user's query."""
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
        prompt_template = self.prompt_service.get_prompt("genai_query_prompt.txt")
        msg_content = prompt_template + f"\n\n{user_query}\n\n{docs_content}"
        response = llm.invoke([HumanMessage(content=msg_content)])
        answer = response.content

        print("--- Initial Answer Generated ---")
        return {"initial_answer": answer}

    def _generate_final_answer(self, state: State) -> Dict[str, List[AIMessage]]:
        """
        Generate the final response using teacher prompt, initial answer, and retrieval
        results.
        """
        print("--- Teacher: Generating Response with Context ---")
        user_query = state.get("user_query")
        initial_answer = state.get("initial_answer")

        if not user_query or not initial_answer:
            print(
                "Error: missing user_query or initial_answer in generate_final_answer"
            )
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

        template = self.prompt_service.get_prompt("teacher_prompt.txt")
        template += f"<query>{user_query}</query>\n"
        template += f"<answer>{initial_answer}</answer>\n"
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

    def _cleanup_messages(self, state: State) -> Dict[str, List[RemoveMessage]]:
        """
        Clean up the message history by removing tool calls, tool messages, and 'PASS'
        messages."""
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
                isinstance(message, AIMessage)
                and message.content.strip().lower() == "pass"
            ):
                message_to_remove_ids.append(message.id)

        # Create RemoveMessage objects for each index to remove
        remove_messages = [RemoveMessage(id=id) for id in message_to_remove_ids]
        return {"messages": remove_messages}

    @staticmethod
    def _route_after_query(state: State) -> Literal["tools", "generate_initial_answer"]:
        """Route based on whether the AI message has tool calls."""
        messages = state.get("messages", [])
        if not messages:
            print(
                "--- Routing Warning: No messages found in state "
                "for route_after_query ---"
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
        print(
            "--- Routing: Pre-retrieve -> Generate Initial Answer (No tool calls) ---"
        )
        return "generate_initial_answer"

    async def stream_chat_response(
        self, chat_id: str, user_message: str
    ) -> AsyncGenerator[str, None]:
        """
        Stream responses from the AI.

        Args:
            chat_id: Identifier for the chat/collection
            user_message: The message from the user

        Yields:
            Tokens from the AI response as they're generated
        """
        try:
            # Prepare input for the graph
            stream_input = {
                "user_query": user_message,
                "collection_name": chat_id,
                "messages": [HumanMessage(content=user_message)],
            }

            # Stream the response from the graph with token-by-token streaming
            current_node = None
            async for chunk, metadata in self.graph.astream(
                stream_input,
                stream_mode="messages",  # Stream token by token
            ):
                current_node = metadata["langgraph_node"]

                # Only stream tokens from the generate_final_answer node
                if current_node == "generate_final_answer" and isinstance(
                    chunk, AIMessage
                ):
                    # For token-by-token streaming, the content is the new token
                    yield chunk.content

        except Exception as e:
            error_msg = f"Error in stream_chat_response: {str(e)}"
            print(f"Error processing message for chat {chat_id}: {e}")
            traceback.print_exc()
            yield f"[ERROR: {error_msg}]"


# Create a singleton instance
lang_graph_service = LangGraphService()


async def stream_chat_response(
    chat_id: str, user_message: str
) -> AsyncGenerator[str, None]:
    """
    Stream chat responses from the LangGraphService.
    Args:
        chat_id: Identifier for the chat/collection
        user_message: The message from the user
    Yields:
        Tokens from the AI response as they're generated
    """
    async for token in lang_graph_service.stream_chat_response(chat_id, user_message):
        yield token
