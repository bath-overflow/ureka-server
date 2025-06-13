import traceback
from typing import Annotated, AsyncGenerator, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
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

    def _build_graph(self) -> CompiledStateGraph:
        """Build and compile the LangGraph."""
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("load_chat_history", self._load_chat_history)
        graph_builder.add_node("pre_retrieve", self._pre_retrieve)
        graph_builder.add_node("tools", self._create_tool_node())
        graph_builder.add_node("generate_initial_answer", self._generate_initial_answer)
        graph_builder.add_node("generate_final_answer", self._generate_final_answer)

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
        graph_builder.add_edge("generate_final_answer", END)

        return graph_builder.compile()

    @staticmethod
    def _create_tool_node() -> ToolNode:
        """Create a tool node for document retrieval."""

        @tool(response_format="content_and_artifact")
        def retrieve(
            query: str,
            collection_name: Annotated[str, InjectedState("collection_name")],
        ):
            """
            Retrieves relevant documents uploaded by the user based on the query.

            Args:
                query: The query to search for relevant documents.
                collection_name: The name of the collection to search in.
            """
            print(
                f"--- Retrieving documents for query: '{query}' in collection: "
                f"'{collection_name}' ---"
            )
            retrieved_docs = vector_store.get_documents(
                collection_name=collection_name, query=query
            )
            print(
                f"--- Retrieved {len(retrieved_docs)} documents from collection "
                f"'{collection_name}' ---"
            )

            content = ""
            if retrieved_docs:
                content = "\n\n".join(
                    f"<source>{doc.metadata.get('source', 'N/A')}</source>"
                    + f"<content>{doc.page_content}</content>"
                    for doc in retrieved_docs
                )

            return content, retrieved_docs

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
        Given the full conversation history, decide whether to retrieve documents
        to answer the latest user query. If so, generate a history-informed
        search query.
        """
        all_messages = state.get("messages", [])
        if not all_messages:
            print("Error: No messages found in state for pre_retrieve")
            raise ValueError("Messages not found in state for pre_retrieve")

        collection_name = state.get("collection_name")
        if not collection_name:
            print("Error: collection_name not found in state for pre_retrieve")
            raise ValueError("Collection name not found in state")

        llm_with_tools = llm.bind_tools(
            [self._create_tool_node().tools_by_name["retrieve"]]
        )

        instruction = self.prompt_service.get_prompt("pre_retrieve_prompt.txt")

        history = ""
        for message in all_messages:
            if isinstance(message, HumanMessage):
                history += f"[Student]: {message.content}\n"
            elif (
                isinstance(message, AIMessage)
                and not message.tool_calls
                and not message.content.strip().lower() == "pass"
            ):
                history += f"[Teacher]: {message.content}\n"

        full_prompt = f"{instruction}\n{history}\n"

        print(f"--- Pre-retrieve: Analyzing {len(all_messages)} messages. ---")

        # Invoke the LLM with the instruction and the full conversation history
        response = llm_with_tools.invoke([HumanMessage(content=full_prompt)])

        # The response is an AIMessage, which might contain tool calls
        # or the content "PASS".
        return {"messages": [response]}

    def _generate_initial_answer(self, state: State) -> Dict[str, str]:
        """
        Generate the initial concise answer to the user's query,
        using conversation history.
        """
        user_query = state.get("user_query")
        if not user_query:
            print("Error: user_query not found in state for generate_initial_answer")
            raise ValueError("User query not found in state")

        all_messages = state.get("messages", [])
        if not all_messages:
            print("Error: No messages found in state for generate_initial_answer")
            raise ValueError("Messages not found in state")

        history = ""
        for message in all_messages:
            if isinstance(message, HumanMessage):
                history += f"[Student]: {message.content}\n"
            elif (
                isinstance(message, AIMessage)
                and not message.tool_calls
                and not message.content.strip().lower() == "pass"
            ):
                history += f"[Teacher]: {message.content}\n"

        # Check if there are tool messages (retrieval results)
        recent_tool_messages = []
        for message in reversed(all_messages):  # Check all_messages for ToolMessages
            if isinstance(message, ToolMessage):
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format retrieved documents if any
        docs_content = ""
        if tool_messages:
            docs_content = "\n".join(tool_msg.content for tool_msg in tool_messages)
            docs_content = f"<reference>{docs_content}</reference>\n"

        # Generate initial answer
        instruction = self.prompt_service.get_prompt("genai_query_prompt.txt")

        full_prompt = f"{instruction}\n{docs_content}\n{history}\n"
        response = llm.invoke([HumanMessage(content=full_prompt)])
        answer = response.content

        print(f"--- Initial Answer Generated (for query: '{user_query}') ---")
        return {"initial_answer": answer}

    def _generate_final_answer(self, state: State) -> Dict[str, List[AIMessage]]:
        """
        Generate the final response using teacher prompt, initial answer, and retrieval
        results.
        """
        print("--- Teacher: Generating Response with Context ---")
        user_query = state.get("user_query")
        if not user_query:
            print("Error: user_query not found in state for generate_final_answer")
            raise ValueError("User query not found in state")
        initial_answer = state.get("initial_answer")
        if not initial_answer:
            print("Error: initial_answer not found in state for generate_final_answer")
            raise ValueError("Initial answer not found in state")
        all_messages = state.get("messages", [])
        if not all_messages:
            print("Error: No messages found in state for generate_final_answer")
            raise ValueError("Messages not found in state")

        recent_tool_messages = []
        for message in reversed(all_messages):
            if isinstance(message, ToolMessage):
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format retrieved documents if any
        reference = ""
        if tool_messages:
            reference = "\n".join(tool_msg.content for tool_msg in tool_messages)

        dialogue_history = ""
        for message in all_messages:
            if isinstance(message, HumanMessage):
                dialogue_history += f"[Student]: {message.content}\n"
            elif (
                isinstance(message, AIMessage)
                and not message.tool_calls
                and not message.content.strip().lower() == "pass"
            ):
                dialogue_history += f"[Teacher]: {message.content}\n"

        instruction = self.prompt_service.get_prompt("teacher_prompt.txt")

        prompt = instruction.format(
            dialogue_history=dialogue_history,
            reference=reference,
            key_ideas=initial_answer,
        )

        response = llm.invoke([HumanMessage(prompt)])

        return {"messages": [response]}

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
            user_message: The message from the user. This is already written in the
            chat history.

        Yields:
            Tokens from the AI response as they're generated
        """
        try:
            # Prepare input for the graph
            stream_input = {
                "user_query": user_message,
                "collection_name": chat_id,
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


class SimpleChatGraphService(LangGraphService):
    """
    A simplified version of LangGraphService for basic chat functionality.
    This service does not include document retrieval or complex routing.
    Thus the graph is as follows:
    load chat history -> generate response -> END
    """

    def _build_graph(self) -> CompiledStateGraph:
        """Build and compile the simplified LangGraph."""
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("load_chat_history", self._load_chat_history)
        graph_builder.add_node("generate_final_answer", self._generate_response)

        # Set entry point
        graph_builder.set_entry_point("load_chat_history")

        # Add edges
        graph_builder.add_edge("load_chat_history", "generate_final_answer")
        graph_builder.add_edge("generate_final_answer", END)

        return graph_builder.compile()

    def _generate_response(self, state: State) -> Dict[str, List[AIMessage]]:
        """
        Generate a response based on the chat history.
        This is a simplified version that does not use tools or retrieval.
        """
        all_messages = state.get("messages", [])
        if not all_messages:
            print("Error: No messages found in state for generate_response")
            raise ValueError("Messages not found in state for generate_response")

        history = ""
        for message in all_messages:
            if isinstance(message, HumanMessage):
                history += f"[User]: {message.content}\n"
            elif (
                isinstance(message, AIMessage)
                and not message.tool_calls
                and not message.content.strip().lower() == "pass"
            ):
                history += f"[Teacher]: {message.content}\n"

        instruction = self.prompt_service.get_prompt("simple_teacher_prompt.txt")

        prompt = instruction.format(dialogue_history=history)

        response = llm.invoke([HumanMessage(content=prompt)])

        return {"messages": [response]}


# Create a singleton instance for the simplified chat service
simple_chat_graph_service = SimpleChatGraphService()


async def stream_simple_chat_response(
    chat_id: str, user_message: str
) -> AsyncGenerator[str, None]:
    """
    Stream chat responses from the SimpleChatGraphService.
    Args:
        chat_id: Identifier for the chat/collection
        user_message: The message from the user
    Yields:
        Tokens from the AI response as they're generated
    """
    async for token in simple_chat_graph_service.stream_chat_response(
        chat_id, user_message
    ):
        yield token
