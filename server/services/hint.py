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
from datetime import datetime
from server.models.chat_model import ChatMessage, ChatHistory

# 테스트용 ChatHistory 생성
test_chat_history = ChatHistory(
    id="12",
    messages=[
        ChatMessage(
            role="user",
            message="What is prompt engineering?",
            created_at=datetime.utcnow().isoformat(),
        ),
        ChatMessage(
            role="assistant",
            message="Can you describe what prompt engineering aims to achieve when working with language models?",
            created_at=datetime.utcnow().isoformat(),
        ),
        ChatMessage(
            role="user",
            message="It aims to design inputs that guide the model to produce better outputs.",
            created_at=datetime.utcnow().isoformat(),
        ),
        ChatMessage(
            role="assistant",
            message="Can you explain some techniques commonly used in prompt engineering to improve model responses?",
            created_at=datetime.utcnow().isoformat(),
        )
    ],
)


# --- State Definition ---
class State(TypedDict):
    prev_question: Optional[str]
    hint: Optional[str]
    references: Optional[List[str]]
    initial_answer: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]
    collection_name: Optional[str]


class HintService:
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
        graph_builder.add_node("generate_hint", self._generate_hint)

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
        graph_builder.add_edge("generate_initial_answer", "generate_hint")
        graph_builder.add_edge("generate_hint", END)

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
        messages, _ = self._get_messages_and_last_ai_question(collection_name)

        print(f"--- Loaded {len(messages)} messages from chat history ---")
        return {"messages": messages}

    def _pre_retrieve(self, state: State) -> Dict[str, List[AnyMessage]]:
        """
        Given the query, decide whether to retrieve documents or use LLM knowledge.
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

        instruction = self.prompt_service.get_prompt("hint_pre_retrieve_prompt.txt")

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
        """Generate the initial concise answer to previous question."""
        prev_question = state.get("prev_question")
        if not prev_question:
            print("Error: prev_question not found in state for generate_initial_answer")
            raise ValueError("Previous question not found in state")

        all_messages = state.get("messages", [])
        if not all_messages:
            print("Error: No messages found in state for generate_initial_answer")
            raise ValueError("Messages not found in state")

        # Check if there are tool messages (retrieval results)
        recent_tool_messages = []
        for message in reversed(all_messages):
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
        prompt_template = self.prompt_service.get_prompt("hint_query_prompt.txt")
        full_prompt = prompt_template + f"\n\n{prev_question}\n\n{docs_content}"
        response = llm.invoke([HumanMessage(content=full_prompt)])
        answer = response.content

        print("--- Initial Answer Generated ---")
        return {
            "initial_answer": answer,
            "references": docs_content,
        }

    def _generate_hint(self, state: State) -> Dict[str, list[str]]:
        """
        Generate the final response using hint prompt, initial answer, and retrieval
        results.
        """
        print("--- Teacher: Generating Response with Context ---")
        prev_question = state.get("prev_question")
        if not prev_question:
            print("Error: prev_question not found in state for generate_hint")
            raise ValueError("Previous question not found in state")
        
        initial_answer = state.get("initial_answer")
        if not initial_answer:
            print("Error: initial_answer not found in state for generate_hint")
            raise ValueError("Initial answer not found in state")
        
        all_messages = state.get("messages", [])
        if not all_messages:
            print("Error: No messages found in state for generate_hint")
            raise ValueError("Messages not found in state")

        docs_content = state.get("references", "")
        print(f"PREV QUESTION:\n{prev_question}")
        print(f"INITIAL ANSWER:\n{initial_answer}")
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

        instruction = self.prompt_service.get_prompt("hint_prompt.txt")

        full_prompt = "\n".join(
            [
                instruction,
                f"<answer>{initial_answer}</answer>",
                docs_content,
                history + "[Teacher]: ",
            ]
        )
        
        response = llm.invoke([HumanMessage(full_prompt)])
        
        # reference에서 source만 추출
        import re
        source_pattern = r"<source>(.*?)</source>"
        sources = re.findall(source_pattern, docs_content)
        sources = list(set(sources)) if sources else ["None"]

        return {
            "messages": [response],
            "references": sources,
        }

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

    def _get_messages_and_last_ai_question(self, chat_id: str) -> tuple[list[AnyMessage], str]:
        """
        Fetch chat history and return LangChain message list and last AI question.
        """
        from server.routers.chat import chat_service
        
        #chat_history = chat_service.get_history(chat_id)
        chat_history = test_chat_history
        
        if chat_history is None:
            raise ValueError(f"No chat history found for collection '{chat_id}'")

        # Convert to messages
        messages = [
            HumanMessage(content=msg.message) if msg.role == "user" else AIMessage(content=msg.message)
            for msg in chat_history.messages
        ]

        if not messages:
            raise ValueError("Chat history is empty")

        # Get the most recent AI message (used as prev_question)
        prev_question = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                prev_question = msg.content
                break

        if not prev_question:
            raise ValueError("No previous AI question found in chat history")

        return messages, prev_question

    async def generate_hint_response(
        self, chat_id: str
    ) -> tuple[str, list[str]]:
        """
        Generate hint responses from the AI.
        Args:
            chat_id: Identifier for the chat/collection
        Yields:
            The full generated hint response as a string.
            Refered lecture notes.
        """
        try:
            # Get previous question from llm
            messages, prev_question = self._get_messages_and_last_ai_question(chat_id)
            
            # Prepare input for the graph
            graph_input = {
                "prev_question": prev_question,
                "collection_name": chat_id,
                "messages": messages,
            }

            # Run the graph fully (non-streaming)
            final_state = await self.graph.ainvoke(graph_input)

            # Extract final message (typically from generate_hint)
            messages = final_state.get("messages", [])
            hint = "[No hint response generated.]"
            for message in reversed(messages):
                if isinstance(message, AIMessage):
                    hint = message.content.strip()

            references = final_state.get("references", [])
            if not references:
                references = ["None"]

            return hint, references

        except Exception as e:
            error_msg = f"Error in generate_chat_response: {str(e)}"
            print(f"Error processing message for chat {chat_id}: {e}")
            traceback.print_exc()
            return f"[ERROR: {error_msg}]"
