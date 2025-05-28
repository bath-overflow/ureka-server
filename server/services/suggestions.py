from typing import Dict, List, Literal, Optional, TypedDict
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from server.routers.chat import chat_service
from server.services.llm import llm
from datetime import datetime
from server.services.prompt import PromptService
from server.models.chat_model import ChatMessage, ChatHistory

# 테스트용 ChatHistory 생성
test_chat_history = ChatHistory(
    id="123",
    messages=[
        ChatMessage(
            role="user",
            message="What is TCP?",
            created_at=datetime.utcnow().isoformat(),
        ),
        ChatMessage(
            role="assistant",
            message="Can you describe what TCP does in the network stack?",
            created_at=datetime.utcnow().isoformat(),
        ),
        ChatMessage(
            role="user",
            message="TCP does 3-way handshake and controls congestion.",
            created_at=datetime.utcnow().isoformat(),
        ),
        ChatMessage(
            role="assistant",
            message="Can you explain how congestion control works in TCP?",
            created_at=datetime.utcnow().isoformat(),
        )
    ]
)


class SuggestionState(TypedDict):
    collection_name: str
    messages: List[AnyMessage]
    suggested_questions: Optional[List[str]]

class SuggestionService:
    def __init__(self):
        self.graph = self._build_graph()
        self.prompt_service = PromptService()

    def _build_graph(self):
        graph = StateGraph(SuggestionState)
        graph.add_node("load_chat_history", self._load_chat_history)
        graph.add_node("generate_suggestions", self._generate_suggestions)
        graph.set_entry_point("load_chat_history")
        graph.add_edge("load_chat_history", "generate_suggestions")
        graph.add_edge("generate_suggestions", END)
        return graph.compile()

    def _load_chat_history(self, state: SuggestionState) -> Dict:
        collection_name = state.get("collection_name")
        chat_history = test_chat_history
        #chat_history = chat_service.get_history(collection_name)
        if chat_history is None:
            raise ValueError(f"No chat history found for '{collection_name}'")

        messages = [
            HumanMessage(content=msg.message)
            if msg.role == "user" else AIMessage(content=msg.message)
            for msg in chat_history.messages
        ]

        return {"messages": messages}

    def _generate_suggestions(self, state: SuggestionState) -> Dict:
        messages = state.get("messages", [])
        history_str = self._format_history(messages)

        prompt_template = self.prompt_service.get_prompt("suggestion_prompt.txt")
        full_prompt = prompt_template + f"\n\n{history_str}"

        response = llm.invoke([HumanMessage(content=full_prompt)])
        lines = [line.strip("- ") for line in response.content.strip().split("\n") if line.strip()]
        return {"suggested_questions": lines}

    def _format_history(self, messages: List[AnyMessage]) -> str:
        history = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history += f"[Student]: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history += f"[Tutor]: {msg.content}\n"
        return history

