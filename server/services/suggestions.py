from typing import Dict, List, Optional, TypedDict
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from datetime import datetime

from server.routers.chat import chat_service
from server.services.llm import llm
from server.services.prompt import PromptService
from server.models.chat_model import ChatMessage, ChatHistory
from server.repositories.document_store import get_documents_by_project
from server.utils.db import minio_client

# 테스트용 chat history
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

# 테스트용 lecture text
test_lecture_text = """
TCP (Transmission Control Protocol) is a core protocol of the Internet protocol suite. 
It originated in the initial network implementation in which it complemented the Internet Protocol (IP). 
Therefore, the entire suite is commonly referred to as TCP/IP. 
TCP provides reliable, ordered, and error-checked delivery of a stream of data between applications running on hosts communicating via an IP network.
"""


class SuggestionState(TypedDict):
    collection_name: str
    messages: List[AnyMessage]
    suggested_questions: Optional[List[str]]

class SuggestionService:
    def __init__(self, db):
        self.db = db
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
        #chat_history = test_chat_history  
        chat_history = chat_service.get_history(collection_name)

        if chat_history is None or not chat_history.messages:
            return {"messages": [], "collection_name": collection_name}

        messages = [
            HumanMessage(content=msg.message)
            if msg.role == "user" else AIMessage(content=msg.message)
            for msg in chat_history.messages
        ]

        return {"messages": messages, "collection_name": collection_name}

    def _generate_suggestions(self, state: SuggestionState) -> Dict:
        messages = state.get("messages", [])
        collection_name = state.get("collection_name")

        if not messages:
            # Lecture-based suggestion (no chat history)
            return self._suggest_from_lecture(collection_name)
        else:
            # Chat history-based suggestion
            return self._suggest_from_chat(messages)

    def _suggest_from_chat(self, messages: List[AnyMessage]) -> Dict:
        history_str = self._format_history(messages)

        prompt_template = self.prompt_service.get_prompt("suggestion_chat_prompt.txt")
        full_prompt = prompt_template + f"\n\n{history_str}"

        response = llm.invoke([HumanMessage(content=full_prompt)])
        questions = [line.strip("- ").strip() for line in response.content.strip().split("\n") if line.strip()]
        return {"suggested_questions": questions}

    def _suggest_from_lecture(self, collection_name: str) -> Dict:
        summary = self._summarize_lecture(collection_name)
        questions = self._generate_questions_from_summary(summary)
        return {"suggested_questions": questions}

    def _summarize_lecture(self, collection_name: str) -> str:
        documents = get_documents_by_project(self.db, collection_name)
        if not documents:
            return "No lecture document found for this session."

        # get latest document name from db
        latest_doc = documents[0]
        bucket_name = "documents"
        object_name = latest_doc.filename

        try:
            # read file from minIO
            response = minio_client.get_object(bucket_name, object_name)
            file_content = response.read().decode("utf-8")
        except Exception as e:
            print(f"Error reading from MinIO: {e}")
            return "Failed to retrieve lecture material."

        prompt_template = self.prompt_service.get_prompt("lecture_summary_prompt.txt")
        full_prompt = prompt_template + f"\n\n{file_content[:2000]}"

        response = llm.invoke([HumanMessage(content=full_prompt)])
        return response.content.strip()

    def _generate_questions_from_summary(self, summary: str) -> List[str]:

        prompt_template = self.prompt_service.get_prompt("suggestion_lecture_prompt.txt")
        full_prompt = prompt_template + f"\n\n{summary}"

        response = llm.invoke([HumanMessage(content=full_prompt)])
        return [line.strip("- ").strip() for line in response.content.strip().split("\n") if line.strip()]

    def _format_history(self, messages: List[AnyMessage]) -> str:
        history = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history += f"[Student]: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history += f"[Tutor]: {msg.content}\n"
        return history
