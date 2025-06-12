from typing import Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, StateGraph

from server.repositories.document_store import get_documents_by_project
from server.routers.chat import chat_service
from server.services.document import get_markdown_content
from server.services.llm import llm
from server.services.prompt import PromptService
from server.utils.db import get_db

# test_file_content = """
# 프롬프팅(Prompting)은 인공지능 언어 모델에게 원하는 응답을 얻기 위해 입력 문장을 설계하는 기술입니다. 프롬프트의 구조, 문맥, 명확성은 모델의 응답 품질에 큰 영향을 미칩니다. 예를 들어, "AI란 무엇인가요?"라는 단순 질문보다 "AI의 정의와 역사적 발전 과정을 간단히 설명해 주세요."라는 구체적인 프롬프트가 더 나은 응답을 이끌어냅니다.

# 프롬프팅 기법에는 Zero-shot, One-shot, Few-shot 프롬프팅이 있으며, 이들은 예시 제공 유무에 따라 나뉩니다. Zero-shot은 예시 없이 질문만 던지는 방식이고, Few-shot은 여러 개의 예시를 통해 모델의 기대 응답 형태를 유도합니다.

# 또한, 역할 기반 프롬프팅(Role Prompting), 체인 오브 쏘트(Chain of Thought) 프롬프팅 등 다양한 고급 기법들이 존재합니다. 역할 기반 프롬프팅은 모델에게 특정 인물이나 역할을 부여하여 보다 일관된 응답을 유도하고, 체인 오브 쏘트는 문제 해결 과정을 단계별로 유도해 복잡한 문제를 푸는 데 효과적입니다.

# 프롬프팅은 단순히 질문을 던지는 것이 아니라, 언어 모델과의 상호작용에서 원하는 결과를 끌어내는 설계 능력이라고 할 수 있습니다.
# """


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

        chat_history = chat_service.get_history(collection_name)

        if chat_history is None or not chat_history.messages:
            return {"messages": [], "collection_name": collection_name}

        messages = [
            (
                HumanMessage(content=msg.message)
                if msg.role == "user"
                else AIMessage(content=msg.message)
            )
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
        questions = [
            line.strip("- ").strip()
            for line in response.content.strip().split("\n")
            if line.strip()
        ]
        return {"suggested_questions": questions}

    def _suggest_from_lecture(self, collection_name: str) -> Dict:
        summary = self._summarize_lecture(collection_name)
        if not summary:
            questions = "추천 질문을 생성하기 위해, 강의자료를 먼저 올려 주세요."
        else:
            questions = self._generate_questions_from_summary(summary)
        return {"suggested_questions": questions}

    def _summarize_lecture(self, collection_name: str) -> str:
        relational_db = get_db()
        documents = get_documents_by_project(self.db, collection_name)
        if not documents:
            return None

        file_content = ""
        # file_content = test_file_content

        for doc in documents:
            try:
                # read markdown contents
                content = get_markdown_content(
                    relational_db, collection_name, doc.filename
                )
                if content.strip():
                    file_content += f"\n\n# Document: {doc.filename}\n{content.strip()}"
            except Exception as e:
                print(f"Failed to read {doc.filename} from MinIO: {e}")
                continue

        if not file_content.strip():
            raise ValueError("All lecture documents are empty or failed to load.")

        prompt_template = self.prompt_service.get_prompt("lecture_summary_prompt.txt")
        full_prompt = prompt_template + f"\n\n{file_content}"

        response = llm.invoke([HumanMessage(content=full_prompt)])
        return response.content.strip()

    def _generate_questions_from_summary(self, summary: str) -> List[str]:

        prompt_template = self.prompt_service.get_prompt(
            "suggestion_lecture_prompt.txt"
        )
        full_prompt = prompt_template + f"\n\n{summary}"

        response = llm.invoke([HumanMessage(content=full_prompt)])
        return [
            line.strip("- ").strip()
            for line in response.content.strip().split("\n")
            if line.strip()
        ]

    def _format_history(self, messages: List[AnyMessage]) -> str:
        history = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history += f"[Student]: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history += f"[Tutor]: {msg.content}\n"
        return history
