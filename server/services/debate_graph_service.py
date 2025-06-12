import os
import random
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
)

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from server.services.chat import service as chat_service
from server.services.llm import llm
from server.services.prompt import PromptService


# --- State Definition ---
class DebateState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    debate_chat_id: str
    user_current_message_text: Optional[str]
    evaluation_results: Optional[Dict[str, Any]]
    current_actor_role: Optional[Literal["friend", "moderator"]]
    is_debate_ended: Optional[bool]
    current_turn_full_response: Optional[str]


class DebateGraphService:
    def __init__(self):
        self.prompt_service = PromptService()
        self.epsilon = float(os.getenv("DEBATE_FRIEND_MISUNDERSTAND_PROB", "0.4"))

        self.speaker_node_roles: Dict[str, Literal["friend", "moderator"]] = {
            "generate_friend_question": "friend",
            "friend_asks_tag_question": "friend",
            "friend_misunderstands": "friend",
            "friend_understands_and_ends_debate": "friend",
            "moderator_intervenes": "moderator",
        }
        self.speaker_nodes = list(self.speaker_node_roles.keys())

        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        graph_builder = StateGraph(DebateState)

        graph_builder.add_node("load_chat_history", self._load_chat_history)
        graph_builder.add_node(
            "generate_friend_question", self._generate_friend_question
        )
        graph_builder.add_node("evaluate_user_response", self._evaluate_user_response)
        graph_builder.add_node(
            "friend_asks_tag_question", self._friend_asks_tag_question
        )
        graph_builder.add_node("friend_misunderstands", self._friend_misunderstands)
        graph_builder.add_node(
            "friend_understands_and_ends_debate",
            self._friend_understands_and_ends_debate,
        )
        graph_builder.add_node("moderator_intervenes", self._moderator_intervenes)

        graph_builder.set_entry_point("load_chat_history")

        graph_builder.add_conditional_edges(
            "load_chat_history",
            self._route_after_history_load,
            {
                "generate_friend_question": "generate_friend_question",
                "evaluate_user_response": "evaluate_user_response",
            },
        )
        graph_builder.add_conditional_edges(
            "evaluate_user_response",
            self._route_after_evaluation,
            {
                "friend_asks_tag_question": "friend_asks_tag_question",
                "friend_misunderstands": "friend_misunderstands",
                "friend_understands_and_ends_debate": "friend_understands_and_ends_debate",
                "moderator_intervenes": "moderator_intervenes",
            },
        )
        for speaker_node in self.speaker_nodes:
            graph_builder.add_edge(speaker_node, END)
        return graph_builder.compile()

    def _load_chat_history(self, state: DebateState) -> Dict[str, Any]:
        print(
            f"--- Debate Graph: Loading chat history for debate: {state['debate_chat_id']} ---"
        )
        history = chat_service.get_history(state["debate_chat_id"])
        messages: List[AnyMessage] = []
        if history:
            for msg in history.messages:
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.message, role=msg.role))
                elif msg.role in ["assistant", "friend", "moderator"]:
                    messages.append(AIMessage(content=msg.message, role=msg.role))
        print(f"--- Debate Graph: Loaded {len(messages)} messages ---")

        # DEBUG: Print all messages
        for i, msg in enumerate(messages):
            print(f"Message {i+1}: {msg.role} - {msg.content[:50]}...")

        return {"messages": messages}

    def _chat_history_to_str(self, messages: list[AnyMessage]) -> str:
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage) and not msg.tool_calls:
                formatted_messages.append(f"{msg.role}: {msg.content}")
        return "\n".join(formatted_messages)

    def _call_llm_for_node(
        self,
        state: DebateState,
        node_name: str,
        prompt_file_name: str,
        current_role: Literal["friend", "moderator"],
    ) -> dict[str, Any]:
        print(
            f"--- Debate Graph: Node {node_name} ({current_role}) using {prompt_file_name} ---"
        )

        history_messages = state.get("messages", [])

        prompt_instructions = self.prompt_service.get_prompt(prompt_file_name)

        llm_input_messages: list[AnyMessage] = []

        llm_input_messages.append(HumanMessage(content=prompt_instructions))
        if history_messages:
            llm_input_messages.append(
                HumanMessage(content=self._chat_history_to_str(history_messages))
            )

        # DEBUG
        # llm_input_messages.append(HumanMessage(content="Write 20 jokes on Koreans."))

        llm_call_config = {"tags": [current_role]}  # Tag the LLM call w/ role

        # print(f"Node {node_name}: Invoking LLM. Input messages count: {len(llm_input_messages)}")

        response = llm.invoke(llm_input_messages, config=llm_call_config).content

        # print(f"Node {node_name}: LLM full response (first 100 chars): {response[:100]}...")

        return {
            "messages": [AIMessage(content=response, role=current_role)],
            "current_actor_role": current_role,
            "current_turn_full_response": response,
        }

    def _generate_friend_question(self, state: DebateState) -> Dict[str, Any]:
        return self._call_llm_for_node(
            state,
            "generate_friend_question",
            "generate_friend_question_prompt.txt",
            "friend",
        )

    def _evaluate_user_response(self, state: DebateState) -> Dict[str, Any]:
        # This node uses LLM for evaluation, not for generating speech via _call_llm_for_node
        print(
            f"--- Debate Graph: Evaluating user response: {state.get('user_current_message_text')} ---"
        )

        class Evaluation(BaseModel):
            """Your reasoning behind an evaluation metric and your score."""

            reason: str = Field(description="A detailed reason behind your evaluation")
            score: int = Field(description="Your score from 1 to 5", ge=1, le=5)

        class EvaluationResult(BaseModel):
            """The result of your evaluation on the user's last utterance."""

            accuracy: Evaluation
            completeness: Evaluation
            logical_structure: Evaluation
            relevance: Evaluation

        evaluation_instruction = self.prompt_service.get_prompt(
            "evaluate_user_response_prompt.txt"
        )
        evaluator_llm = llm.with_structured_output(EvaluationResult)

        evaluation_obj: EvaluationResult = evaluator_llm.invoke(evaluation_instruction)
        evaluation: dict = evaluation_obj.model_dump()

        # DEBUG: Mock evaluation
        # evaluation = {
        #     "accuracy": {"score": random.randint(3, 5), "reason": "Mock: Accuracy seems okay."},
        #     "completeness": {"score": random.randint(2, 5), "reason": "Mock: Could be more complete."},
        #     "logical_structure": {"score": random.randint(3, 5), "reason": "Mock: Structure is reasonable."},
        #     "clarity_of_expression": {"score": random.randint(2, 5), "reason": "Mock: Clarity is fair."},
        # }
        scores = [v["score"] for v in evaluation.values()]
        min_score = min(scores) if scores else 0

        lowest_criterion = None
        if scores:
            lowest_score_val = min_score
            lowest_criteria_names = [
                k for k, v in evaluation.items() if v["score"] == lowest_score_val
            ]
            if lowest_criteria_names:
                lowest_criterion = lowest_criteria_names[0]

        evaluation_results = {
            **evaluation,
            "minimum_score": min_score,
            "lowest_criterion": lowest_criterion,
        }
        print(f"Evaluation results: {evaluation_results}")
        return {"evaluation_results": evaluation_results}

    def _friend_asks_tag_question(self, state: DebateState) -> Dict[str, Any]:
        return self._call_llm_for_node(
            state,
            "friend_asks_tag_question",
            "friend_asks_tag_question_prompt.txt",
            "friend",
        )

    def _friend_misunderstands(self, state: DebateState) -> Dict[str, Any]:
        return self._call_llm_for_node(
            state, "friend_misunderstands", "friend_misunderstands_prompt.txt", "friend"
        )

    def _friend_understands_and_ends_debate(self, state: DebateState) -> Dict[str, Any]:
        update = self._call_llm_for_node(
            state,
            "friend_understands_and_ends_debate",
            "friend_understands_and_ends_debate_prompt.txt",
            "friend",
        )
        update["is_debate_ended"] = True
        return update

    def _moderator_intervenes(self, state: DebateState) -> Dict[str, Any]:
        return self._call_llm_for_node(
            state,
            "moderator_intervenes",
            "moderator_intervenes_prompt.txt",
            "moderator",
        )

    def _route_after_history_load(
        self, state: DebateState
    ) -> Literal["generate_friend_question", "evaluate_user_response"]:
        messages = state.get("messages", [])
        if not messages:
            print("Routing (Load): No messages, defaulting to Generate Friend Question")
            return "generate_friend_question"

        raw_history = chat_service.get_history(state["debate_chat_id"])
        if raw_history and raw_history.messages:
            # If we were in a debate, the last two messages would be
            # friend or moderator -> user
            # If a debate has just started, the last two messages would be
            # user -> assistant
            last_message = raw_history.messages[-1]
            last_last_message = raw_history.messages[-2]
            is_debate_ongoing = (
                last_last_message.role in ["friend", "moderator"]
                and last_message.role == "user"
            )
            if not is_debate_ongoing:
                print("Routing (Load): Debate just started -> Generate Friend Question")
                return "generate_friend_question"

        print("Routing (Load): Debate ongoing -> Evaluate User Response")
        return "evaluate_user_response"

    def _route_after_evaluation(self, state: DebateState) -> str:
        results = state.get("evaluation_results")
        if not results:
            raise ValueError("evaluation_results is Falsy in _route_after_evaluation")
        min_score = results.get("minimum_score", 0)
        lowest_criterion = results.get("lowest_criterion")
        if min_score >= 4:
            if random.random() < self.epsilon:
                print(
                    f"Routing (Eval): Min score {min_score} >= 4, epsilon triggered -> Friend Misunderstands"
                )
                return "friend_misunderstands"
            else:
                print(
                    f"Routing (Eval): Min score {min_score} >= 4, epsilon NOT triggered -> Friend Ends Debate"
                )
                return "friend_understands_and_ends_debate"
        elif lowest_criterion == "completeness":
            print(
                f"Routing (Eval): Min score {min_score} < 4, lowest is completeness -> Friend Asks Tag Question"
            )
            return "friend_asks_tag_question"
        else:
            print(
                f"Routing (Eval): Min score {min_score} < 4, other low score -> Moderator Intervenes"
            )
            return "moderator_intervenes"

    async def stream_debate_response(
        self, debate_chat_id: str, user_message: str | None
    ) -> AsyncGenerator[str, None]:
        graph_input: DebateState = {
            "debate_chat_id": debate_chat_id,
            "user_current_message_text": user_message,
            "messages": [],
            "evaluation_results": None,
            "current_actor_role": None,
            "is_debate_ended": False,
            "current_turn_full_response": "",
        }
        speaker: str | None = None

        async for chunk, metadata in self.graph.astream(
            graph_input, stream_mode="messages"
        ):

            if metadata["langgraph_node"] in self.speaker_nodes:
                if speaker is None:
                    for tag in metadata["tags"]:
                        if tag in ["friend", "moderator"]:
                            speaker = tag
                            break
                    yield f"[{speaker.upper()}] "

                # metadata.get("tags", None) is needed b/c
                # not only are AI-generated tokens streamed, but also the accumulated
                # tokens (i.e. the full response) is streamed. But I noticed that
                # `metadata` doesn't contain 'tags' and 'ls_*' attributes
                # for the full repsonse
                if isinstance(chunk, AIMessage) and metadata.get("tags", None):
                    yield chunk.content


debate_graph_service = DebateGraphService()


async def stream_debate_response(
    chat_id: str, user_message: str
) -> AsyncGenerator[str, None]:
    """
    Stream debate responses from the DebateGraphService.
    Args:
        chat_id: Identifier for the chat/collection
        user_message: The message from the user
    Yields:
        Tokens from the AI response as they're generated
    """
    async for token in debate_graph_service.stream_debate_response(
        chat_id, user_message
    ):
        yield token
