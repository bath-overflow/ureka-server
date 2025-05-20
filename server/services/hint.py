from server.services.langgraph_service import lang_graph_service

class HintService:
    """Service for generating hints with chat history."""

    async def generate_hint(self, chat_id: str, original_question: str) -> str:
        """
        Generate a hint based on the user's latest message and full chat history.

        Args:
            chat_id (str): The chat session identifier.
            original_question (str): The original question from the user.

        Returns:
            str: A generated hint as a string.
        """
        # LangGraph의 응답을 받아 토큰 단위로 스트리밍하고 문자열로 이어 붙임
        hint_tokens = []
        async for token in lang_graph_service.stream_chat_response(chat_id, original_question):
            hint_tokens.append(token)
        
        hint = "".join(hint_tokens).strip()
        return hint

    # def generate_hint(self, chat_id: str, message: str) -> str:
    #     """
    #     Generate a helpful learning hint based on the user's input message using an LLM.

    #     Args:
    #         chat_id (str): The ID of the chat session.
    #         message (str): The user's input message.

    #     Returns:
    #         str: A hint generated to guide the user without directly answering the question.
    #     """
        
    #     prompt = (
    #         f"A student asked the following question:\n"
    #         f"'{message}'\n"
    #         f"Provide a concise hint that encourages the student to think critically and discover the answer themselves. "
    #         f"The hint should not give away the answer directly, but guide them to the right concept."
    #     )

    #     try:
    #         response = llm.invoke([
    #             HumanMessage(content=prompt)
    #         ])
    #         return response.content.strip()
    #     except Exception as e:
    #         return f"An error occurred while generating a hint: {str(e)}"