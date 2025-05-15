from server.services.langgraph_service import LangGraphService
from server.services.llm import llm
from langchain.schema import HumanMessage

class HintService:
    """Service for generating learning hints."""

    def generate_hint(self, chat_id: str, message: str) -> str:
        """
        Generate a helpful learning hint based on the user's input message using an LLM.

        Args:
            chat_id (str): The ID of the chat session.
            message (str): The user's input message.

        Returns:
            str: A hint generated to guide the user without directly answering the question.
        """
        
        prompt = (
            f"A student asked the following question:\n"
            f"'{message}'\n"
            f"Provide a concise hint that encourages the student to think critically and discover the answer themselves. "
            f"The hint should not give away the answer directly, but guide them to the right concept."
        )

        try:
            response = llm.invoke([
                HumanMessage(content=prompt)
            ])
            return response.content.strip()
        except Exception as e:
            return f"An error occurred while generating a hint: {str(e)}"