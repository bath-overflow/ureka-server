from langchain.chat_models import init_chat_model

from server.utils.config import GOOGLE_API_KEY

llm = init_chat_model(
    model="gemini-2.0-flash", model_provider="google_genai", api_key=GOOGLE_API_KEY
)
