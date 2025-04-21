from langchain_openai import ChatOpenAI

from server.utils.config import OPENAI_API_KEY

open_api_llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base="https://api.openai.com/v1/chat/completions",
    openai_api_version="2023-10-01",
)
