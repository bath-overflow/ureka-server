import os
from urllib.parse import quote_plus

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from server.models.chat import ChatHistory, ChatMessage

HOST = os.getenv("MONGO_HOST", "localhost")
PORT = os.getenv("MONGO_PORT", "27017")
DATABASE_NAME = os.getenv("MONGO_DB", "mydatabase")
MONGO_INITDB_ROOT_USERNAME = os.getenv("MONGO_INITDB_ROOT_USERNAME", "root")
MONGO_INITDB_ROOT_PASSWORD = os.getenv("MONGO_INITDB_ROOT_PASSWORD", "rootpassword")

uri = "mongodb://%s:%s@%s" % (
    quote_plus(MONGO_INITDB_ROOT_USERNAME),
    quote_plus(MONGO_INITDB_ROOT_PASSWORD),
    HOST,
)

client = MongoClient(uri)
db: Database = client[DATABASE_NAME]
chat_history_collection: Collection = db["chathistory"]


def create_chat_history(chat_id: str) -> ChatHistory:
    """
    Create a new chat history document and insert it into the database.
    """
    chat_history = ChatHistory(id=chat_id, messages=[])
    chat_history_collection.insert_one(chat_history.model_dump())
    return chat_history


def get_chat_history(chat_id: str) -> ChatHistory | None:
    """
    Retrieve a chat history by project_id.
    """
    data = chat_history_collection.find_one({"id": chat_id})
    if data:
        return ChatHistory(**data)
    return None


def append_chat_message(chat_id: str, message: ChatMessage) -> ChatHistory | None:
    """
    Append a chat message to the chat history.
    """
    result = chat_history_collection.update_one(
        {"id": chat_id}, {"$push": {"messages": message.model_dump()}}
    )
    if result.modified_count == 1:
        return get_chat_history(chat_id)
    return None


if __name__ == "__main__":
    # Example usage
    chat_id = "1"
    create_chat_history(chat_id)
    chat_history = get_chat_history(chat_id)
    print(chat_history)
    message = ChatMessage(
        id="1",
        role="user",
        message="Hello, how can I help you?",
        created_at="2023-01-01T00:00:00Z",
    )
    append_chat_message(chat_id, message)
    chat_history = get_chat_history(chat_id)
    print(chat_history)
