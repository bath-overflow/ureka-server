from fastapi import FastAPI

from server.routers.chat import chat_router
from server.routers.document import document_router

app = FastAPI(root_path="/server")

app.include_router(document_router, prefix="/documents", tags=["documents"])
app.include_router(chat_router, tags=["chat"])
