from fastapi import FastAPI

from server.routers.chat import chat_router
from server.routers.document import router as document_router
from server.routers.project import router as project_router

app = FastAPI()

app.include_router(document_router, tags=["documents"])
app.include_router(chat_router, tags=["chat"])
app.include_router(project_router, tags=["projects"])
