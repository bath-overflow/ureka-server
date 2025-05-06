from contextlib import asynccontextmanager

from fastapi import FastAPI

from server.routers.document import document_router
from server.routers.langgraph_chat import chat_router
from server.services.langgraph_service import conn


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    """
    # Code that should be run before the app starts taking requests
    yield
    # Code that should be run after the application finishes handling requests,
    # right before the shutdown
    await conn.close()


app = FastAPI(lifespan=lifespan)


app.include_router(document_router, prefix="/documents", tags=["documents"])
app.include_router(chat_router, prefix="/projects", tags=["chat"])
