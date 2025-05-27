import os

from fastapi import FastAPI

import server.models.document_model
import server.models.project_model
from server.routers.chat import chat_router
from server.routers.hint import hint_router
from server.routers.recommend import recommend_router
from server.routers.document import router as document_router
from server.routers.project import router as project_router
from server.utils.db import Base, engine

_ = server.models.project_model
_ = server.models.document_model

if os.environ.get("ENVIRONMENT") != "test":
    Base.metadata.create_all(bind=engine)


app = FastAPI()

app.include_router(document_router, tags=["documents"])
app.include_router(chat_router, tags=["chat"])
app.include_router(hint_router, tags=["hint"])
app.include_router(recommend_router, tags=["recommend"])
app.include_router(project_router, tags=["projects"])
