from fastapi import FastAPI

from server.models.document import Document  # 다른 테이블도 마찬가지
from server.models.project import Project  # 반드시 임포트해야 등록됨
from server.routers.chat import chat_router
from server.routers.document import router as document_router
from server.routers.project import router as project_router
from server.utils.db import Base, engine  # Base는 declarative_base()

# Create all tables in the database
_ = Document
_ = Project
Base.metadata.create_all(bind=engine)


app = FastAPI()

app.include_router(document_router, tags=["documents"])
app.include_router(chat_router, tags=["chat"])
app.include_router(project_router, tags=["projects"])
