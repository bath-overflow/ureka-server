from fastapi import FastAPI
from routers.document import document_router

app = FastAPI()


app.include_router(document_router, prefix="/documents", tags=["documents"])
