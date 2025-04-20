from fastapi import FastAPI

from server.routers.document import document_router

app = FastAPI()


app.include_router(document_router, prefix="/documents", tags=["documents"])
