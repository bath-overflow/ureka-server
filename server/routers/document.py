from fastapi import APIRouter, HTTPException, UploadFile

from server.repositories.documnet_store import save_pdf_to_db

document_router = APIRouter()


@document_router.post("/{id}/upload-pdf")
async def upload_pdf(id: int, file: UploadFile = None):
    """
    Upload a PDF file and save it to the database.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    # Read the file content
    file_content = await file.read()

    # Save the file to MinIO or any other storage
    # 일단 로컬에 저장
    file_name = f"{id}_{file.filename}"
    with open(file_name, "wb") as f:
        f.write(file_content)
    # Save the file URL to the database
    file_url = f"http://localhost:9000/{file_name}"
    await save_pdf_to_db(id, file_name, file_url)
    # Return a success response

    return {"message": "PDF uploaded successfully"}
