import os
import tempfile
from io import BytesIO, StringIO
from typing import Annotated, Tuple

import magic_pdf.model as model_config
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from loguru import logger
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import DataWriter
from magic_pdf.data.dataset import ImageDataset, PymuDocDataset
from magic_pdf.data.read_api import read_local_images, read_local_office
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.operators.models import InferenceResult
from magic_pdf.operators.pipes import PipeResult
from minio import Minio
from vector_store import vector_store

model_config.__use_inside_model__ = True

app = FastAPI()

# MinIO configuration
MINIO_URL = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
IS_PROD = os.getenv("ENVIRONMENT") == "production"

if IS_PROD:
    MINIO_URL = "minio.cspc.me"

MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")

minio_client = Minio(
    MINIO_URL,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=IS_PROD,
)


# Ensure buckets exist
def ensure_bucket_exists(bucket_name: str):
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)


pdf_extensions = [".pdf"]
office_extensions = [".ppt", ".pptx", ".doc", ".docx"]
image_extensions = [".png", ".jpg", ".jpeg"]


class MemoryDataWriter(DataWriter):
    def __init__(self):
        self.buffer = StringIO()

    def write(self, path: str, data: bytes) -> None:
        if isinstance(data, str):
            self.buffer.write(data)
        else:
            self.buffer.write(data.decode("utf-8"))

    def write_string(self, path: str, data: str) -> None:
        self.buffer.write(data)

    def get_value(self) -> str:
        return self.buffer.getvalue()

    def close(self):
        self.buffer.close()


class MinioDataWriter(DataWriter):
    """Data writer implementation for MinIO storage"""

    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        ensure_bucket_exists(bucket_name)

    def write(self, path: str, data: bytes) -> None:
        """
        Write data to MinIO storage
        Args:
            path: Object name in the bucket
            data: Data to be written
        """
        if isinstance(data, str):
            data_bytes = data.encode("utf-8")
        else:
            data_bytes = data

        minio_client.put_object(
            bucket_name=self.bucket_name,
            object_name=path,
            data=BytesIO(data_bytes),
            length=len(data_bytes),
        )

    def write_string(self, path: str, data: str) -> None:
        """
        Write string data to MinIO storage
        Args:
            path: Object name in the bucket
            data: String data to be written
        """
        data_bytes = data.encode("utf-8")
        minio_client.put_object(
            bucket_name=self.bucket_name,
            object_name=path,
            data=BytesIO(data_bytes),
            length=len(data_bytes),
            content_type="text/plain",
        )


def init_writers(
    file: UploadFile,
    project_id: str,
) -> Tuple[DataWriter, DataWriter, bytes, str]:
    """
    Initialize writers based on path type

    Args:
        file: Uploaded file object
        project_id: Project ID of the file

    Returns:
        Tuple[md_writer, image_writer, file_bytes, file_extension]
    """
    # Read the uploaded file content
    file_bytes = file.file.read()
    if file.filename is None:
        raise ValueError("File name is missing.")
    file_extension = os.path.splitext(file.filename)[1].lower()

    # Create writers for "markdowns" and "images" buckets
    markdown_writer = MinioDataWriter("markdowns")
    image_writer = MinioDataWriter("images")

    return markdown_writer, image_writer, file_bytes, file_extension


def process_file(
    file_bytes: bytes,
    file_extension: str,
    image_writer: DataWriter,
) -> tuple[InferenceResult, PipeResult]:
    """
    Process PDF file content

    Args:
        file_bytes: Binary content of file
        file_extension: file extension
        image_writer: Image writer

    Returns:
        Tuple[InferenceResult, PipeResult]: Returns inference result and pipeline result
    """

    ds: ImageDataset | PymuDocDataset | None = None
    if file_extension in pdf_extensions:
        ds = PymuDocDataset(file_bytes)
    elif file_extension in office_extensions:
        # Process Office files
        temp_dir = tempfile.mkdtemp()
        with open(os.path.join(temp_dir, f"temp_file.{file_extension}"), "wb") as f:
            f.write(file_bytes)
        ds = read_local_office(temp_dir)[0]
    elif file_extension in image_extensions:
        # Process image files
        temp_dir = tempfile.mkdtemp()
        with open(os.path.join(temp_dir, f"temp_file.{file_extension}"), "wb") as f:
            f.write(file_bytes)
        ds = read_local_images(temp_dir)[0]

    if ds is None:
        raise ValueError("Unsupported file type or failed to read file.")

    infer_result: InferenceResult | None = None
    pipe_result: PipeResult | None = None

    if ds.classify() == SupportedPdfParseMethod.OCR:
        infer_result = ds.apply(doc_analyze, ocr=True)
        if infer_result is None:
            raise ValueError("Failed to parse document (OCR).")
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    else:
        infer_result = ds.apply(doc_analyze, ocr=False)
        if infer_result is None:
            raise ValueError("Failed to parse document (non-OCR).")
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    return infer_result, pipe_result


def split_markdown_into_documents(
    markdown_content: str, source_name: str
) -> list[Document]:
    """
    Split markdown content into LangChain Document objects using headers.

    Args:
        markdown_content: The markdown content to split
        source_name: Name to use as the source in document metadata

    Returns:
        List of Document objects
    """
    # Define headers to split on
    headers_to_split_on = [
        ("#", "header1"),
        ("##", "header2"),
        ("###", "header3"),
        ("####", "header4"),
    ]

    # Create the splitter
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )

    # Split the markdown content
    docs = markdown_splitter.split_text(markdown_content)

    # Add source metadata
    for doc in docs:
        doc.metadata["source"] = source_name

    return docs


@app.post(
    "/file_parse",
    summary="Parse file and return markdown content",
)
async def file_parse(
    file: Annotated[UploadFile, File()],
    project_id: Annotated[str, Form()],
):
    """
    Execute the process of converting a document file to markdown content.
    Also saves the content to vector store for later retrieval.

    Args:
        file: The file to be parsed (PDF, Office document, or image)
        project_id: Project ID for the file
    """
    if file.filename is None:
        return JSONResponse({"error": "File name is missing."}, status_code=400)

    try:
        # Initialize readers/writers and get file content
        md_writer, image_writer, file_bytes, file_extension = init_writers(
            file=file, project_id=project_id
        )

        # Process the file
        infer_result, pipe_result = process_file(
            file_bytes, file_extension, image_writer
        )

        # Use MemoryDataWriter to get results
        md_content_writer = MemoryDataWriter()

        # Use PipeResult's dump method to get data
        pipe_result.dump_md(md_content_writer, "", "")

        # Get content
        md_content = md_content_writer.get_value()

        # Save markdown content to MinIO
        file_name_without_ext = os.path.splitext(file.filename)[0]
        markdown_path = f"{project_id}/{file_name_without_ext}.md"
        md_writer.write_string(markdown_path, md_content)

        # Clean up memory writers
        md_content_writer.close()

        # Split markdown into documents
        docs = split_markdown_into_documents(md_content, source_name=file.filename)

        # Add documents to vector store
        doc_ids = vector_store.add_documents(project_id, docs)
        logger.info(
            f"Added {len(doc_ids)} document chunks to vector store collection "
            + project_id
        )

        return JSONResponse({"md_content": md_content}, status_code=200)

    except Exception as e:
        logger.exception(e)
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
