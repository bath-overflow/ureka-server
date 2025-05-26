import os
import tempfile
from typing import Annotated

import magic_pdf.model as model_config
import uvicorn
from data_reader_writer import MemoryDataWriter, image_writer, markdown_writer
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
from vector_store import vector_store

model_config.__use_inside_model__ = True

app = FastAPI()

pdf_extensions = [".pdf"]
office_extensions = [".ppt", ".pptx", ".doc", ".docx"]
image_extensions = [".png", ".jpg", ".jpeg"]


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
    Save the markdown content and image files to object storage.
    Split the markdown content into documents and add them to the vector store.

    Args:
        file: The file to be parsed (PDF, Office document, or image)
        project_id: Project ID for the file
    """
    if file.filename is None:
        return JSONResponse({"error": "File name is missing."}, status_code=400)

    try:
        # Get the file content and extension
        file_bytes = file.file.read()
        file_extension = os.path.splitext(file.filename)[1].lower()

        # Process the file
        _, pipe_result = process_file(file_bytes, file_extension, image_writer)

        # Use MemoryDataWriter to get results
        md_content_writer = MemoryDataWriter()

        # Use PipeResult's dump method to get data
        pipe_result.dump_md(md_content_writer, "", "")

        # Get content
        md_content = md_content_writer.get_value()

        # Save markdown content to MinIO
        file_name_without_ext = os.path.splitext(file.filename)[0]
        markdown_path = f"{project_id}/{file_name_without_ext}.md"
        markdown_writer.write_string(markdown_path, md_content)

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
