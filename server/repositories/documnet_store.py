import os
from io import BytesIO

import asyncpg
from fastapi import UploadFile
from minio import Minio

# Database connection settings
DATABASE_URL = "postgresql://username:password@localhost/dbname"

# MinIO client settings
MINIO_URL = os.environ.get("MINIO_URL", "localhost:9000")

minio_client = Minio(
    MINIO_URL,
    access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
    secure=False,
)


async def upload_file_to_minio(bucket_name: str, file: UploadFile, file_name: str):
    """
    Upload a file to MinIO and return the file URL.
    """
    # Ensure the bucket exists
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    # Upload the file
    file_content = await file.read()
    minio_client.put_object(
        bucket_name, file_name, BytesIO(file_content), len(file_content)
    )

    # Generate the file URL
    file_url = f"http://{MINIO_URL}/{bucket_name}/{file_name}"
    return file_url


async def get_db_connection():
    return await asyncpg.connect(DATABASE_URL)


async def save_pdf_to_db(note_id: int, file_name: str, file_url: str):
    """
    Save the MinIO file URL to the database.
    """
    conn = await get_db_connection()
    try:
        await conn.execute(
            """
            INSERT INTO documents (note_id, file_name, file_url)
            VALUES ($1, $2, $3)
            """,
            note_id,
            file_name,
            file_url,
        )
    finally:
        await conn.close()
