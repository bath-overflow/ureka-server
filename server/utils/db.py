# utils/db.py

import os
from io import BytesIO

from fastapi import UploadFile
from minio import Minio
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# PostgreSQL connection settings
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "db")
DB_USER = os.environ.get("DB_USER", "user")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# MinIO client settings
MINIO_URL = os.environ.get("MINIO_URL", "localhost:9000")

minio_client = Minio(
    MINIO_URL,
    access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
    secure=False,
)


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


Base = declarative_base()


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
