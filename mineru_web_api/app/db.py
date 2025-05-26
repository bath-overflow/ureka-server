import os

from minio import Minio

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
