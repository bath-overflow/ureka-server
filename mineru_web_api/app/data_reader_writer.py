from io import BytesIO, StringIO

from db import ensure_bucket_exists, minio_client
from magic_pdf.data.data_reader_writer import DataWriter


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


markdown_writer = MinioDataWriter("markdowns")
image_writer = MinioDataWriter("images")
