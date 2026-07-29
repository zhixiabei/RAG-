from hashlib import sha256
from io import BytesIO
from typing import BinaryIO

from minio import Minio
from urllib3 import PoolManager, Retry, Timeout


class MinioObjectStore:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool, bucket: str):
        self.bucket = bucket
        http_client = PoolManager(timeout=Timeout(connect=2.0, read=300.0), retries=Retry(total=0))
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            http_client=http_client,
        )

    def ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def put_bytes(self, object_key: str, content: bytes, content_type: str) -> None:
        self.put_stream(object_key, BytesIO(content), len(content), content_type)

    def put_stream(self, object_key: str, stream: BinaryIO, length: int, content_type: str) -> None:
        self.client.put_object(self.bucket, object_key, stream, length, content_type=content_type)

    def calculate_hash(self, object_key: str) -> str:
        response = self.client.get_object(self.bucket, object_key)
        digest = sha256()
        try:
            for chunk in response.stream(amt=1024 * 1024):
                digest.update(chunk)
        finally:
            response.close()
            response.release_conn()
        return digest.hexdigest()

    def delete_object(self, object_key: str) -> None:
        self.client.remove_object(self.bucket, object_key)
