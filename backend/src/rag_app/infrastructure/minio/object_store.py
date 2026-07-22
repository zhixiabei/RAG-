from io import BytesIO

from minio import Minio
from urllib3 import PoolManager, Retry, Timeout


class MinioObjectStore:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool, bucket: str):
        self.bucket = bucket
        http_client = PoolManager(timeout=Timeout(connect=2.0, read=5.0), retries=Retry(total=0))
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
        self.client.put_object(self.bucket, object_key, BytesIO(content), len(content), content_type=content_type)
