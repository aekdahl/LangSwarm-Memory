from google.cloud import storage
import asyncio
from datetime import datetime
from .base import SharedMemoryBase


class GCSSharedMemory(SharedMemoryBase):
    """
    Google Cloud Storage (GCS)-backed shared memory with optional thread-safe and async-safe operations.
    """

    def __init__(self, bucket_name, prefix="shared_memory/", thread_safe=True, async_safe=False):
        """
        Initialize GCS client and thread/async locks.

        Parameters:
        - bucket_name (str): GCS bucket name.
        - prefix (str): Prefix for keys in the bucket.
        - thread_safe (bool): Enable thread-safe operations.
        - async_safe (bool): Enable async-safe operations.
        """
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.prefix = prefix
        self.thread_safe = thread_safe or async_safe

        if self.thread_safe:
            if async_safe:
                self.lock = asyncio.Lock()
            else:
                self.lock = Lock()

    def _get_blob(self, key):
        return self.bucket.blob(f"{self.prefix}{key}")

    def _get_default_metadata(self, metadata):
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "agent": metadata.get("agent"),
            "timestamp": metadata.get("timestamp", now),
            "llm": metadata.get("llm"),
            "action": metadata.get("action"),
            "confidence": metadata.get("confidence"),
            "query": metadata.get("query"),
        }

    def read_context(self, context_id):
        """
        Read all entries for a given context_id in timestamp order.
        """
        def _read():
            blobs = sorted(self.client.list_blobs(self.bucket, prefix=f"{self.prefix}{context_id}:"))
            return {
                blob.name[len(self.prefix):]: eval(blob.download_as_text())
                for blob in blobs
            }

        if self.thread_safe:
            with self.lock:
                return _read()
        else:
            return _read()

    def write(self, value, metadata=None, context_id=None):
        metadata = metadata or {}
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        key = f"{context_id}:{timestamp}"
        entry = {"value": value, "metadata": self._get_default_metadata(metadata)}

        def _write():
            blob = self._get_blob(key)
            blob.upload_from_string(str(entry))

        if self.thread_safe:
            with self.lock:
                _write()
        else:
            _write()

    def delete(self, key):
        """
        Delete a key in GCS.
        """
        def _delete():
            blob = self._get_blob(key)
            if blob.exists():
                blob.delete()

        if self.thread_safe:
            with self.lock:
                _delete()
        else:
            _delete()

    def clear(self):
        """
        Clear all keys in GCS.
        """
        def _clear():
            blobs = self.client.list_blobs(self.bucket, prefix=self.prefix)
            for blob in blobs:
                blob.delete()

        if self.thread_safe:
            with self.lock:
                _clear()
        else:
            _clear()
