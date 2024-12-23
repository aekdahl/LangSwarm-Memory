from google.cloud import storage
from .base import SharedMemoryBase
import json

class GCSSharedMemory(SharedMemoryBase):
    """
    Google Cloud Storage (GCS)-backed shared memory.
    """

    def __init__(self, bucket_name, prefix="shared_memory/"):
        """
        Initialize GCS client and bucket.

        Parameters:
        - bucket_name (str): GCS bucket name.
        - prefix (str): Prefix for keys in the bucket.
        """
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.prefix = prefix

    def _get_blob(self, key):
        return self.bucket.blob(f"{self.prefix}{key}")

    def read(self, key=None):
        """
        Read memory from GCS.

        Parameters:
        - key: Optional key to fetch specific data. If None, fetch all memory.

        Returns:
        - The value for the key if specified, else all memory as a dictionary.
        """
        if key:
            blob = self._get_blob(key)
            if not blob.exists():
                return None
            return blob.download_as_text()

        # List all keys with the prefix and return their values
        blobs = self.client.list_blobs(self.bucket, prefix=self.prefix)
        return {blob.name[len(self.prefix):]: blob.download_as_text() for blob in blobs}

    def write(self, key, value):
        """
        Write to GCS.

        Parameters:
        - key (str): Key to write.
        - value (str): Value to associate with the key.
        """
        blob = self._get_blob(key)
        blob.upload_from_string(value)

    def delete(self, key):
        """
        Delete a key in GCS.

        Parameters:
        - key (str): Key to delete.
        """
        blob = self._get_blob(key)
        if blob.exists():
            blob.delete()

    def clear(self):
        """
        Clear all keys in GCS.
        """
        blobs = self.client.list_blobs(self.bucket, prefix=self.prefix)
        for blob in blobs:
            blob.delete()

    def similarity_search(self, query, top_k=5):
        raise NotImplementedError("Similarity search is not supported for Google Cloud Storage.")
