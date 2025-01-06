from langswarm.memory.adapters.database_adapter import DatabaseAdapter
import sqlite3
import redis
from chromadb import Client
from chromadb.config import Settings
from google.cloud import storage


class SQLiteAdapter(DatabaseAdapter):
    def __init__(self, db_path="memory.db"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    metadata TEXT
                )
                """
            )
            conn.commit()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def add_documents(self, documents):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for doc in documents:
                key = doc.get("key", "")
                value = doc.get("text", "")
                metadata = str(doc.get("metadata", {}))
                cursor.execute(
                    "INSERT OR REPLACE INTO memory (key, value, metadata) VALUES (?, ?, ?)",
                    (key, value, metadata),
                )
            conn.commit()

    def query(self, query, filters=None):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            sql_query = "SELECT key, value, metadata FROM memory WHERE value LIKE ?"
            params = [f"%{query}%"]

            if filters:
                for field, value in filters.items():
                    sql_query += f" AND metadata LIKE ?"
                    params.append(f'%"{field}": "{value}"%')

            cursor.execute(sql_query, params)
            rows = cursor.fetchall()
            return [
                {"key": row[0], "text": row[1], "metadata": eval(row[2])} for row in rows
            ]

    def delete(self, document_ids):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for doc_id in document_ids:
                cursor.execute("DELETE FROM memory WHERE key = ?", (doc_id,))
            conn.commit()


class RedisAdapter(DatabaseAdapter):
    def __init__(self, redis_url="redis://localhost:6379/0"):
        self.client = redis.StrictRedis.from_url(redis_url)

    def add_documents(self, documents):
        for doc in documents:
            key = doc.get("key", "")
            value = doc.get("text", "")
            metadata = doc.get("metadata", {})
            self.client.set(key, str({"value": value, "metadata": metadata}))

    def query(self, query, filters=None):
        keys = self.client.keys("*")
        results = []
        for key in keys:
            entry = eval(self.client.get(key).decode())
            if query.lower() in entry["value"].lower():
                if filters and not all(
                    entry["metadata"].get(k) == v for k, v in filters.items()
                ):
                    continue
                results.append({"key": key.decode(), **entry})
        return results

    def delete(self, document_ids):
        for doc_id in document_ids:
            self.client.delete(doc_id)


class ChromaDBAdapter(DatabaseAdapter):
    def __init__(self, collection_name="shared_memory", persist_directory=None):
        self.client = Client(Settings(persist_directory=persist_directory))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents):
        for doc in documents:
            key = doc.get("key", "")
            value = doc.get("text", "")
            metadata = doc.get("metadata", {})
            self.collection.add(ids=[key], documents=[value], metadatas=[metadata])

    def query(self, query, filters=None):
        results = self.collection.query(query_text=query)
        if filters:
            return [
                {
                    "key": res["id"],
                    "text": res["document"],
                    "metadata": res["metadata"],
                }
                for res in results
                if all(
                    res["metadata"].get(k) == v for k, v in filters.items()
                )
            ]
        return results

    def delete(self, document_ids):
        for doc_id in document_ids:
            self.collection.delete(ids=[doc_id])


class GCSAdapter(DatabaseAdapter):
    def __init__(self, bucket_name, prefix="shared_memory/"):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.prefix = prefix

    def add_documents(self, documents):
        for doc in documents:
            key = f"{self.prefix}{doc.get('key', '')}"
            value = doc.get("text", "")
            metadata = doc.get("metadata", {})
            blob = self.bucket.blob(key)
            blob.upload_from_string(str({"value": value, "metadata": metadata}))

    def query(self, query, filters=None):
        blobs = list(self.client.list_blobs(self.bucket, prefix=self.prefix))
        results = []
        for blob in blobs:
            entry = eval(blob.download_as_text())
            if query.lower() in entry["value"].lower():
                if filters and not all(
                    entry["metadata"].get(k) == v for k, v in filters.items()
                ):
                    continue
                results.append({"key": blob.name[len(self.prefix):], **entry})
        return results

    def delete(self, document_ids):
        for doc_id in document_ids:
            blob = self.bucket.blob(f"{self.prefix}{doc_id}")
            if blob.exists():
                blob.delete()