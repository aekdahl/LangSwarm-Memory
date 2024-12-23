from chromadb import Client
from chromadb.config import Settings
import asyncio
from threading import Lock
from datetime import datetime

from chromadb import Client
from chromadb.config import Settings
from datetime import datetime
import asyncio
from threading import Lock


class ChromaDBSharedMemory:
    """
    ChromaDB-backed shared memory with optional thread-safe and async-safe operations.
    """

    def __init__(self, collection_name="shared_memory", persist_directory=None, thread_safe=True, async_safe=False):
        """
        Initialize ChromaDB client and thread/async locks.

        Parameters:
        - collection_name (str): Name of the ChromaDB collection.
        - persist_directory (str): Directory for ChromaDB persistence.
        - thread_safe (bool): Enable thread-safe operations.
        - async_safe (bool): Enable async-safe operations.
        """
        self.client = Client(Settings(persist_directory=persist_directory))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.thread_safe = thread_safe or async_safe

        if self.thread_safe:
            self.lock = asyncio.Lock() if async_safe else Lock()

    def _generate_entry_id(self, context_id):
        """
        Generate a unique ID for an entry using context_id and timestamp.
        """
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        return f"{context_id}:{timestamp}"

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
        Read all entries for a given context_id, ordered by timestamp.

        Parameters:
        - context_id (str): The context ID to fetch.

        Returns:
        - Ordered dictionary of key-value pairs.
        """
        def _read():
            results = self.collection.get(
                where={"context_id": context_id},
                sort_by="timestamp",
                sort_order="asc"
            )
            return {
                result_id: {
                    "value": document,
                    "metadata": metadata
                }
                for result_id, document, metadata in zip(results["ids"], results["documents"], results["metadatas"])
            }

        if self.thread_safe:
            with self.lock:
                return _read()
        else:
            return _read()

    def write(self, value, metadata=None, context_id=None):
        """
        Write a value to ChromaDB with metadata and context_id.

        Parameters:
        - value (str): Value to store.
        - metadata (dict): Optional metadata.
        - context_id (str): Context ID for grouping entries.
        """
        metadata = metadata or {}
        entry_id = self._generate_entry_id(context_id)
        entry_metadata = {
            **metadata,
            "context_id": context_id,
            "timestamp": entry_id.split(":")[1],  # Extract timestamp from the ID
        }

        def _write():
            self.collection.add(ids=[entry_id], documents=[value], metadatas=[entry_metadata])

        if self.thread_safe:
            with self.lock:
                _write()
        else:
            _write()

    def delete_context(self, context_id):
        """
        Delete all entries for a given context_id.

        Parameters:
        - context_id (str): The context ID to delete.
        """
        def _delete():
            self.collection.delete(where={"context_id": context_id})

        if self.thread_safe:
            with self.lock:
                _delete()
        else:
            _delete()

    def clear(self):
        """
        Clear all entries in the ChromaDB collection.
        """
        def _clear():
            self.collection.delete()

        if self.thread_safe:
            with self.lock:
                _clear()
        else:
            _clear()

    def similarity_search(self, query, top_k=5):
        """
        Perform similarity search in ChromaDB.

        Parameters:
        - query (str): Query string to search.
        - top_k (int): Number of top similar results to return.

        Returns:
        - List of top_k most similar documents.
        """
        def _search():
            query_vector = self.client.encode(query)
            results = self.collection.query(query_embeddings=[query_vector], n_results=top_k)
            return results["documents"]

        if self.thread_safe:
            with self.lock:
                return _search()
        else:
            return _search()
