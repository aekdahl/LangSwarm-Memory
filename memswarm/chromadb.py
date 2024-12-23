from chromadb import Client
from chromadb.config import Settings
import asyncio
from threading import Lock
from datetime import datetime

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

    def read(self, key=None):
        """
        Read memory from ChromaDB.

        Parameters:
        - key (str): Key to fetch. If None, fetch all memory.

        Returns:
        - List of documents if key is None, otherwise the specific document.
        """
        def _read():
            if key:
                result = self.collection.get(ids=[key])
                if result["documents"]:
                    return result["documents"][0]
                return None
            return self.collection.get()["documents"]

        if self.thread_safe:
            with self.lock:
                return _read()
        else:
            return _read()

    def write(self, key, value, metadata=None):
        """
        Write a key-value pair to ChromaDB with metadata.

        Parameters:
        - key (str): Key to store.
        - value (str): Value to store.
        - metadata (dict): Optional metadata.
        """
        metadata = metadata or {}
        entry_metadata = self._get_default_metadata(metadata)

        def _write():
            self.collection.add(ids=[key], documents=[value], metadatas=[entry_metadata])

        if self.thread_safe:
            with self.lock:
                _write()
        else:
            _write()

    def delete(self, key):
        """
        Delete a key in ChromaDB.

        Parameters:
        - key (str): Key to delete.
        """
        def _delete():
            self.collection.delete(ids=[key])

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
