from .base import SharedMemoryBase

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
except ImportError:
    cosine_similarity = None
    SentenceTransformer = None

import asyncio
from threading import Lock
from datetime import datetime
            
class InMemorySharedMemory(SharedMemoryBase):
    """
    A basic in-memory implementation of shared memory.
    """

    def __init__(self, thread_safe = True, async_safe = False):
        self.memory = {}
        self.thread_safe = thread_safe or async_safe
        
        if self.thread_safe:
            if async_safe:
                self.lock = asyncio.Lock()
            else:
                self.lock = Lock()

        if SentenceTransformer:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embeddings = {}

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
        Read the memory.

        Parameters:
        - key: Optional key to fetch specific data. If None, fetch all memory.

        Returns:
        - The value for the key if specified, else all memory as a dictionary.
        """
        if self.thread_safe:
            with self.lock:
                if key:
                    return self.memory.get(key)
                return self.memory.copy()
        else:
        if key:
            return self.memory.get(key)
        return self.memory.copy()
            
    def write(self, key, value, metadata=None):
        """
        Write a key-value pair to memory.
        """
        metadata = metadata or {}
        _value = {
            "value": value,
            "metadata": self._get_default_metadata(metadata),
        }
        if self.thread_safe:
            with self.lock:
                self.memory[key] = _value
                if SentenceTransformer:
                    self.embeddings[key] = self.model.encode(_value)
        else:
            self.memory[key] = _value
            if SentenceTransformer:
                self.embeddings[key] = self.model.encode(_value)
            
    def delete(self, key):
        """
        Delete a specific key from memory.
        """
        if self.thread_safe:
            with self.lock:
                if key in self.memory:
                    del self.memory[key]
        else:
            if key in self.memory:
                del self.memory[key]
                
    def clear(self):
        """
        Clear all memory.
        """
        if self.thread_safe:
            with self.lock:
                self.memory.clear()
        else:
            self.memory.clear()
            
    def similarity_search(self, query, top_k=5):
        if cosine_similarity is not None and SentenceTransformer is ot None:
            query_vector = self.model.encode(query)
            similarities = {
                key: cosine_similarity([query_vector], [vector])[0][0]
                for key, vector in self.embeddings.items()
            }
            sorted_keys = sorted(similarities, key=similarities.get, reverse=True)
            return [(key, similarities[key]) for key in sorted_keys[:top_k]]
        else:
            return []
