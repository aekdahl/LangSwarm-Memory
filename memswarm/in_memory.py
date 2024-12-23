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

    def write_scope(self, value, metadata=None, context_id=None):
        metadata = metadata or {}
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        key = f"{context_id}:{timestamp}"
        entry_metadata = {
            "agent_id": metadata.get("agent_id"),
            "group_id": metadata.get("group_id"),
            "timestamp": timestamp,
            **metadata,
        }
        entry = {"value": value, "metadata": entry_metadata}

        if self.thread_safe:
            with self.lock:
                self.memory[key] = entry
        else:
            self.memory[key] = entry

    def read_context(self, context_id):
        """
        Read all entries for a given context_id in timestamp order.
        """
        def _read():
            return {
                key: value
                for key, value in sorted(self.memory.items())
                if key.startswith(f"{context_id}:")
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

        if self.thread_safe:
            with self.lock:
                self.memory[key] = entry
        else:
            self.memory[key] = entry
            
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
