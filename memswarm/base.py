from abc import ABC, abstractmethod
from datetime import datetime
from threading import Lock
import asyncio

class SharedMemoryBase(ABC):
    """
    Abstract base class for shared memory implementations.
    """

    @abstractmethod
    def read(self, key=None):
        """Read the entire memory or a specific key."""
        pass

    @abstractmethod
    def write(self, key, value, metadata=None):
        """
        Write a key-value pair to memory with optional metadata.

        Parameters:
        - key (str): Key to store.
        - value (str): Value to store.
        - metadata (dict, optional): Metadata about the entry. Fields:
          - agent (str): Agent identifier.
          - timestamp (str): ISO 8601 timestamp (default: now).
          - llm (str): Optional LLM used.
          - action (str): Optional action type (e.g., query, response).
          - confidence (float): Optional confidence score.
          - query (str): Optional original query leading to this entry.
        """
        pass

    @abstractmethod
    def delete(self, key):
        """Delete a specific key from memory."""
        pass

    @abstractmethod
    def clear(self):
        """Clear all memory."""
        pass


class ThreadSafeSharedMemory(SharedMemoryBase):
    """
    A thread-safe wrapper for shared memory implementations.
    """

    def __init__(self, memory: SharedMemoryBase):
        self.memory = memory
        self.lock = Lock()

    def read(self, key=None):
        with self.lock:
            return self.memory.read(key)

    def write(self, key, value):
        with self.lock:
            self.memory.write(key, value)

    def delete(self, key):
        with self.lock:
            self.memory.delete(key)

    def clear(self):
        with self.lock:
            self.memory.clear()

    @abstractmethod
    def similarity_search(self, query, top_k=5):
        """
        Perform similarity search on stored data.

        Parameters:
        - query (str): Query string to search.
        - top_k (int): Number of top similar results to return.

        Returns:
        - List of most similar documents.
        """
        raise NotImplementedError("Similarity search is not supported for this backend.")


class AsyncSafeSharedMemory(SharedMemoryBase):
    """
    An async-safe wrapper for shared memory implementations.
    """

    def __init__(self, memory: SharedMemoryBase):
        self.memory = memory
        self.lock = asyncio.Lock()

    async def read(self, key=None):
        async with self.lock:
            return self.memory.read(key)

    async def write(self, key, value):
        async with self.lock:
            self.memory.write(key, value)

    async def delete(self, key):
        async with self.lock:
            self.memory.delete(key)

    async def clear(self):
        async with self.lock:
            self.memory.clear()

