from abc import ABC, abstractmethod
from threading import Lock

class SharedMemoryBase(ABC):
    """
    Abstract base class for shared memory implementations.
    """

    @abstractmethod
    def read(self, key=None):
        """Read the entire memory or a specific key."""
        pass

    @abstractmethod
    def write(self, key, value):
        """Write a key-value pair to memory."""
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
