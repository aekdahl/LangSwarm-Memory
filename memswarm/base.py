from abc import ABC, abstractmethod

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
