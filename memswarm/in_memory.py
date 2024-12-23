from threading import Lock
from .base import SharedMemoryBase

class InMemorySharedMemory(SharedMemoryBase):
    """
    Thread-safe in-memory shared memory.
    """

    def __init__(self):
        self.memory = {}
        self.lock = Lock()

    def read(self):
        with self.lock:
            return self.memory.copy()

    def write(self, key, value):
        with self.lock:
            self.memory[key] = value

    def delete(self, key):
        with self.lock:
            if key in self.memory:
                del self.memory[key]

    def clear(self):
        with self.lock:
            self.memory.clear()
