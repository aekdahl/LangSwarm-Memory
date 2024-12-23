from .base import SharedMemoryBase

class HybridSharedMemory(SharedMemoryBase):
    """
    Hybrid memory combines in-memory caching with a persistent backend.
    """

    def __init__(self, cache, backend):
        self.cache = cache
        self.backend = backend

    def read(self, key=None):
        # Try reading from cache first
        value = self.cache.read(key)
        if value is not None:
            return value

        # Fallback to backend if not found in cache
        value = self.backend.read(key)
        if value is not None:
            self.cache.write(key, value)  # Cache the result
        return value

    def write(self, key, value):
        self.cache.write(key, value)
        self.backend.write(key, value)

    def delete(self, key):
        self.cache.delete(key)
        self.backend.delete(key)

    def clear(self):
        self.cache.clear()
        self.backend.clear()
