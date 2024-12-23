import redis
from .base import SharedMemoryBase

class RedisSharedMemory(SharedMemoryBase):
    """
    Redis-backed shared memory.
    """

    def __init__(self, redis_url="redis://localhost:6379/0"):
        self.client = redis.StrictRedis.from_url(redis_url)

    def read(self, key=None):
        if key:
            return self.client.get(key).decode() if self.client.get(key) else None
        return {key.decode(): self.client.get(key).decode() for key in self.client.keys()}

    def write(self, key, value):
        self.client.set(key, value)

    def delete(self, key):
        self.client.delete(key)

    def clear(self):
        self.client.flushdb()

