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


class OptimizedRedisMemory(SharedMemoryBase):
    """
    Redis-backed shared memory with atomic transactions.
    """

    def __init__(self, redis_url="redis://localhost:6379/0"):
        """
        Initialize Redis client.

        Parameters:
        - redis_url (str): Redis connection URL.
        """
        self.client = redis.StrictRedis.from_url(redis_url)

    def read(self, key=None):
        """
        Read memory from Redis.

        Parameters:
        - key: Optional key to fetch specific data. If None, fetch all memory.

        Returns:
        - The value for the key if specified, else all memory as a dictionary.
        """
        if key:
            value = self.client.get(key)
            return value.decode() if value else None

        return {key.decode(): self.client.get(key).decode() for key in self.client.keys()}

    def write(self, key, value):
        """
        Write to Redis using an atomic transaction.

        Parameters:
        - key (str): Key to write.
        - value (str): Value to associate with the key.
        """
        with self.client.pipeline() as pipe:
            pipe.set(key, value)
            pipe.execute()

    def delete(self, key):
        """
        Delete a key in Redis.

        Parameters:
        - key (str): Key to delete.
        """
        with self.client.pipeline() as pipe:
            pipe.delete(key)
            pipe.execute()

    def clear(self):
        """
        Clear all keys in Redis.
        """
        self.client.flushdb()
