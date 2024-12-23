import redis
import asyncio
from datetime import datetime
from .base import SharedMemoryBase

class RedisSharedMemory(SharedMemoryBase):
    """
    Redis-backed shared memory with optional thread-safe and async-safe operations.
    """

    def __init__(self, redis_url="redis://localhost:6379/0", thread_safe=True, async_safe=False):
        """
        Initialize Redis client and thread/async locks.

        Parameters:
        - redis_url (str): Redis connection URL.
        - thread_safe (bool): Enable thread-safe operations.
        - async_safe (bool): Enable async-safe operations.
        """
        self.client = redis.StrictRedis.from_url(redis_url)
        self.thread_safe = thread_safe or async_safe

        if self.thread_safe:
            if async_safe:
                self.lock = asyncio.Lock()
            else:
                self.lock = Lock()

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
        Read memory from Redis.

        Parameters:
        - key (str): Key to fetch. If None, fetch all memory.

        Returns:
        - The value for the key if specified, else all memory as a dictionary.
        """
        def _read():
            if key:
                data = self.client.get(key)
                return eval(data.decode()) if data else None
            return {
                k.decode(): eval(self.client.get(k).decode())
                for k in self.client.keys()
            }

        if self.thread_safe:
            with self.lock:
                return _read()
        else:
            return _read()

    def write(self, key, value, metadata=None):
        """
        Write a key-value pair to Redis with metadata.

        Parameters:
        - key (str): Key to store.
        - value (str): Value to store.
        - metadata (dict): Optional metadata.
        """
        metadata = metadata or {}
        entry = {
            "value": value,
            "metadata": self._get_default_metadata(metadata),
        }

        def _write():
            self.client.set(key, str(entry))

        if self.thread_safe:
            with self.lock:
                _write()
        else:
            _write()

    def delete(self, key):
        """
        Delete a key in Redis.
        """
        def _delete():
            self.client.delete(key)

        if self.thread_safe:
            with self.lock:
                _delete()
        else:
            _delete()

    def clear(self):
        """
        Clear all keys in Redis.
        """
        def _clear():
            self.client.flushdb()

        if self.thread_safe:
            with self.lock:
                _clear()
        else:
            _clear()
