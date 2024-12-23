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

    def read_scope(self, agent_id=None, group_id=None, context_id=None):
        """
        Read memory entries scoped to a specific agent, group, or context.
        """
        def _filter_scope():
            keys = self.client.keys(f"{context_id}:*" if context_id else "*")
            result = {}
            for key in sorted(keys):  # Ensure lexicographical order
                entry = eval(self.client.get(key).decode())
                metadata = entry.get("metadata", {})
                if agent_id and metadata.get("agent_id") == agent_id:
                    result[key.decode()] = entry
                elif group_id and metadata.get("group_id") == group_id:
                    result[key.decode()] = entry
            return result
    
        if self.thread_safe:
            with self.lock:
                return _filter_scope()
        else:
            return _filter_scope()

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

        def _write():
            self.client.set(key, str(entry))

        if self.thread_safe:
            with self.lock:
                _write()
        else:
            _write()

    def read_context(self, context_id):
        """
        Read all entries for a given context_id in timestamp order.
        """
        def _read():
            keys = sorted(self.client.keys(f"{context_id}:*"))
            return {
                key.decode(): eval(self.client.get(key).decode())
                for key in keys
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
