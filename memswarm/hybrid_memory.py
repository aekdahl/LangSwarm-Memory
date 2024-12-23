from .base import SharedMemoryBase

class HybridSharedMemory(SharedMemoryBase):
    """
    Hybrid memory combines in-memory caching with a persistent backend.
    """

    def __init__(self, cache, backend):
        self.cache = cache
        self.backend = backend

    def write_scope(self, value, metadata=None, context_id=None):
        """
        Write a value to both in-memory and persistent backends with metadata for agent, group, and context scoping.

        Parameters:
        - value (str): Value to store.
        - metadata (dict): Metadata containing `agent_id` and `group_id`.
        - context_id (str): Context ID for grouping entries.
        """
        self.cache.write_scope(value, metadata, context_id)
        self.backend.write_scope(value, metadata, context_id)

    def read_scope(self, agent_id=None, group_id=None, context_id=None):
        """
        Read memory scoped to a specific agent, group, or context from both in-memory and persistent backends.

        Parameters:
        - agent_id (str): ID of the agent requesting the scope.
        - group_id (str): ID of the group requesting the scope.
        - context_id (str, optional): Restrict results to a specific context.

        Returns:
        - Dictionary of key-value pairs visible to the agent or group within the context.
        """
        # Read scoped data from in-memory and persistent backends
        in_memory_data = self.cache.read_scope(agent_id, group_id, context_id)
        persistent_data = self.backend.read_scope(agent_id, group_id, context_id)

        # Merge and prioritize in-memory data
        combined_data = {**persistent_data, **in_memory_data}
        return combined_data
        
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
