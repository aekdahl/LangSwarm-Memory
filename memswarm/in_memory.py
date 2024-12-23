from .base import SharedMemoryBase

class InMemorySharedMemory(SharedMemoryBase):
    """
    A basic in-memory implementation of shared memory.
    """

    def __init__(self):
        self.memory = {}

    def read(self, key=None):
        """
        Read the memory.

        Parameters:
        - key: Optional key to fetch specific data. If None, fetch all memory.

        Returns:
        - The value for the key if specified, else all memory as a dictionary.
        """
        if key:
            return self.memory.get(key)
        return self.memory.copy()

    def write(self, key, value):
        """
        Write a key-value pair to memory.
        """
        self.memory[key] = value

    def delete(self, key):
        """
        Delete a specific key from memory.
        """
        if key in self.memory:
            del self.memory[key]

    def clear(self):
        """
        Clear all memory.
        """
        self.memory.clear()

    def similarity_search(self, query, top_k=5):
        query_vector = self.model.encode(query)
        similarities = {
            key: cosine_similarity([query_vector], [vector])[0][0]
            for key, vector in self.embeddings.items()
        }
        sorted_keys = sorted(similarities, key=similarities.get, reverse=True)
        return [(key, similarities[key]) for key in sorted_keys[:top_k]]
