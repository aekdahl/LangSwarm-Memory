from .base import SharedMemoryBase

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
except ImportError:
    cosine_similarity = None
    SentenceTransformer = None

class InMemorySharedMemory(SharedMemoryBase):
    """
    A basic in-memory implementation of shared memory.
    """

    def __init__(self):
        self.memory = {}
        if SentenceTransformer:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embeddings = {}

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
        if SentenceTransformer:
            self.embeddings[key] = self.model.encode(value)

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
        if cosine_similarity is not None and SentenceTransformer is ot None:
            query_vector = self.model.encode(query)
            similarities = {
                key: cosine_similarity([query_vector], [vector])[0][0]
                for key, vector in self.embeddings.items()
            }
            sorted_keys = sorted(similarities, key=similarities.get, reverse=True)
            return [(key, similarities[key]) for key in sorted_keys[:top_k]]
        else:
            return []
