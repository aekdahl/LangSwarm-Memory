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
    def write(self, key, value, metadata=None):
        """
        Write a key-value pair to memory with optional metadata.

        Parameters:
        - key (str): Key to store.
        - value (str): Value to store.
        - metadata (dict, optional): Metadata about the entry. Fields:
          - agent (str): Agent identifier.
          - timestamp (str): ISO 8601 timestamp (default: now).
          - llm (str): Optional LLM used.
          - action (str): Optional action type (e.g., query, response).
          - confidence (float): Optional confidence score.
          - query (str): Optional original query leading to this entry.
        """
        pass

    @abstractmethod
    def delete(self, key):
        """Delete a specific key from memory."""
        pass

    @abstractmethod
    def clear(self):
        """Clear all memory."""
        pass

    @abstractmethod
    def similarity_search(self, query, top_k=5):
        """
        Perform similarity search on stored data.

        Parameters:
        - query (str): Query string to search.
        - top_k (int): Number of top similar results to return.

        Returns:
        - List of most similar documents.
        """
        raise NotImplementedError("Similarity search is not supported for this backend.")

