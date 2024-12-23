from abc import ABC, abstractmethod

class SharedMemoryBase(ABC):
    """
    Abstract base class for shared memory implementations.
    """

    @abstractmethod
    def read(self, key=None, context_id=None):
        """
        Read memory. If context_id is provided, fetch all entries in the context.

        Parameters:
        - key: Specific key to fetch (optional).
        - context_id: Fetch all entries in the given context (optional).
        """
        pass

    @abstractmethod
    def write(self, key, value, metadata=None, context_id=None):
        """
        Write a key-value pair to memory with metadata and context_id.

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
        - context_id: ID of the context (optional).
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

