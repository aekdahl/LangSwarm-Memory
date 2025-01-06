import threading

class ThreadSafeAdapter:
    def __init__(self, adapter):
        """
        Wraps an existing adapter to make it thread-safe.

        Args:
            adapter: The adapter to wrap (e.g., FAISSBackend, PineconeBackend, etc.).
        """
        self.adapter = adapter
        self.lock = threading.RLock()

    def add_documents(self, documents):
        """
        Thread-safe method to add documents to the underlying adapter.

        Args:
            documents (list): List of documents to add.
        """
        with self.lock:
            return self.adapter.add_documents(documents)

    def query(self, query, top_k=5):
        """
        Thread-safe method to query the underlying adapter.

        Args:
            query (str): Query string.
            top_k (int): Number of top results to retrieve.

        Returns:
            list: Query results.
        """
        with self.lock:
            return self.adapter.query(query, top_k)

    def delete(self, ids):
        """
        Thread-safe method to delete documents from the underlying adapter.

        Args:
            ids (list): List of document IDs to delete.
        """
        with self.lock:
            return self.adapter.delete(ids)

    def __getattr__(self, attr):
        """
        Delegate any other methods or attributes to the underlying adapter.
        
        This allows access to other methods (if needed) without redefining them.
        """
        return getattr(self.adapter, attr)
