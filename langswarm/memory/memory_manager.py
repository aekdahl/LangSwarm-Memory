import functools

class MemoryManager:
    def __init__(self, backends=None, **kwargs):
        """
        Initialize MemoryManager with multiple backends.

        Args:
            backends (list): List of backend configurations. Each entry can specify
                            the backend type (e.g., "langchain", "llama_index") and
                            corresponding parameters.
        """
        self.adapters = []
        if backends:
            for backend in backends:
                if backend["type"] == "langchain":
                    self.adapters.append(LangChainAdapter(**backend.get("params", {})))
                elif backend["type"] == "llama_index":
                    self.adapters.append(LlamaIndexAdapter(**backend.get("params", {})))
                else:
                    raise ValueError(f"Unsupported backend: {backend['type']}")

    def add_documents(self, documents):
        for adapter in self.adapters:
            adapter.add_documents(documents)

    def query(self, query, filters=None):
        results = []
        for adapter in self.adapters:
            results.extend(adapter.query(query, filters))
        return results

    def delete(self, document_ids):
        for adapter in self.adapters:
            adapter.delete(document_ids)



"""
The SharedMemoryManager will unify and orchestrate memory backends, enabling seamless integration with the LangSwarm ecosystem. Its primary role is to handle shared memory operations across multiple backends and provide a consistent interface for:

Centralized and Federated Memory: Supporting both centralized memory (e.g., for global indices) and federated memory (e.g., agent-specific memory).
Thread-Safe Operations: Ensuring safe concurrent access to shared memory.
Multi-Backend Orchestration: Allowing flexible switching and management of memory backends (e.g., FAISS, Pinecone, Elasticsearch).
"""
class SharedMemoryManager:
    def __init__(self, backend_configs, thread_safe=True):
        """
        Initializes the shared memory manager.

        Args:
            backend_configs (list): List of backend configurations. Each entry specifies the backend type 
                                    (e.g., "faiss", "pinecone") and its parameters.
            thread_safe (bool): Whether to make the manager thread-safe.
        """
        self.backends = self._initialize_backends(backend_configs)
        self.lock = threading.RLock() if thread_safe else None

    def _initialize_backends(self, backend_configs):
        """
        Initializes memory backends based on configurations.

        Args:
            backend_configs (list): List of configurations for each backend.

        Returns:
            list: List of initialized backend instances.
        """
        initialized_backends = []
        for config in backend_configs:
            if config["type"] == "faiss":
                initialized_backends.append(FaissAdapter(**config.get("params", {})))
            elif config["type"] == "pinecone":
                initialized_backends.append(PineconeAdapter(**config.get("params", {})))
            else:
                raise ValueError(f"Unsupported backend type: {config['type']}")
        return initialized_backends

    def _example_usage(self):
        """
        Example usage of the SharedMemoryManager.
        """
        backend_configs = [
            {"type": "faiss", "params": {"dimension": 128}},
            {"type": "pinecone", "params": {"api_key": "your-api-key", "environment": "us-west1"}}
        ]
        manager = SharedMemoryManager(backend_configs)
        documents = [{"text": "Document 1"}, {"text": "Document 2"}]
        manager.add_documents(documents)
        results = manager.query("query text")
        print("Results:", results)

    def _thread_safe(method):
        """Decorator to add thread-safety to methods if enabled."""
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, 'lock') and self.lock:  # Ensure instance has a lock attribute
                with self.lock:
                    return method(self, *args, **kwargs)
            return method(self, *args, **kwargs)
        return wrapper

    @_thread_safe
    def add_documents(self, documents):
        for backend in self.backends:
            backend.add_documents(documents)

    @_thread_safe
    def query(self, query, top_k=5):
        results = []
        for backend in self.backends:
            results.extend(backend.query(query, top_k))
        return results

    @_thread_safe
    def delete(self, ids):
        for backend in self.backends:
            backend.delete(ids)

