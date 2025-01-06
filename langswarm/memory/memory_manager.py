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
    def __init__(self, backends, thread_safe=True):
        """
        Initializes the shared memory manager.
        
        Args:
            backends (list): List of backend adapters to use.
            thread_safe (bool): Whether to make the manager thread-safe.
        """
        self.backends = backends
        self.lock = threading.RLock() if thread_safe else None

    def _thread_safe(method):
        """Decorator to add thread-safety to methods if enabled."""
        def wrapper(self, *args, **kwargs):
            if self.lock:
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

