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
