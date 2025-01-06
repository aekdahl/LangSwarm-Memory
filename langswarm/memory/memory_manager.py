class MemoryManager:
    def __init__(self, backend="langchain", **kwargs):
        if backend == "langchain":
            self.adapter = LangChainAdapter(**kwargs)
        elif backend == "llama_index":
            self.adapter = LlamaIndexAdapter(index_path=kwargs.get("index_path", "index.json"))
        else:
            raise ValueError("Unsupported backend")

    def add_documents(self, documents):
        self.adapter.add_documents(documents)

    def query(self, query, filters=None):
        return self.adapter.query(query, filters)

    def delete(self, document_ids):
        self.adapter.delete(document_ids)
