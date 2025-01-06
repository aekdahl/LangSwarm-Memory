from .database_adapter import DatabaseAdapter

try:
    from llama_index import GPTSimpleVectorIndex, Document
except ImportError:
    GPTSimpleVectorIndex = None
    Document = None

class LoadFromDiskAdapter(DatabaseAdapter):
    def __init__(self, index_path="index.json"):
        if all(var is not None for var in (GPTSimpleVectorIndex, Document)):
            try:
                self.index = GPTSimpleVectorIndex.load_from_disk(index_path)
            except FileNotFoundError:
                self.index = GPTSimpleVectorIndex([])
        else:
            xxx

    def add_documents(self, documents):
        docs = [Document(text=doc["text"], metadata=doc.get("metadata", {})) for doc in documents]
        self.index.insert(docs)
        self.index.save_to_disk()

    def query(self, query, filters=None):
        results = self.index.query(query)
        if filters:
            results = [res for res in results if all(res.extra_info.get(k) == v for k, v in filters.items())]
        return results

    def delete(self, document_ids):
        raise NotImplementedError("Delete functionality not implemented for LlamaIndex")
