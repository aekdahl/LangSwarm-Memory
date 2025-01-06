from .database_adapter import DatabaseAdapter

try:
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings
    import pinecone
except ImportError:
    Pinecone = None
    OpenAIEmbeddings = None
    pinecone = None

class PineconeAdapter(DatabaseAdapter):
    def __init__(self, *args, **kwargs):
        if all(var is not None for var in (Pinecone, OpenAIEmbeddings, pinecone)):
            pinecone.init(api_key=kwargs["api_key"], environment=kwargs["environment"])
            self.db = Pinecone(index_name=kwargs["index_name"], embedding_function=OpenAIEmbeddings())
        else:
            raise ValueError("Unsupported vector database. Make sure LangChain and Pinecone packages are installed.")

    def add_documents(self, documents):
        texts = [doc["text"] for doc in documents]
        metadata = [doc.get("metadata", {}) for doc in documents]
        self.db.add_texts(texts, metadatas=metadata)

    def query(self, query, filters=None):
        return self.db.similarity_search(query, filter=filters)

    def delete(self, document_ids):
        for doc_id in document_ids:
            self.db.delete(doc_id)
