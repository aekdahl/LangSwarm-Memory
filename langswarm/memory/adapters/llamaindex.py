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
            raise ValueError("Unsupported database. Make sure LlamaIndex is installed.")

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

try:
    import pinecone
    from llama_index import PineconeIndex, Document
except ImportError:
    pinecone = None
    PineconeIndex = None
    Document = None

class LlamaIndexPineconeAdapter(LlamaIndexAdapter):
    """
    Adapter for Pinecone integration with LlamaIndex.

    Setup:
        1. Install Pinecone: `pip install pinecone-client`.
        2. Initialize Pinecone with your API key and environment:
           ```
           pinecone.init(api_key="your-api-key", environment="your-environment")
           ```

    Usage:
        Add, query, and manage documents in a Pinecone-backed vector index.
    """
    def __init__(self, index_name="pinecone-index"):
        if pinecone is None or PineconeIndex is None:
            raise ImportError("Pinecone or LlamaIndex is not installed. Please install the required packages.")

        self.index_name = index_name
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=768)  # Update dimension based on your embedding model
        self.index = PineconeIndex(index_name=index_name)

    def add_documents(self, documents):
        docs = [Document(text=doc["text"], metadata=doc.get("metadata", {})) for doc in documents]
        self.index.insert(docs)

    def query(self, query_text):
        return self.index.query(query_text)

    def delete(self, document_ids):
        self.index.delete(document_ids)


try:
    from llama_index import WeaviateIndex, Document
except ImportError:
    WeaviateIndex = None
    Document = None

class LlamaIndexWeaviateAdapter(LlamaIndexAdapter):
    """
    Adapter for Weaviate integration with LlamaIndex.

    Setup:
        1. Install Weaviate client: `pip install weaviate-client`.
        2. Ensure you have a running Weaviate instance and its URL.

    Usage:
        Add, query, and manage documents in a Weaviate-backed vector index.
    """
    def __init__(self, weaviate_url):
        if WeaviateIndex is None:
            raise ImportError("Weaviate or LlamaIndex is not installed. Please install the required packages.")

        self.index = WeaviateIndex(weaviate_url=weaviate_url)

    def add_documents(self, documents):
        docs = [Document(text=doc["text"], metadata=doc.get("metadata", {})) for doc in documents]
        self.index.insert(docs)

    def query(self, query_text):
        return self.index.query(query_text)

    def delete(self, document_ids):
        raise NotImplementedError("Document deletion is not yet supported for Weaviate.")

class WeaviateAdapter(DatabaseAdapter):
    def __init__(self, *args, **kwargs):
        if Weaviate:
            self.db = Weaviate(client=kwargs["client"])
        else:
            raise ValueError("Weaviate package is not installed.")

    def add_documents(self, documents):
        for doc in documents:
            self.db.add_text(doc["text"], metadata=doc.get("metadata", {}))

    def add_documents_with_metadata(self, documents, metadata):
        for doc, meta in zip(documents, metadata):
            self.db.add_text(doc, metadata=meta)

    def query(self, query, filters=None):
        return self.db.query(query, filters=filters)

    def query_by_metadata(self, metadata_query, top_k=5):
        return self.db.query_by_metadata(metadata_query, top_k=top_k)

    def delete(self, document_ids):
        for doc_id in document_ids:
            self.db.delete_by_id(doc_id)

    def delete_by_metadata(self, metadata_query):
        self.db.delete_by_metadata(metadata_query)





try:
    from llama_index import FAISSIndex, Document
except ImportError:
    FAISSIndex = None
    Document = None

class LlamaIndexFAISSAdapter(LlamaIndexAdapter):
    """
    Adapter for FAISS integration with LlamaIndex.

    Setup:
        1. Install FAISS: `pip install faiss-cpu`.
        2. Initialize a FAISS index for local vector storage.

    Usage:
        Add, query, and manage documents in a FAISS-backed vector index.
    """
    def __init__(self, index_path="faiss_index.json"):
        if FAISSIndex is None:
            raise ImportError("FAISS or LlamaIndex is not installed. Please install the required packages.")

        try:
            self.index = FAISSIndex.load_from_disk(index_path)
        except FileNotFoundError:
            self.index = FAISSIndex([])

    def add_documents(self, documents):
        docs = [Document(text=doc["text"], metadata=doc.get("metadata", {})) for doc in documents]
        self.index.insert(docs)
        self.index.save_to_disk("faiss_index.json")

    def query(self, query_text):
        return self.index.query(query_text)

    def delete(self, document_ids):
        raise NotImplementedError("Document deletion is not yet supported for FAISS.")

class FAISSAdapter(DatabaseAdapter):
    def __init__(self, *args, **kwargs):
        if FAISS:
            self.db = FAISS.from_documents([], kwargs["embedding_function"])
        else:
            raise ValueError("FAISS package is not installed.")

    def add_documents(self, documents):
        texts = [doc["text"] for doc in documents]
        metadata = [doc.get("metadata", {}) for doc in documents]
        self.db.add_texts(texts, metadatas=metadata)

    def add_documents_with_metadata(self, documents, metadata):
        self.db.add_texts(documents, metadatas=metadata)

    def query(self, query, filters=None):
        # FAISS does not natively support metadata filtering; mock functionality
        results = self.db.similarity_search(query)
        if filters:
            return [doc for doc in results if all(doc["metadata"].get(k) == v for k, v in filters.items())]
        return results

    def query_by_metadata(self, metadata_query, top_k=5):
        results = self.db.similarity_search(query=None, k=top_k)
        return [doc for doc in results if all(doc["metadata"].get(k) == v for k, v in metadata_query.items())]

    def delete(self, document_ids):
        # FAISS does not support deletion by document ID
        raise NotImplementedError("Deletion by document ID is not supported in FAISS.")

    def delete_by_metadata(self, metadata_query):
        # Mock functionality for metadata-based deletion
        raise NotImplementedError("Deletion by metadata is not supported in FAISS.")



try:
    from llama_index import SQLDatabase, SQLIndex
except ImportError:
    SQLDatabase = None
    SQLIndex = None

class LlamaIndexSQLAdapter(LlamaIndexAdapter):
    """
    Adapter for SQL integration with LlamaIndex.

    Setup:
        1. Install a SQL database driver (e.g., `pip install sqlite`).
        2. Create and configure your database URI.

    Usage:
        Add, query, and manage documents in a SQL-backed index.
    """
    def __init__(self, database_uri, index_path="sql_index.json"):
        if SQLDatabase is None or SQLIndex is None:
            raise ImportError("SQLDatabase or LlamaIndex is not installed. Please install the required packages.")

        self.sql_db = SQLDatabase(database_uri=database_uri)
        try:
            self.index = SQLIndex.load_from_disk(index_path)
        except FileNotFoundError:
            self.index = SQLIndex([], sql_database=self.sql_db)

    def add_documents(self, documents):
        for doc in documents:
            self.sql_db.insert({"text": doc["text"], **doc.get("metadata", {})})
        self.index.refresh()

    def query(self, query_text):
        return self.index.query(query_text)

    def delete(self, document_ids):
        raise NotImplementedError("Document deletion is not yet supported for SQL.")
