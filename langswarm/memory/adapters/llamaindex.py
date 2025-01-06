from .database_adapter import DatabaseAdapter

try:
    from llama_index import Document
except ImportError:
    Document = None
    
try:
    from llama_index import GPTSimpleVectorIndex
except ImportError:
    GPTSimpleVectorIndex = None

try:
    import pinecone
    from llama_index import PineconeIndex
except ImportError:
    pinecone = None
    PineconeIndex = None

try:
    from llama_index import WeaviateIndex
except ImportError:
    WeaviateIndex = None

try:
    from llama_index import FAISSIndex
except ImportError:
    FAISSIndex = None

try:
    from llama_index import SQLDatabase, SQLIndex
except ImportError:
    SQLDatabase = None
    SQLIndex = None


class LlamaIndexDiskAdapter(DatabaseAdapter):
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

    def capabilities(self) -> Dict[str, bool]:
        return {
            "vector_search": True,
            "metadata_filtering": True,
            "semantic_search": True,
        }


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

    def capabilities(self) -> Dict[str, bool]:
        return {
            "vector_search": True,
            "metadata_filtering": True,
            "semantic_search": True,
        }


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

    def capabilities(self) -> Dict[str, bool]:
        return {
            "vector_search": True,
            "metadata_filtering": True,
            "semantic_search": True,
        }


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

    def capabilities(self) -> Dict[str, bool]:
        return {
            "vector_search": True,
            "metadata_filtering": False,  # FAISS lacks native metadata filtering.
            "semantic_search": True,
        }


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

    def capabilities(self) -> Dict[str, bool]:
        return {
            "vector_search": False,
            "metadata_filtering": True,
            "semantic_search": False,
        }
