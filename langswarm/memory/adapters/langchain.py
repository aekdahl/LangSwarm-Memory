from .database_adapter import DatabaseAdapter

try:
    from langchain.embeddings.openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None
    
try:
    from langchain.vectorstores import Pinecone
    import pinecone
except ImportError:
    Pinecone = None

try:
    from langchain.vectorstores import Weaviate
except ImportError:
    Weaviate = None

try:
    from langchain.vectorstores import Milvus
except ImportError:
    Milvus = None

try:
    from langchain.vectorstores import Qdrant
except ImportError:
    Qdrant = None

try:
    from langchain.vectorstores import SQLite
except ImportError:
    SQLite = None


class PineconeAdapter(DatabaseAdapter):
    def __init__(self, *args, **kwargs):
        if all(var is not None for var in (Pinecone, OpenAIEmbeddings)):
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


class WeaviateAdapter(DatabaseAdapter):
    """
    When working with LangChain's Weaviate integration, you can optionally provide 
    this client if you already have a preconfigured or specialized Weaviate client 
    setup. Otherwise, LangChain can initialize its own connection to the Weaviate 
    instance based on the url and authentication details provided.
    """
    def __init__(self, *args, **kwargs):
        if all(var is not None for var in (Weaviate, OpenAIEmbeddings)):
            self.db = Weaviate(
                url=kwargs["weaviate_url"],
                embedding_function=OpenAIEmbeddings(),
                client=kwargs.get("weaviate_client", None)  # Optional: Add Weaviate client instance if needed
            )
        else:
            raise ValueError("Unsupported vector database. Make sure LangChain and Weaviate packages are installed.")

    def add_documents(self, documents):
        texts = [doc["text"] for doc in documents]
        metadata = [doc.get("metadata", {}) for doc in documents]
        self.db.add_texts(texts, metadatas=metadata)

    def query(self, query, filters=None):
        return self.db.similarity_search(query, filter=filters)

    def delete(self, document_ids):
        # Not directly supported in LangChain's Weaviate implementation
        raise NotImplementedError("Document deletion is not yet supported in WeaviateAdapter.")


class MilvusAdapter(DatabaseAdapter):
    def __init__(self, *args, **kwargs):
        if all(var is not None for var in (Milvus, OpenAIEmbeddings)):
            self.db = Milvus(
                embedding_function=OpenAIEmbeddings(),
                collection_name=kwargs["collection_name"],
                connection_args={
                    "host": kwargs["milvus_host"],
                    "port": kwargs["milvus_port"]
                }
            )
        else:
            raise ValueError("Unsupported vector database. Make sure LangChain and Milvus packages are installed.")

    def add_documents(self, documents):
        texts = [doc["text"] for doc in documents]
        metadata = [doc.get("metadata", {}) for doc in documents]
        self.db.add_texts(texts, metadatas=metadata)

    def query(self, query, filters=None):
        return self.db.similarity_search(query, filter=filters)

    def delete(self, document_ids):
        # Not directly supported in LangChain's Milvus implementation
        raise NotImplementedError("Document deletion is not yet supported in MilvusAdapter.")


class QdrantAdapter(DatabaseAdapter):
    def __init__(self, *args, **kwargs):
        if all(var is not None for var in (Qdrant, OpenAIEmbeddings)):
            self.db = Qdrant(
                host=kwargs["qdrant_host"],
                port=kwargs["qdrant_port"],
                embedding_function=OpenAIEmbeddings(),
                collection_name=kwargs["collection_name"]
            )
        else:
            raise ValueError("Unsupported vector database. Make sure LangChain and Qdrant packages are installed.")

    def add_documents(self, documents):
        texts = [doc["text"] for doc in documents]
        metadata = [doc.get("metadata", {}) for doc in documents]
        self.db.add_texts(texts, metadatas=metadata)

    def query(self, query, filters=None):
        return self.db.similarity_search(query, filter=filters)

    def delete(self, document_ids):
        self.db.delete(ids=document_ids)


class SQLiteAdapter(DatabaseAdapter):
    def __init__(self, *args, **kwargs):
        if all(var is not None for var in (SQLite, OpenAIEmbeddings)):
            self.db = SQLite(
                embedding_function=OpenAIEmbeddings(),
                database_path=kwargs["database_path"],
                table_name=kwargs["table_name"]
            )
        else:
            raise ValueError("Unsupported database. Make sure LangChain and SQLite packages are installed.")

    def add_documents(self, documents):
        texts = [doc["text"] for doc in documents]
        metadata = [doc.get("metadata", {}) for doc in documents]
        self.db.add_texts(texts, metadatas=metadata)

    def query(self, query, filters=None):
        return self.db.similarity_search(query, filter=filters)

    def delete(self, document_ids):
        self.db.delete(ids=document_ids)
