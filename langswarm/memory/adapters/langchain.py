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

try:
    from langchain.vectorstores import Redis
    import redis
except ImportError:
    Redis = None

try:
    from langchain.vectorstores import Chroma
    import chromadb
except ImportError:
    Chroma = None


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

    def add_documents_with_metadata(self, documents, metadata):
        self.db.add_texts(documents, metadatas=metadata)
        
    def query(self, query, filters=None):
        return self.db.similarity_search(query, filter=filters)

    def query_by_metadata(self, metadata_query, top_k=5):
        return self.db.similarity_search(query=None, filter=metadata_query, k=top_k)
        
    def delete(self, document_ids):
        for doc_id in document_ids:
            self.db.delete(doc_id)

    def delete_by_metadata(self, metadata_query):
        results = self.db.similarity_search(query=None, filter=metadata_query, k=1000)
        ids_to_delete = [doc["id"] for doc in results]
        for doc_id in ids_to_delete:
            self.db.delete(doc_id)
            
    def capabilities(self) -> Dict[str, bool]:
        return {
            "vector_search": True,  # Pinecone supports vector-based similarity search.
            "metadata_filtering": True,  # Metadata filtering is available.
            "semantic_search": True,  # Embedding-based semantic search supported via OpenAIEmbeddings.
        }


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

    def add_documents_with_metadata(self, documents, metadata):
        for doc, meta in zip(documents, metadata):
            self.db.add_text(doc, metadata=meta)

    def query(self, query, filters=None):
        return self.db.similarity_search(query, filter=filters)
        # return self.db.query(query, filters=filters) <-- Should we use the simple query instead?        

    def query_by_metadata(self, metadata_query, top_k=5):
        return self.db.query_by_metadata(metadata_query, top_k=top_k)

    def delete(self, document_ids):
        try:
            for doc_id in document_ids:
                self.db.delete_by_id(doc_id)
        except:
            # Not directly supported in LangChain's Weaviate implementation
            raise NotImplementedError("Document deletion is not yet supported in WeaviateAdapter.")

    def delete_by_metadata(self, metadata_query):
        self.db.delete_by_metadata(metadata_query)

    def capabilities(self) -> Dict[str, bool]:
        return {
            "vector_search": True,  # Weaviate supports vector-based similarity search.
            "metadata_filtering": True,  # Metadata filtering is available.
            "semantic_search": True,  # Semantic search is supported with embeddings.
        }


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

    def add_documents_with_metadata(self, documents, metadata):
        self.db.add_texts(documents, metadatas=metadata)

    def query(self, query, filters=None):
        return self.db.similarity_search(query, filter=filters)

    def query_by_metadata(self, metadata_query, top_k=5):
        return self.db.query_by_metadata(metadata_query, top_k=top_k)

    def delete(self, document_ids):
        try:
            for doc_id in document_ids:
                self.db.delete_by_id(doc_id)
        except:
            # Not directly supported in LangChain's Milvus implementation
            raise NotImplementedError("Document deletion is not yet supported in MilvusAdapter.")

    def delete_by_metadata(self, metadata_query):
        self.db.delete_by_metadata(metadata_query)

    def capabilities(self) -> Dict[str, bool]:
        return {
            "vector_search": True,  # Milvus supports vector-based similarity search.
            "metadata_filtering": True,  # Metadata filtering is available.
            "semantic_search": True,  # Semantic search is supported with embeddings.
        }


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

    def capabilities(self) -> Dict[str, bool]:
        return {
            "vector_search": True,  # Qdrant supports vector-based similarity search.
            "metadata_filtering": True,  # Metadata filtering is available.
            "semantic_search": True,  # Embedding-based semantic search supported.
        }


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

    def capabilities(self) -> Dict[str, bool]:
        return {
            "vector_search": True,  # LangChain's SQLite integration supports vector-based search.
            "metadata_filtering": True,  # Metadata filtering is implemented via SQL queries.
            "semantic_search": True,  # Embedding-based semantic search supported.
        }


class RedisAdapter(DatabaseAdapter):
    def __init__(self, *args, **kwargs):
        if Redis:
            self.db = Redis(index_name=kwargs["index_name"], redis_url=kwargs["redis_url"])
        else:
            raise ValueError("Redis package is not installed.")

    def add_documents(self, documents):
        texts = [doc["text"] for doc in documents]
        metadata = [doc.get("metadata", {}) for doc in documents]
        self.db.add_texts(texts, metadatas=metadata)

    def add_documents_with_metadata(self, documents, metadata):
        self.db.add_texts(documents, metadatas=metadata)

    def query(self, query, filters=None):
        return self.db.similarity_search(query, filter=filters)

    def query_by_metadata(self, metadata_query, top_k=5):
        return self.db.similarity_search(query=None, filter=metadata_query, k=top_k)

    def delete(self, document_ids):
        for doc_id in document_ids:
            self.db.delete(doc_id)

    def delete_by_metadata(self, metadata_query):
        self.db.delete(filter=metadata_query)


class ChromaAdapter(DatabaseAdapter):
    def __init__(self, *args, **kwargs):
        if Chroma:
            self.db = Chroma(
                collection_name=kwargs["collection_name"],
                embedding_function=kwargs["embedding_function"],
            )
        else:
            raise ValueError("Chroma package is not installed.")

    def add_documents(self, documents):
        texts = [doc["text"] for doc in documents]
        metadata = [doc.get("metadata", {}) for doc in documents]
        self.db.add_texts(texts, metadatas=metadata)

    def add_documents_with_metadata(self, documents, metadata):
        self.db.add_texts(documents, metadatas=metadata)

    def query(self, query, filters=None):
        return self.db.similarity_search(query, filter=filters)

    def query_by_metadata(self, metadata_query, top_k=5):
        return self.db.similarity_search(query=None, filter=metadata_query, k=top_k)

    def delete(self, document_ids):
        for doc_id in document_ids:
            self.db.delete(doc_id)

    def delete_by_metadata(self, metadata_query):
        self.db.delete(filter=metadata_query)
