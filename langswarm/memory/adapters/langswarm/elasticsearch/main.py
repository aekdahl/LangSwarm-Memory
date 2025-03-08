class ElasticsearchAdapter(DatabaseAdapter):
    """
    An Elasticsearch adapter for document storage and retrieval.
    
    This retriever enables:
    - Full-text search and metadata-based filtering.
    - Vector search for similarity matching (if enabled).
    - Scalable storage for structured and unstructured data.
    
    Use cases:
    - Storing and retrieving AI-generated knowledge graphs.
    - Enabling hybrid search with metadata and embeddings.
    - Querying structured text data in real-time.
    
    - Usage format:

Replace `action` and parameters as needed.
    """
    
    def __init__(self, identifier, *args, **kwargs):
        self.identifier = identifier
        self.brief = (
            f"ElasticsearchRetriever"
        )
        super().__init__(
            name="ElasticsearchRetriever",
            description=(
                "This retriever allows querying and storing documents in Elasticsearch. "
                "It supports full-text search, metadata filtering, and can be extended for vector search."
            ),
            instruction="""
- Actions and Parameters:
    - `add_documents`: Store documents in Elasticsearch.
      - Parameters:
        - `documents` (List[Dict]): A list of dictionaries with `"text"` and optional `"metadata"`.
    
    - `query`: Perform a search query.
      - Parameters:
        - `query` (str): The text query for full-text search.
        - `filters` (Dict, optional): Metadata-based filtering criteria.
    
    - `delete`: Remove documents by ID.
      - Parameters:
        - `document_ids` (List[str]): A list of document IDs to delete.

- Usage format:

Replace `action` and parameters as needed.
            """
        )
        if Elasticsearch:
            self.db = Elasticsearch(kwargs["connection_string"])
        else:
            raise ValueError("Elasticsearch package is not installed.")

    def run(self, payload, action="query"):
        """
        Execute retrieval actions.
        :param payload: Dict - The input query parameters.
        :param action: str - The action to perform: 'query', 'add_documents', or 'delete'.
        :return: str - The result of the action.
        """
        if action == "query":
            return self.query(**payload)
        elif action == "add_documents":
            return self.add_documents(**payload)
        elif action == "delete":
            return self.delete(**payload)
        else:
            return (
                f"Unsupported action: {action}. Available actions are:\n\n"
                f"{self.instruction}"
            )
        
    def add_documents(self, documents):
        for doc in documents:
            self.db.index(index="documents", body={"text": doc["text"], "metadata": doc.get("metadata", {})})

    def query(self, query, filters=None):
        body = {"query": {"match": {"text": query}}}
        if filters:
            body["query"] = {"bool": {"must": [{"match": {"text": query}}], "filter": [{"term": filters}]}}
        result = self.db.search(index="documents", body=body)
        return self.standardize_output(
            text=result["_source"]["text"],
            source="Elasticsearch",
            metadata={k: v for k, v in result["_source"].items() if k != "text"},
            id=result["_id"],
            relevance_score=result.get("_score")
        )

    def delete(self, document_ids):
        for doc_id in document_ids:
            self.db.delete(index="documents", id=doc_id)

    def delete_by_metadata(self, metadata_query):
        body = {"query": {"bool": {"filter": [{"term": metadata_query}]}}}
        self.db.delete_by_query(index="documents", body=body)

    def capabilities(self) -> Dict[str, bool]:
        return {
            "vector_search": True,  # Elasticsearch supports vector search with extensions like dense_vector.
            "metadata_filtering": True,  # Strong metadata filtering capabilities.
            "semantic_search": True,  # Can be configured for semantic search using embeddings.
        }
