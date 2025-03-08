class RedisAdapter(DatabaseAdapter):
    """
    A fast key-value document store for structured retrieval using Redis.
    
    This retriever enables:
    - Storing and retrieving text documents efficiently.
    - Performing keyword searches with metadata filtering.
    - Managing document storage with insertion, querying, and deletion operations.
    
    Use cases:
    - Storing structured memory for AI agents.
    - Fast retrieval of past interactions or logs.
    - Querying metadata-enriched Redis storage.
    
    - Usage format:

Replace `action` and parameters as needed.
    """
    def __init__(self, identifier, redis_url="redis://localhost:6379/0"):
        self.identifier = identifier
        self.brief = (
            f"RedisRetriever"
        )
        super().__init__(
            name="RedisRetriever",
            description=(
                "This retriever enables document storage and retrieval using Redis. "
                "It supports keyword-based searching and metadata filtering. "
                "Ideal for real-time caching and AI memory management."
            ),
            instruction="""
- Actions and Parameters:
    - `add_documents`: Add documents to Redis.
      - Parameters:
        - `documents` (List[Dict]): A list of dictionaries with keys `"key"`, `"text"`, and optional `"metadata"`.
    
    - `query`: Perform a keyword-based Redis search.
      - Parameters:
        - `query` (str): The text query for retrieval.
        - `filters` (Dict, optional): Metadata filters for refining results.
    
    - `delete`: Remove documents from Redis by ID.
      - Parameters:
        - `document_ids` (List[str]): A list of document keys to remove.

- Usage format:

Replace `action` and parameters as needed.
            """
        )
        if any(var is None for var in (redis)):
            raise ValueError("Unsupported database. Make sure sqlite3 is installed.")
            
        self.client = redis.StrictRedis.from_url(redis_url)

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
            key = doc.get("key", "")
            value = doc.get("text", "")
            metadata = doc.get("metadata", {})
            self.client.set(key, str({"value": value, "metadata": metadata}))

    def query(self, query, filters=None):
        keys = self.client.keys("*")
        results = []
        for key in keys:
            entry = eval(self.client.get(key).decode())
            if query.lower() in entry["value"].lower():
                if filters and not all(
                    entry["metadata"].get(k) == v for k, v in filters.items()
                ):
                    continue
                results.append({"key": key.decode(), **entry})

        return self.standardize_output(
            text=[result["value"] for result in results],
            source="Redis",
            metadata=[result["metadata"] for result in results],
            id=[result["key"] for result in results]
        )

    def delete(self, document_ids):
        for doc_id in document_ids:
            self.client.delete(doc_id)

    def capabilities(self) -> Dict[str, bool]:
        return {
            "vector_search": False,  # Redis requires vector extensions like RediSearch for this.
            "metadata_filtering": True,  # Supports metadata-based filtering if implemented.
            "semantic_search": False,  # No built-in semantic search support.
        }
