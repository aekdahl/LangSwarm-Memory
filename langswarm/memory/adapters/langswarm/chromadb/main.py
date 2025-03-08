class ChromaDBAdapter(DatabaseAdapter):
    """
    A high-performance vector database adapter for semantic search using ChromaDB.
    
    This retriever enables:
    - Storing and retrieving vector-embedded documents.
    - Performing semantic and metadata-based search.
    - Managing indexed collections efficiently.
    
    Use cases:
    - AI memory retrieval and context-aware responses.
    - Fast, scalable semantic search for LLMs.
    - Querying documents with metadata-based filtering.
    
    - Usage format:

Replace `action` and parameters as needed.
    """
    
    def __init__(self, identifier, collection_name="shared_memory", persist_directory=None, brief=None):
        self.identifier = identifier
        self.brief = brief or (
            f"The {identifier} adapter enables semantic search in the {collection_name} collection"
        )
        super().__init__(
            name="ChromaDBRetriever",
            description=(
                f"{identifier} adapter enables semantic search using ChromaDB, in the {collection_name} collection."
            ),
            instruction="""
- Actions and Parameters:
    - `add_documents`: Store documents in ChromaDB.
      - Parameters:
        - `documents` (List[Dict]): A list of dictionaries with `"key"`, `"text"`, and `"metadata"`.
    
    - `query`: Perform a semantic search.
      - Parameters:
        - `query` (str): The search query.
        - `filters` (Dict, optional): Metadata filters for refining results.
        - `n` (int, optional): Number of results to retrieve (default: 5).
    
    - `delete`: Remove documents by ID.
      - Parameters:
        - `document_ids` (List[str]): A list of document IDs to delete.

- Usage format:

Replace `action` and parameters as needed.
            """
        )
        if ChromaDB is None:
            raise ValueError("Unsupported database. Make sure ChromaDB is installed.")
        if persist_directory:
            self.client = ChromaDB(Settings(persist_directory=persist_directory))
        else:
            self.client = ChromaDB(Settings())
        self.collection = self.client.get_or_create_collection(name=collection_name)

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
            self.collection.add(ids=[key], documents=[value], metadatas=[metadata])

    def query(self, query, filters=None, n=5):
        results = self.collection.query(query_texts=query)
        if filters:
            return [
                {
                    "key": res["id"],
                    "text": res["document"],
                    "metadata": res["metadata"],
                }
                for res in results
                if all(
                    res["metadata"].get(k) == v for k, v in filters.items()
                )
            ]
        
        return self.standardize_output(
            text=results["documents"][0],
            source="ChromaDB",
            metadata=results["metadatas"][0],
            id=results["ids"][0]
        )

    def delete(self, document_ids):
        for doc_id in document_ids:
            self.collection.delete(ids=[doc_id])

    def capabilities(self) -> Dict[str, bool]:
        return {
            "vector_search": True,  # Chroma supports vector-based search.
            "metadata_filtering": True,  # Metadata filtering is a core feature.
            "semantic_search": True,  # Supports embeddings for semantic search.
        }

    def _build_filter_conditions(self, filters: Dict):
        """
        Construct filter conditions for metadata queries.
        
        Args:
            filters (Dict): Filtering conditions.

        Returns:
            Filter: Qdrant filter object.
        """
        conditions = []
        for key, value in filters.items():
            if isinstance(value, (int, float)):
                conditions.append(FieldCondition(key=key, range=Range(gte=value)))
            else:
                conditions.append(FieldCondition(key=key, match={"value": value}))
        
        return Filter(must=conditions)
