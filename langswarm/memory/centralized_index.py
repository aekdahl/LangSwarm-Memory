from datetime import datetime, timedelta
try:
    from llama_index import GPTSimpleVectorIndex, Document
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    GPTSimpleVectorIndex = None
    Document = None
    LLAMA_INDEX_AVAILABLE = False


class CentralizedIndex:
    def __init__(self, index_path="memory_index.json", expiration_days=None):
        """
        Centralized index for long-term memory and shared knowledge.

        :param index_path: Path to store the index file.
        :param expiration_days: Number of days before memory fades (optional).
        """
        self.index_path = index_path
        self.expiration_days = expiration_days
        self._indexing_is_available = LLAMA_INDEX_AVAILABLE

        if not LLAMA_INDEX_AVAILABLE:
            self.index = None
            print("LlamaIndex is not installed. Memory indexing features are disabled.")
            return

        # Try to load an existing index or create a new one
        try:
            self.index = GPTSimpleVectorIndex.load_from_disk(index_path)
        except FileNotFoundError:
            self.index = GPTSimpleVectorIndex([])

    @property
    def indexing_is_available(self):
        """Check if indexing is available."""
        return self._indexing_is_available

    def add_documents(self, docs):
        """
        Add documents to the centralized index.

        :param docs: List of documents with text and optional metadata.
        """
        if not self.indexing_is_available:
            print("Indexing features are unavailable.")
            return

        documents = [
            Document(text=doc["text"], metadata={
                **doc.get("metadata", {}),
                "timestamp": datetime.now().isoformat()  # Add a timestamp to each document
            })
            for doc in docs
        ]
        self.index.insert(documents)
        self.index.save_to_disk(self.index_path)

    def query(self, query_text, metadata_filter=None):
        """
        Query the index with optional metadata filtering.

        :param query_text: The text query.
        :param metadata_filter: Dictionary of metadata filters (optional).
        :return: Filtered results based on the query and metadata.
        """
        if not self.indexing_is_available:
            print("Indexing features are unavailable.")
            return []

        results = self.index.query(query_text)

        # Apply metadata filtering if specified
        if metadata_filter:
            results = [
                res for res in results
                if all(res.extra_info.get(key) == value for key, value in metadata_filter.items())
            ]
        return results

    def purge_expired_documents(self):
        """
        Remove documents from the index that have exceeded the expiration period.
        """
        if not self.indexing_is_available or self.expiration_days is None:
            return

        now = datetime.now()
        valid_documents = []
        for doc in self.index.documents:
            timestamp = doc.extra_info.get("timestamp")
            if timestamp:
                doc_time = datetime.fromisoformat(timestamp)
                if (now - doc_time) <= timedelta(days=self.expiration_days):
                    valid_documents.append(doc)

        self.index = GPTSimpleVectorIndex(valid_documents)
        self.index.save_to_disk(self.index_path)
