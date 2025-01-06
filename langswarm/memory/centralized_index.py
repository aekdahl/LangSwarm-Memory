import importlib
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

    def _clean_expired_documents(self):
        """
        Internal method to clean up expired documents.
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
    
        # Update the index with valid documents only
        self.index = GPTSimpleVectorIndex(valid_documents)
        self.index.save_to_disk(self.index_path)
    
    def _validate_and_normalize_metadata(self, metadata):
        """
        Validates and normalizes metadata.
    
        Args:
            metadata (dict): Metadata dictionary.
    
        Returns:
            dict: Normalized metadata with lowercase keys.
        """
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary.")
    
        return {str(key).lower(): value for key, value in metadata.items()}
    
    def add_documents(self, docs):
        """
        Add documents to the centralized index with metadata validation.
    
        :param docs: List of documents with text and optional metadata.
        """
        if not self.indexing_is_available:
            print("Indexing features are unavailable.")
            return
    
        self._clean_expired_documents()
    
        documents = [
            Document(text=doc["text"], metadata={
                **self._validate_and_normalize_metadata(doc.get("metadata", {})),
                "timestamp": datetime.now().isoformat()  # Add a timestamp to each document
            })
            for doc in docs
        ]
        self.index.insert(documents)
        self.index.save_to_disk(self.index_path)
    
    def query(self, query_text, metadata_filter=None):
        """
        Query the index with metadata validation and filtering.
    
        :param query_text: The text query.
        :param metadata_filter: Dictionary of metadata filters (optional).
        :return: Filtered results based on the query and metadata.
        """
        self._clean_expired_documents()
    
        if not self.indexing_is_available:
            print("Indexing features are unavailable.")
            return []
    
        results = self.index.query(query_text)
    
        # Apply metadata filtering if specified
        if metadata_filter:
            normalized_filter = self._validate_and_normalize_metadata(metadata_filter)
            results = [
                res for res in results
                if all(res.extra_info.get(key) == value for key, value in normalized_filter.items())
            ]
        return results


    def purge_expired_documents(self):
        """
        Remove documents from the index that have exceeded the expiration period.
        """
        self._clean_expired_documents()





class HybridCentralizedIndex:
    """
    HybridCentralizedIndex manages multiple backends, enabling hybrid retrieval and indexing.
    """

    def __init__(self, config):
        """
        Initialize the hybrid index with the given configuration.
        
        Args:
            config (dict): Configuration specifying backends and their settings.
        """
        self.config = config
        self.backends = {}
        self._check_dependencies()
        self._initialize_backends()

    def _check_dependencies(self):
        """
        Verify that required libraries for enabled backends are installed.
        """
        dependency_map = {
            "faiss": "faiss",
            "elasticsearch": "elasticsearch",
            "pinecone": "pinecone-client",
        }

        for backend, library in dependency_map.items():
            if self.config.get(backend, {}).get("enabled", False):
                if not self._is_library_installed(library):
                    raise ImportError(
                        f"The library '{library}' is required for the '{backend}' backend "
                        f"but is not installed. Please install it using `pip install {library}`."
                    )

    @staticmethod
    def _is_library_installed(library):
        """
        Check if a library is installed.
        
        Args:
            library (str): Name of the library to check.
        
        Returns:
            bool: True if installed, False otherwise.
        """
        try:
            importlib.import_module(library)
            return True
        except ImportError:
            return False

    def _validate_backend_config(self, backend_name, required_fields):
        """
        Validate that the backend configuration contains the required fields.
    
        Args:
            backend_name (str): Name of the backend (e.g., "faiss").
            required_fields (list): List of required configuration fields.
    
        Raises:
            ValueError: If any required field is missing or invalid.
        """
        config = self.config.get(backend_name, {})
        for field in required_fields:
            if field not in config or not config[field]:
                raise ValueError(f"Missing or invalid field '{field}' in {backend_name} configuration.")
    
    def _initialize_backends(self):
        """
        Initialize the configured backends with validated configurations.
        """
        if self.config.get("faiss", {}).get("enabled", False):
            self._validate_backend_config("faiss", ["dimension"])
            self.backends["faiss"] = self._initialize_faiss()
    
        if self.config.get("elasticsearch", {}).get("enabled", False):
            self._validate_backend_config("elasticsearch", ["host"])
            self.backends["elasticsearch"] = self._initialize_elasticsearch()
    
        if self.config.get("pinecone", {}).get("enabled", False):
            self._validate_backend_config("pinecone", ["api_key", "environment", "index_name"])
            self.backends["pinecone"] = self._initialize_pinecone()


    def _initialize_faiss(self):
        """
        Initialize the FAISS backend with error handling.
        """
        try:
            from faiss import IndexFlatL2
            config = self.config["faiss"]
            return IndexFlatL2(config["dimension"])
        except ImportError:
            raise ImportError("FAISS is not installed. Please install it with `pip install faiss-cpu`.")
        except KeyError as e:
            raise ValueError(f"FAISS configuration is missing key: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FAISS backend: {e}")
    
    def _initialize_elasticsearch(self):
        """
        Initialize the Elasticsearch backend with error handling.
        """
        try:
            from elasticsearch import Elasticsearch
            config = self.config["elasticsearch"]
            return Elasticsearch([config["host"]])
        except ImportError:
            raise ImportError("Elasticsearch is not installed. Please install it with `pip install elasticsearch`.")
        except KeyError as e:
            raise ValueError(f"Elasticsearch configuration is missing key: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Elasticsearch backend: {e}")
    
    def _initialize_pinecone(self):
        """
        Initialize the Pinecone backend with error handling.
        """
        try:
            import pinecone
            config = self.config["pinecone"]
            pinecone.init(api_key=config["api_key"], environment=config["environment"])
            return pinecone.Index(config["index_name"])
        except ImportError:
            raise ImportError("Pinecone is not installed. Please install it with `pip install pinecone-client`.")
        except KeyError as e:
            raise ValueError(f"Pinecone configuration is missing key: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone backend: {e}")


    def add_to_index(self, data, backend="faiss", **kwargs):
        """
        Add data to the specified backend.
        
        Args:
            data (Any): Data to be indexed.
            backend (str): Backend to use (default: 'faiss').
            **kwargs: Additional arguments for backend-specific methods.
        """
        if backend not in self.backends:
            raise ValueError(f"Backend '{backend}' is not initialized or supported.")
        if backend == "faiss":
            self.backends[backend].add(data)
        elif backend == "elasticsearch":
            self.backends[backend].index(index=self.config[backend]["index_name"], body=data)
        elif backend == "pinecone":
            self.backends[backend].upsert(data, **kwargs)

    def query(self, query_text, backend=None, filters=None, top_k=5):
        """
        Query the specified backend or all enabled backends.

        Args:
            query_text (str): Query string or vector.
            backend (str): Specific backend to query. If None, queries all enabled backends.
            filters (dict): Optional filters for querying.
            top_k (int): Number of top results to retrieve.

        Returns:
            list: Combined and deduplicated results from the backends.
        """
        results = []

        if backend:
            # Query a specific backend
            if backend not in self.backends:
                raise ValueError(f"Backend '{backend}' is not initialized or supported.")
            results.extend(self._query_backend(backend, query_text, filters, top_k))
        else:
            # Query all enabled backends
            for backend_name, backend_instance in self.backends.items():
                results.extend(self._query_backend(backend_name, query_text, filters, top_k))

        # Deduplicate and sort results if necessary
        results = self._deduplicate_and_sort_results(results, top_k)
        return results

    def _query_backend(self, backend, query_text, filters, top_k):
        """
        Query a specific backend with the provided parameters.

        Args:
            backend (str): Backend name.
            query_text (str): Query string or vector.
            filters (dict): Optional filters for querying.
            top_k (int): Number of top results to retrieve.

        Returns:
            list: Query results from the backend.
        """
        backend_instance = self.backends.get(backend)

        if backend == "faiss":
            # FAISS query logic
            return backend_instance.search(query_text, top_k)
        elif backend == "elasticsearch":
            # Elasticsearch query logic
            body = {"query": {"match": {"text": query_text}}}
            if filters:
                body["query"] = {"bool": {"must": [{"match": {"text": query_text}}], "filter": [{"term": filters}]}}
            return backend_instance.search(index=self.config["elasticsearch"]["index_name"], body=body)["hits"]["hits"]
        elif backend == "pinecone":
            # Pinecone query logic
            return backend_instance.query(vector=query_text, top_k=top_k, filter=filters)
        else:
            raise ValueError(f"Unsupported backend '{backend}' for querying.")

    def _deduplicate_and_sort_results(self, results, top_k):
        """
        Deduplicate and sort query results.

        Args:
            results (list): Raw query results.
            top_k (int): Number of top results to retain.

        Returns:
            list: Deduplicated and sorted results.
        """
        # Deduplicate results based on a unique identifier (e.g., "id" or "text")
        seen = set()
        deduplicated_results = []
        for result in results:
            identifier = result.get("id") or result.get("text")
            if identifier not in seen:
                seen.add(identifier)
                deduplicated_results.append(result)

        # Sort results by score or other criteria
        sorted_results = sorted(deduplicated_results, key=lambda x: x.get("score", 0), reverse=True)

        # Return the top_k results
        return sorted_results[:top_k]
