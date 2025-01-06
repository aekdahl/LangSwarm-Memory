import importlib


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
