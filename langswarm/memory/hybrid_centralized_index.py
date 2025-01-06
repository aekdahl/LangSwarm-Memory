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

    def _initialize_backends(self):
        """
        Initialize the configured backends.
        """
        if self.config.get("faiss", {}).get("enabled", False):
            self.backends["faiss"] = self._initialize_faiss()

        if self.config.get("elasticsearch", {}).get("enabled", False):
            self.backends["elasticsearch"] = self._initialize_elasticsearch()

        if self.config.get("pinecone", {}).get("enabled", False):
            self.backends["pinecone"] = self._initialize_pinecone()

    def _initialize_faiss(self):
        from faiss import IndexFlatL2
        config = self.config["faiss"]
        return IndexFlatL2(config["dimension"])

    def _initialize_elasticsearch(self):
        from elasticsearch import Elasticsearch
        config = self.config["elasticsearch"]
        return Elasticsearch([config["host"]])

    def _initialize_pinecone(self):
        import pinecone
        config = self.config["pinecone"]
        pinecone.init(api_key=config["api_key"], environment=config["environment"])
        return pinecone.Index(config["index_name"])

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

    def query(self, query, backend="faiss", **kwargs):
        """
        Query the specified backend.
        
        Args:
            query (str): Query string or vector.
            backend (str): Backend to use (default: 'faiss').
            **kwargs: Additional arguments for bac
