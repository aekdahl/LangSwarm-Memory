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

    def query(self, query, backend="faiss", **kwargs):
        """
        Query the specified backend.
        
        Args:
            query (str): Query string or vector.
            backend (str): Backend to use (default: 'faiss').
            **kwargs: Additional arguments for bac
