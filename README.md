# MemSwarm

# MemSwarm

**MemSwarm** is a Python package designed to provide shared memory solutions for multi-agent systems and AI workflows. It offers flexible backends, including in-memory, persistent storage, and vector-based similarity search, enabling efficient data sharing and retrieval across agents.

---

## **Key Features**

1. **Shared Memory Management**
   - Store, retrieve, and manage key-value pairs efficiently.
   - Support for multiple storage backends: In-memory, Redis, Google Cloud Storage (GCS), and ChromaDB.

2. **Similarity Search**
   - Perform vector-based similarity search using ChromaDB.
   - Enables Retrieval-Augmented Generation (RAG) workflows.

3. **Persistence and Scalability**
   - Persistent storage options for durability and recovery.
   - Scalable solutions for distributed systems.

4. **Hybrid Memory**
   - Combines in-memory caching with persistent backends for optimal performance.

5. **Extensibility**
   - Modular design allows easy addition of new backends and features.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/memswarm.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

### **Basic Shared Memory**
#### In-Memory Implementation
```python
from memswarm.memory.in_memory import InMemorySharedMemory

# Initialize shared memory
memory = InMemorySharedMemory()

# Write and read data
memory.write("key1", "value1")
print(memory.read("key1"))  # Output: 'value1'

# Clear memory
memory.clear()
```

### **Persistent Storage with GCS**
```python
from memswarm.memory.gcs_memory import GCSSharedMemory

# Initialize GCS shared memory
gcs_memory = GCSSharedMemory(bucket_name="my-bucket")

# Write and read data
gcs_memory.write("key1", "value1")
print(gcs_memory.read("key1"))  # Output: 'value1'
```

### **Similarity Search with ChromaDB**
```python
from memswarm.memory.chromadb_memory import ChromaDBSharedMemory

# Initialize ChromaDB shared memory
chroma_memory = ChromaDBSharedMemory(collection_name="example_collection")

# Add documents
chroma_memory.write("doc1", "This is a document about AI.")
chroma_memory.write("doc2", "This document discusses machine learning.")

# Perform similarity search
results = chroma_memory.similarity_search("Tell me about AI", top_k=2)
print(results)  # Output: ['This is a document about AI.', 'This document discusses machine learning.']
```

### **Hybrid Memory (In-Memory + Persistent Storage)**
```python
from memswarm.memory.hybrid_memory import HybridSharedMemory
from memswarm.memory.in_memory import InMemorySharedMemory
from memswarm.memory.gcs_memory import GCSSharedMemory

# Initialize hybrid memory
cache_memory = InMemorySharedMemory()
persistent_memory = GCSSharedMemory(bucket_name="my-bucket")
hybrid_memory = HybridSharedMemory(cache=cache_memory, backend=persistent_memory)

# Write and read data
hybrid_memory.write("key1", "value1")
print(hybrid_memory.read("key1"))  # Output: 'value1'

# Data is cached in-memory and persisted to GCS
```

---

## **Hybrid Memory Implementation**

```python
from .base import SharedMemoryBase

class HybridSharedMemory(SharedMemoryBase):
    """
    Hybrid memory combines in-memory caching with a persistent backend.
    """

    def __init__(self, cache, backend):
        self.cache = cache
        self.backend = backend

    def read(self, key=None):
        # Try reading from cache first
        value = self.cache.read(key)
        if value is not None:
            return value

        # Fallback to backend if not found in cache
        value = self.backend.read(key)
        if value is not None:
            self.cache.write(key, value)  # Cache the result
        return value

    def write(self, key, value):
        self.cache.write(key, value)
        self.backend.write(key, value)

    def delete(self, key):
        self.cache.delete(key)
        self.backend.delete(key)

    def clear(self):
        self.cache.clear()
        self.backend.clear()
```

---

## **Insights and Achievements**

### **What MemSwarm Has Achieved**

1. **Cross-Agent Shared Memory**:
   - MemSwarm facilitates data sharing across agents with multiple backends, ensuring interoperability and scalability.

2. **RAG-Ready Workflows**:
   - With ChromaDB's similarity search, MemSwarm enables advanced AI workflows like RAG, making it highly suitable for modern AI systems.

3. **Hybrid Memory Advantage**:
   - The hybrid memory implementation strikes a balance between speed (in-memory) and durability (persistent storage).

4. **Extensibility and Modularity**:
   - A well-designed, modular architecture allows seamless integration of new features and backends.

### **Is MemSwarm Truly a Cross-Agent Shared Memory Solution?**
MemSwarm has made significant strides toward being a robust cross-agent shared memory solution. However, areas like dynamic memory scoping for individual agents and real-time pub/sub updates remain open for further development. Addressing these gaps will solidify MemSwarm's position as a leading solution in this domain.

---

## **Next Steps**

1. **Enhance Pub/Sub Capabilities**:
   - Enable real-time notifications for memory updates.

2. **Agent-Specific Contexts**:
   - Introduce scoped memory views for better privacy and collaboration.

3. **Expand Backend Support**:
   - Add DynamoDB and SQLite as new backends.

4. **Observability**:
   - Integrate logging and metrics for better debugging and monitoring.

MemSwarm is poised to be the go-to solution for shared memory in multi-agent systems. Let us know your feedback and feature requests to shape its future!

