# Hybrid Shared Memory System with Scoped Access

## Overview
This project provides a hybrid shared memory system combining multiple storage backends (e.g., in-memory, SQLite, Redis, ChromaDB, GCS) with dynamic scoping capabilities. It supports fine-grained access for individual agents and groups, enabling efficient and context-aware memory sharing in multi-agent systems.

---

## Features

### Core Features
- **Hybrid Memory**: Combines fast in-memory caching with persistent backends for durability.
- **Dynamic Scoping**:
  - Scoped reads and writes based on `agent_id`, `group_id`, and `context_id`.
  - Granular control over memory visibility and organization.
- **Multi-Backend Support**:
  - **In-Memory**: Lightweight and fast.
  - **SQLite**: Persistent, lightweight database.
  - **Redis**: High-performance, distributed memory.
  - **ChromaDB**: Vector-based memory with advanced metadata filtering.
  - **Google Cloud Storage (GCS)**: Persistent cloud storage.
- **Thread and Async Safety**: Ensures consistent operations in concurrent environments.

---

## Installation

### Prerequisites
- Python 3.8+
- Install required packages:

```bash
pip install -r requirements.txt
```

### Requirements
- For ChromaDB: Install `chromadb`
- For Redis: Install `redis-py`
- For GCS: Install `google-cloud-storage`

---

## Usage

### Initialization

#### In-Memory Backend
```python
from memory_backends.in_memory import InMemorySharedMemory

memory = InMemorySharedMemory()
```

#### SQLite Backend
```python
from memory_backends.sqlite_memory import SQLiteSharedMemory

memory = SQLiteSharedMemory(db_path="shared_memory.db")
```

#### Hybrid Memory
```python
from memory_backends.hybrid_memory import HybridMemory
from memory_backends.in_memory import InMemorySharedMemory
from memory_backends.sqlite_memory import SQLiteSharedMemory

in_memory = InMemorySharedMemory()
persistent = SQLiteSharedMemory(db_path="shared_memory.db")

hybrid_memory = HybridMemory(in_memory, persistent)
```

---

### Write Scoped Entries

```python
hybrid_memory.write_scope(
    value="The capital of France is Paris.",
    metadata={"agent_id": "Agent1", "group_id": "GroupA"},
    context_id="ctx_1234"
)

hybrid_memory.write_scope(
    value="France is in Europe.",
    metadata={"agent_id": "Agent2", "group_id": "GroupA"},
    context_id="ctx_1234"
)
```

---

### Read Scoped Entries

#### Read Entries for a Specific Agent
```python
agent_data = hybrid_memory.read_scope(agent_id="Agent1")
print(agent_data)
```

#### Read Entries for a Specific Group
```python
group_data = hybrid_memory.read_scope(group_id="GroupA")
print(group_data)
```

#### Read Entries for a Specific Context
```python
context_data = hybrid_memory.read_scope(context_id="ctx_1234")
print(context_data)
```

#### Combine Scopes
```python
agent_context_data = hybrid_memory.read_scope(agent_id="Agent1", context_id="ctx_1234")
print(agent_context_data)
```

---

## Backend-Specific Features

### In-Memory Shared Memory
- Lightweight and fast.
- Ideal for temporary or volatile data storage.

### SQLite Shared Memory
- Persistent storage for durable data.
- Useful for small to medium datasets.

### Redis Shared Memory
- High-performance distributed memory.
- Suitable for large-scale, real-time applications.

### ChromaDB Shared Memory
- Vector-based memory for similarity search.
- Advanced metadata filtering.

### Google Cloud Storage (GCS) Shared Memory
- Persistent cloud-based storage.
- Ideal for distributed systems with shared resources.

---

## **Next Steps**

1. **Enhance Pub/Sub Capabilities**:
   - Enable real-time notifications for memory updates.

2. **Expand Backend Support**:
   - Add new backends.

3. **Observability**:
   - Integrate logging and metrics for better debugging and monitoring.

MemSwarm is poised to be the go-to solution for shared memory in multi-agent systems. Let us know your feedback and feature requests to shape its future!

---

## Future Enhancements
- **Access Control**: Add `readable_by` and `writable_by` metadata for granular permission management.
- **Dynamic Group Management**: Support for adding/removing agents from groups at runtime.
- **Performance Optimization**: Benchmark and improve query performance for large-scale datasets.

---

## Contributing
1. Fork the repository.
2. Create a new branch for your feature.
3. Submit a pull request with a detailed explanation.

---

## License
This project is licensed under the MIT License.
