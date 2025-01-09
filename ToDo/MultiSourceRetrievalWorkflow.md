# MultiSourceRetrievalWorkflow - Integration with LangChain and LlamaIndex

## Overview
The `MultiSourceRetrievalWorkflow` class in the **LangSwarm-Memory** package provides a flexible mechanism for dynamically managing multiple retrieval sources. This enhancement allows seamless integration with **LangChain** and **LlamaIndex**, enabling developers to utilize the power of multi-source retrieval while leveraging the advanced features provided by these frameworks.

## Features
- **Dynamic Source Management**: Add or remove retrieval sources at runtime.
- **Integration with Frameworks**: Easily adapt to LangChain or LlamaIndex using a `to_framework` method.
- **Centralized Control**: Manage retrieval logic within a single workflow class while delegating the heavy lifting to LangChain or LlamaIndex.

---

## Code Implementation

### MultiSourceRetrievalWorkflow
```python
class MultiSourceRetrievalWorkflow:
    """
    A workflow to manage multiple retrieval sources dynamically and integrate with frameworks like LangChain or LlamaIndex.
    """
    def __init__(self, sources=None):
        """
        Initialize the workflow with optional sources.

        Args:
            sources (list): A list of retrieval source instances.
        """
        self.sources = sources or []

    def add_source(self, source):
        """
        Add a new retrieval source.

        Args:
            source: A retrieval source instance.
        """
        self.sources.append(source)

    def remove_source(self, source_name):
        """
        Remove a retrieval source by its name.

        Args:
            source_name (str): Name of the source to remove.
        """
        self.sources = [s for s in self.sources if s.name != source_name]

    def retrieve(self, query):
        """
        Retrieve data from all sources and combine results.

        Args:
            query (str): The query string.

        Returns:
            list: Combined results from all sources.
        """
        results = []
        for source in self.sources:
            results.extend(source.query(query))
        return results

    def to_framework(self, framework="langchain"):
        """
        Convert the workflow into a format compatible with LangChain or LlamaIndex.

        Args:
            framework (str): The target framework, either "langchain" or "llamaindex".

        Returns:
            object: A framework-compatible retrieval object.
        """
        if framework == "langchain":
            from langchain.chains import RetrievalQA
            from langchain.vectorstores.base import VectorStore

            class MultiSourceLangChain(VectorStore):
                """
                LangChain-compatible wrapper for MultiSourceRetrievalWorkflow.
                """
                def __init__(self, workflow):
                    self.workflow = workflow

                def similarity_search(self, query, k=10):
                    results = self.workflow.retrieve(query)
                    return results[:k]

            return MultiSourceLangChain(self)

        elif framework == "llamaindex":
            from llama_index import QueryEngine

            class MultiSourceLlamaIndex(QueryEngine):
                """
                LlamaIndex-compatible wrapper for MultiSourceRetrievalWorkflow.
                """
                def __init__(self, workflow):
                    self.workflow = workflow

                def query(self, query):
                    return self.workflow.retrieve(query)

            return MultiSourceLlamaIndex(self)

        else:
            raise ValueError(f"Unsupported framework: {framework}")
```

---

## Usage Examples

### Example 1: Basic Usage
```python
# Initialize the workflow
workflow = MultiSourceRetrievalWorkflow()

# Add sources
workflow.add_source(my_langchain_retriever)
workflow.add_source(my_llamaindex_retriever)

# Retrieve data
results = workflow.retrieve("What is LangSwarm?")
print("Combined Results:", results)
```

### Example 2: Convert to LangChain
```python
# Convert to LangChain-compatible object
langchain_retriever = workflow.to_framework("langchain")

# Use LangChain's RetrievalQA with the retriever
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA(retriever=langchain_retriever, llm=my_llm)
response = qa_chain.run("What is LangSwarm?")
print("LangChain Response:", response)
```

### Example 3: Convert to LlamaIndex
```python
# Convert to LlamaIndex-compatible object
llamaindex_retriever = workflow.to_framework("llamaindex")

# Use the LlamaIndex query engine
response = llamaindex_retriever.query("What is LangSwarm?")
print("LlamaIndex Response:", response)
```

---

## Documentation

### Methods

#### `add_source(self, source)`
Add a new retrieval source to the workflow.

- **Args**:
  - `source`: A retrieval source instance.

#### `remove_source(self, source_name)`
Remove a retrieval source by its name.

- **Args**:
  - `source_name (str)`: The name of the source to remove.

#### `retrieve(self, query)`
Retrieve data from all sources and combine results.

- **Args**:
  - `query (str)`: The query string.

- **Returns**:
  - `list`: Combined results from all sources.

#### `to_framework(self, framework)`
Convert the workflow into a format compatible with LangChain or LlamaIndex.

- **Args**:
  - `framework (str)`: The target framework, either "langchain" or "llamaindex".

- **Returns**:
  - `object`: A framework-compatible retrieval object.

---

## Testing

### Test Cases

#### Test 1: Adding and Removing Sources
- Verify that sources can be dynamically added and removed.

#### Test 2: Retrieval from Multiple Sources
- Validate that data is retrieved correctly from all sources.

#### Test 3: Integration with LangChain
- Ensure that the `to_framework("langchain")` method produces a compatible retriever.

#### Test 4: Integration with LlamaIndex
- Ensure that the `to_framework("llamaindex")` method produces a compatible retriever.

---

## Conclusion
The **MultiSourceRetrievalWorkflow** is a versatile tool that simplifies the management of multiple retrieval sources and integrates seamlessly with leading frameworks. Its dynamic capabilities and compatibility make it a valuable addition to any Retrieval-Augmented Generation (RAG) pipeline.
