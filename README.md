
# LangSwarm-Memory

LangSwarm-Memory is a modular and extensible memory management framework designed to support multi-agent systems in diverse AI-driven applications. It provides powerful tools for retrieval, reranking, and workflow orchestration, making it ideal for autonomous AI solutions.

## Features

- **Retrieval Workflows**: Hybrid, temporal, federated, cross-domain, and more.
- **Advanced Reranking**: Multi-agent and combined reranking workflows for improved relevance.
- **Domain-Specific Templates**: Pre-built workflows for biomedical research, customer support, legal documents, and more.
- **Pluggable Adapters**: Integrations with LangChain, LlamaIndex, and various databases (e.g., Pinecone, Redis, SQLite, etc.).

## Installation

To install LangSwarm-Memory, use pip:

```bash
pip install langswar-memory
```

## Usage

### Example: Hybrid Retrieval and Reranking Workflow

```python
from langswar_memory.templates.hybrid import HybridRetrievalRerankingWorkflow

# Configuration
dense_config = {"backend": "pinecone", "api_key": "your-api-key"}
sparse_config = {"backend": "bm25"}
reranker_configs = [{"reranker": "SemanticReranker", "params": {"model_name": "all-MiniLM-L6-v2"}}]
documents = [
    {"text": "What is LangSwarm?", "metadata": {"source": "docs"}},
    {"text": "LangSwarm is a multi-agent framework.", "metadata": {"source": "tutorials"}}
]

# Initialize and run the workflow
workflow = HybridRetrievalRerankingWorkflow(dense_config, sparse_config, reranker_configs, documents)
response = workflow.run("What is LangSwarm?")
print(response)
```

## Components

### Templates
- **`hybrid.py`**: Combines dense and sparse retrieval with reranking.
- **`biomed.py`**: Biomedical literature retrieval and ranking.
- **`legal.py`**: Legal document retrieval for case preparation.
- **`chatbot.py`**: Knowledge-based chatbot workflows.
- **... and more**

### Rerankers
- Multi-agent and combined reranking strategies for improved document scoring.

### Adapters
- Database integrations (e.g., Pinecone, SQLite, Redis, etc.).
- LangChain and LlamaIndex connectors.

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/your-repo/langswarm-memory).

## License

LangSwarm-Memory is licensed under the [MIT License](LICENSE).
