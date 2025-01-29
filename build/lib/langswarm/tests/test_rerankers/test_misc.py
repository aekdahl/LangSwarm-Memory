from langswarm.memory.rerankers.misc import BM25Reranker, MetadataReranker

def test_bm25_reranker():
    documents = [{"text": "example text 1"}, {"text": "example text 2"}]
    reranker = BM25Reranker(documents)
    query = "example"

    result = reranker.rerank(query, documents)
    assert isinstance(result, list)
    assert all("score" in doc for doc in result)

def test_metadata_reranker():
    documents = [{"text": "doc1", "metadata": {"key": 1}}, {"text": "doc2", "metadata": {"key": 2}}]
    reranker = MetadataReranker(metadata_field="key")

    result = reranker.rerank("example query", documents)
    assert isinstance(result, list)
    assert result[0]["metadata"]["key"] >= result[1]["metadata"]["key"]
