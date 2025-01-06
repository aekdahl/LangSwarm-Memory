import pytest
from langswarm.memory.rerankers.openai import OpenAIReranker

def test_openai_reranker_init():
    reranker = OpenAIReranker(model_name="gpt-4")
    assert reranker.llm is not None

def test_openai_reranker_rerank():
    reranker = OpenAIReranker(model_name="gpt-4")
    query = "example query"
    documents = [{"text": "doc1"}, {"text": "doc2"}]

    result = reranker.rerank(query, documents)
    assert isinstance(result, list)
    assert all("text" in doc for doc in result)
