from langswarm.memory.adapters.langchain import PineconeAdapter

def test_pinecone_adapter():
    adapter = PineconeAdapter(api_key="dummy", environment="test", index_name="test-index")
    documents = [{"text": "example", "metadata": {"key": "value"}}]
    adapter.add_documents(documents)

    result = adapter.query("example")
    assert isinstance(result, list)
    assert "text" in result[0]
