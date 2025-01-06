from langswarm.memory.adapters.database_adapter import BaseExampleAdapter

def test_base_example_adapter():
    adapter = BaseExampleAdapter()
    adapter.connect(config={})
    data = {"id": "test", "value": "example"}
    assert adapter.insert(data)
    assert len(adapter.query({"id": "test"})) == 1
    assert adapter.delete("test")
