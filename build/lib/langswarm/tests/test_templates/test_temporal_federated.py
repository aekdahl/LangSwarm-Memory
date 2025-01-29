from langswarm.memory.templates.temporal_federated import TemporalFederatedWorkflow

def test_temporal_federated_workflow():
    dense_config = {"backend": "mock"}
    sparse_config = {"backend": "mock"}
    retriever_configs = [{"retriever": "mock", "params": {}}]
    workflow = TemporalFederatedWorkflow(dense_config, sparse_config, "timestamp", retriever_configs)

    response = workflow.run("example query", "2025-01-01", "2025-01-31")
    assert isinstance(response, list)
