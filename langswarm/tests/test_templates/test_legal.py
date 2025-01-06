from langswarm.memory.templates.legal import LegalDocumentWorkflow

def test_legal_workflow():
    retriever_config = {"backend": "mock"}
    reranker_config = {"model_name": "mock"}
    legal_data = ["mock legal data"]

    workflow = LegalDocumentWorkflow(retriever_config, reranker_config, legal_data)
    response = workflow.run("What is LangSwarm?")
    assert isinstance(response, str)
