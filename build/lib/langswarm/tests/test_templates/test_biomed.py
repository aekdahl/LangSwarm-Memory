from langswarm.memory.templates.biomed import BiomedicalSearchWorkflow

def test_biomedical_workflow():
    retriever_config = {"backend": "mock"}
    reranker_config = {"model_name": "mock"}
    biomedical_data = ["mock biomedical data"]

    workflow = BiomedicalSearchWorkflow(retriever_config, reranker_config, biomedical_data)
    response = workflow.run("What is AI in healthcare?")
    assert isinstance(response, str)
