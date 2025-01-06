from langswarm.memory.templates.research import ResearchAssistantWorkflow

def test_research_workflow():
    retriever_config = {"backend": "mock"}
    reranker_config = {"model_name": "mock"}
    research_data = ["mock research data"]

    workflow = ResearchAssistantWorkflow(retriever_config, reranker_config, research_data)
    response = workflow.run("What is reinforcement learning?")
    assert isinstance(response, str)
