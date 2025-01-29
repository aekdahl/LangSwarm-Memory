import pytest
from langswarm.memory.rerankers.workflows import CombinedRerankingWorkflow, MultiAgentRerankingWorkflow

def test_combined_reranking_workflow():
    rerankers = [MockReranker1(), MockReranker2()]
    workflow = CombinedRerankingWorkflow(rerankers)
    query = "example query"
    documents = [{"text": "doc1"}, {"text": "doc2"}]
    
    result = workflow.run(query, documents)
    assert isinstance(result, list)
    assert all("text" in doc and "score" in doc for doc in result)

def test_multi_agent_reranking_workflow():
    agents = [MockAgent1(), MockAgent2()]
    workflow = MultiAgentRerankingWorkflow(agents)
    query = "example query"
    documents = [{"text": "doc1"}, {"text": "doc2"}]

    result = workflow.run(query, documents)
    assert isinstance(result, list)
    assert all("text" in doc and "score" in doc for doc in result)
