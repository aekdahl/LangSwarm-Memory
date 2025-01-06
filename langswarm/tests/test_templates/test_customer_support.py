from langswarm.memory.templates.customer_support import CustomerSupportWorkflow

def test_customer_support_workflow():
    retriever_config = {"backend": "mock"}
    reranker_config = {"model_name": "mock"}
    faq_data = ["mock faq data"]
    ticket_data = ["mock ticket data"]

    workflow = CustomerSupportWorkflow(retriever_config, retriever_config, reranker_config, faq_data, ticket_data)
    response = workflow.run("How do I reset my password?")
    assert isinstance(response, str)
