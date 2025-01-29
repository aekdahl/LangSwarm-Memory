from langswarm.memory.templates.chatbot import ChatbotWorkflow

def test_chatbot_workflow():
    retriever_config = {"backend": "mock"}
    reranker_config = {"model_name": "mock"}
    data_files = ["mock_data.txt"]

    workflow = ChatbotWorkflow(retriever_config, reranker_config, data_files)
    response = workflow.run("What is LangSwarm?")
    assert isinstance(response, str)
