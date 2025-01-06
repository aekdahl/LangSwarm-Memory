class OpenAIReranker(BaseReranker):
    def __init__(self, model_name="gpt-4"):
        from langchain.llms import OpenAI
        self.llm = OpenAI(model_name=model_name)

    def rerank(self, query, documents):
        prompt = f"Rerank the following documents based on their relevance to the query:\n\nQuery: {query}\n"
        for idx, doc in enumerate(documents, 1):
            prompt += f"{idx}. {doc['text']}\n"
        prompt += "\nReturn the sorted order of document indices."

        response = self.llm(prompt)
        ranking = map(int, response.split())  # Extract indices
        return [documents[i - 1] for i in ranking]
