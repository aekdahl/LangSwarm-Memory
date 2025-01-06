class HuggingFaceReranker(BaseReranker):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer, util
        self.model = SentenceTransformer(model_name)

    def rerank(self, query, documents):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        results = []
        for doc in documents:
            doc_embedding = self.model.encode(doc['text'], convert_to_tensor=True)
            score = util.pytorch_cos_sim(query_embedding, doc_embedding).item()
            results.append({"text": doc["text"], "metadata": doc.get("metadata", {}), "score": score})
        return sorted(results, key=lambda x: x["score"], reverse=True)
