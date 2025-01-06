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


class HuggingFaceSemanticReranker(BaseReranker):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer, util
        self.model = SentenceTransformer(model_name)

    def rerank(self, query, documents):
        """
        Rerank documents based on semantic similarity to the query.
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        results = []
        for doc in documents:
            doc_embedding = self.model.encode(doc["text"], convert_to_tensor=True)
            score = util.pytorch_cos_sim(query_embedding, doc_embedding).item()
            results.append({"text": doc["text"], "metadata": doc.get("metadata", {}), "score": score})
        return sorted(results, key=lambda x: x["score"], reverse=True)


class HuggingFaceDPRReranker(BaseReranker):
    def __init__(self, model_name="facebook/dpr-question_encoder-single-nq-base"):
        from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder
        self.query_model = DPRQuestionEncoder.from_pretrained(model_name)
        self.query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
        self.context_model = DPRContextEncoder.from_pretrained(model_name)

    def rerank(self, query, documents):
        """
        Rerank documents using Dense Passage Retrieval (DPR).
        """
        query_inputs = self.query_tokenizer(query, return_tensors="pt")
        query_embedding = self.query_model(**query_inputs).pooler_output

        results = []
        for doc in documents:
            context_inputs = self.query_tokenizer(doc["text"], return_tensors="pt")
            context_embedding = self.context_model(**context_inputs).pooler_output
            score = (query_embedding * context_embedding).sum().item()
            results.append({"text": doc["text"], "metadata": doc.get("metadata", {}), "score": score})
        return sorted(results, key=lambda x: x["score"], reverse=True)


