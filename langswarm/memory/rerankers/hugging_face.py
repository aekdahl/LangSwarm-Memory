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
    """
    Reranks documents based on semantic similarity using Hugging Face models.

    Supports pre-trained and fine-tuned models for specific domains.

    Pre-Trained and Domain-Specific Models:
    ---------------------------------------
    General Models:
        - "all-MiniLM-L6-v2": Lightweight, general-purpose semantic similarity.
        - "all-mpnet-base-v2": High-performance general-purpose model.
    
    Domain-Specific Models:
        - Biomedical:
            - "sci-bert/scibert-scivocab-uncased": Optimized for scientific research.
            - "biobert-v1.1": Fine-tuned for biomedical text.
        - Legal:
            - "nlpaueb/legal-bert-base-uncased": Designed for legal documents.
        - Finance:
            - "finbert": Fine-tuned for financial data.
        - Customer Support:
            - "paraphrase-multilingual-mpnet-base-v2": Multilingual customer query handling.

    Usage Example:
    --------------
    1. Initialize the reranker:
        reranker = HuggingFaceSemanticReranker(model_name="biobert-v1.1")

    2. Provide query and documents:
        query = "What are the effects of this drug on the immune system?"
        documents = [
            {"text": "This drug enhances immune response in patients with cancer."},
            {"text": "The medication targets immune cells to reduce inflammation."},
        ]

    3. Perform reranking:
        results = reranker.rerank(query, documents)

    Returns:
        A list of documents sorted by relevance score.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the reranker with a Hugging Face model.

        Args:
            model_name (str): Name of the Hugging Face model to use.
        """
        from sentence_transformers import SentenceTransformer, util
        self.model = SentenceTransformer(model_name)

    def rerank(self, query, documents):
        """
        Rerank documents based on semantic similarity to the query.

        Args:
            query (str): The query string.
            documents (list): List of documents with 'text' and optional 'metadata'.

        Returns:
            list: Documents sorted by relevance score.
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


