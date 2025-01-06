class CombinedRerankingWorkflow:
    """
    Combines multiple reranking strategies (e.g., semantic similarity and metadata-based scoring)
    into a unified reranking workflow.

    Attributes:
        rerankers (list): A list of reranker instances.
        weights (list): Corresponding weights for each reranker.
    """
    def __init__(self, rerankers, weights=None):
        """
        Initialize the workflow with rerankers and their weights.

        Args:
            rerankers (list): List of reranker instances (subclasses of BaseReranker).
            weights (list): List of weights for each reranker (default: equal weights).
        """
        self.rerankers = rerankers
        if weights is None:
            self.weights = [1.0 / len(rerankers)] * len(rerankers)
        else:
            self.weights = weights

    def run(self, query, documents):
        """
        Perform combined reranking.

        Args:
            query (str): The query string.
            documents (list): List of documents to rerank.

        Returns:
            list: Documents sorted by combined scores.
        """
        # Initialize scores for each document
        scores = {doc["text"]: 0.0 for doc in documents}

        # Iterate over each reranker and aggregate scores
        for reranker, weight in zip(self.rerankers, self.weights):
            ranked_docs = reranker.rerank(query, documents)
            for doc in ranked_docs:
                scores[doc["text"]] += doc["score"] * weight

        # Sort documents by combined scores
        combined_results = [{"text": doc, "score": score} for doc, score in scores.items()]
        return sorted(combined_results, key=lambda x: x["score"], reverse=True)
