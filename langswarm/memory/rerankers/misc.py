from memory.rerankers.rerank import BaseReranker  # Import missing base class.

class BM25Reranker(BaseReranker):
    def __init__(self, documents):
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi([doc["text"].split() for doc in documents])

    def rerank(self, query, documents):
        """
        Rerank documents using BM25 scoring.
        """
        scores = self.bm25.get_scores(query.split())
        for doc, score in zip(documents, scores):
            doc["score"] = score
        return sorted(documents, key=lambda x: x["score"], reverse=True)


class MetadataReranker(BaseReranker):
    def __init__(self, metadata_field, reverse=True):
        self.metadata_field = metadata_field
        self.reverse = reverse

    def rerank(self, query, documents):
        """
        Rerank documents based on a metadata field.
        """
        return sorted(
            documents,
            key=lambda doc: doc.get("metadata", {}).get(self.metadata_field, 0),
            reverse=self.reverse
        )
