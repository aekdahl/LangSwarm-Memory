class TemporalRetriever:
    """
    Retrieve documents based on temporal constraints (e.g., by timestamps).
    """
    def __init__(self, retriever, timestamp_field):
        """
        Args:
            retriever (object): Base retriever (e.g., dense, sparse).
            timestamp_field (str): Field containing document timestamps.
        """
        self.retriever = retriever
        self.timestamp_field = timestamp_field

    def query(self, query, start_time=None, end_time=None):
        """
        Retrieve documents within a temporal range.

        Args:
            query (str): Query string.
            start_time (str): Start time in ISO 8601 format.
            end_time (str): End time in ISO 8601 format.

        Returns:
            list: Retrieved documents matching the temporal range.
        """
        all_results = self.retriever.query(query)
        filtered_results = [
            doc for doc in all_results
            if self._is_within_range(doc[self.timestamp_field], start_time, end_time)
        ]
        return filtered_results

    def _is_within_range(self, timestamp, start_time, end_time):
        """Helper function to check if a timestamp is within a given range."""
        if start_time and timestamp < start_time:
            return False
        if end_time and timestamp > end_time:
            return False
        return True


class FederatedRetriever:
    """
    Retrieve documents from multiple retrievers (federated search).
    """
    def __init__(self, retrievers):
        """
        Args:
            retrievers (list): List of retrievers to federate.
        """
        self.retrievers = retrievers

    def query(self, query):
        """
        Federate queries across all retrievers.

        Args:
            query (str): Query string.

        Returns:
            list: Combined results from all retrievers.
        """
        results = []
        for retriever in self.retrievers:
            results.extend(retriever.query(query))
        return self._deduplicate_results(results)

    def _deduplicate_results(self, results):
        """Remove duplicates based on document IDs."""
        seen = set()
        deduplicated = []
        for result in results:
            doc_id = result.get("id")
            if doc_id not in seen:
                seen.add(doc_id)
                deduplicated.append(result)
        return deduplicated
