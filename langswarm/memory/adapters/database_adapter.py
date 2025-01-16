from abc import ABC, abstractmethod
from typing import Dict

class DatabaseAdapter(ABC):
    """
    Abstract base class for database adapters.

    Defines the interface that all database adapters must implement.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Ensure all subclasses implement their initialization logic.
        """
        pass

    @abstractmethod
    def add_documents(self, data):
        """
        Insert data into the database.

        Args:
            data (dict): The data to insert.

        Returns:
            bool: True if the operation was successful, False otherwise.

        Raises:
            ValueError: If data is invalid.
        """
        pass

    @abstractmethod
    def query(self, filters):
        """
        Query the database using the given filters.

        Args:
            filters (dict): A dictionary of query filters.

        Returns:
            list: A list of results matching the filters.

        Raises:
            ValueError: If filters are invalid.
        """
        pass

    @abstractmethod
    def delete(self, identifier):
        """
        Delete a record from the database.

        Args:
            identifier (str): The unique identifier of the record to delete.

        Returns:
            bool: True if the operation was successful, False otherwise.

        Raises:
            KeyError: If the identifier does not exist.
        """
        pass

    @abstractmethod
    def capabilities(self) -> Dict[str, bool]:
        raise NotImplementedError

    # Adding get_relevant_documents() for LangChain integration
    def get_relevant_documents(self, query: str, k: int = 5, filters: Dict = None) -> List[Dict]:
        """
        Retrieve the most relevant documents using the query() method.

        Args:
            query (str): The query string for retrieval.
            k (int): The number of documents to retrieve (default is 5).
            filters (Dict): Additional filters for querying metadata (optional).

        Returns:
            List[Dict]: A list of relevant documents.
        """
        # Combine query string and filters for more advanced backends
        query_filters = {"query": query}
        if filters:
            query_filters.update(filters)

        # Query the database and limit results to k
        results = self.query(filters=query_filters)[:k]
        return results
