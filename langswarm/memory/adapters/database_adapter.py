from abc import ABC, abstractmethod
from typing import Dict

class DatabaseAdapter(ABC):
    """
    Abstract base class for database adapters.

    Defines the interface that all database adapters must implement.
    """

    @abstractmethod
    def connect(self, config):
        """
        Establish a connection to the database.

        Args:
            config (dict): Configuration dictionary with required connection details.

        Returns:
            None

        Raises:
            ConnectionError: If the connection cannot be established.
        """
        pass

    @abstractmethod
    def insert(self, data):
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

# Example implementation of the DatabaseAdapter
class BaseExampleAdapter(DatabaseAdapter):
    """
    Example implementation of the DatabaseAdapter for demonstration purposes.

    This is a simple in-memory adapter that stores data in a Python dictionary.
    """

    def __init__(self):
        self._storage = {}

    def connect(self, config):
        """
        Simulate a connection to a database.

        Args:
            config (dict): Configuration dictionary (not used in this example).

        Returns:
            None
        """
        print("Connected to the in-memory database.")

    def insert(self, data):
        """
        Insert data into the in-memory database.

        Args:
            data (dict): The data to insert.

        Returns:
            bool: True if successful, False otherwise.
        """
        if 'id' not in data:
            raise ValueError("Data must contain an 'id' field.")
        self._storage[data['id']] = data
        return True

    def query(self, filters):
        """
        Query the in-memory database.

        Args:
            filters (dict): A dictionary of filters (e.g., {'field': 'value'}).

        Returns:
            list: A list of matching records.
        """
        results = []
        for record in self._storage.values():
            if all(record.get(k) == v for k, v in filters.items()):
                results.append(record)
        return results

    def delete(self, identifier):
        """
        Delete a record from the in-memory database.

        Args:
            identifier (str): The unique identifier of the record.

        Returns:
            bool: True if successful, False otherwise.
        """
        if identifier not in self._storage:
            raise KeyError(f"No record found with id '{identifier}'.")
        del self._storage[identifier]
        return True
