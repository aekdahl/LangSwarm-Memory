class SQLiteAdapter(DatabaseAdapter):
    """
    A lightweight document store for managing structured text retrieval using SQLite.
    
    This retriever enables:
    - Storing and retrieving text documents efficiently.
    - Performing SQL-based keyword searches with metadata filtering.
    - Managing document storage with insertion, querying, and deletion operations.
    
    Use cases:
    - Storing structured memory for AI agents.
    - Fast retrieval of past interactions or logs.
    - Querying metadata-enriched text databases.
    
    - Usage format:
START>>>
{
  "calls": [
    {
      "type": "rag",
      "method": "request",
      "instance_name": "",
      "action": "Retrieve function documentation",
      "parameters": {}
    }
  ]
}
<<<END

or

START>>>
{
  "calls": [
    {
      "type": "rag",
      "method": "execute",
      "instance_name": "<rag_name>",
      "action": "<action_name>",
      "parameters": {params_dictionary}
    }
  ]
}
<<<END
Replace `rag_name`, `action_name`, and parameters as needed.
    """
    
    def __init__(self, identifier, db_path="memory.db"):
        self.identifier = identifier
        self.brief = (
            f"SQLiteRetriever"
        )
        super().__init__(
            name="SQLiteRetriever",
            description=(
                "This retriever enables document storage and retrieval using SQLite. "
                "It supports keyword-based searching, metadata filtering, and structured querying. "
                "Ideal for managing structured agent memory or log-based retrieval."
            ),
            instruction="""
- Actions and Parameters:
    - `add_documents`: Add documents to SQLite.
      - Parameters:
        - `documents` (List[Dict]): A list of dictionaries with keys `"key"`, `"text"`, and optional `"metadata"`.
    
    - `query`: Perform a keyword-based SQL search.
      - Parameters:
        - `query` (str): The text query for retrieval.
        - `filters` (Dict, optional): Metadata filters for refining results.
    
    - `delete`: Remove documents from SQLite by ID.
      - Parameters:
        - `document_ids` (List[str]): A list of document keys to remove.

- Usage format:
START>>>
{
  "calls": [
    {
      "type": "rag",
      "method": "request",
      "instance_name": "",
      "action": "Retrieve function documentation",
      "parameters": {}
    }
  ]
}
<<<END

or

START>>>
{
  "calls": [
    {
      "type": "rag",
      "method": "execute",
      "instance_name": "<rag_name>",
      "action": "<action_name>",
      "parameters": {params_dictionary}
    }
  ]
}
<<<END
Replace `rag_name`, `action_name`, and parameters as needed.
            """
        )
        if any(var is None for var in (sqlite3)):
            raise ValueError("Unsupported database. Make sure sqlite3 is installed.")
            
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    metadata TEXT
                )
                """
            )
            conn.commit()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def run(self, payload, action="query"):
        """
        Execute retrieval actions.
        :param payload: Dict - The input query parameters.
        :param action: str - The action to perform: 'query', 'add_documents', or 'delete'.
        :return: str - The result of the action.
        """
        if action == "query":
            return self.query(**payload)
        elif action == "add_documents":
            return self.add_documents(**payload)
        elif action == "delete":
            return self.delete(**payload)
        else:
            return (
                f"Unsupported action: {action}. Available actions are:\n\n"
                f"{self.instruction}"
            )
        
    def add_documents(self, documents):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for doc in documents:
                key = doc.get("key", "")
                value = doc.get("text", "")
                metadata = str(doc.get("metadata", {}))
                cursor.execute(
                    "INSERT OR REPLACE INTO memory (key, value, metadata) VALUES (?, ?, ?)",
                    (key, value, metadata),
                )
            conn.commit()

    def query(self, query, filters=None):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            sql_query = "SELECT key, value, metadata FROM memory WHERE value LIKE ?"
            params = [f"%{query}%"]

            if filters:
                for field, value in filters.items():
                    sql_query += f" AND metadata LIKE ?"
                    params.append(f'%"{field}": "{value}"%')

            cursor.execute(sql_query, params)
            rows = cursor.fetchall()
            return self.standardize_output(
                    text=[row[1] for row in rows],
                    source="SQLite",
                    metadata=[eval(row[2]) for row in rows],
                    id=[row[0] for row in rows]
                )

    def delete(self, document_ids):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for doc_id in document_ids:
                cursor.execute("DELETE FROM memory WHERE key = ?", (doc_id,))
            conn.commit()

    def capabilities(self) -> Dict[str, bool]:
        return {
            "vector_search": False,  # SQLite does not support vector-based search.
            "metadata_filtering": True,  # Supports metadata filtering through SQL queries.
            "semantic_search": False,  # Requires external embeddings for semantic capabilities.
        }
