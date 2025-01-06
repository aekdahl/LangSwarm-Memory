class DatabaseAdapter:
    def add_documents(self, documents):
        raise NotImplementedError("Subclasses must implement add_documents()")

    def query(self, query, filters=None):
        raise NotImplementedError("Subclasses must implement query()")

    def delete(self, document_ids):
        raise NotImplementedError("Subclasses must implement delete()")
