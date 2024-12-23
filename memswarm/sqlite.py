import sqlite3
import asyncio
from datetime import datetime
from .base import SharedMemoryBase

class SQLiteSharedMemory(SharedMemoryBase):
    """
    SQLite-backed shared memory with optional thread-safe and async-safe operations.
    """

    def __init__(self, db_path=":memory:", thread_safe=True, async_safe=False):
        """
        Initialize SQLite database and thread/async locks.

        Parameters:
        - db_path (str): Path to the SQLite database file. Defaults to in-memory.
        - thread_safe (bool): Enable thread-safe operations.
        - async_safe (bool): Enable async-safe operations.
        """
        self.db_path = db_path
        self.thread_safe = thread_safe or async_safe

        if self.thread_safe:
            self.lock = asyncio.Lock() if async_safe else Lock()

        self._initialize_db()

    def _initialize_db(self):
        """
        Initialize the SQLite database schema.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            conn.commit()

    def _get_connection(self):
        """
        Get a new SQLite connection.
        """
        return sqlite3.connect(self.db_path)

    def _get_default_metadata(self, metadata):
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "agent": metadata.get("agent"),
            "timestamp": metadata.get("timestamp", now),
            "llm": metadata.get("llm"),
            "action": metadata.get("action"),
            "confidence": metadata.get("confidence"),
            "query": metadata.get("query"),
        }

    def read(self, key=None):
        """
        Read memory from SQLite.

        Parameters:
        - key (str): Key to fetch. If None, fetch all memory.

        Returns:
        - The value for the key if specified, else all memory as a dictionary.
        """
        def _read():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if key:
                    cursor.execute("SELECT value, metadata FROM memory WHERE key = ?", (key,))
                    row = cursor.fetchone()
                    if row:
                        return {"value": row[0], "metadata": eval(row[1])}
                    return None
                else:
                    cursor.execute("SELECT key, value, metadata FROM memory")
                    rows = cursor.fetchall()
                    return {
                        row[0]: {"value": row[1], "metadata": eval(row[2])}
                        for row in rows
                    }

        if self.thread_safe:
            with self.lock:
                return _read()
        else:
            return _read()

    def write(self, key, value, metadata=None):
        """
        Write a key-value pair to SQLite with metadata.

        Parameters:
        - key (str): Key to store.
        - value (str): Value to store.
        - metadata (dict): Optional metadata.
        """
        metadata = metadata or {}
        entry_metadata = str(self._get_default_metadata(metadata))

        def _write():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO memory (key, value, metadata)
                    VALUES (?, ?, ?)
                """, (key, value, entry_metadata))
                conn.commit()

        if self.thread_safe:
            with self.lock:
                _write()
        else:
            _write()

    def delete(self, key):
        """
        Delete a key in SQLite.

        Parameters:
        - key (str): Key to delete.
        """
        def _delete():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM memory WHERE key = ?", (key,))
                conn.commit()

        if self.thread_safe:
            with self.lock:
                _delete()
        else:
            _delete()

    def clear(self):
        """
        Clear all entries in SQLite.
        """
        def _clear():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM memory")
                conn.commit()

        if self.thread_safe:
            with self.lock:
                _clear()
        else:
            _clear()
