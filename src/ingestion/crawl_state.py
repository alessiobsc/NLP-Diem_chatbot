import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from requests import Response

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Use a thread-local connection to ensure thread safety for SQLite
_thread_local = threading.local()


class CrawlStateManager:
    """
    Manages the state of crawled URLs using a SQLite database to support
    incremental crawling and avoid re-fetching unchanged content.

    This class is thread-safe and can be used as a context manager.
    """

    def __init__(self, db_path: str = "db/crawl_state.db"):
        self.db_path = db_path
        # Ensure the directory for the database exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Gets a thread-local database connection."""
        if not hasattr(_thread_local, "conn"):
            _thread_local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            _thread_local.conn.row_factory = sqlite3.Row
        return _thread_local.conn

    def _init_db(self):
        """Initializes the database and creates the table if it doesn't exist."""
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS crawled_pages (
                        url TEXT PRIMARY KEY,
                        etag TEXT,
                        last_modified TEXT,
                        crawled_at TEXT NOT NULL,
                        metadata_json TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize crawl state database: {e}")
            raise

    def get_url_info(self, url: str) -> Optional[sqlite3.Row]:
        """Retrieves all stored information for a given URL."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM crawled_pages WHERE url = ?", (url,))
            return cursor.fetchone()
        except Exception as e:
            logger.warning(f"Could not get URL info for {url} from state DB: {e}")
            return None

    def update_url_state(self, url: str, response: Response, metadata: Optional[Dict[str, Any]] = None):
        """
        Updates the state of a URL in the database after a successful fetch.
        Extracts ETag and Last-Modified from the response headers.
        """
        etag = response.headers.get("ETag")
        last_modified = response.headers.get("Last-Modified")
        # Use timezone-aware datetime in UTC as recommended in modern Python
        crawled_at = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(metadata) if metadata else None

        try:
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT INTO crawled_pages (url, etag, last_modified, crawled_at, metadata_json)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(url) DO UPDATE SET
                        etag = excluded.etag,
                        last_modified = excluded.last_modified,
                        crawled_at = excluded.crawled_at,
                        metadata_json = excluded.metadata_json
                """, (url, etag, last_modified, crawled_at, metadata_json))
                conn.commit()
        except Exception as e:
            logger.warning(f"Could not update URL state for {url} in DB: {e}")

    @staticmethod
    def close():
        """Closes the thread-local database connection."""
        if hasattr(_thread_local, "conn"):
            _thread_local.conn.close()
            del _thread_local.conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
