import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

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
                        metadata_json TEXT,
                        content_hash TEXT,
                        indexed_parent_ids_json TEXT,
                        indexed_at TEXT
                    )
                """)
                existing_columns = {
                    row["name"]
                    for row in conn.execute("PRAGMA table_info(crawled_pages)").fetchall()
                }
                for column_name, column_type in {
                    "content_hash": "TEXT",
                    "indexed_parent_ids_json": "TEXT",
                    "indexed_at": "TEXT",
                }.items():
                    if column_name not in existing_columns:
                        conn.execute(f"ALTER TABLE crawled_pages ADD COLUMN {column_name} {column_type}")
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

    def get_all_urls(self) -> List[str]:
        """Retrieves all crawled URLs from the database."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT url FROM crawled_pages")
            return [row['url'] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get all URLs from state DB: {e}")
            return []

    def remove_url(self, url: str):
        """Removes a URL's state from the database."""
        try:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM crawled_pages WHERE url = ?", (url,))
                conn.commit()
                logger.info(f"Removed URL state for {url} from the database.")
        except Exception as e:
            logger.error(f"Failed to remove URL state for {url}: {e}")

    def update_index_state(self, url: str, content_hash: str, parent_ids: List[str]):
        """Stores the indexed content hash and parent docstore IDs for a source URL."""
        indexed_at = datetime.now(timezone.utc).isoformat()
        parent_ids_json = json.dumps(parent_ids)

        try:
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT INTO crawled_pages (
                        url, etag, last_modified, crawled_at,
                        metadata_json, content_hash, indexed_parent_ids_json, indexed_at
                    )
                    VALUES (?, NULL, NULL, ?, NULL, ?, ?, ?)
                    ON CONFLICT(url) DO UPDATE SET
                        content_hash = excluded.content_hash,
                        indexed_parent_ids_json = excluded.indexed_parent_ids_json,
                        indexed_at = excluded.indexed_at
                """, (url, indexed_at, content_hash, parent_ids_json, indexed_at))
                conn.commit()
        except Exception as e:
            logger.warning(f"Could not update index state for {url}: {e}")

    def get_index_state(self, url: str) -> Dict[str, Any]:
        """Returns indexed hash and parent IDs for a URL, if present."""
        row = self.get_url_info(url)
        if not row:
            return {"content_hash": None, "parent_ids": []}

        parent_ids = []
        try:
            raw_parent_ids = row["indexed_parent_ids_json"]
            if raw_parent_ids:
                parent_ids = json.loads(raw_parent_ids)
        except Exception:
            parent_ids = []

        return {
            "content_hash": row["content_hash"],
            "parent_ids": parent_ids,
        }

    def clear_index_state(self, url: str):
        """Clears only indexing metadata while preserving crawl cache headers."""
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    UPDATE crawled_pages
                    SET content_hash = NULL,
                        indexed_parent_ids_json = NULL,
                        indexed_at = NULL
                    WHERE url = ?
                """, (url,))
                conn.commit()
        except Exception as e:
            logger.warning(f"Could not clear index state for {url}: {e}")

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
