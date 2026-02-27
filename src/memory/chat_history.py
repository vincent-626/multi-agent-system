"""Per-user chat message history persisted to SQLite.

Stored in the same database file as long-term memory facts so the system
has only one SQLite file to manage.

SQLite schema
-------------
Table: chat_messages
  id          INTEGER PRIMARY KEY AUTOINCREMENT
  user_id     TEXT NOT NULL
  question    TEXT NOT NULL
  answer      TEXT NOT NULL
  sources     TEXT NOT NULL  -- JSON array of source file names
  web_sources TEXT NOT NULL  -- JSON array of URLs
  confidence  TEXT NOT NULL
  timestamp   TEXT NOT NULL  -- ISO-8601 UTC

Index: idx_chat_user_id ON chat_messages(user_id)
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from src.config import LONG_TERM_MEMORY_DB
from src.schemas import FinalResponse

logger = logging.getLogger(__name__)


@contextmanager
def _connect() -> Generator[sqlite3.Connection, None, None]:
    path = Path(LONG_TERM_MEMORY_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the chat_messages table and index if they do not exist."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                question    TEXT NOT NULL,
                answer      TEXT NOT NULL,
                sources     TEXT NOT NULL,
                web_sources TEXT NOT NULL,
                confidence  TEXT NOT NULL,
                timestamp   TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_user_id ON chat_messages(user_id)"
        )


init_db()


def save_message(user_id: str, question: str, response: FinalResponse) -> None:
    """Persist a completed Q&A exchange for *user_id*.

    Args:
        user_id:  Opaque user identifier.
        question: The original user question.
        response: The completed :class:`~src.schemas.FinalResponse`.
    """
    now = datetime.now(tz=timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO chat_messages
                (user_id, question, answer, sources, web_sources, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                question,
                response.answer,
                json.dumps(response.sources),
                json.dumps(response.web_sources),
                response.confidence,
                now,
            ),
        )


def get_messages(user_id: str, limit: int = 50) -> list[dict]:
    """Return the most recent *limit* messages for *user_id*, oldest first.

    Args:
        user_id: Opaque user identifier.
        limit:   Maximum number of messages to return.

    Returns:
        List of dicts with keys ``question``, ``answer``, ``sources``,
        ``web_sources``, ``confidence``, and ``timestamp``.
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT question, answer, sources, web_sources, confidence, timestamp
            FROM chat_messages
            WHERE user_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [
        {
            "question": row["question"],
            "answer": row["answer"],
            "sources": json.loads(row["sources"]),
            "web_sources": json.loads(row["web_sources"]),
            "confidence": row["confidence"],
            "timestamp": row["timestamp"],
        }
        for row in rows
    ]
