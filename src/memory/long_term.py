"""Persistent per-user long-term memory.

After each conversation the LLM extracts memorable facts about the user —
preferences, background, ongoing projects, constraints — and stores them in
SQLite.  On the next session those facts are retrieved and injected into the
system prompt so the assistant remembers who it is talking to.

This is different from a Q&A cache: facts describe *the user*, not past
answers, so they improve every future response rather than only short-
circuiting repeated questions.

SQLite schema
-------------
Table: memory_facts
  id        INTEGER PRIMARY KEY AUTOINCREMENT
  user_id   TEXT NOT NULL
  fact      TEXT NOT NULL
  timestamp TEXT NOT NULL  -- ISO-8601 UTC

Index: idx_facts_user_id ON memory_facts(user_id)
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from src.config import LONG_TERM_MEMORY_DB
from src.schemas import FinalResponse

logger = logging.getLogger(__name__)


# ── Connection helper ──────────────────────────────────────────────────────────

@contextmanager
def _connect() -> Generator[sqlite3.Connection, None, None]:
    """Open a short-lived SQLite connection, commit on success, close always."""
    path = Path(LONG_TERM_MEMORY_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── Schema initialisation ──────────────────────────────────────────────────────

def init_db() -> None:
    """Create the SQLite memory_facts table and index if they do not exist."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_facts (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   TEXT NOT NULL,
                fact      TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_user_id ON memory_facts(user_id)"
        )


# Initialise on import so callers never have to think about it.
init_db()


# ── Public API ─────────────────────────────────────────────────────────────────

def get_facts(user_id: str) -> list[str]:
    """Return all stored facts for *user_id*, oldest first.

    Args:
        user_id: Opaque user identifier.

    Returns:
        List of fact strings in chronological order.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT fact FROM memory_facts WHERE user_id = ? ORDER BY timestamp ASC",
            (user_id,),
        ).fetchall()
    return [row["fact"] for row in rows]


def extract_and_save(user_id: str, question: str, response: FinalResponse) -> list[str]:
    """Extract memorable facts from a conversation and persist them.

    Runs a lightweight LLM pass over the Q&A to pull out anything worth
    remembering about the user (preferences, context, background, projects).
    Results are appended to the user's fact store in SQLite.

    Args:
        user_id:  Opaque user identifier.
        question: The original user question.
        response: The completed :class:`~src.schemas.FinalResponse`.

    Returns:
        List of fact strings that were saved (may be empty).
    """
    import src.ollama_client as ollama
    from src.config import FAST_MODEL

    prompt = (
        f"Conversation:\n"
        f"User: {question}\n"
        f"Assistant: {response.answer[:600]}\n\n"
        "Extract facts about the user worth remembering for future sessions.\n"
        "Focus on: preferences, background, ongoing projects, constraints, personal details.\n"
        "Only extract what the user explicitly stated or clearly implied.\n"
        "Each fact should be one concise sentence.\n"
        "If nothing memorable was revealed, return an empty list.\n"
        'Respond with a JSON array of strings only. Example: ["User prefers Python", "User is deploying to a VPS"]'
    )

    try:
        raw = ollama.chat(
            prompt=prompt,
            model=FAST_MODEL,
            think=False,
            system=(
                "You extract memorable facts about users from conversations. "
                "Be concise and specific. Only record what the user revealed about themselves."
            ),
        )
        facts = ollama.parse_json_list(raw)
    except Exception as exc:
        logger.warning("Fact extraction failed: %s", exc)
        return []

    if not facts:
        return []

    now = datetime.now(tz=timezone.utc).isoformat()
    with _connect() as conn:
        conn.executemany(
            "INSERT INTO memory_facts (user_id, fact, timestamp) VALUES (?, ?, ?)",
            [(user_id, fact, now) for fact in facts],
        )

    logger.info("Saved %d fact(s) for user %s.", len(facts), user_id[:8])
    return facts


def format_for_prompt(facts: list[str]) -> str:
    """Format user facts for injection into the system prompt.

    Args:
        facts: List of fact strings as returned by :func:`get_facts`.

    Returns:
        A multi-line string, or an empty string when *facts* is empty.
    """
    if not facts:
        return ""
    lines = ["What I know about this user:"]
    for fact in facts:
        lines.append(f"- {fact}")
    return "\n".join(lines)
