"""Persistent per-user long-term memory backed by Qdrant.

After each conversation the LLM extracts memorable facts about the user —
preferences, background, ongoing projects, constraints — and stores them as
embedding vectors in Qdrant.  On the next session the top-k most relevant
facts (by cosine similarity to the current query) are retrieved and injected
into the system prompt so the assistant remembers who it is talking to.

This approach gives us three things over the old SQLite store:
- Relevance filtering: only facts semantically related to the query are
  injected, keeping the context window focused.
- Semantic deduplication: a new fact is skipped when it is too similar to
  an existing one (cosine similarity ≥ MEMORY_DEDUP_THRESHOLD).
- Scalability: Qdrant handles an unbounded fact store efficiently.

Each point in the collection has the payload:
    {
        "fact":      str,   -- the fact sentence
        "user_id":   str,   -- opaque user identifier
        "timestamp": str,   -- ISO-8601 UTC
    }
"""

import logging
from datetime import datetime, timedelta, timezone

import src.clients.ollama_client as ollama
import src.clients.qdrant_client as qdrant

from src.config import (
    FAST_MODEL,
    MEMORY_COLLECTION,
    MEMORY_DEDUP_THRESHOLD,
    MEMORY_FACT_TTL_DAYS,
    MEMORY_TOP_K,
    TEMPERATURE_JSON,
)
from src.schemas import FinalResponse

logger = logging.getLogger(__name__)


# ── Collection initialisation ─────────────────────────────────────────────────

def _init_collection() -> None:
    """Create the memory collection in Qdrant if it does not exist."""
    try:
        qdrant.create_memory_collection(MEMORY_COLLECTION)
    except Exception as exc:
        logger.warning("Could not initialise memory collection: %s", exc)


_init_collection()


# ── Public API ────────────────────────────────────────────────────────────────

def get_facts(user_id: str, query: str) -> list[str]:
    """Return the top-k most relevant stored facts for *user_id*.

    Embeds *query* and performs a vector search filtered by user_id and TTL.
    Only facts within MEMORY_FACT_TTL_DAYS and above the default score
    threshold are returned.

    Args:
        user_id: Opaque user identifier.
        query:   The current user question (used for relevance ranking).

    Returns:
        List of fact strings ordered by relevance (most relevant first).
    """
    cutoff = (
        datetime.now(tz=timezone.utc) - timedelta(days=MEMORY_FACT_TTL_DAYS)
    ).isoformat()

    try:
        query_vector = ollama.embed(query)
        results = qdrant.search_memory(
            collection=MEMORY_COLLECTION,
            query_vector=query_vector,
            user_id=user_id,
            top_k=MEMORY_TOP_K,
            cutoff_timestamp=cutoff,
        )
    except Exception as exc:
        logger.warning("Memory retrieval failed: %s", exc)
        return []

    return [r["fact"] for r in results]


def extract_and_save(user_id: str, question: str, response: FinalResponse) -> list[str]:
    """Extract memorable facts from a conversation and persist them.

    Runs a lightweight LLM pass over the Q&A to pull out anything worth
    remembering about the user.  Each candidate fact is embedded and checked
    for near-duplicates already in the store (cosine similarity ≥
    MEMORY_DEDUP_THRESHOLD).  Novel facts are upserted into Qdrant.

    Args:
        user_id:  Opaque user identifier.
        question: The original user question.
        response: The completed :class:`~src.schemas.FinalResponse`.

    Returns:
        List of fact strings that were saved (may be empty).
    """

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
            temperature=TEMPERATURE_JSON,
            system=(
                "You extract memorable facts about users from conversations. "
                "Be concise and specific. Only record what the user revealed about themselves."
            ),
        )
        candidates = ollama.parse_json_list(raw)
    except Exception as exc:
        logger.warning("Fact extraction failed: %s", exc)
        return []

    if not candidates:
        return []

    now = datetime.now(tz=timezone.utc).isoformat()
    saved: list[str] = []

    for fact in candidates:
        try:
            fact_vector = ollama.embed(fact)

            # Semantic dedup: skip if a very similar fact already exists
            existing = qdrant.search_memory(
                collection=MEMORY_COLLECTION,
                query_vector=fact_vector,
                user_id=user_id,
                top_k=1,
                score_threshold=MEMORY_DEDUP_THRESHOLD,
            )
            if existing:
                logger.debug("Skipping duplicate fact (score=%.3f): %s", existing[0]["score"], fact)
                continue

            qdrant.upsert_memory(
                collection=MEMORY_COLLECTION,
                fact=fact,
                vector=fact_vector,
                payload={"user_id": user_id, "timestamp": now},
            )
            saved.append(fact)
        except Exception as exc:
            logger.warning("Failed to save fact '%s': %s", fact[:60], exc)

    if saved:
        logger.info("Saved %d fact(s) for user %s.", len(saved), user_id[:8])
    return saved


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
