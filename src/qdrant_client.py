"""Thin wrapper around the qdrant-client library."""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.config import QDRANT_URL
from src.retry import qdrant_retry

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None

# ── Client ────────────────────────────────────────────────────────────────────

def _get_client() -> QdrantClient:
    """Return a cached Qdrant client, creating it on first call."""
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


# ── Public API ────────────────────────────────────────────────────────────────

@qdrant_retry
def create_collection(name: str, vector_size: int = 768) -> None:
    """Create a Qdrant collection if it does not already exist.

    Args:
        name:        Collection name.
        vector_size: Dimensionality of the embedding vectors (default 768 for
                     nomic-embed-text).
    """
    client = _get_client()
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )
        print(f"[Qdrant] Created collection '{name}' (vector_size={vector_size})")
    else:
        print(f"[Qdrant] Collection '{name}' already exists — skipping creation")


@qdrant_retry
def upsert(collection: str, points: list[dict]) -> None:
    """Insert or update points in a collection.

    Args:
        collection: Target collection name.
        points:     List of dicts with keys ``id``, ``vector``, ``payload``.
                    The payload should contain ``text``, ``source_file``, and
                    ``chunk_index``.
    """
    client = _get_client()
    qdrant_points = [
        qmodels.PointStruct(
            id=p["id"],
            vector=p["vector"],
            payload=p["payload"],
        )
        for p in points
    ]
    client.upsert(collection_name=collection, points=qdrant_points)


@qdrant_retry
def search_with_filter(
    collection: str,
    query_vector: list[float],
    query_filter: qmodels.Filter,
    top_k: int = 5,
    score_threshold: float | None = None,
) -> list[tuple[int, float]]:
    """Search a collection with a payload filter, returning (point_id, score) pairs.

    Used for memory search where the caller needs to join results back to SQLite
    by point ID.

    Args:
        collection:      Collection name.
        query_vector:    Embedding of the query.
        query_filter:    Qdrant filter applied before scoring (e.g. user_id match).
        top_k:           Maximum number of results to return.
        score_threshold: Minimum similarity score; results below this are dropped.

    Returns:
        List of ``(point_id, score)`` tuples, highest score first.
    """
    client = _get_client()
    results = client.search(
        collection_name=collection,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=False,  # IDs are enough; metadata lives in SQLite
    )
    return [(int(r.id), r.score) for r in results]


@qdrant_retry
def search(
    collection: str,
    query_vector: list[float],
    top_k: int = 5,
    score_threshold: float | None = None,
) -> list[dict]:
    """Search a collection and return the top-k results above *score_threshold*.

    Args:
        collection:      Collection name.
        query_vector:    Embedding of the query.
        top_k:           Number of results to return.
        score_threshold: Minimum cosine similarity score; results below this
                         are excluded. If None, all results are returned.

    Returns:
        List of dicts with keys ``text``, ``source_file``, ``chunk_index``,
        and ``score``.
    """
    client = _get_client()
    results = client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )
    out = []
    for r in results:
        assert r.payload is not None, (
            f"Qdrant result {r.id} has no payload — was it ingested correctly?"
        )
        out.append({
            "text": r.payload.get("text", ""),
            "source_file": r.payload.get("source_file", ""),
            "chunk_index": r.payload.get("chunk_index", 0),
            "score": r.score,
        })
    return out
