"""Thin wrapper around the qdrant-client library."""

import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.config import QDRANT_URL
from src.clients.retry import qdrant_retry

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
    """Create a Qdrant collection with named dense + sparse vectors if it does not exist.

    Uses named vector configs so that hybrid (dense + sparse) search can be
    performed via RRF fusion at query time.

    Args:
        name:        Collection name.
        vector_size: Dimensionality of the dense embedding vectors (default 768
                     for nomic-embed-text).
    """
    client = _get_client()
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": qmodels.VectorParams(
                    size=vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": qmodels.SparseVectorParams(),
            },
        )
        print(f"[Qdrant] Created collection '{name}' (vector_size={vector_size})")
    else:
        print(f"[Qdrant] Collection '{name}' already exists — skipping creation")


@qdrant_retry
def upsert(collection: str, points: list[dict]) -> None:
    """Insert or update points in a collection.

    Args:
        collection: Target collection name.
        points:     List of dicts with keys ``id``, ``vector``, ``payload``,
                    and optionally ``sparse_indices`` / ``sparse_values``.
                    The payload should contain ``text``, ``source_file``, and
                    ``chunk_index``.
    """
    client = _get_client()
    qdrant_points = []
    for p in points:
        vector: dict = {"dense": p["vector"]}
        if "sparse_indices" in p and "sparse_values" in p:
            vector["sparse"] = qmodels.SparseVector(
                indices=p["sparse_indices"],
                values=p["sparse_values"],
            )
        qdrant_points.append(
            qmodels.PointStruct(
                id=p["id"],
                vector=vector,
                payload=p["payload"],
            )
        )
    client.upsert(collection_name=collection, points=qdrant_points)



@qdrant_retry
def search(
    collection: str,
    query_vector: list[float],
    sparse_indices: list[int] | None = None,
    sparse_values: list[float] | None = None,
    top_k: int = 5,
    score_threshold: float | None = None,
) -> list[dict]:
    """Search a collection using hybrid (dense + sparse) RRF fusion.

    When ``sparse_indices`` and ``sparse_values`` are provided, runs a
    two-branch prefetch (dense HNSW + sparse keyword) and combines them with
    Reciprocal Rank Fusion.  Falls back to dense-only search when sparse
    vectors are omitted.

    ``score_threshold`` is applied to the dense prefetch branch only (cosine
    similarity gate before RRF).  It has no effect on the sparse branch or the
    final RRF scores, which are on a different scale.

    Args:
        collection:      Collection name.
        query_vector:    Dense embedding of the query.
        sparse_indices:  Token indices of the sparse query vector.
        sparse_values:   TF weights corresponding to *sparse_indices*.
        top_k:           Number of results to return.
        score_threshold: Minimum cosine similarity for dense prefetch candidates.

    Returns:
        List of dicts with keys ``text``, ``source_file``, ``chunk_index``,
        and ``score``.
    """
    client = _get_client()

    if sparse_indices is not None and sparse_values is not None:
        prefetch = [
            qmodels.Prefetch(
                query=qmodels.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
                using="sparse",
                limit=top_k * 3,
            ),
            qmodels.Prefetch(
                query=query_vector,
                using="dense",
                limit=top_k * 3,
                score_threshold=score_threshold,
            ),
        ]
        response = client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        results = response.points
    else:
        results = client.search(
            collection_name=collection,
            query_vector=("dense", query_vector),
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


@qdrant_retry
def create_memory_collection(name: str, vector_size: int = 768) -> None:
    """Create a dense-only memory collection if it does not exist.

    Memory facts use dense-only search (no sparse vectors) — they are short
    natural-language sentences where semantic similarity is what matters.

    Args:
        name:        Collection name.
        vector_size: Embedding dimensionality (default 768 for nomic-embed-text).
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
        logger.info("[Qdrant] Created memory collection '%s'", name)


@qdrant_retry
def upsert_memory(collection: str, fact: str, vector: list[float], payload: dict) -> None:
    """Store a single memory fact as a Qdrant point.

    Args:
        collection: Target collection name.
        fact:       The fact text (stored in payload for retrieval).
        vector:     Dense embedding of *fact*.
        payload:    Arbitrary metadata (e.g. ``user_id``, ``timestamp``).
    """
    client = _get_client()
    point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{payload.get('user_id', '')}:{fact}"))
    client.upsert(
        collection_name=collection,
        points=[
            qmodels.PointStruct(
                id=point_id,
                vector=vector,
                payload={"fact": fact, **payload},
            )
        ],
    )


@qdrant_retry
def search_memory(
    collection: str,
    query_vector: list[float],
    user_id: str,
    top_k: int = 10,
    score_threshold: float | None = None,
    cutoff_timestamp: float | None = None,
) -> list[dict]:
    """Retrieve the most relevant memory facts for *user_id*.

    Filters by user_id (and optionally a minimum timestamp) via Qdrant
    payload filters, then ranks by cosine similarity to *query_vector*.

    Args:
        collection:       Collection name.
        query_vector:     Dense embedding of the current query.
        user_id:          Only return facts belonging to this user.
        top_k:            Maximum number of facts to return.
        score_threshold:  Minimum cosine similarity to include a fact.
        cutoff_timestamp: Unix timestamp (seconds); exclude facts older than this.

    Returns:
        List of dicts with keys ``fact``, ``timestamp``, and ``score``.
    """
    client = _get_client()

    must_conditions: list = [
        qmodels.FieldCondition(
            key="user_id",
            match=qmodels.MatchValue(value=user_id),
        )
    ]
    if cutoff_timestamp:
        must_conditions.append(
            qmodels.FieldCondition(
                key="timestamp",
                range=qmodels.Range(gte=cutoff_timestamp),
            )
        )

    results = client.search(
        collection_name=collection,
        query_vector=query_vector,
        query_filter=qmodels.Filter(must=must_conditions),
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )

    out = []
    for r in results:
        if r.payload is None:
            continue
        out.append({
            "fact": r.payload.get("fact", ""),
            "timestamp": r.payload.get("timestamp", ""),
            "score": r.score,
        })
    return out
