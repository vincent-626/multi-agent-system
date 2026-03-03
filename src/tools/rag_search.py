"""RAG retrieval tool — hybrid dense + sparse search over Qdrant."""

import asyncio

import src.clients.ollama_client as ollama
import src.clients.qdrant_client as qdrant
from src.config import COLLECTION_NAME, RAG_SCORE_THRESHOLD, TOP_K
from src.tools.sparse import compute_sparse


async def rag_search(query: str) -> list[dict]:
    """Embed *query* and return the top matching chunks from Qdrant.

    Uses hybrid dense + sparse RRF fusion with a score threshold on the dense
    branch. Returns an empty list on any failure.

    Args:
        query: The natural-language question or search string.

    Returns:
        List of dicts with keys ``text``, ``source_file``, ``chunk_index``,
        and ``score``.
    """
    try:
        query_vector = await asyncio.to_thread(ollama.embed, query)
        sparse_indices, sparse_values = await asyncio.to_thread(compute_sparse, query)
        return await asyncio.to_thread(
            qdrant.search,
            COLLECTION_NAME,
            query_vector,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            top_k=TOP_K,
            score_threshold=RAG_SCORE_THRESHOLD,
        )
    except Exception:
        return []
