"""RAG retrieval tool — hybrid dense + sparse search over Qdrant."""

import asyncio

import src.clients.ollama_client as ollama
import src.clients.qdrant_client as qdrant
from src.config import (
    COLLECTION_NAME,
    FAST_MODEL,
    HYDE_ENABLED,
    RAG_SCORE_THRESHOLD,
    TEMPERATURE_SYNTHESIS,
    TOP_K,
)
from src.tools.sparse import compute_sparse


async def _hypothetical_document(query: str) -> str:
    """Generate a short hypothetical answer to *query* for HyDE embedding.

    The hypothetical document is written as if it were extracted from a
    relevant source — phrased like document content rather than a question —
    so its embedding lands closer to actual document chunks in vector space.
    """
    return await asyncio.to_thread(
        ollama.chat,
        prompt=f"Question: {query}\n\nWrite a short, factual paragraph that directly answers this question. Be concise (2-4 sentences). Write as if extracted from a reference document.",
        model=FAST_MODEL,
        think=False,
        temperature=TEMPERATURE_SYNTHESIS,
    )


async def rag_search(query: str) -> list[dict]:
    """Embed *query* and return the top matching chunks from Qdrant.

    When HYDE_ENABLED, generates a hypothetical answer and embeds that for
    the dense vector instead of the raw query. The sparse (BM25) vector
    always uses the original query so keyword matching is not degraded.

    Uses hybrid dense + sparse RRF fusion with a score threshold on the dense
    branch. Returns an empty list on any failure.

    Args:
        query: The natural-language question or search string.

    Returns:
        List of dicts with keys ``text``, ``source_file``, ``chunk_index``,
        and ``score``.
    """
    try:
        if HYDE_ENABLED:
            hypothesis = await _hypothetical_document(query)
            dense_text = ollama.strip_thinking(hypothesis)
        else:
            dense_text = query

        query_vector, sparse = await asyncio.gather(
            asyncio.to_thread(ollama.embed, dense_text),
            asyncio.to_thread(compute_sparse, query),
        )
        sparse_indices, sparse_values = sparse

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
