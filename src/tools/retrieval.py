"""RAG retrieval tool — embeds a query and searches Qdrant."""

from __future__ import annotations

import src.ollama_client as ollama
import src.qdrant_client as qdrant
from src.config import COLLECTION_NAME, TOP_K


def rag_search(query: str) -> list[dict]:
    """Embed *query* and return the top matching chunks from Qdrant.

    Args:
        query: The natural-language question or search string.

    Returns:
        List of dicts with keys ``text``, ``source_file``, ``chunk_index``,
        and ``score``.  Returns an empty list if no results are found.
    """
    vector = ollama.embed(query)
    results = qdrant.search(
        collection=COLLECTION_NAME,
        query_vector=vector,
        top_k=TOP_K,
    )
    return results
