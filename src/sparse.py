"""Sparse vector computation for hybrid (dense + sparse) search.

Delegates to FastEmbed's BM25 sparse embedding model, which produces
TF-IDF weighted sparse vectors with corpus-level IDF calibration.

Advantages over a hand-rolled TF approach:
- IDF component: common tokens (e.g. "skills", "experience") are
  down-weighted globally, making discriminative terms (e.g. proper
  names, rare keywords) stand out.
- Proper tokenization: handles punctuation, casing, and morphology
  more robustly than a simple regex split.
- No index-collision risk: vocabulary indices are assigned by the model,
  not via CRC32 hashing.

The model ("Qdrant/bm25") is a few MB and is downloaded to
~/.cache/fastembed on first use.
"""

from __future__ import annotations

from fastembed import SparseTextEmbedding

_model: SparseTextEmbedding | None = None


def _get_model() -> SparseTextEmbedding:
    """Return a cached BM25 model, loading it on first call."""
    global _model
    if _model is None:
        _model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _model


def compute_sparse(text: str) -> tuple[list[int], list[float]]:
    """Return ``(indices, values)`` for a BM25-weighted sparse vector.

    Suitable for use as a Qdrant ``SparseVector``.

    Args:
        text: The input text to encode.

    Returns:
        A ``(indices, values)`` pair.  Both lists are empty when the text
        contains no usable tokens.
    """
    model = _get_model()
    embedding = next(iter(model.embed([text])))
    return embedding.indices.tolist(), embedding.values.tolist()
