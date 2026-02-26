"""Document ingestion pipeline.

Usage (CLI):
    python -m src.ingest <filepath>
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from pdfminer.high_level import extract_text as _pdf_extract

import src.ollama_client as ollama
import src.qdrant_client as qdrant
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME
from src.sparse import compute_sparse


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_text(path: Path) -> str:
    """Extract plain text from a PDF or .txt file.

    Args:
        path: Path to the document.

    Returns:
        The extracted text as a single string.

    Raises:
        ValueError: for unsupported file types.
    """
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        return _pdf_extract(str(path))

    raise ValueError(
        f"Unsupported file type '{suffix}'. Only .txt and .pdf are supported."
    )


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Recursively split *text* into chunks of at most *size* characters.

    Split order: double-newline → newline → ". " → character-level.

    Args:
        text:    The input text.
        size:    Maximum characters per chunk.
        overlap: Number of characters to carry over between chunks.

    Returns:
        List of non-empty chunk strings.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, sep_index: int) -> list[str]:
        if len(text) <= size or sep_index >= len(separators):
            return [text]
        sep = separators[sep_index]
        parts = text.split(sep) if sep else list(text)
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If the part itself is too large, recurse with next separator
                if len(part) > size:
                    chunks.extend(_split(part, sep_index + 1))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks

    raw_chunks = _split(text.strip(), 0)

    # Apply overlap: each chunk starts with the tail of the previous one
    if overlap <= 0 or len(raw_chunks) <= 1:
        return [c for c in raw_chunks if c.strip()]

    overlapped: list[str] = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        tail = raw_chunks[i - 1][-overlap:]
        overlapped.append(tail + raw_chunks[i])

    return [c for c in overlapped if c.strip()]


# ── Point ID generation ───────────────────────────────────────────────────────

def _point_id(source_file: str, chunk_index: int) -> int:
    """Generate a stable integer point ID from source file + chunk index."""
    digest = hashlib.md5(f"{source_file}:{chunk_index}".encode()).hexdigest()
    return int(digest[:16], 16) % (2**63)  # keep within signed 64-bit range


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_file(filepath: str | Path) -> int:
    """Ingest a single document into the Qdrant vector store.

    Steps: extract → chunk → embed → upsert.

    Args:
        filepath: Path to the document (.pdf or .txt).

    Returns:
        Number of chunks ingested.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"[Ingest] Processing '{path.name}' …")

    # 1. Extract text
    text = _extract_text(path)
    print(f"[Ingest] Extracted {len(text):,} characters.")

    # 2. Chunk
    chunks = _chunk_text(text)
    print(f"[Ingest] Split into {len(chunks)} chunks.")

    # 3. Ensure collection exists
    qdrant.create_collection(COLLECTION_NAME)

    # 4. Embed + sparse encode + upsert in batches
    points: list[dict] = []
    for i, chunk in enumerate(chunks):
        if i % 10 == 0:
            print(f"[Ingest] Embedding chunk {i+1}/{len(chunks)} …", end="\r")
        vector = ollama.embed(chunk)
        sparse_indices, sparse_values = compute_sparse(chunk)
        points.append(
            {
                "id": _point_id(path.name, i),
                "vector": vector,
                "sparse_indices": sparse_indices,
                "sparse_values": sparse_values,
                "payload": {
                    "text": chunk,
                    "source_file": path.name,
                    "chunk_index": i,
                },
            }
        )

    print()  # newline after \r progress
    qdrant.upsert(COLLECTION_NAME, points)
    print(f"[Ingest] Done. Upserted {len(points)} chunks from '{path.name}'.")
    return len(points)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingest <filepath>")
        sys.exit(1)
    n = ingest_file(sys.argv[1])
    print(f"Ingested {n} chunks.")
