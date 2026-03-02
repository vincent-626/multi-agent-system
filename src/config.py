"""Central configuration — all constants and env-var overrides live here."""

import os

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3")          # capable model — synthesis, conversational
FAST_MODEL: str = os.getenv("FAST_MODEL", "qwen3:1.7b")   # fast model — structured JSON, summaries
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_THINK: bool = os.getenv("LLM_THINK", "true").lower() == "true"

# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME: str = "documents"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50
TOP_K: int = 5
RAG_SCORE_THRESHOLD: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.55"))

# ── Agent behaviour ───────────────────────────────────────────────────────────
MAX_AGENT_ITERATIONS: int = 8
MAX_RESEARCH_ITERATIONS: int = int(os.getenv("MAX_RESEARCH_ITERATIONS", "2"))
MAX_WORKER_STEPS: int = int(os.getenv("MAX_WORKER_STEPS", "5"))
CONFIDENCE_THRESHOLD: float = 0.6  # below this the orchestrator delegates

# ── Memory ────────────────────────────────────────────────────────────────────
LONG_TERM_MEMORY_DB: str = os.getenv("LONG_TERM_MEMORY_DB", "memory/long_term.db")

# ── Auth ──────────────────────────────────────────────────────────────────────
API_KEY: str = os.getenv("API_KEY", "")  # empty string = auth disabled

# ── Rate limiting ─────────────────────────────────────────────────────────────
RATE_LIMIT_QUERY: str = os.getenv("RATE_LIMIT_QUERY", "10/minute")
RATE_LIMIT_INGEST: str = os.getenv("RATE_LIMIT_INGEST", "5/minute")
RATE_LIMIT_HISTORY: str = os.getenv("RATE_LIMIT_HISTORY", "30/minute")

# ── Web search ────────────────────────────────────────────────────────────────
WEB_SEARCH_MAX_RESULTS: int = 5

# ── arXiv search ──────────────────────────────────────────────────────────────
ARXIV_MAX_RESULTS: int = int(os.getenv("ARXIV_MAX_RESULTS", "5"))
