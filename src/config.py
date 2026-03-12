"""Central configuration — all constants and env-var overrides live here."""

import os

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3")          # capable model — synthesis, conversational
FAST_MODEL: str = os.getenv("FAST_MODEL", "qwen3:1.7b")   # fast model — structured JSON, summaries
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_THINK: bool = os.getenv("LLM_THINK", "true").lower() == "true"
TEMPERATURE_JSON: float = float(os.getenv("TEMPERATURE_JSON", "0.1"))   # structured JSON calls
TEMPERATURE_SYNTHESIS: float = float(os.getenv("TEMPERATURE_SYNTHESIS", "0.7"))  # free-text answer

# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME: str = "documents"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50
TOP_K: int = 5
RAG_SCORE_THRESHOLD: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.55"))
HYDE_ENABLED: bool = os.getenv("HYDE_ENABLED", "true").lower() == "true"

# ── Agent behaviour ───────────────────────────────────────────────────────────
MAX_RESEARCH_ITERATIONS: int = int(os.getenv("MAX_RESEARCH_ITERATIONS", "2"))
MAX_WORKER_STEPS: int = int(os.getenv("MAX_WORKER_STEPS", "5"))

# ── Memory ────────────────────────────────────────────────────────────────────
LONG_TERM_MEMORY_DB: str = os.getenv("LONG_TERM_MEMORY_DB", "memory/long_term.db")
MEMORY_FACT_TTL_DAYS: int = int(os.getenv("MEMORY_FACT_TTL_DAYS", "90"))
MEMORY_COLLECTION: str = "user_memory"
MEMORY_TOP_K: int = int(os.getenv("MEMORY_TOP_K", "10"))
CHAT_HISTORY_TURNS: int = int(os.getenv("CHAT_HISTORY_TURNS", "3"))
MEMORY_DEDUP_THRESHOLD: float = float(os.getenv("MEMORY_DEDUP_THRESHOLD", "0.92"))

# ── Auth ──────────────────────────────────────────────────────────────────────
API_KEY: str = os.getenv("API_KEY", "")  # empty string = auth disabled

# ── Rate limiting ─────────────────────────────────────────────────────────────
RATE_LIMIT_QUERY: str = os.getenv("RATE_LIMIT_QUERY", "10/minute")
RATE_LIMIT_INGEST: str = os.getenv("RATE_LIMIT_INGEST", "60/minute")
RATE_LIMIT_HISTORY: str = os.getenv("RATE_LIMIT_HISTORY", "30/minute")

# ── Web search ────────────────────────────────────────────────────────────────
WEB_SEARCH_MAX_RESULTS: int = 5
WEB_SEARCH_TIMEOUT: int = int(os.getenv("WEB_SEARCH_TIMEOUT", "30"))

# ── arXiv search ──────────────────────────────────────────────────────────────
ARXIV_MAX_RESULTS: int = int(os.getenv("ARXIV_MAX_RESULTS", "5"))
ARXIV_SEARCH_TIMEOUT: int = int(os.getenv("ARXIV_SEARCH_TIMEOUT", "30"))
