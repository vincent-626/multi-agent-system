# Multi-Agent System

A production-minded multi-agent AI system demonstrating orchestrator/specialist architecture, structured outputs, per-user memory, confidence-based routing, web search, and real-time streaming — all running **fully locally** with no cloud APIs.

---

## Architecture

```
User
 │
 ▼
Browser UI  (vanilla JS, SSE streaming)
 │
 ▼
FastAPI Server  (src/server.py)
 │
 ▼
Orchestrator  (src/agents/orchestrator.py)
 ├── Long-Term Memory   (per-user facts → SQLite)
 ├── Short-Term Memory  (in-session step trace)
 │
 ├─[1] Decompose ──────► LLM: split into sub-questions
 │
 ├─[2] Retrieve ───────► Ollama (embed) + Qdrant (search)
 │      per sub-question  chunks injected as evidence
 │
 ├─[3] Gap Analysis ───► LLM: is evidence sufficient?
 │      (loops back)       follow-up questions or web queries
 │
 ├─[4] Web Search ─────► DuckDuckGo (no API key)
 │      on gap queries    results injected as evidence
 │
 └─[5] Synthesise ─────► LLM: final answer from all evidence
```

---

## Prerequisites

- [Docker](https://www.docker.com/) & Docker Compose
- [uv](https://docs.astral.sh/uv/) for local development only

---

## Setup

### Mac (local)

Docker on Mac cannot access the GPU, so Ollama runs natively to get Metal acceleration. Only Qdrant and the app run in Docker.

```bash
# 1. Install and start Ollama natively
brew install ollama
ollama serve &

# 2. Pull the required models
ollama pull qwen3
ollama pull qwen3:1.7b
ollama pull nomic-embed-text

# 3. Copy env file and start the stack
cp .env.example .env
docker-compose up --build
```

The app connects to native Ollama via `host.docker.internal:11434`.

### Linux / cloud server

Ollama runs as a container alongside the app. Uncomment the `ollama` and `ollama-init` services in `docker-compose.yml` and set `OLLAMA_BASE_URL=http://ollama:11434` in the app's environment block. For GPU support, also uncomment the `deploy` block (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)).

```bash
cp .env.example .env
docker-compose up --build
```

On first run `ollama-init` pulls `qwen3`, `qwen3:1.7b`, and `nomic-embed-text` before the app starts. Models are stored in the `ollama_data` Docker volume and are not re-downloaded on subsequent starts.

---

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Local development

```bash
# Install dependencies
uv sync

# Run the server with auto-reload
uv run uvicorn src.server:app --reload

# Ingest a document
uv run python -m src.ingest docs/my_document.pdf
```

---

## Usage

### Via the UI

1. **Upload a document** — drag a PDF or TXT onto the sidebar upload zone. The system will chunk, embed, and store it in Qdrant automatically.
2. **Ask a question** — type in the chat box and press Enter or click **Ask**.
3. **Watch the reasoning trace** expand in real time as agents work.
4. **See the final answer** with confidence, sources, and any web links.

Each browser generates a UUID on first visit (stored in `localStorage`) that scopes its long-term memory — questions and answers are never shared between users.

### CLI document ingestion

```bash
uv run python -m src.ingest docs/my_document.pdf
```

### Demo knowledge base

The reference knowledge base is populated with Wikipedia articles on particle physics. The domain was chosen deliberately: technical terminology (quarks, leptons, bosons, hadronisation) stress-tests keyword discrimination in the sparse vectors; heavy cross-referencing between concepts (e.g. the Standard Model article referencing Higgs, QCD, and electroweak unification) produces natural multi-hop questions; and every factual claim is independently verifiable, making retrieval quality straightforward to evaluate.

### Adding a dependency

```bash
uv add <package>   # updates pyproject.toml and uv.lock
```

---

## Design Decisions

**No LangChain / LangGraph**
Every agent call, tool invocation, and routing decision is explicit Python. There are no hidden abstractions to debug, no dependency on a framework's release cadence, and the full reasoning trace is trivially inspectable.

**Structured Pydantic outputs**
Agents return typed models (`RAGResult`, `OrchestratorDecision`, …) rather than raw strings. This enforces a contract between components, makes unit-testing straightforward, and causes parse failures to be loud rather than silent.

**Research agent architecture**
The orchestrator operates as a research agent: it decomposes the question into sub-questions, retrieves evidence for each independently, runs a gap analysis to identify what is still missing, issues follow-up queries (doc retrieval or web search), and finally synthesises across all gathered evidence. This handles multi-hop questions that a single retrieval pass would miss — e.g. "compare the risk profiles in these two reports and identify what neither covers".

The loop is bounded by `MAX_RESEARCH_ITERATIONS` (default 2) to prevent runaway behaviour. This architecture requires a capable reasoning model; the default is `qwen3:8b`, which produces credible decompositions and gap analyses. Smaller models (1–3b) tend to produce shallow sub-questions and unreliable gap detection.

**Retrieval outside the routing loop**
Document retrieval (embed + vector search) runs unconditionally before the orchestrator loop whenever documents exist. The retrieved chunks are injected directly into the loop prompt as context; the orchestrator LLM never decides *whether* to search — it only decides what to do *after* seeing the results.

This deliberately contradicts the more common agentic pattern of routing to a RAG specialist on demand. The routing-first approach suffers from a fundamental flaw: a small model asked "should I search?" before seeing any evidence will frequently guess wrong and skip retrieval entirely, answering from its own weights even when relevant documents exist. Retrieval is cheap (one embedding call + one vector search — no LLM); routing decisions are not free and are unreliable at small model sizes.

*Trade-off:* retrieval always runs even for questions the documents cannot answer (e.g. arithmetic, current events). This costs one embedding call and one vector search per request when documents are loaded. The upside — guaranteed document visibility — is worth the cost for the target use case.

**Confidence-based routing**
The orchestrator assigns a `confidence_score` to every decision. After a low-confidence result it can fall back to web search. Systems that acknowledge uncertainty are far safer in regulated or high-stakes contexts.

**SSE streaming**
The frontend opens a single `fetch()` stream and receives agent steps as they happen. Users see the reasoning trace build in real time rather than staring at a blank screen.

**Hybrid search (dense + sparse RRF)**
Document retrieval combines two complementary vector representations fused via Reciprocal Rank Fusion (RRF):

- **Dense vectors** (nomic-embed-text, 768d, HNSW): capture semantic meaning and paraphrase-level similarity.
- **Sparse vectors** (BM25 via FastEmbed `Qdrant/bm25`, `src/sparse.py`): capture exact keyword matches with proper TF-IDF weighting, especially effective for proper nouns and uncommon terms.

This combination fixes a systematic failure mode of pure semantic search: a resume chunk containing generic skills and experience bullets looks semantically similar to *any* question about *any* person's background. BM25's IDF component automatically down-weights corpus-wide common tokens ("skills", "experience") and amplifies discriminative ones (names, rare keywords), so "who is Alice Smith" retrieves Alice Smith's documents rather than the nearest semantic neighbour.

Sparse vectors are produced by FastEmbed's BM25 model (~few MB, downloaded to `~/.cache/fastembed` on first use). The score threshold applies to the dense prefetch branch only (cosine similarity gate before RRF); the final RRF score fuses rank positions from both branches independently of absolute similarity.

*Trade-off:* the hybrid schema stores two vectors per chunk (~2× the storage vs. dense-only) and requires two prefetch passes at query time. For the target workload (thousands of chunks, millisecond-class HNSW search) this is negligible.

**Qdrant for document RAG**
Document chunks are embedded and stored in Qdrant rather than a general-purpose database for two reasons. First, Qdrant's HNSW index makes approximate nearest-neighbour search over thousands of chunks fast without a full scan. Second, Qdrant's payload filtering lets the RAG specialist scope searches to specific collections or metadata fields without a separate SQL join.

Postgres + pgvector is the more common production choice and consolidates everything into one service. However, since SQLite carries zero operational cost (it is just a file), the real infrastructure comparison is Qdrant vs Postgres. Qdrant is meaningfully lighter to operate: it ships with sensible defaults, requires no WAL tuning, no `pg_hba.conf`, and no connection pool management. For an on-premises deployment where you want to minimise moving parts, Qdrant + SQLite is a pragmatic choice.

**Per-user long-term memory: fact extraction**
The system extracts *facts about the user* from each conversation — preferences, background, ongoing projects, constraints — and stores them in SQLite.  At the start of every subsequent session those facts are retrieved and injected into the system prompt, so the assistant remembers who it is talking to rather than just what it has previously answered.Facts accumulate slowly — a few per conversation — and are small enough that the full set can be injected without semantic search or a vector store.  Plain SQLite is sufficient.

**uv for dependency management**
`uv` manages the virtualenv and produces a `uv.lock` file that pins every transitive dependency at exact versions. `uv sync --frozen` in Docker ensures the container always installs exactly what was tested locally.

**Exponential backoff on external calls**
All Ollama and Qdrant calls are wrapped with `backoff.expo` (max 4 attempts, full jitter). Retry logic lives in `src/retry.py` as shared decorators (`http_retry`, `qdrant_retry`) so each client stays focused on its own logic.

**Ollama over vLLM**
Ollama is used for local model serving rather than vLLM. vLLM's headline advantages — PagedAttention and continuous batching — maximise throughput when many requests arrive concurrently and can be merged into a single forward pass. This system's research loop is inherently sequential (decompose → retrieve → gap analysis → synthesise), so there is no batch to form per user, and low concurrency means PagedAttention's KV cache savings are irrelevant. Ollama also has first-class Apple Silicon support via Metal; vLLM has no Metal backend and would fall back to CPU-only inference on macOS, making it impractical for local development.

*When to reconsider:* a dedicated Linux server with a high-VRAM NVIDIA GPU serving many concurrent users is the point at which vLLM's throughput advantage becomes meaningful. At that scale, full-precision or AWQ-quantised models (vLLM's sweet spot) also outperform GGUF quantisation.

**DuckDuckGo for web search**
No API key, no rate-limit tiers for moderate use, and entirely client-side — keeping the system fully self-hostable. The `duckduckgo-search` package wraps the public DDGS API.

---

## Known Limitations / Future Work

- **Simple API key auth** — set `API_KEY` in `.env` to enable; the key is passed as a `?api_key=` query parameter and stored in `localStorage`. Sufficient for a demo; not appropriate for a public or multi-tenant deployment (use OAuth or a proper identity layer instead).
- **UUID identity is not authenticated** — anyone who knows a user's UUID can query their memory; acceptable for a demo, not for a multi-tenant deployment.
- **DuckDuckGo reliability** — the unofficial DDGS API can be rate-limited or blocked; a production deployment should use a paid search API (Brave, Tavily, …).
- **Observability** — add [Langfuse](https://langfuse.com/) (self-hostable) for persistent trace history, per-span latency breakdowns, and cross-session analytics. The existing SSE trace covers real-time visibility but traces are lost on page refresh; Langfuse also enables building an evaluation dataset from real queries.
- **Model routing** — the current two-tier setup (`LLM_MODEL` for synthesis/conversational, `FAST_MODEL` for structured JSON tasks) is a static, hand-coded split. A more sophisticated router would classify each task at runtime — considering query complexity, required output format, and confidence requirements — and select from a wider model menu (e.g. a mid-tier model for gap analysis on complex topics, a larger model only for multi-document synthesis). Routing could also be cost-aware, falling back to a larger model only when a smaller one produces a low-confidence or malformed output.
- **RAG score threshold calibration** — `RAG_SCORE_THRESHOLD` (default 0.55, overridable via env var) is currently set by intuition. For a production system, the threshold should be tuned empirically: collect a representative query set, plot the score distribution of relevant vs. irrelevant chunks, and pick a threshold that maximises recall while holding precision above an acceptable floor. A RAGAS evaluation run provides exactly this data.
- **Memory: selective injection** — all stored facts are injected on every request. With many sessions this wastes tokens on irrelevant context; a production system would filter facts by relevance (semantic search) or recency before injection.
- **Memory: contradiction handling** — facts are only appended, never updated. If a user says "I use a VPS" then later "I switched to bare metal", both facts persist. A production system would detect conflicts and supersede outdated facts.
- **Memory: consolidation** — related facts accumulate independently over time and can become noisy. Periodically merging similar facts (e.g. via an LLM summarisation pass) would keep the fact store concise.
- **Memory: triggered extraction** — fact extraction runs after every conversation, including ones where nothing memorable was shared (e.g. a calculator query). Skipping extraction when the conversation is unlikely to contain user-relevant facts would save an LLM call per request.
- **Memory: user management** — there is no interface to view, edit, or delete individual facts. Relevant for transparency and GDPR right-to-erasure.
