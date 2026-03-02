# Multi-Agent System

A production-minded multi-agent AI system for particle physics research: an orchestrator decomposes questions and dispatches homogeneous **ResearchWorker** agents **in parallel** — one per sub-question — each running a **ReAct loop** (Reason → Act → Observe → repeat) with access to all tools. A synthesis agent then combines their evidence into a final answer. Demonstrates structured outputs, per-user memory, multi-path routing, and real-time SSE streaming — running **fully locally** with Ollama and Qdrant (no cloud APIs required for the core system).

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
 │
 ├─[1] Long-term memory ──► SQLite: inject known user facts into context
 │
 ├─[2] Decompose ──────────► LLM_MODEL: classify intent + plan sub-questions
 │                            │
 │              ┌─────────────┼──────────────────────┐
 │              ▼             ▼                      ▼
 │        conversational   tool call            sub-questions
 │              │         (calculator /              │
 │              │          unit_converter)            │
 │              ▼             ▼                      ▼
 │           respond      execute              [3] Research loop
 │           directly     → SynthesisAgent          │
 │                                    ┌──────────────┴──────────────┐
 │                                    │  asyncio.gather (parallel)   │
 │                                    ▼         ▼           ▼        │
 │                              Worker(q1)  Worker(q2)  Worker(q3)   │
 │                                    │         │           │        │
 │                              ┌─────▼─────────▼───────────▼─────┐ │
 │                              │         ReAct loop               │ │
 │                              │  ┌──────────────────────────┐    │ │
 │                              │  │ Reason: pick next tool   │    │ │
 │                              │  │ Act:    call tool        │    │ │
 │                              │  │ Observe: append result   │    │ │
 │                              │  │ repeat up to MAX_WORKER_ │    │ │
 │                              │  │        STEPS or "done"   │    │ │
 │                              │  └──────────────────────────┘    │ │
 │                              │  Tools available per worker:      │ │
 │                              │  • rag_search  (Qdrant hybrid)    │ │
 │                              │  • web_search  (DuckDuckGo)       │ │
 │                              │  • arxiv_search (arXiv API)       │ │
 │                              │  • calculator  (AST-safe)         │ │
 │                              │  • unit_converter                 │ │
 │                              └──────────────────────────────────┘ │
 │                                    │                              │
 │                                    ▼                              │
 │                              EvidenceBundle[]                     │
 │                                    │                              │
 │                                    ▼                              │
 │                             Gap analysis ──── sufficient? ──► break
 │                             (FAST_MODEL)         │
 │                                                  └──────────────┘
 │                                                   (loop back with
 │                                                    new questions)
 │
 └─[4] SynthesisAgent ─────► LLM_MODEL: final answer from all EvidenceBundles
```

**Agents:**
- `Orchestrator` — decomposes questions, dispatches ResearchWorkers, runs gap analysis, coordinates the loop
- `ResearchWorker` — runs a multi-turn ReAct loop; the LLM decides which tool to call next based on the question and prior tool results; one worker per sub-question, all in parallel
- `SynthesisAgent` — combines all `EvidenceBundle` results into a final answer

**Two-model split:**
- `LLM_MODEL` (`qwen3`, default) — decomposition, ReAct tool-selection steps, synthesis, conversational replies
- `FAST_MODEL` (`qwen3:1.7b`, default) — gap analysis (structured JSON, speed matters more than depth)

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

Use `docker-compose.cloud.yml`, which runs Ollama as a container with full GPU support via the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

**Automated setup:**

1. Edit `startup.sh` and set `REPO_URL` to your fork.
2. Copy it to the instance and run:

```bash
bash startup.sh
```

This installs dependencies, clones the repo, creates `.env`, and starts all services. On first run `ollama-init` pulls `qwen3`, `qwen3:1.7b`, and `nomic-embed-text` before the app starts. Models are stored in the `ollama_data` Docker volume and are not re-downloaded on subsequent starts.

**Manual start:**

```bash
cp .env.example .env
docker compose -f docker-compose.cloud.yml up --build -d
```

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

1. **Ask a question** — type in the chat box and press Enter or click the arrow button.
2. **Watch the reasoning trace** expand in real time as workers step through their ReAct loops.
3. **See the final answer** with sources and any web links.

Each browser generates a UUID on first visit (stored in `localStorage`) that scopes its long-term memory — questions and answers are never shared between users.

### CLI document ingestion

```bash
uv run python -m src.ingest docs/my_document.pdf
```

### Demo knowledge base

The reference knowledge base is populated with Wikipedia articles on particle physics. The domain was chosen deliberately: technical terminology (quarks, leptons, bosons, hadronisation) stress-tests keyword discrimination in the sparse vectors; heavy cross-referencing between concepts (e.g. the Standard Model article referencing Higgs, QCD, and electroweak unification) produces natural multi-hop questions; and every factual claim is independently verifiable, making retrieval quality straightforward to evaluate.

---

## Evaluation

The eval pipeline measures RAG quality using [RAGAS](https://docs.ragas.io/) with `gpt-4o-mini` as the judge.

**Metrics (all reference-free):**
- **Faithfulness** — is the answer grounded in the retrieved chunks?
- **Response Relevancy** — does the answer actually address the question?
- **Context Precision** — are the retrieved chunks relevant to the question?

**Prerequisites:** Ollama and Qdrant must be running, at least one document must be ingested, and an OpenAI API key must be exported (`gpt-4o-mini` costs a few cents per full run).

```bash
# 1. Install eval dependencies
uv sync --extra eval

# 2. Set your OpenAI API key
export OPENAI_API_KEY=sk-...

# 3. Run the full evaluation (orchestrator + scoring)
uv run python -m eval.run_eval

# 3b. Re-run scoring only from a previous results.json (skips the orchestrator)
uv run python -m eval.run_eval --score-only
```

The runner executes each question in `eval/golden_dataset.json` through the full orchestrator, then scores the results with RAGAS. Raw answers and retrieved contexts are saved to `eval/results.json` after step 1. Final scores (overall, per-category, per-question) are saved to `eval/scores.json`.

**Interpreting scores** (0–1 scale, higher is better):
- `> 0.7` — good; retrieval and generation are well-aligned
- `0.5–0.7` — acceptable; some gaps in grounding or relevance
- `< 0.5` — investigate; likely retrieval misses or hallucination

Note: faithfulness and context precision are not meaningful for calculator and unit converter questions (no retrieved context is expected). Focus on the `factual`, `multi_hop`, and `arxiv_search` categories for RAG quality.

**Adding questions to the golden dataset:**

Edit `eval/golden_dataset.json` and add entries with a `question` and `category`:

```json
{"id": 99, "question": "Your question here", "category": "factual"}
```

Supported categories: `factual`, `multi_hop`, `out_of_scope`, `calculator`, `unit_converter`, `arxiv_search`.

---

## Design Decisions

**No LangChain / LangGraph**
Every agent call, tool invocation, and routing decision is explicit Python. There are no hidden abstractions to debug, no dependency on a framework's release cadence, and the full reasoning trace is trivially inspectable.

**Structured Pydantic outputs**
Agents return typed models (`ResearchPlan`, `WorkerToolCall`, `GapAnalysis`, `EvidenceBundle`, `FinalResponse`, …) rather than raw strings. This enforces a contract between components, makes unit-testing straightforward, and causes parse failures to be loud rather than silent.

`EvidenceBundle` is the shared data contract between ResearchWorkers and the orchestrator: it carries the sub-question, the formatted context string, document source files, web URLs, and raw chunk texts (used by the eval pipeline). `WorkerToolCall` is the per-step JSON schema the LLM must produce inside the ReAct loop, specifying which tool to call and why.

**Four-path routing via LLM classification**
The orchestrator classifies every input in a single LLM call before doing any other work:

1. **Conversational** — greetings, chit-chat, meta questions → answered directly without RAG
2. **Calculator** — arithmetic, math functions, physics constant expressions → evaluated by an AST-based safe calculator, never `eval()`
3. **Unit converter** — unit conversion requests (energy, mass, cross-section, length, …) → handled by a lookup-table converter
4. **Research** — everything else → full research loop (ResearchWorkers)

Paths 1–3 are fast-paths that bypass the research loop entirely. The alternative — a dedicated lightweight classifier — would require labelled training data and retraining whenever a new route is added; prompting is cheaper to maintain and handles novel input gracefully.

**Homogeneous ResearchWorkers with ReAct loops**
For research questions, the orchestrator dispatches one `ResearchWorker` per sub-question, all running concurrently via `asyncio.gather`. Each worker runs a multi-turn ReAct loop (up to `MAX_WORKER_STEPS`, default 5):

1. The worker's LLM receives the sub-question and the conversation history so far
2. It produces a `WorkerToolCall` JSON object: which tool to call and why
3. The tool is executed and its result appended as a user message
4. The loop continues until the LLM emits `"tool": "done"` or the step cap is reached

The key design property is that **source selection is an emergent decision** made by the LLM after seeing the question, not a hardcoded routing rule made before any evidence is gathered. A question about a recent experimental result might prompt the worker to try `rag_search` first, then `web_search` if the documents lack recent data, then `arxiv_search` for the paper — the same worker, the same loop, different tool sequence.

This replaces the previous architecture of three fixed specialist agents (`RAGAgent`, `WebSearchAgent`, `ArxivAgent`) that pre-committed to a single source per iteration. Those agents had no LLM reasoning and no ability to adapt their search strategy; they were routing logic dressed up as agents.

*Trade-off:* each ReAct step is a synchronous LLM call, so a worker that uses three tools makes three sequential LLM calls. The previous specialists made zero LLM calls during retrieval (only during summarisation). The latency cost is real but justified: the worker can use any combination of tools and can try alternative queries when the first attempt fails, something the fixed specialists could not do.

**Asyncio cooperative safety for shared state**
All workers within a research iteration share a single `seen_chunks: set[tuple[str, int]]` for cross-worker deduplication. Mutations to this set are safe without locks because asyncio is cooperative: set mutations happen synchronously within a single coroutine's turn and cannot interleave with mutations in another coroutine. No chunk is ever returned to two workers in the same iteration.

**Gap analysis produces follow-up questions, not source queries**
After each research iteration, `FAST_MODEL` evaluates whether the accumulated evidence is sufficient to answer the original question. If not, it produces a list of `follow_up_questions` — the next generation of sub-questions for the next batch of ResearchWorkers. It does not specify which sources to use; that decision belongs to the workers. This separation keeps the gap-analysis schema (`GapAnalysis`) simple and source-agnostic.

**Two-model split**
`LLM_MODEL` (`qwen3` by default) handles tasks requiring coherent reasoning: decomposition, ReAct tool-selection steps, synthesis, and conversational replies. `FAST_MODEL` (`qwen3:1.7b` by default) handles gap analysis, where the output is a small structured JSON object and speed matters more than depth. Both are configurable via environment variables.

ReAct steps use `think=False` (disabling extended chain-of-thought output) because the step output must be parseable JSON; reasoning quality is provided by the model itself in the `reasoning` field of `WorkerToolCall`, not by a separate thinking block.

**SSE streaming**
The frontend opens a single `fetch()` stream and receives agent steps as they happen. Users see the reasoning trace build in real time — including each worker's ReAct steps — rather than staring at a blank screen for the full duration of the research loop.

**Hybrid search (dense + sparse RRF)**
Document retrieval combines two complementary vector representations fused via Reciprocal Rank Fusion (RRF):

- **Dense vectors** (nomic-embed-text, 768d, HNSW): capture semantic meaning and paraphrase-level similarity.
- **Sparse vectors** (BM25 via FastEmbed `Qdrant/bm25`, `src/sparse.py`): capture exact keyword matches with proper TF-IDF weighting, especially effective for proper nouns and uncommon terms.

This combination fixes a systematic failure mode of pure semantic search: a resume chunk containing generic skills and experience bullets looks semantically similar to *any* question about *any* person's background. BM25's IDF component automatically down-weights corpus-wide common tokens ("skills", "experience") and amplifies discriminative ones (names, rare keywords), so "who is Alice Smith" retrieves Alice Smith's documents rather than the nearest semantic neighbour.

Sparse vectors are produced by FastEmbed's BM25 model (~few MB, downloaded to `~/.cache/fastembed` on first use). The score threshold applies to the dense prefetch branch only (cosine similarity gate before RRF); the final RRF score fuses rank positions from both branches independently of absolute similarity.

*Trade-off:* the hybrid schema stores two vectors per chunk (~2× the storage vs. dense-only) and requires two prefetch passes at query time. For the target workload (thousands of chunks, millisecond-class HNSW search) this is negligible.

**Qdrant for document RAG**
Document chunks are embedded and stored in Qdrant rather than a general-purpose database for two reasons. First, Qdrant's HNSW index makes approximate nearest-neighbour search over thousands of chunks fast without a full scan. Second, Qdrant's payload filtering lets searches scope to specific collections or metadata fields without a separate SQL join.

Postgres + pgvector is the more common production choice and consolidates everything into one service. However, since SQLite carries zero operational cost (it is just a file), the real infrastructure comparison is Qdrant vs Postgres. Qdrant is meaningfully lighter to operate: it ships with sensible defaults, requires no WAL tuning, no `pg_hba.conf`, and no connection pool management. For an on-premises deployment where you want to minimise moving parts, Qdrant + SQLite is a pragmatic choice.

**Per-user long-term memory: fact extraction**
The system extracts *facts about the user* from each conversation — preferences, background, ongoing projects, constraints — and stores them in SQLite. At the start of every subsequent session those facts are retrieved and injected into the system prompt, so the assistant remembers who it is talking to rather than just what it has previously answered. Facts accumulate slowly — a few per conversation — and are small enough that the full set can be injected without semantic search or a vector store. Plain SQLite is sufficient.

**uv for dependency management**
`uv` manages the virtualenv and produces a `uv.lock` file that pins every transitive dependency at exact versions. `uv sync --frozen` in Docker ensures the container always installs exactly what was tested locally.

**Exponential backoff on external calls**
All Ollama and Qdrant calls are wrapped with `backoff.expo` (max 4 attempts, full jitter). Retry logic lives in `src/retry.py` as shared decorators (`http_retry`, `qdrant_retry`) so each client stays focused on its own logic.

**Ollama over vLLM**
Ollama is used for local model serving rather than vLLM. vLLM's headline advantages — PagedAttention and continuous batching — maximise throughput when many requests arrive concurrently and can be merged into a single forward pass. This system's research loop is inherently sequential (decompose → research → gap analysis → synthesise), so there is no batch to form per user, and low concurrency means PagedAttention's KV cache savings are irrelevant. Ollama also has first-class Apple Silicon support via Metal; vLLM has no Metal backend and would fall back to CPU-only inference on macOS, making it impractical for local development.

*When to reconsider:* a dedicated Linux server with a high-VRAM NVIDIA GPU serving many concurrent users is the point at which vLLM's throughput advantage becomes meaningful. At that scale, full-precision or AWQ-quantised models (vLLM's sweet spot) also outperform GGUF quantisation.

**DuckDuckGo for web search**
No API key, no rate-limit tiers for moderate use, and entirely client-side — keeping the system fully self-hostable. The `duckduckgo-search` package wraps the public DDGS API.

---

## Known Limitations / Future Work

- **Simple API key auth** — set `API_KEY` in `.env` to enable; the key is passed as a `?api_key=` query parameter and stored in `localStorage`. Sufficient for a demo; not appropriate for a public or multi-tenant deployment (use OAuth or a proper identity layer instead).
- **UUID identity is not authenticated** — anyone who knows a user's UUID can query their memory; acceptable for a demo, not for a multi-tenant deployment.
- **DuckDuckGo reliability** — the unofficial DDGS API can be rate-limited or blocked; a production deployment should use a paid search API (Brave, Tavily, …).
- **Observability** — add [Langfuse](https://langfuse.com/) (self-hostable) for persistent trace history, per-span latency breakdowns, and cross-session analytics. The existing SSE trace covers real-time visibility but traces are lost on page refresh; Langfuse also enables building an evaluation dataset from real queries.
- **ReAct latency** — each worker tool-selection step is a sequential LLM call. A worker that calls three tools makes three round-trips to the LLM before returning evidence. Caching tool results (especially embedding + Qdrant searches) and batching tool calls where the model is confident would reduce latency.
- **Model routing** — the two-tier split (`LLM_MODEL` for ReAct/synthesis/conversational, `FAST_MODEL` for gap analysis) is a static, hand-coded split. A more sophisticated router would classify each task at runtime — considering query complexity, required output format, and confidence requirements — and select from a wider model menu.
- **Caching** — the research pipeline makes several LLM calls per query with no caching. Embedding results, RAG results, and full responses are all candidates for caching to reduce latency and compute cost on repeated or similar queries.
- **RAG score threshold calibration** — `RAG_SCORE_THRESHOLD` (default 0.55, overridable via env var) is currently set by intuition. For a production system, the threshold should be tuned empirically: collect a representative query set, plot the score distribution of relevant vs. irrelevant chunks, and pick a threshold that maximises recall while holding precision above an acceptable floor. A RAGAS evaluation run provides exactly this data.
- **Memory: selective injection** — all stored facts are injected on every request. With many sessions this wastes tokens on irrelevant context; a production system would filter facts by relevance (semantic search) or recency before injection.
- **Memory: contradiction handling** — facts are only appended, never updated. If a user says "I use a VPS" then later "I switched to bare metal", both facts persist. A production system would detect conflicts and supersede outdated facts.
- **Memory: consolidation** — related facts accumulate independently over time and can become noisy. Periodically merging similar facts (e.g. via an LLM summarisation pass) would keep the fact store concise.
- **Memory: triggered extraction** — fact extraction runs after every conversation, including ones where nothing memorable was shared (e.g. a calculator query). Skipping extraction when the conversation is unlikely to contain user-relevant facts would save an LLM call per request.
- **Memory: user management** — there is no interface to view, edit, or delete individual facts. Relevant for transparency and GDPR right-to-erasure.
