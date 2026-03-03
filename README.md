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
 │              │          unit_converter)           │
 │              ▼             ▼                      ▼
 │           respond      execute              [3] Research loop
 │           directly     → SynthesisAgent           │
 │                                    ┌──────────────┴──────────────┐
 │                                    │  asyncio.gather (parallel)  │
 │                                    ▼         ▼           ▼       │
 │                              Worker(q1)  Worker(q2)  Worker(q3)  │
 │                                    │         │           │       │
 │                              ┌─────▼─────────▼───────────▼─────┐ │
 │                              │         ReAct loop              │ │
 │                              │  ┌──────────────────────────┐   │ │
 │                              │  │ Reason: pick next tool   │   │ │
 │                              │  │ Act:    call tool        │   │ │
 │                              │  │ Observe: append result   │   │ │
 │                              │  │ repeat up to MAX_WORKER_ │   │ │
 │                              │  │        STEPS or "done"   │   │ │
 │                              │  └──────────────────────────┘   │ │
 │                              │  Tools available per worker:    │ │
 │                              │  • rag_search  (Qdrant hybrid)  │ │
 │                              │  • web_search  (DuckDuckGo)     │ │
 │                              │  • arxiv_search (arXiv API)     │ │
 │                              │  • calculator  (AST-safe)       │ │
 │                              │  • unit_converter               │ │
 │                              └─────────────────────────────────┘ │
 │                                    │                             │
 │                                    ▼                             │
 │                              EvidenceBundle[]                    │
 │                                    │                             │
 │                                    ▼                             │
 │                             Gap analysis ──── sufficient? ────────► break
 │                             (FAST_MODEL)         │               │
 │                                                  └───────────────┘
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

1. Edit `scripts/startup.sh` and set `REPO_URL` to your fork.
2. Copy it to the instance and run:

```bash
bash scripts/startup.sh
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

Note: faithfulness and context precision are not meaningful for calculator and unit converter questions (no retrieved context is expected). Focus on the `factual` and `multi_hop` categories for RAG quality.

**Benchmark results** (particle physics Wikipedia knowledge base, `qwen3` + `qwen3:1.7b`, judge: `gpt-4o-mini`):

| Category | Faithfulness | Answer Relevancy | Context Precision |
|---|---|---|---|
| factual | 0.81 | 0.82 | 0.57 |
| multi_hop | 0.86 | 0.74 | 0.93 |
| out_of_scope | 0.44 | 0.00 | 0.65 |
| **overall** | **0.75** | **0.64** | **0.68** |

A few things worth noting about these numbers:

- **`out_of_scope` answer relevancy is 0.0 by design.** These questions have no answer in the knowledge base (e.g. "what GPA is required for CERN's summer programme?"). The correct system behaviour is to say "I don't have this information" — which RAGAS scores as irrelevant because it's looking for a substantive answer. A system that scored well here would be hallucinating.
- **`multi_hop` context precision (0.93) is the strongest signal.** These questions require connecting multiple concepts across documents (e.g. strong force → gluons → quark confinement), and the ResearchWorker consistently retrieves the right chunks across sub-questions.
- **`factual` context precision (0.57) is the weakest.** A few specific factual questions (LHC circumference, CERN member states) retrieved chunks that didn't contain the answer — either the knowledge base lacks those specific facts, or the worker's query wasn't targeted enough. These are the clearest candidates for knowledge base expansion.

**Adding questions to the golden dataset:**

Edit `eval/golden_dataset.json` and add entries with a `question` and `category`:

```json
{"id": 99, "question": "Your question here", "category": "factual"}
```

Supported categories: `factual`, `multi_hop`, `out_of_scope`, `calculator`, `unit_converter`.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **No LangChain / LangGraph** | Every agent call and routing decision is explicit Python. No hidden abstractions, no framework dependency, full reasoning trace is trivially inspectable. |
| **Structured Pydantic outputs** | Agents return typed models (`WorkerToolCall`, `EvidenceBundle`, `GapAnalysis`, …) rather than raw strings. Contracts between components are explicit; parse failures are loud, not silent. |
| **Four-path routing** | A single LLM classification call routes to: *conversational* (direct reply), *calculator* (AST evaluator, never `eval()`), *unit converter* (lookup table), or *research* (full loop). Fast-paths skip the research loop entirely. |
| **Homogeneous ResearchWorkers** | Each sub-question gets a worker running a ReAct loop with access to all tools. Source selection is an emergent LLM decision based on the question — not a hardcoded routing rule. All workers run concurrently via `asyncio.gather`. Replaces three fixed specialist agents that pre-committed to a single source with no reasoning. *Trade-off:* each tool call is a sequential LLM round-trip; flexibility costs latency. |
| **Two-model split** | `LLM_MODEL` (`qwen3`) for decomposition, ReAct steps, and synthesis. `FAST_MODEL` (`qwen3:1.7b`) for gap analysis where a small JSON response is all that's needed. Both overridable via env vars. |
| **Hybrid search (dense + sparse RRF)** | Dense vectors (nomic-embed-text) handle semantic similarity; sparse BM25 vectors handle exact keyword matches. Fused via RRF. Pure semantic search fails on proper nouns and rare terms — BM25's IDF weighting fixes this. ~2× storage cost, negligible at this scale. |
| **Qdrant + SQLite** | Qdrant for vector search, SQLite for memory and chat history. Postgres + pgvector consolidates both but adds operational overhead (WAL tuning, connection pooling). Qdrant + SQLite has fewer moving parts for a self-hosted deployment. |
| **Per-user long-term memory** | Facts (preferences, background, projects) are extracted from each conversation and injected into subsequent sessions. The user is remembered across sessions, not just within one. |
| **Ollama over vLLM** | vLLM's batching advantages require high concurrency to matter. This system's loop is sequential per user, so there's no batch to form. Ollama has first-class Metal support on macOS; vLLM falls back to CPU. Right trade-off for local development; revisit on a dedicated multi-user GPU server. |

---

## Known Limitations / Future Work

- **ReAct latency** — each tool-selection step is a sequential LLM call; a worker using three tools makes three round-trips. Caching embedding and retrieval results for repeated queries would help.
- **DuckDuckGo reliability** — the unofficial DDGS API can be rate-limited or blocked. A production deployment should use a paid search API (Brave, Tavily, …).
- **RAG threshold not calibrated** — `RAG_SCORE_THRESHOLD` (default 0.55) is set by intuition. It should be tuned against a representative query set, using RAGAS results to find the precision/recall trade-off point.
- **Observability** — SSE traces are lost on page refresh. Adding [Langfuse](https://langfuse.com/) (self-hostable) would give persistent trace history, per-span latency, and a real query dataset for eval.
- **Auth is demo-grade** — API key via query parameter, UUID identity unauthenticated. Fine for local use; a multi-tenant deployment needs a proper identity layer.
- **Memory limitations** — facts are appended but never updated or deduplicated, all facts are injected on every request regardless of relevance, and there is no UI to view or delete them.
