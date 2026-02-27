"""FastAPI application with SSE streaming, file ingestion, and health endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from pathlib import Path

import requests as _requests
from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile
from fastapi.security import APIKeyQuery
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.agents.orchestrator import Orchestrator
from src.config import API_KEY, OLLAMA_BASE_URL, QDRANT_URL
from src.ingest import ingest_file
from src.memory.short_term import ShortTermMemory
from src.schemas import AgentStep, FinalResponse, QueryRequest

logger = logging.getLogger(__name__)

# ── Auth ──────────────────────────────────────────────────────────────────────

_api_key_query = APIKeyQuery(name="api_key", auto_error=False)


def _check_api_key(key: str | None = Security(_api_key_query)) -> None:
    """Reject requests with a wrong or missing key when API_KEY is configured."""
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")


app = FastAPI(title="Multi-Agent System", version="1.0.0")

# Serve the single-file frontend
_static_dir = Path(__file__).parent.parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root() -> HTMLResponse:
    """Serve the frontend UI."""
    index = _static_dir / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return HTMLResponse(content=index.read_text(encoding="utf-8"))


@app.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...), _: None = Depends(_check_api_key)) -> dict:
    """Upload and ingest a document into the vector store.

    Accepts a .pdf or .txt file via multipart/form-data.

    Returns:
        ``{"status": "ok", "chunks": N}`` on success.
    """
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    dest = docs_dir / (file.filename or "upload.txt")
    content = await file.read()
    dest.write_bytes(content)

    try:
        n_chunks = await asyncio.to_thread(ingest_file, dest)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"status": "ok", "chunks": n_chunks}


@app.post("/query")
async def query_endpoint(body: QueryRequest, _: None = Depends(_check_api_key)) -> StreamingResponse:
    """Process a question and stream back agent steps + final answer via SSE.

    Each Server-Sent Event is a JSON object on a ``data:`` line.

    Event types:
    - ``step``  — one agent reasoning step
    - ``final`` — the synthesised answer with sources and confidence
    - ``error`` — something went wrong
    """
    return StreamingResponse(
        _stream_query(body.question, body.user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health() -> dict:
    """Check that Ollama and Qdrant are reachable.

    Returns:
        ``{"status": "ok", "ollama": bool, "qdrant": bool}``
    """
    async def _check(url: str) -> bool:
        try:
            r = await asyncio.to_thread(_requests.get, url, timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    ollama_ok, qdrant_ok = await asyncio.gather(
        _check(f"{OLLAMA_BASE_URL}/api/tags"),
        _check(f"{QDRANT_URL}/healthz"),
    )
    return {"status": "ok", "ollama": ollama_ok, "qdrant": qdrant_ok}


# ── SSE helper ────────────────────────────────────────────────────────────────

def _sse(event: dict) -> str:
    """Encode *event* as an SSE ``data:`` line."""
    return f"data: {json.dumps(event)}\n\n"


def _stream_query(question: str, user_id: str):
    """Generator that runs the orchestrator and yields SSE events.

    Args:
        question: The user's question.
        user_id:  Opaque client UUID used to scope long-term memory.

    Yields:
        SSE-formatted strings.
    """
    memory = ShortTermMemory()
    orchestrator = Orchestrator(memory=memory)

    try:
        for item in orchestrator.run(question=question, short_term=memory, user_id=user_id):
            if isinstance(item, AgentStep):
                yield _sse(
                    {
                        "type": "step",
                        "agent": item.agent,
                        "action": item.action,
                        "input": item.input,
                        "output": item.output,
                        "tool_used": item.tool_used,
                        "thinking": item.thinking,
                    }
                )
            elif isinstance(item, FinalResponse):
                yield _sse(
                    {
                        "type": "final",
                        "answer": item.answer,
                        "sources": item.sources,
                        "web_sources": item.web_sources,
                        "confidence": item.confidence,
                        "from_memory": item.from_memory,
                    }
                )
    except Exception as exc:
        tb = traceback.format_exc()
        yield _sse({"type": "error", "message": str(exc), "traceback": tb})

    yield "data: [DONE]\n\n"
