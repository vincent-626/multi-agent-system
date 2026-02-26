"""Thin HTTP wrapper over the Ollama REST API.

Uses plain `requests` — no Ollama SDK dependency.
"""

from __future__ import annotations

import json
import logging
from typing import Generator, Type, TypeVar

import requests
from pydantic import BaseModel

from src.config import EMBED_MODEL, LLM_MODEL, LLM_THINK, OLLAMA_BASE_URL
from src.retry import http_retry

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)

# ── Public API ────────────────────────────────────────────────────────────────

@http_retry
def embed(text: str) -> list[float]:
    """Return the embedding vector for *text* using the configured embed model.

    Retries up to 3 times with exponential backoff on transient errors.

    Raises:
        RuntimeError: if Ollama is unreachable after all retries.
    """
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


@http_retry
def chat(prompt: str, system: str = "", think: bool | None = None, timeout: int = 120) -> str:
    """Non-streaming chat call. Returns the full response string.

    Retries up to 3 times with exponential backoff on transient errors.

    Args:
        prompt: The user message.
        system: Optional system prompt.
        think:  Override the global LLM_THINK setting for this call.
                Pass False for structured JSON calls where thinking adds
                latency without improving output quality.

    Raises:
        RuntimeError: if Ollama is unreachable after all retries.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": LLM_MODEL, "messages": messages, "stream": False, "think": LLM_THINK if think is None else think},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


@http_retry
def _open_stream(messages: list[dict]) -> requests.Response:
    """Open a streaming HTTP connection to Ollama, with retry on transient errors.

    Separated from :func:`chat_stream` because backoff decorators cannot wrap
    generator functions — they need a regular function that either returns or
    raises on each attempt.
    """
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": LLM_MODEL, "messages": messages, "stream": True, "think": LLM_THINK},
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()
    return resp


def chat_stream(prompt: str, system: str = "") -> Generator[str, None, None]:
    """Streaming chat call. Yields response tokens as they arrive.

    The initial HTTP connection is retried with exponential backoff; mid-stream
    failures are not retried (the client will see a truncated response).

    Args:
        prompt: The user message.
        system: Optional system prompt.

    Yields:
        Individual token strings from the model.

    Raises:
        RuntimeError: if Ollama is unreachable after all retries.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = _open_stream(messages)

    for line in resp.iter_lines():
        if line:
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token
            if chunk.get("done"):
                break


# ── JSON parsing ──────────────────────────────────────────────────────────────

def extract_thinking(text: str) -> str:
    """Return the content inside the first <think>…</think> block, or ''."""
    start = text.find("<think>")
    end = text.find("</think>")
    if start == -1 or end == -1:
        return ""
    return text[start + len("<think>"):end].strip()


def strip_thinking(text: str) -> str:
    """Remove the <think>…</think> block from *text*."""
    start = text.find("<think>")
    end = text.find("</think>")
    if start == -1 or end == -1:
        return text
    return (text[:start] + text[end + len("</think>"):]).strip()


def parse_json_list(response: str) -> list[str]:
    """Strip markdown fences and think blocks, then parse a JSON array.

    Args:
        response: Raw LLM output expected to contain a JSON array of strings.

    Returns:
        List of strings, or an empty list on any parse failure.
    """
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    if "<think>" in text:
        start = text.find("</think>")
        if start != -1:
            text = text[start + len("</think>"):].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(item) for item in data if item and isinstance(item, str)]


def parse_json_response(response: str, model: Type[T]) -> T:
    """Strip markdown fences, parse JSON, validate against *model*.

    Args:
        response: Raw LLM output (may be wrapped in ```json … ``` fences).
        model:    The Pydantic model class to validate against.

    Returns:
        A validated instance of *model*.

    Raises:
        ValueError: if the JSON cannot be parsed or fails validation.
    """
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()

    # Strip <think>…</think> blocks emitted by qwen3
    if "<think>" in text:
        start = text.find("</think>")
        if start != -1:
            text = text[start + len("</think>"):].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse JSON from LLM response.\n"
            f"Error: {exc}\n"
            f"Response was:\n{response[:500]}"
        ) from exc

    try:
        return model.model_validate(data)
    except Exception as exc:
        raise ValueError(
            f"LLM response did not match schema {model.__name__}.\n"
            f"Error: {exc}\n"
            f"Parsed data: {data}"
        ) from exc
