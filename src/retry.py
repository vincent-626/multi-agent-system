"""Shared retry decorators for external service calls.

Usage:
    from src.retry import http_retry, qdrant_retry

    @http_retry
    def call_ollama(): ...

    @qdrant_retry
    def call_qdrant(): ...
"""

from __future__ import annotations

import logging

import backoff
import backoff.types
import httpx
import requests
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)

# ── Ollama / requests-based HTTP ──────────────────────────────────────────────

_HTTP_RETRYABLE = (requests.ConnectionError, requests.Timeout, requests.HTTPError)


def _http_giveup(exc: Exception) -> bool:
    """Give up immediately on 4xx — these won't fix themselves on retry."""
    return (
        isinstance(exc, requests.HTTPError)
        and exc.response is not None
        and exc.response.status_code < 500
    )


def _http_on_giveup(details: backoff.types.Details) -> None:
    logger.error(
        "HTTP call failed after %d attempt(s): %s", details["tries"], details.get("exception")
    )


http_retry = backoff.on_exception(
    backoff.expo,
    _HTTP_RETRYABLE,
    max_tries=4,
    jitter=backoff.full_jitter,
    giveup=_http_giveup,
    on_giveup=_http_on_giveup,
)

# ── Qdrant / httpx-based ──────────────────────────────────────────────────────

_QDRANT_RETRYABLE = (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError, UnexpectedResponse)


def _qdrant_giveup(exc: Exception) -> bool:
    """Give up on our own assertions or Qdrant 4xx errors."""
    if isinstance(exc, AssertionError):
        return True
    if isinstance(exc, UnexpectedResponse) and exc.status_code is not None and exc.status_code < 500:
        return True
    return False


def _qdrant_on_giveup(details: backoff.types.Details) -> None:
    logger.error(
        "Qdrant call failed after %d attempt(s): %s", details["tries"], details.get("exception")
    )


qdrant_retry = backoff.on_exception(
    backoff.expo,
    _QDRANT_RETRYABLE,
    max_tries=4,
    jitter=backoff.full_jitter,
    giveup=_qdrant_giveup,
    on_giveup=_qdrant_on_giveup,
)
