"""DuckDuckGo web search tool — no API key required."""

from duckduckgo_search import DDGS

from src.config import WEB_SEARCH_MAX_RESULTS
from src.schemas import WebSearchResult


def web_search(query: str) -> WebSearchResult:
    """Search the web using DuckDuckGo and return structured results.

    Uses the ``duckduckgo-search`` package (DDGS).  No API key is required.
    The caller (orchestrator) is responsible for LLM-summarising the results.

    Args:
        query: The search query string.

    Returns:
        A :class:`WebSearchResult` containing raw results and an empty summary
        (the orchestrator fills in the summary after calling the LLM).  On any
        failure a graceful error result is returned instead of raising.
    """
    try:
        results: list[dict] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=WEB_SEARCH_MAX_RESULTS):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    }
                )

        if not results:
            return WebSearchResult(
                query=query,
                results=[],
                summary="No results found for this query.",
                confidence="low",
            )

        return WebSearchResult(
            query=query,
            results=results,
            summary="",  # orchestrator fills this in
            confidence="medium",
        )

    except Exception as exc:  # noqa: BLE001
        return WebSearchResult(
            query=query,
            results=[],
            summary=f"Web search failed: {exc}",
            confidence="low",
        )
