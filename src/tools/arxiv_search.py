"""arXiv paper search tool — no API key required.

Uses the official ``arxiv`` Python client to search the arXiv preprint server.
Particularly useful for particle physics (hep-ph, hep-ex, hep-th) queries.

Common HEP category codes
--------------------------
hep-ph  — High Energy Physics - Phenomenology
hep-ex  — High Energy Physics - Experiment
hep-th  — High Energy Physics - Theory
hep-lat — Lattice High Energy Physics
nucl-ex — Nuclear Experiment
nucl-th — Nuclear Theory
gr-qc   — General Relativity and Quantum Cosmology
astro-ph.HE — High Energy Astrophysical Phenomena
"""

import arxiv

from src.config import ARXIV_MAX_RESULTS


def arxiv_search(
    query: str,
    max_results: int = ARXIV_MAX_RESULTS,
    categories: list[str] | None = None,
    since_year: int | None = None,
) -> list[dict]:
    """Search arXiv and return structured paper metadata.

    Args:
        query:       Full-text search query (titles, abstracts, authors).
        max_results: Maximum number of papers to return.
        categories:  Optional list of arXiv category codes to restrict the
                     search (e.g. ``["hep-ph", "hep-ex"]``).  When provided,
                     results are limited to papers in at least one of the
                     listed categories.
        since_year:  Optional earliest publication year (inclusive). When set,
                     only papers submitted from January 1st of that year
                     onwards are returned.

    Returns:
        List of dicts, each containing:
        ``title``, ``authors`` (list[str]), ``abstract``, ``url``,
        ``pdf_url``, ``published`` (YYYY-MM-DD), ``categories`` (list[str]).
        Returns an empty list on failure.
    """
    full_query = query
    if categories:
        cat_filter = " OR ".join(f"cat:{c}" for c in categories)
        full_query = f"({full_query}) AND ({cat_filter})"
    if since_year:
        full_query = f"({full_query}) AND submittedDate:[{since_year}0101 TO *]"

    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=full_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for r in client.results(search):
            results.append({
                "title": r.title,
                "authors": [a.name for a in r.authors],
                "abstract": r.summary,
                "url": r.entry_id,
                "pdf_url": r.pdf_url,
                "published": r.published.strftime("%Y-%m-%d") if r.published else "unknown",
                "categories": r.categories,
            })
        return results
    except Exception:  # noqa: BLE001
        return []
