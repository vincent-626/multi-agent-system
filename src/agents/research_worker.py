"""ResearchWorker — a homogeneous agent that runs a ReAct loop to answer one sub-question.

Each worker has access to all tools (rag_search, web_search, arxiv_search, calculator,
unit_converter) and decides which to call based on the question at hand.  Source
selection is an emergent decision from the LLM, not hardcoded routing logic.
"""

import asyncio
import logging

import src.ollama_client as ollama
import src.qdrant_client as qdrant
from src.agents.base import BaseAgent
from src.config import (
    COLLECTION_NAME,
    LLM_MODEL,
    MAX_WORKER_STEPS,
    RAG_SCORE_THRESHOLD,
    TOP_K,
)
from src.memory.short_term import ShortTermMemory
from src.schemas import AgentStep, EvidenceBundle, WorkerToolCall
from src.sparse import compute_sparse
from src.tools.arxiv_search import arxiv_search
from src.tools.calculator import calculate
from src.tools.unit_converter import convert as unit_convert
from src.tools.web_search import web_search

logger = logging.getLogger(__name__)

_WORKER_SYSTEM = """\
You are a research worker agent. Your job is to answer ONE research question by choosing
the right tools, calling them in sequence, and stopping when you have enough evidence.

Available tools:

1. rag_search — search the local document knowledge base
   args: {"query": "<search string>"}

2. web_search — search the live web (DuckDuckGo, no API key needed)
   args: {"query": "<search string>"}

3. arxiv_search — search academic papers on arXiv
   args: {"query": "<search string>", "since_year": <int or null>}

4. calculator — evaluate a mathematical expression (supports physics constants)
   args: {"expression": "<expression>"}

5. unit_converter — convert between physical units
   args: {"value": <number>, "from": "<unit>", "to": "<unit>"}

6. done — stop and return the evidence you have gathered
   args: {}

Strategy:
- Start with rag_search; it is fast and uses the local knowledge base.
- Use web_search for current events, news, or facts not in documents.
- Use arxiv_search for academic papers and recent research results.
- Use calculator / unit_converter for numeric questions.
- Call done as soon as you have sufficient evidence. Do NOT keep searching if
  the question is already answered.
- Each tool call should use a DIFFERENT, more targeted query if the previous one
  returned little information.

IMPORTANT: Respond ONLY with valid JSON (no markdown, no extra text):
{
  "tool": "<tool name>",
  "args": { ... },
  "reasoning": "<why you chose this tool and these args>"
}
"""


class ResearchWorker(BaseAgent):
    """Runs a ReAct loop to answer a single sub-question using any available tool.

    Args:
        memory: Shared short-term memory for the current session.
    """

    def __init__(self, memory: ShortTermMemory) -> None:
        super().__init__(name="ResearchWorker", system_prompt=_WORKER_SYSTEM, memory=memory)

    async def run(
        self,
        question: str,
        seen_chunks: set[tuple[str, int]],
    ) -> tuple[EvidenceBundle, list[AgentStep]]:
        """Run the ReAct loop for *question* and return evidence + trace steps.

        Args:
            question:    The sub-question to answer.
            seen_chunks: Mutable set of (source_file, chunk_index) already seen
                         across all workers this iteration (deduplication).

        Returns:
            Tuple of (:class:`~src.schemas.EvidenceBundle`, list of
            :class:`~src.schemas.AgentStep`) so the orchestrator can yield
            the per-step trace to the SSE stream.
        """
        messages: list[dict] = [
            {"role": "system", "content": _WORKER_SYSTEM},
            {"role": "user", "content": f"Research question: {question}"},
        ]

        tool_results: list[str] = []
        sources: list[str] = []
        web_sources: list[str] = []
        raw_texts: list[str] = []
        steps: list[AgentStep] = []

        for step_idx in range(MAX_WORKER_STEPS):
            try:
                raw = await asyncio.to_thread(
                    ollama.chat_messages,
                    messages,
                    False,  # think=False — reliable JSON tool selection
                    300,
                    LLM_MODEL,
                )
            except Exception as exc:
                logger.warning("[ResearchWorker] LLM call failed at step %d: %s", step_idx, exc)
                break

            try:
                tool_call = ollama.parse_json_response(raw, WorkerToolCall)
            except ValueError as exc:
                logger.warning("[ResearchWorker] JSON parse failed at step %d: %s", step_idx, exc)
                break

            steps.append(self._log_step(
                action=tool_call.tool,
                input_text=question,
                output_text=tool_call.reasoning,
            ))

            if tool_call.tool == "done":
                break

            result = await self._execute_tool(
                tool_call, seen_chunks, sources, web_sources, raw_texts
            )
            tool_results.append(result)

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Tool result:\n{result}"})

        context = "\n\n".join(tool_results) if tool_results else ""
        return EvidenceBundle(
            question=question,
            context=context,
            sources=sources,
            web_sources=web_sources,
            raw_texts=raw_texts,
        ), steps

    async def _execute_tool(
        self,
        tool_call: WorkerToolCall,
        seen_chunks: set[tuple[str, int]],
        sources: list[str],
        web_sources: list[str],
        raw_texts: list[str],
    ) -> str:
        """Dispatch *tool_call* to the appropriate tool and return a formatted result string."""
        tool = tool_call.tool
        args = tool_call.args

        if tool == "rag_search":
            query = args.get("query", "")
            hits = await self._do_rag_search(query)
            new_hits = [
                h for h in hits
                if (h["source_file"], h["chunk_index"]) not in seen_chunks
            ]
            for h in new_hits:
                seen_chunks.add((h["source_file"], h["chunk_index"]))
                sources.append(h["source_file"])
                raw_texts.append(h["text"])
            # Deduplicate sources list preserving order
            seen_src: set[str] = set()
            deduped: list[str] = []
            for s in sources:
                if s not in seen_src:
                    seen_src.add(s)
                    deduped.append(s)
            sources[:] = deduped
            return self._format_rag_hits(new_hits)

        if tool == "web_search":
            query = args.get("query", "")
            result = await asyncio.to_thread(web_search, query)
            if not result.results:
                return "No web results found."
            web_sources.extend(r["url"] for r in result.results if r.get("url"))
            lines = [
                f"- {r['title']} ({r['url']}): {r['snippet']}"
                for r in result.results
            ]
            return "\n".join(lines)

        if tool == "arxiv_search":
            query = args.get("query", "")
            since_year = args.get("since_year")
            results = await asyncio.to_thread(arxiv_search, query, since_year=since_year)
            if not results:
                return "No arXiv papers found."
            web_sources.extend(r["url"] for r in results)
            parts = []
            for r in results:
                authors = ", ".join(r["authors"][:3])
                if len(r["authors"]) > 3:
                    authors += " et al."
                abstract = r["abstract"][:400] + "..." if len(r["abstract"]) > 400 else r["abstract"]
                parts.append(
                    f"Title: {r['title']}\n"
                    f"Authors: {authors}\n"
                    f"Abstract: {abstract}\n"
                    f"URL: {r['url']}"
                )
            return "\n\n".join(parts)

        if tool == "calculator":
            expression = args.get("expression", "")
            return calculate(expression)

        if tool == "unit_converter":
            return unit_convert(
                args.get("value", 0),
                args.get("from", ""),
                args.get("to", ""),
            )

        return f"Unknown tool: {tool}"

    async def _do_rag_search(self, query: str) -> list[dict]:
        """Embed *query* and search Qdrant using hybrid dense + sparse RRF."""
        try:
            query_vector = await asyncio.to_thread(ollama.embed, query)
            sparse_indices, sparse_values = await asyncio.to_thread(compute_sparse, query)
            return await asyncio.to_thread(
                qdrant.search,
                COLLECTION_NAME,
                query_vector,
                sparse_indices=sparse_indices,
                sparse_values=sparse_values,
                top_k=TOP_K,
                score_threshold=RAG_SCORE_THRESHOLD,
            )
        except Exception:
            return []

    def _format_rag_hits(self, hits: list[dict]) -> str:
        """Format Qdrant hits into a readable context string."""
        if not hits:
            return "No relevant document chunks found."
        parts = [
            f"[{i}] (source: {h['source_file']}, score: {h['score']:.3f})\n{h['text']}"
            for i, h in enumerate(hits, start=1)
        ]
        return "\n\n".join(parts)
