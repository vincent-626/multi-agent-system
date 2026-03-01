"""
Research orchestrator — decomposes questions, retrieves evidence iteratively,
identifies gaps, and synthesises a final answer.
"""

import asyncio
from collections.abc import AsyncGenerator

import src.ollama_client as ollama
import src.qdrant_client as qdrant
from src.agents.base import BaseAgent
from src.config import COLLECTION_NAME, FAST_MODEL, LLM_MODEL, MAX_RESEARCH_ITERATIONS, RAG_SCORE_THRESHOLD, TOP_K
from src.memory.chat_history import save_message
from src.memory.long_term import extract_and_save, format_for_prompt, get_facts
from src.memory.short_term import ShortTermMemory
from src.schemas import AgentStep, FinalResponse, GapAnalysis, ResearchPlan, WebSearchResult
from src.sparse import compute_sparse
from src.tools.calculator import calculate
from src.tools.unit_converter import convert as unit_convert
from src.tools.web_search import web_search

# ── Prompts ───────────────────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM = """\
You are a research planning agent. Your first job is to classify the input, then plan accordingly.

Classification rules:
- If the input is a greeting, chit-chat, a thank-you, or a meta question about the assistant
  (e.g. "hi", "hello", "thanks", "what can you do?"), set is_conversational=true and leave
  sub_questions and tool_call empty.
- If the input requires any calculation — arithmetic, math functions (sqrt, log, sin…), or physics
  constants (m_e, hbar, c, alpha…) — set tool_call to invoke the calculator.
- If the input asks to convert a value between units (e.g. "convert 500 MeV to GeV",
  "how many fb is 1 pb?", "what is 7 TeV in J?"), set tool_call to invoke the unit_converter.
  Supported units: eV/keV/MeV/GeV/TeV/PeV, J, erg, b/mb/μb/nb/pb/fb/ab,
  m/cm/mm/μm/nm/pm/fm, Å, ly, pc, eV/c²/MeV/c²/GeV/c², u/amu, kg, g,
  MeV/c/GeV/c, s/ms/μs/ns/ps/fs, K.
- Otherwise, break the question into 2–4 specific sub-questions that together fully answer it.
  If the question is already atomic, return it as-is in a list of one.

Respond ONLY with valid JSON (no markdown, no extra text):
{
  "is_conversational": false,
  "sub_questions": ["...", "..."],
  "tool_call": null
}

When a tool is needed, set tool_call like these examples:
  calculator:     {"tool": "calculator", "args": {"expression": "sqrt(m_p**2 + 500**2)"}, "reasoning": "..."}
  unit_converter: {"tool": "unit_converter", "args": {"value": 500, "from": "MeV", "to": "GeV"}, "reasoning": "..."}
"""

_GAP_SYSTEM = """\
You are a research quality agent. Given a question and all evidence gathered so far,
decide whether the evidence is sufficient to write a complete answer.

If gaps exist, output specific follow-up questions — either for document retrieval or for a live web search.

Rules:
- Be honest: if the evidence is sufficient, say so.
- Keep follow-up questions tightly scoped — do not repeat or rephrase queries already attempted.
- Prefer follow_up_questions for document-based gaps.
- Use web_search_queries only for time-sensitive or factual gaps not covered by documents.
- Respond ONLY with valid JSON (no markdown, no extra text):
{
  "is_sufficient": true | false,
  "reasoning": "why the evidence is or is not sufficient",
  "follow_up_questions": [],
  "web_search_queries": []
}"""

_SYNTH_SYSTEM = """\
You are a synthesis agent. Given a research question and all gathered evidence,
write a clear, well-structured final answer for the user.
- Cite source files where relevant.
- If evidence is conflicting, acknowledge it.
- Be concise but complete. Do not pad with filler."""


class Orchestrator(BaseAgent):
    """Research orchestrator that decomposes questions and iteratively gathers evidence.

    Args:
        memory: Shared short-term memory for the current session.
    """

    def __init__(self, memory: ShortTermMemory) -> None:
        super().__init__(
            name="Orchestrator",
            system_prompt=_DECOMPOSE_SYSTEM,
            memory=memory,
        )

    async def run(
        self,
        question: str,
        short_term: ShortTermMemory,
        user_id: str = "anonymous",
    ) -> AsyncGenerator[AgentStep | FinalResponse, None]:
        """Run the research loop, yielding steps and a final response.

        Args:
            question:   The user's question.
            short_term: The session's short-term memory.
            user_id:    Opaque identifier used to scope long-term memory lookups.

        Yields:
            :class:`~src.schemas.AgentStep` for each reasoning step, then a
            :class:`~src.schemas.FinalResponse` as the final item.
        """
        print(f"\n{'='*60}")
        print(f"[Orchestrator] Question: {question}")
        print(f"{'='*60}")

        # ── 1. Long-term memory ───────────────────────────────────────────────
        facts = get_facts(user_id)
        memory_context = format_for_prompt(facts)

        # ── 2. Decompose question into sub-questions ───────────────────────────
        plan = await self._decompose(question, memory_context)
        yield self._log_step(
            action="decompose",
            input_text=question,
            output_text=f"{len(plan.sub_questions)} sub-questions: {plan.sub_questions}",
        )

        # ── 3. Conversational fast-path ───────────────────────────────────────
        if plan.is_conversational or (not plan.sub_questions and not plan.requires_calculator):
            raw = await asyncio.to_thread(
                ollama.chat,
                question,
                system="You are a helpful assistant. Reply conversationally and concisely.",
                think=False,
            )
            answer = ollama.strip_thinking(raw)
            yield self._log_step(action="respond", input_text=question, output_text=answer)
            response = FinalResponse(
                answer=answer, steps=short_term.get_history(),
                sources=[], web_sources=[], confidence="high", from_memory=False,
            )
            yield response
            await asyncio.to_thread(save_message, user_id, question, response)
            return

        # ── 4. Direct tool fast-path ──────────────────────────────────────────
        if plan.tool_call:
            tc = plan.tool_call
            if tc.tool == "calculator":
                expression = tc.args.get("expression", "")
                result = calculate(expression)
                yield self._log_step(
                    action="use_calculator",
                    input_text=expression,
                    output_text=result,
                    tool_used="calculator",
                )
                answer, thinking = await self._synthesise(question, memory_context, [
                    {"question": expression, "context": result, "sources": [], "web_sources": []}
                ])
                yield self._log_step(action="synthesise", input_text=question, output_text=answer, thinking=thinking)
                response = FinalResponse(
                    answer=answer, steps=short_term.get_history(),
                    sources=[], web_sources=[], confidence="high", from_memory=False,
                )
                yield response
                await asyncio.to_thread(save_message, user_id, question, response)
                await asyncio.to_thread(extract_and_save, user_id, question, response)
                return
            elif tc.tool == "unit_converter":
                args = tc.args
                result = unit_convert(args.get("value", 0), args.get("from", ""), args.get("to", ""))
                yield self._log_step(
                    action="unit_conversion",
                    input_text=f"{args.get('value')} {args.get('from')} → {args.get('to')}",
                    output_text=result,
                    tool_used="unit_converter",
                )
                response = FinalResponse(
                    answer=result, steps=short_term.get_history(),
                    sources=[], web_sources=[], confidence="high", from_memory=False,
                )
                yield response
                await asyncio.to_thread(save_message, user_id, question, response)
                return

        # ── 5. Research loop ──────────────────────────────────────────────────
        evidence: list[dict] = []
        all_sources: list[str] = []
        all_web_sources: list[str] = []
        all_context_texts: list[str] = []

        # Track seen chunks by (source_file, chunk_index) to avoid redundancy
        seen_chunks: set[tuple[str, int]] = set()

        pending_doc_queries = list(plan.sub_questions)
        pending_web_queries: list[str] = []

        for iteration in range(MAX_RESEARCH_ITERATIONS + 1):
            prev_seen_count = len(seen_chunks)

            # ── 5a. Retrieve for each pending doc query ───────────────────────
            for sq in pending_doc_queries:
                hits = await self._retrieve(sq)

                # Deduplicate: keep only chunks not seen in previous retrievals
                new_hits = [
                    h for h in hits
                    if (h["source_file"], h["chunk_index"]) not in seen_chunks
                ]
                for h in new_hits:
                    seen_chunks.add((h["source_file"], h["chunk_index"]))

                ctx = self._format_hits(new_hits)
                srcs = list(dict.fromkeys(h["source_file"] for h in new_hits))
                all_sources.extend(srcs)
                all_context_texts.extend(h["text"] for h in new_hits)
                evidence.append({"question": sq, "context": ctx, "sources": srcs, "web_sources": []})

                skipped = len(hits) - len(new_hits)
                if not hits:
                    status = "No relevant chunks found"
                elif skipped:
                    status = f"{len(new_hits)} new chunks, {skipped} duplicate(s) skipped"
                else:
                    status = f"{len(new_hits)} new chunks"

                yield self._log_step(
                    action="retrieve",
                    input_text=sq,
                    output_text=status,
                    tool_used="rag_search",
                )

            # ── 5b. Web search for each pending web query ─────────────────────
            for wq in pending_web_queries:
                summary, urls = await self._web_search(wq)
                all_web_sources.extend(urls)
                evidence.append({"question": wq, "context": summary, "sources": [], "web_sources": urls})
                yield self._log_step(
                    action="web_search",
                    input_text=wq,
                    output_text=summary or "No results",
                    tool_used="web_search",
                )

            # ── 5c. Early stopping: retrieval exhausted ───────────────────────
            if iteration >= MAX_RESEARCH_ITERATIONS:
                break

            retrieval_exhausted = bool(pending_doc_queries) and len(seen_chunks) == prev_seen_count
            if retrieval_exhausted and not pending_web_queries and iteration > 0:
                yield self._log_step(
                    action="gap_analysis",
                    input_text=question,
                    output_text="Retrieval exhausted — no new chunks found, proceeding to synthesis",
                )
                break

            # ── 5d. Gap analysis with coverage context ────────────────────────
            attempted_queries = [e["question"] for e in evidence]
            try:
                gap = await self._identify_gaps(question, evidence, attempted_queries, len(seen_chunks))
            except ValueError:
                yield self._log_step(
                    action="gap_analysis",
                    input_text=question,
                    output_text="Parse failed — proceeding to synthesis",
                )
                break
            yield self._log_step(
                action="gap_analysis",
                input_text=question,
                output_text=f"sufficient={gap.is_sufficient} | {gap.reasoning}",
            )

            if gap.is_sufficient:
                break

            pending_doc_queries = gap.follow_up_questions
            pending_web_queries = gap.web_search_queries

            if not pending_doc_queries and not pending_web_queries:
                break

        # ── 6. Synthesise ─────────────────────────────────────────────────────
        answer, thinking = await self._synthesise(question, memory_context, evidence)
        yield self._log_step(
            action="synthesise",
            input_text=question,
            output_text=answer,
            thinking=thinking,
        )

        # ── 7. Yield final response and persist facts ─────────────────────────
        response = FinalResponse(
            answer=answer,
            steps=short_term.get_history(),
            sources=list(dict.fromkeys(all_sources)),
            web_sources=list(dict.fromkeys(all_web_sources)),
            confidence="high" if any(e["sources"] or e["web_sources"] for e in evidence) else "medium",
            from_memory=False,
            contexts=all_context_texts,
        )
        yield response
        await asyncio.to_thread(save_message, user_id, question, response)
        await asyncio.to_thread(extract_and_save, user_id, question, response)

    # ── helpers ───────────────────────────────────────────────────────────────

    async def _decompose(self, question: str, memory_context: str) -> ResearchPlan:
        """Break *question* into sub-questions via an LLM call."""
        prompt = (
            f"{memory_context}\n\n" if memory_context else ""
        ) + f"Question: {question}\n\nDecompose this into sub-questions. Respond with JSON only."
        raw = await asyncio.to_thread(ollama.chat, prompt, system=_DECOMPOSE_SYSTEM, think=False, model=LLM_MODEL)
        try:
            return ollama.parse_json_response(raw, ResearchPlan)
        except ValueError:
            return ResearchPlan(sub_questions=[question])

    async def _retrieve(self, query: str) -> list[dict]:
        """Embed *query* and search Qdrant using hybrid dense + sparse RRF."""
        try:
            query_vector = await asyncio.to_thread(ollama.embed, query)
            sparse_indices, sparse_values = await asyncio.to_thread(compute_sparse, query)
            return await asyncio.to_thread(
                qdrant.search,
                COLLECTION_NAME, query_vector,
                sparse_indices=sparse_indices,
                sparse_values=sparse_values,
                top_k=TOP_K,
                score_threshold=RAG_SCORE_THRESHOLD,
            )
        except Exception:
            return []

    def _format_hits(self, hits: list[dict]) -> str:
        """Format a list of Qdrant hits into a readable context string."""
        parts = [
            f"[{i}] (source: {h['source_file']}, score: {h['score']:.3f})\n{h['text']}"
            for i, h in enumerate(hits, start=1)
        ]
        return "\n\n".join(parts)

    async def _web_search(self, query: str) -> tuple[str, list[str]]:
        """Run a web search and summarise results. Returns (summary, urls)."""
        result = await asyncio.to_thread(web_search, query)
        if not result.results:
            return "No results found.", []

        snippets = "\n".join(f"- {r['title']}: {r['snippet']}" for r in result.results)
        summary_raw = await asyncio.to_thread(
            ollama.chat,
            f"Summarise these search results for: {query}\n\n{snippets}\n\nGive a concise, factual summary.",
            system="You are a helpful assistant that summarises web search results.",
            think=False,
            model=FAST_MODEL,
        )
        urls = [r["url"] for r in result.results if r.get("url")]
        return ollama.strip_thinking(summary_raw), urls

    async def _identify_gaps(
        self,
        question: str,
        evidence: list[dict],
        attempted_queries: list[str],
        unique_chunk_count: int,
    ) -> GapAnalysis:
        """Ask the LLM whether the gathered evidence is sufficient.

        Passes a coverage summary so the model avoids generating follow-up
        questions that overlap with what has already been retrieved.
        """
        evidence_text = "\n\n---\n\n".join(
            f"Sub-question: {e['question']}\nEvidence:\n{e['context'] or 'No relevant information found.'}"
            for e in evidence
        )
        coverage = (
            f"Retrieval summary: {unique_chunk_count} unique document chunks retrieved so far.\n"
            f"Queries already attempted: {attempted_queries}\n"
            "Do not generate follow-up questions that overlap with or rephrase the above queries."
        )
        prompt = (
            f"Original question: {question}\n\n"
            f"{coverage}\n\n"
            f"Evidence gathered so far:\n{evidence_text}\n\n"
            "Is this sufficient to fully answer the original question? "
            "If not, what specific gaps remain? Respond with JSON only."
        )
        raw = await asyncio.to_thread(ollama.chat, prompt, system=_GAP_SYSTEM, think=False, model=FAST_MODEL)
        return ollama.parse_json_response(raw, GapAnalysis)

    async def _synthesise(
        self, question: str, memory_context: str, evidence: list[dict]
    ) -> tuple[str, str]:
        """Synthesise a final answer from all gathered evidence."""
        evidence_text = "\n\n---\n\n".join(
            f"Sub-question: {e['question']}\nEvidence:\n{e['context'] or 'No relevant information found.'}"
            for e in evidence
        )
        prompt = (
            (f"{memory_context}\n\n" if memory_context else "")
            + f"Research question: {question}\n\n"
            f"All gathered evidence:\n{evidence_text}\n\n"
            "Write a clear, complete, well-structured answer based on the evidence above."
        )
        raw = await asyncio.to_thread(ollama.chat, prompt, system=_SYNTH_SYSTEM, think=False, timeout=600)
        return ollama.strip_thinking(raw), ollama.extract_thinking(raw)
