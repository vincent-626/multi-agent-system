"""
Research orchestrator — decomposes questions, dispatches parallel ResearchWorkers,
identifies gaps, and synthesises a final answer.
"""

import asyncio
from collections.abc import AsyncGenerator
from datetime import date

import src.clients.ollama_client as ollama
from src.agents.base import BaseAgent
from src.agents.research_worker import ResearchWorker
from src.agents.synthesis_agent import SynthesisAgent
from src.config import FAST_MODEL, LLM_MODEL, MAX_RESEARCH_ITERATIONS
from src.memory.chat_history import save_message
from src.memory.long_term import extract_and_save, format_for_prompt, get_facts
from src.memory.short_term import ShortTermMemory
from src.schemas import AgentStep, EvidenceBundle, FinalResponse, GapAnalysis, ResearchPlan
from src.tools.calculator import calculate
from src.tools.unit_converter import convert as unit_convert

# ── Prompts ───────────────────────────────────────────────────────────────────

def _decompose_system() -> str:
    return (
        f"Today's date is {date.today().isoformat()}.\n"
        "You are a research planning agent. Your first job is to classify the input, then plan accordingly.\n"
        "\n"
        "Classification rules:\n"
        "- If the input is a greeting, chit-chat, a thank-you, or a meta question about the assistant\n"
        "  (e.g. \"hi\", \"hello\", \"thanks\", \"what can you do?\"), set is_conversational=true and leave\n"
        "  sub_questions and tool_call empty.\n"
        "- If the input requires any calculation — arithmetic, math functions (sqrt, log, sin…), or physics\n"
        "  constants (m_e, hbar, c, alpha…) — set tool_call to invoke the calculator.\n"
        "- If the input asks to convert a value between units (e.g. \"convert 500 MeV to GeV\",\n"
        "  \"how many fb is 1 pb?\", \"what is 7 TeV in J?\"), set tool_call to invoke the unit_converter.\n"
        "  Supported units: eV/keV/MeV/GeV/TeV/PeV, J, erg, b/mb/μb/nb/pb/fb/ab,\n"
        "  m/cm/mm/μm/nm/pm/fm, Å, ly, pc, eV/c²/MeV/c²/GeV/c², u/amu, kg, g,\n"
        "  MeV/c/GeV/c, s/ms/μs/ns/ps/fs, K.\n"
        "- Otherwise, break the question into 2–4 specific sub-questions that together fully answer it.\n"
        "  If the question is already atomic, return it as-is in a list of one.\n"
        "\n"
        "Respond ONLY with valid JSON (no markdown, no extra text):\n"
        "{\n"
        "  \"is_conversational\": false,\n"
        "  \"sub_questions\": [\"...\", \"...\"],\n"
        "  \"tool_call\": null\n"
        "}\n"
        "\n"
        "When a tool is needed, set tool_call like these examples:\n"
        "  calculator:     {\"tool\": \"calculator\", \"args\": {\"expression\": \"sqrt(m_p**2 + 500**2)\"}, \"reasoning\": \"...\"}\n"
        "  unit_converter: {\"tool\": \"unit_converter\", \"args\": {\"value\": 500, \"from\": \"MeV\", \"to\": \"GeV\"}, \"reasoning\": \"...\"}"
    )

_GAP_SYSTEM = """\
You are a research quality agent. Given a question and all evidence gathered so far,
decide whether the evidence is sufficient to write a complete answer.

If gaps exist, output specific follow-up questions for workers to investigate.
Workers will decide on their own which sources (documents, web, arXiv) to use.

Rules:
- Be honest: if the evidence is sufficient, say so.
- Keep follow-up questions tightly scoped — do not repeat or rephrase questions already attempted.
- Respond ONLY with valid JSON (no markdown, no extra text):
{
  "isSufficient": true | false,
  "reasoning": "why the evidence is or is not sufficient",
  "gaps": []
}"""


class Orchestrator(BaseAgent):
    """Slim coordinator that decomposes questions and dispatches parallel ResearchWorkers.

    Uses :class:`~src.agents.research_worker.ResearchWorker` agents (one per
    sub-question, all in parallel) and
    :class:`~src.agents.synthesis_agent.SynthesisAgent` for the final answer.

    Args:
        memory: Shared short-term memory for the current session.
    """

    def __init__(self, memory: ShortTermMemory) -> None:
        super().__init__(
            name="Orchestrator",
            system_prompt=_decompose_system(),
            memory=memory,
        )
        self._synth = SynthesisAgent(memory=memory)

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
        if plan.is_conversational or (not plan.sub_questions and not plan.tool_call):
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
                answer, thinking = await self._synth.run(
                    question,
                    [EvidenceBundle(question=expression, context=result)],
                    memory_context,
                )
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
        evidence: list[EvidenceBundle] = []
        all_sources: list[str] = []
        all_web_sources: list[str] = []
        all_context_texts: list[str] = []

        seen_chunks: set[tuple[str, int]] = set()
        pending_questions = list(plan.sub_questions)

        for iteration in range(MAX_RESEARCH_ITERATIONS + 1):
            prev_seen_count = len(seen_chunks)

            # ── 5a. Dispatch one ResearchWorker per question, all in parallel ──
            results = list(await asyncio.gather(
                *(ResearchWorker(memory=self.memory).run(q, seen_chunks)
                  for q in pending_questions)
            ))

            # ── 5b. Collect evidence and yield per-worker ReAct steps ─────────
            for bundle, worker_steps in results:
                # Yield each ReAct step so the trace shows tool-by-tool reasoning
                for step in worker_steps:
                    yield step
                evidence.append(bundle)
                all_sources.extend(bundle.sources)
                all_web_sources.extend(bundle.web_sources)
                all_context_texts.extend(bundle.raw_texts)

            # ── 5c. Early stopping: iteration cap ─────────────────────────────
            if iteration >= MAX_RESEARCH_ITERATIONS:
                break

            # ── 5d. Early stopping: retrieval exhausted ───────────────────────
            if len(seen_chunks) == prev_seen_count and iteration > 0:
                yield self._log_step(
                    action="gap_analysis",
                    input_text=question,
                    output_text="Retrieval exhausted — no new evidence found, proceeding to synthesis",
                )
                break

            # ── 5e. Gap analysis ──────────────────────────────────────────────
            attempted = [e.question for e in evidence]
            try:
                gap = await self._identify_gaps(question, evidence, attempted, len(seen_chunks))
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

            if gap.is_sufficient or not gap.follow_up_questions:
                break

            pending_questions = gap.follow_up_questions

        # ── 6. Synthesise ─────────────────────────────────────────────────────
        answer, thinking = await self._synth.run(question, evidence, memory_context)
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
            confidence="high" if any(e.sources or e.web_sources for e in evidence) else "medium",
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
        raw = await asyncio.to_thread(ollama.chat, prompt, system=_decompose_system(), think=False, model=LLM_MODEL)
        try:
            return ollama.parse_json_response(raw, ResearchPlan)
        except ValueError:
            return ResearchPlan(sub_questions=[question])

    async def _identify_gaps(
        self,
        question: str,
        evidence: list[EvidenceBundle],
        attempted_queries: list[str],
        unique_chunk_count: int,
    ) -> GapAnalysis:
        """Ask the LLM whether the gathered evidence is sufficient."""
        evidence_text = "\n\n---\n\n".join(
            f"Sub-question: {e.question}\nEvidence:\n{e.context or 'No relevant information found.'}"
            for e in evidence
        )
        coverage = (
            f"Retrieval summary: {unique_chunk_count} unique document chunks retrieved so far.\n"
            f"Questions already attempted: {attempted_queries}\n"
            "Do not generate follow-up questions that overlap with or rephrase the above questions."
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
