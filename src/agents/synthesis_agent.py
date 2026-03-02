"""SynthesisAgent — combines gathered evidence into a final answer."""

import asyncio
from datetime import date

import src.ollama_client as ollama
from src.agents.base import BaseAgent
from src.memory.short_term import ShortTermMemory
from src.schemas import EvidenceBundle


def _synth_system() -> str:
    return (
        f"Today's date is {date.today().isoformat()}.\n"
        "You are a synthesis agent. Given a research question and all gathered evidence,\n"
        "write a clear, well-structured final answer for the user.\n"
        "- Cite source files where relevant.\n"
        "- If evidence is conflicting, acknowledge it.\n"
        "- Be concise but complete. Do not pad with filler.\n"
        "- If the evidence does not contain information relevant to the question, say so clearly and honestly.\n"
        "  Do not speculate, invent details, or draw on knowledge beyond what the evidence provides."
    )


class SynthesisAgent(BaseAgent):
    """Synthesises a final answer from all gathered evidence bundles.

    Args:
        memory: Shared short-term memory for the current session.
    """

    def __init__(self, memory: ShortTermMemory) -> None:
        super().__init__(name="SynthesisAgent", system_prompt=_synth_system(), memory=memory)

    async def run(
        self,
        question: str,
        evidence: list[EvidenceBundle],
        memory_context: str,
    ) -> tuple[str, str]:
        """Synthesise a final answer from *evidence*.

        Args:
            question:       The original user question.
            evidence:       All gathered evidence bundles.
            memory_context: Formatted long-term memory facts to inject.

        Returns:
            Tuple of (answer, thinking) where thinking is the raw ``<think>``
            content (empty string if the model produced none).
        """
        evidence_text = "\n\n---\n\n".join(
            f"Sub-question: {e.question}\nEvidence:\n{e.context or 'No relevant information found.'}"
            for e in evidence
        )
        prompt = (
            (f"{memory_context}\n\n" if memory_context else "")
            + f"Research question: {question}\n\n"
            f"All gathered evidence:\n{evidence_text}\n\n"
            "Write a clear, complete, well-structured answer based on the evidence above."
        )
        raw = await asyncio.to_thread(
            ollama.chat, prompt, system=_synth_system(), think=False, timeout=600
        )
        return ollama.strip_thinking(raw), ollama.extract_thinking(raw)
