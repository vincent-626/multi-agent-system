"""RAG Specialist — retrieves and summarises information from ingested documents."""

import src.ollama_client as ollama
import src.qdrant_client as qdrant
from src.agents.base import BaseAgent
from src.config import COLLECTION_NAME, TOP_K
from src.memory.short_term import ShortTermMemory
from src.schemas import RAGResult

_SYSTEM_PROMPT = """\
You are a specialist agent that answers questions strictly from the provided document context.
Rules:
- Only use information present in the context below.
- Always cite which source file(s) your answer comes from.
- Rate your confidence as "high", "medium", or "low" honestly.
- If the context does not contain enough information, say "I don't know" and set confidence to "low".
- Never fabricate facts.
Respond ONLY with valid JSON matching this schema (no markdown fences, no extra text):
{
  "answer": "...",
  "source_chunks": ["chunk text 1", "chunk text 2"],
  "source_files": ["file1.pdf"],
  "confidence": "high" | "medium" | "low",
  "reasoning": "why you are confident or not"
}"""


class RAGSpecialist(BaseAgent):
    """Answers questions by searching the Qdrant vector store.

    Args:
        memory: Shared short-term memory for the current session.
    """

    def __init__(self, memory: ShortTermMemory) -> None:
        super().__init__(
            name="RAG Specialist",
            system_prompt=_SYSTEM_PROMPT,
            memory=memory,
        )

    def run(self, task: str) -> RAGResult:
        """Retrieve relevant chunks and generate a grounded answer.

        Args:
            task: The question or subtask to answer from documents.

        Returns:
            A :class:`~src.schemas.RAGResult` with answer, sources, and
            confidence.
        """
        # 1. Embed the task
        query_vector = ollama.embed(task)

        # 2. Search Qdrant
        hits = qdrant.search(
            collection=COLLECTION_NAME,
            query_vector=query_vector,
            top_k=TOP_K,
        )

        # 3. Handle empty results
        if not hits:
            result = RAGResult(
                answer="No relevant documents found in the knowledge base.",
                source_chunks=[],
                source_files=[],
                confidence="low",
                reasoning="Qdrant returned zero results for this query.",
            )
            self._log_step(
                action="rag_search",
                input_text=task,
                output_text=result.answer,
                tool_used="rag_search",
            )
            return result

        # 4. Build context string with source attribution
        context_parts: list[str] = []
        for i, hit in enumerate(hits, start=1):
            context_parts.append(
                f"[{i}] (source: {hit['source_file']}, score: {hit['score']:.3f})\n{hit['text']}"
            )
        context = "\n\n".join(context_parts)

        # 5. Call LLM
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {task}\n\n"
            "Answer strictly from the context above. Respond with JSON only."
        )
        raw = self._call_llm(prompt)

        # 6. Parse and validate
        try:
            result = ollama.parse_json_response(raw, RAGResult)
        except ValueError as exc:
            result = RAGResult(
                answer=f"Failed to parse RAG response: {exc}",
                source_chunks=[],
                source_files=[],
                confidence="low",
                reasoning="JSON parse error",
            )

        # 7. Log and return
        self._log_step(
            action="rag_search",
            input_text=task,
            output_text=result.answer[:300],
            tool_used="rag_search",
        )
        return result
