"""RAG evaluation runner using RAGAS with a local Ollama judge.

Metrics (all reference-free — no ground-truth answers required):
  - Faithfulness:                  is the answer grounded in the retrieved chunks?
  - ResponseRelevancy:             does the answer actually address the question?
  - LLMContextPrecisionWithoutReference: are the retrieved chunks relevant to the question?

Usage:
    # Install eval dependencies first:
    uv sync --extra eval

    # Run against a local stack (Ollama + Qdrant must be running):
    uv run python -m eval.run_eval

Environment variables (all optional, inherit from .env):
    OLLAMA_BASE_URL  default: http://localhost:11434
    LLM_MODEL        default: qwen3  (used as RAGAS judge)
    EMBED_MODEL      default: nomic-embed-text
"""

import asyncio
import json
import os
from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy,
)

from src.agents.orchestrator import Orchestrator
from src.memory.short_term import ShortTermMemory
from src.schemas import FinalResponse

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


# ── Orchestrator runner ───────────────────────────────────────────────────────

async def _run_question(question: str) -> tuple[str, list[str]]:
    """Run the orchestrator on a single question. Returns (answer, contexts)."""
    memory = ShortTermMemory()
    orchestrator = Orchestrator(memory=memory)
    response: FinalResponse | None = None
    async for item in orchestrator.run(question=question, short_term=memory, user_id="eval"):
        if isinstance(item, FinalResponse):
            response = item
    if response is None:
        return "", []
    return response.answer, response.contexts


async def _collect_results(
    questions: list[str],
) -> list[tuple[str, str, list[str]]]:
    """Run all questions sequentially. Returns list of (question, answer, contexts)."""
    results = []
    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question[:70]}...")
        answer, contexts = await _run_question(question)
        print(f"         {len(contexts)} chunk(s) retrieved | answer: {answer[:80].strip()}...")
        results.append((question, answer, contexts))
    return results


# ── RAGAS setup ───────────────────────────────────────────────────────────────

def _build_ragas_components() -> tuple:
    """Return configured (llm, embeddings, metrics) for RAGAS."""
    # Note: to prevent <think> blocks from breaking RAGAS output parsing,
    # set LLM_THINK=false in your environment before running eval.
    llm = LangchainLLMWrapper(
        ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    )
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    )
    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings),
        LLMContextPrecisionWithoutReference(llm=llm),
    ]
    return llm, embeddings, metrics


def _build_dataset(
    results: list[tuple[str, str, list[str]]],
) -> EvaluationDataset:
    samples: list = [
        SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts if contexts else ["No relevant context retrieved."],
        )
        for question, answer, contexts in results
    ]
    return EvaluationDataset(samples=samples)


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_report(
    results: list[tuple[str, str, list[str]]],
    categories: dict[str, str],
    scores_df,
) -> None:
    metric_cols = [c for c in scores_df.columns if c not in ("user_input", "response", "retrieved_contexts")]

    print("\n" + "=" * 70)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 70)

    for i, (question, _, contexts) in enumerate(results):
        row = scores_df.iloc[i]
        cat = categories[question]
        print(f"\n[{cat}] {question}")
        print(f"  retrieved chunks : {len(contexts)}")
        for col in metric_cols:
            val = row.get(col)
            print(f"  {col:<35} {val:.3f}" if val is not None else f"  {col:<35} N/A")

    print("\n" + "─" * 70)
    print("OVERALL AVERAGES")
    for col in metric_cols:
        if col in scores_df.columns:
            print(f"  {col:<35} {scores_df[col].mean():.3f}")

    print("\nPER-CATEGORY AVERAGES")
    scores_df["_category"] = [categories[q] for q, _, _ in results]
    for cat in ["factual", "multi_hop", "out_of_scope", "calculator", "unit_converter", "arxiv_search"]:
        subset = scores_df[scores_df["_category"] == cat]
        if subset.empty:
            continue
        print(f"  {cat}:")
        for col in metric_cols:
            if col in subset.columns:
                print(f"    {col:<33} {subset[col].mean():.3f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    golden = json.loads(GOLDEN_PATH.read_text())
    questions = [entry["question"] for entry in golden]
    categories = {entry["question"]: entry["category"] for entry in golden}

    print(f"Loaded {len(questions)} questions from {GOLDEN_PATH.name}")
    print(f"Judge model : {LLM_MODEL}")
    print(f"Embed model : {EMBED_MODEL}")
    print(f"Ollama URL  : {OLLAMA_BASE_URL}\n")

    # Step 1: collect answers + contexts from the orchestrator
    print("── Step 1/2: running orchestrator ──────────────────────────────────")
    results = asyncio.run(_collect_results(questions))

    # Persist raw results so the slow orchestrator step isn't re-run on reruns
    RESULTS_PATH.write_text(
        json.dumps(
            [{"question": q, "answer": a, "contexts": c} for q, a, c in results],
            indent=2,
        )
    )
    print(f"\nRaw results saved to {RESULTS_PATH.name}")

    # Step 2: score with RAGAS
    print("\n── Step 2/2: scoring with RAGAS ─────────────────────────────────────")
    _, _, metrics = _build_ragas_components()
    dataset = _build_dataset(results)
    scores = evaluate(dataset=dataset, metrics=metrics)
    scores_df = scores.to_pandas()  # type: ignore[union-attr]

    _print_report(results, categories, scores_df)


if __name__ == "__main__":
    main()
