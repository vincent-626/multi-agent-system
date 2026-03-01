"""RAG evaluation runner using RAGAS with GPT-4o-mini as the judge.

Metrics (all reference-free — no ground-truth answers required):
  - Faithfulness:                  is the answer grounded in the retrieved chunks?
  - ResponseRelevancy:             does the answer actually address the question?
  - LLMContextPrecisionWithoutReference: are the retrieved chunks relevant to the question?

Usage:
    # Install eval dependencies first:
    uv sync --extra eval

    # Set your OpenAI API key:
    export OPENAI_API_KEY=sk-...

    # Run against a local stack (Ollama + Qdrant must be running):
    uv run python -m eval.run_eval

Environment variables:
    OPENAI_API_KEY   required — used as the RAGAS judge
    OLLAMA_BASE_URL  default: http://localhost:11434
    LLM_MODEL        default: qwen3  (orchestrator model, not the judge)
    EMBED_MODEL      default: nomic-embed-text
"""

import asyncio
import json
import os
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

from src.agents.orchestrator import Orchestrator
from src.memory.short_term import ShortTermMemory
from src.schemas import FinalResponse

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3")
FAST_MODEL = os.getenv("FAST_MODEL", "qwen3:1.7b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.json"
SCORES_PATH = Path(__file__).parent / "scores.json"


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
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Export it before running eval.")
    llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL))
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    )
    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings),
        LLMContextPrecisionWithoutReference(llm=llm),
    ]
    return llm, embeddings, metrics


MAX_CONTEXTS_FOR_EVAL = 10  # cap to avoid overly long precision prompts


def _build_dataset(
    results: list[tuple[str, str, list[str]]],
) -> EvaluationDataset:
    samples: list = [
        SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts[:MAX_CONTEXTS_FOR_EVAL] if contexts else ["No relevant context retrieved."],
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


def _save_scores(
    results: list[tuple[str, str, list[str]]],
    categories: dict[str, str],
    scores_df,
) -> None:
    from datetime import datetime
    metric_cols = [c for c in scores_df.columns if c not in ("user_input", "response", "retrieved_contexts", "_category")]
    scores_df["_category"] = [categories[q] for q, _, _ in results]

    per_question = []
    for i, (question, answer, contexts) in enumerate(results):
        row = scores_df.iloc[i]
        per_question.append({
            "question": question,
            "category": categories[question],
            "answer": answer,
            "retrieved_chunks": len(contexts),
            "scores": {col: round(float(row[col]), 4) if row.get(col) is not None else None for col in metric_cols},
        })

    overall = {col: round(float(scores_df[col].mean()), 4) for col in metric_cols if col in scores_df.columns}

    by_category = {}
    for cat in scores_df["_category"].unique():
        subset = scores_df[scores_df["_category"] == cat]
        by_category[cat] = {col: round(float(subset[col].mean()), 4) for col in metric_cols if col in subset.columns}

    output = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "judge_model": JUDGE_MODEL,
        "overall": overall,
        "by_category": by_category,
        "per_question": per_question,
    }
    SCORES_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nScores saved to {SCORES_PATH.name}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-only",
        action="store_true",
        help=f"Skip the orchestrator and score from existing {RESULTS_PATH.name}.",
    )
    args = parser.parse_args()

    golden = json.loads(GOLDEN_PATH.read_text())
    categories = {entry["question"]: entry["category"] for entry in golden}

    print(f"Judge model : {JUDGE_MODEL}")
    print(f"Embed model : {EMBED_MODEL}")
    print(f"Ollama URL  : {OLLAMA_BASE_URL}\n")

    if args.score_only:
        if not RESULTS_PATH.exists():
            raise FileNotFoundError(f"{RESULTS_PATH} not found — run without --score-only first.")
        raw = json.loads(RESULTS_PATH.read_text())
        results = [(r["question"], r["answer"], r["contexts"]) for r in raw]
        print(f"Loaded {len(results)} results from {RESULTS_PATH.name}")
    else:
        questions = [entry["question"] for entry in golden]
        print(f"Loaded {len(questions)} questions from {GOLDEN_PATH.name}\n")

        # Step 1: collect answers + contexts from the orchestrator
        print("── Step 1/2: running orchestrator ──────────────────────────────────")
        results = asyncio.run(_collect_results(questions))

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
    _save_scores(results, categories, scores_df)


if __name__ == "__main__":
    main()
