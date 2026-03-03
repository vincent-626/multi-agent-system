"""Unit tests for Pydantic model validation in src/schemas.py."""

import pytest
from pydantic import ValidationError

from src.schemas import (
    AgentStep,
    EvidenceBundle,
    FinalResponse,
    GapAnalysis,
    QueryRequest,
    ResearchPlan,
    WorkerToolCall,
)


# ── GapAnalysis ───────────────────────────────────────────────────────────────

class TestGapAnalysis:
    def test_all_fields_have_defaults(self):
        gap = GapAnalysis()
        assert gap.is_sufficient is False
        assert gap.reasoning == ""
        assert gap.follow_up_questions == []

    def test_populated_via_aliases(self):
        gap = GapAnalysis.model_validate(
            {"isSufficient": True, "reasoning": "done", "gaps": ["q1", "q2"]}
        )
        assert gap.is_sufficient is True
        assert gap.follow_up_questions == ["q1", "q2"]

    def test_populated_via_field_names(self):
        # populate_by_name=True allows using the Python field names directly
        gap = GapAnalysis(is_sufficient=True, reasoning="ok", follow_up_questions=["q"])
        assert gap.is_sufficient is True
        assert gap.follow_up_questions == ["q"]


# ── WorkerToolCall ────────────────────────────────────────────────────────────

class TestWorkerToolCall:
    @pytest.mark.parametrize("tool", [
        "rag_search", "web_search", "arxiv_search",
        "calculator", "unit_converter", "done",
    ])
    def test_all_valid_tools_accepted(self, tool):
        tc = WorkerToolCall(tool=tool, args={}, reasoning="reason")
        assert tc.tool == tool

    def test_invalid_tool_raises_validation_error(self):
        with pytest.raises(ValidationError):
            WorkerToolCall(tool="invalid_tool", args={}, reasoning="")

    def test_args_can_be_any_dict(self):
        tc = WorkerToolCall(
            tool="rag_search",
            args={"query": "higgs boson", "top_k": 5},
            reasoning="search docs",
        )
        assert tc.args["query"] == "higgs boson"


# ── EvidenceBundle ────────────────────────────────────────────────────────────

class TestEvidenceBundle:
    def test_required_fields_only(self):
        eb = EvidenceBundle(question="What is a quark?", context="A quark is...")
        assert eb.question == "What is a quark?"
        assert eb.context == "A quark is..."
        assert eb.sources == []
        assert eb.web_sources == []
        assert eb.raw_texts == []

    def test_optional_fields_populated(self):
        eb = EvidenceBundle(
            question="q",
            context="ctx",
            sources=["doc.pdf"],
            web_sources=["https://example.com"],
            raw_texts=["chunk text"],
        )
        assert eb.sources == ["doc.pdf"]
        assert eb.web_sources == ["https://example.com"]
        assert eb.raw_texts == ["chunk text"]


# ── AgentStep ─────────────────────────────────────────────────────────────────

class TestAgentStep:
    def test_required_fields(self):
        step = AgentStep(
            agent="ResearchWorker",
            action="rag_search",
            input="What is a quark?",
            output="Found 3 chunks.",
        )
        assert step.tool_used is None
        assert step.thinking == ""

    def test_optional_fields(self):
        step = AgentStep(
            agent="Orchestrator",
            action="decompose",
            input="question",
            output="2 sub-questions",
            tool_used="calculator",
            thinking="<think>...",
        )
        assert step.tool_used == "calculator"
        assert step.thinking == "<think>..."


# ── ResearchPlan ──────────────────────────────────────────────────────────────

class TestResearchPlan:
    def test_defaults(self):
        plan = ResearchPlan()
        assert plan.is_conversational is False
        assert plan.sub_questions == []
        assert plan.tool_call is None

    def test_conversational_flag(self):
        plan = ResearchPlan(is_conversational=True)
        assert plan.is_conversational is True


# ── QueryRequest ──────────────────────────────────────────────────────────────

class TestQueryRequest:
    def test_valid_request(self):
        qr = QueryRequest(question="What is a quark?", user_id="user-123")
        assert qr.user_id == "user-123"

    def test_empty_user_id_raises_validation_error(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="q", user_id="")

    def test_empty_question_is_allowed(self):
        # No constraint on question length
        qr = QueryRequest(question="", user_id="u1")
        assert qr.question == ""
