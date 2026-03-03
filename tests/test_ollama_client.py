"""Unit tests for the pure JSON-parsing helpers in src/clients/ollama_client.py.

These tests do not call Ollama — they only exercise the string manipulation
and Pydantic validation logic that parses raw LLM output.
"""

import pytest

from src.clients.ollama_client import (
    extract_thinking,
    parse_json_list,
    parse_json_response,
    strip_thinking,
)
from src.schemas import GapAnalysis, WorkerToolCall


# ── extract_thinking ──────────────────────────────────────────────────────────

class TestExtractThinking:
    def test_extracts_content_between_tags(self):
        assert extract_thinking("<think>reasoning here</think>answer") == "reasoning here"

    def test_returns_empty_when_no_tags(self):
        assert extract_thinking("plain text with no thinking") == ""

    def test_strips_whitespace_from_extracted_content(self):
        assert extract_thinking("<think>  padded  </think>") == "padded"

    def test_missing_close_tag_returns_empty(self):
        assert extract_thinking("<think>no closing tag") == ""

    def test_missing_open_tag_returns_empty(self):
        assert extract_thinking("no opening tag</think>") == ""

    def test_multiline_thinking(self):
        text = "<think>\nline one\nline two\n</think>result"
        assert extract_thinking(text) == "line one\nline two"


# ── strip_thinking ────────────────────────────────────────────────────────────

class TestStripThinking:
    def test_removes_think_block(self):
        assert strip_thinking("<think>hidden</think>visible") == "visible"

    def test_returns_text_unchanged_when_no_tags(self):
        assert strip_thinking("no thinking here") == "no thinking here"

    def test_strips_surrounding_whitespace_from_result(self):
        assert strip_thinking("<think>hidden</think>  visible  ") == "visible"

    def test_think_block_at_end(self):
        result = strip_thinking("prefix<think>hidden</think>")
        assert result == "prefix"

    def test_missing_open_tag_returns_original(self):
        text = "no opening</think>"
        assert strip_thinking(text) == text

    def test_missing_close_tag_returns_original(self):
        text = "<think>no closing"
        assert strip_thinking(text) == text


# ── parse_json_list ───────────────────────────────────────────────────────────

class TestParseJsonList:
    def test_plain_json_array(self):
        assert parse_json_list('["a", "b", "c"]') == ["a", "b", "c"]

    def test_empty_array(self):
        assert parse_json_list("[]") == []

    def test_strips_markdown_fences(self):
        text = "```json\n[\"a\", \"b\"]\n```"
        assert parse_json_list(text) == ["a", "b"]

    def test_strips_think_block(self):
        text = "<think>internal reasoning</think>\n[\"x\", \"y\"]"
        assert parse_json_list(text) == ["x", "y"]

    def test_invalid_json_returns_empty_list(self):
        assert parse_json_list("not json at all") == []

    def test_json_object_returns_empty_list(self):
        # An object is not a list
        assert parse_json_list('{"key": "value"}') == []

    def test_filters_out_non_string_items(self):
        # Only string items are kept
        result = parse_json_list('["hello", 42, "world", null]')
        assert result == ["hello", "world"]


# ── parse_json_response ───────────────────────────────────────────────────────

class TestParseJsonResponse:
    def test_plain_json_parsed_and_validated(self):
        text = '{"isSufficient": true, "reasoning": "enough", "gaps": []}'
        result = parse_json_response(text, GapAnalysis)
        assert result.is_sufficient is True
        assert result.reasoning == "enough"
        assert result.follow_up_questions == []

    def test_strips_markdown_fences(self):
        text = '```json\n{"isSufficient": false, "reasoning": "missing", "gaps": ["q1"]}\n```'
        result = parse_json_response(text, GapAnalysis)
        assert result.is_sufficient is False
        assert result.follow_up_questions == ["q1"]

    def test_strips_think_block(self):
        text = '<think>Let me think.</think>\n{"isSufficient": true, "reasoning": "ok", "gaps": []}'
        result = parse_json_response(text, GapAnalysis)
        assert result.is_sufficient is True

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            parse_json_response("this is not json", GapAnalysis)

    def test_schema_validation_failure_raises_value_error(self):
        # "invalid_tool" is not in WorkerToolCall's Literal
        bad = '{"tool": "invalid_tool", "args": {}, "reasoning": "test"}'
        with pytest.raises(ValueError, match="did not match schema"):
            parse_json_response(bad, WorkerToolCall)

    def test_valid_worker_tool_call(self):
        text = '{"tool": "rag_search", "args": {"query": "quarks"}, "reasoning": "check docs"}'
        result = parse_json_response(text, WorkerToolCall)
        assert result.tool == "rag_search"
        assert result.args == {"query": "quarks"}
        assert result.reasoning == "check docs"

    def test_done_tool_parsed(self):
        text = '{"tool": "done", "args": {}, "reasoning": "sufficient evidence"}'
        result = parse_json_response(text, WorkerToolCall)
        assert result.tool == "done"

    def test_alias_fields_resolved(self):
        # GapAnalysis uses aliases isSufficient and gaps
        text = '{"isSufficient": false, "reasoning": "need more", "gaps": ["follow-up?"]}'
        result = parse_json_response(text, GapAnalysis)
        assert result.is_sufficient is False
        assert result.follow_up_questions == ["follow-up?"]
