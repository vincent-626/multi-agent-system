"""Unit tests for the document chunking function in src/ingest.py."""

import pytest

from src.ingest import _chunk_text


# ── Basic cases ───────────────────────────────────────────────────────────────

def test_short_text_returned_as_single_chunk():
    chunks = _chunk_text("Hello world", size=100, overlap=0)
    assert chunks == ["Hello world"]


def test_empty_text_returns_empty_list():
    assert _chunk_text("", size=100, overlap=0) == []


def test_whitespace_only_returns_empty_list():
    assert _chunk_text("   \n  \t  ", size=100, overlap=0) == []


# ── Separator priority ────────────────────────────────────────────────────────

def test_splits_on_double_newline_first():
    text = "First paragraph.\n\nSecond paragraph."
    chunks = _chunk_text(text, size=20, overlap=0)
    assert len(chunks) == 2
    assert chunks[0] == "First paragraph."
    assert chunks[1] == "Second paragraph."


def test_splits_on_single_newline_when_double_unavailable():
    text = "Line one\nLine two"
    chunks = _chunk_text(text, size=12, overlap=0)
    assert len(chunks) == 2
    assert chunks[0] == "Line one"
    assert chunks[1] == "Line two"


def test_splits_on_sentence_boundary():
    # No newlines, but ". " is a separator
    text = "First sentence. Second sentence."
    chunks = _chunk_text(text, size=20, overlap=0)
    assert len(chunks) >= 2


def test_splits_on_space_as_last_resort():
    # Single long word that can only be split by space
    text = "hello world foo bar baz"  # 23 chars
    chunks = _chunk_text(text, size=12, overlap=0)
    assert len(chunks) >= 2
    assert all(len(c) <= 12 for c in chunks)


# ── Size constraints ──────────────────────────────────────────────────────────

def test_all_raw_chunks_within_size():
    text = "word " * 200
    chunks = _chunk_text(text, size=50, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)


def test_chunk_count_scales_with_text_length():
    short_chunks = _chunk_text("word " * 20, size=50, overlap=0)
    long_chunks = _chunk_text("word " * 200, size=50, overlap=0)
    assert len(long_chunks) > len(short_chunks)


# ── Overlap ───────────────────────────────────────────────────────────────────

def test_overlap_prepends_tail_of_previous_chunk():
    # With size=6 and "\n\n" separator, raw_chunks = ["aaaa", "bbbb", "cccc"]
    # overlapped[1] = "aaaa"[-2:] + "bbbb" = "aa" + "bbbb" = "aabbbb"
    # overlapped[2] = "bbbb"[-2:] + "cccc" = "bb" + "cccc" = "bbcccc"
    text = "aaaa\n\nbbbb\n\ncccc"
    chunks = _chunk_text(text, size=6, overlap=2)
    assert chunks[0] == "aaaa"
    assert chunks[1] == "aabbbb"
    assert chunks[2] == "bbcccc"


def test_no_overlap_with_single_chunk():
    # Single chunk → overlap section is skipped
    chunks = _chunk_text("short text", size=100, overlap=10)
    assert chunks == ["short text"]


def test_zero_overlap_produces_clean_chunks():
    text = "aaaa\n\nbbbb\n\ncccc"
    chunks = _chunk_text(text, size=6, overlap=0)
    assert chunks == ["aaaa", "bbbb", "cccc"]


# ── Non-empty output guarantee ────────────────────────────────────────────────

def test_no_empty_chunks_in_output():
    text = "para one\n\n\n\npara two\n\n"
    chunks = _chunk_text(text, size=20, overlap=0)
    assert all(c.strip() for c in chunks)
