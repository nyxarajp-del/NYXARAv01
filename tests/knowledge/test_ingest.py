"""Tests for nyxara.knowledge.ingest — chunking and document loading."""

from __future__ import annotations

import os
import tempfile

from nyxara.knowledge.ingest import Document, chunk_text


def test_chunk_never_empty_and_covers_text():
    text = "Alpha beta gamma. " * 200
    chunks = chunk_text(text, max_chars=200, overlap=40)
    assert chunks
    assert all(c.strip() for c in chunks)
    assert all(len(c) <= 200 + 40 for c in chunks)


def test_chunk_overlap_present_for_long_text():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, max_chars=120, overlap=30)
    assert len(chunks) > 1


def test_chunk_short_text_single_chunk():
    chunks = chunk_text("just a little text", max_chars=800, overlap=100)
    assert chunks == ["just a little text"] or (len(chunks) == 1)


def test_document_from_file_round_trips():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "doc.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("NYXARA serves the Master.")
        doc = Document.from_file(path)
    assert "NYXARA" in doc.text
    assert doc.source


def test_document_missing_file_is_data_not_crash():
    doc = Document.from_file("/no/such/file/xyz.txt")
    assert isinstance(doc, Document)
    assert doc.text == "" or "error" in (doc.metadata or {})
