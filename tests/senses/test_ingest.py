"""Tests for nyxara.senses.ingest."""

from __future__ import annotations

import os
import tempfile

import pytest

from nyxara.senses.ingest import Chunk, Chunker, DocumentType, Ingestor


def _ing(**kw):
    return Ingestor(chunker=Chunker(**kw)) if kw else Ingestor()


# -------------------- type detection -------------------- #
def test_detect_by_extension():
    assert Ingestor.detect_type("a.md") is DocumentType.MARKDOWN
    assert Ingestor.detect_type("a.csv") is DocumentType.CSV
    assert Ingestor.detect_type("a.json") is DocumentType.JSON
    assert Ingestor.detect_type("a.py") is DocumentType.CODE
    assert Ingestor.detect_type("a.pdf") is DocumentType.PDF
    assert Ingestor.detect_type("a.txt") is DocumentType.TEXT


def test_detect_json_from_content():
    assert Ingestor.detect_type(content='{"a": 1}') is DocumentType.JSON
    assert Ingestor.detect_type(content="[1, 2, 3]") is DocumentType.JSON


def test_detect_invalid_json_not_json():
    assert Ingestor.detect_type(content="{not valid") is not DocumentType.JSON


def test_detect_markdown_from_content():
    assert Ingestor.detect_type(content="# Heading\nbody") is DocumentType.MARKDOWN


def test_detect_html_from_content():
    assert Ingestor.detect_type(content="<html><body>x</body></html>") is DocumentType.HTML


def test_detect_csv_from_content():
    assert Ingestor.detect_type(content="a,b,c\n1,2,3") is DocumentType.CSV


def test_detect_default_text():
    assert Ingestor.detect_type(content="just some prose without structure") is DocumentType.TEXT


# -------------------- markdown -------------------- #
def test_ingest_markdown_headings():
    d = _ing().ingest_text("# A\ntext\n## B\nmore", source="x.md")
    assert d.doc_type is DocumentType.MARKDOWN
    assert d.metadata["headings"] == ["A", "B"]


def test_ingest_markdown_strips_syntax():
    d = _ing().ingest_text("# Title\n**bold** and `code` and [link](http://x)", source="x.md")
    assert "#" not in d.text and "*" not in d.text and "`" not in d.text
    assert "link" in d.text and "http://x" not in d.text


# -------------------- csv / tsv -------------------- #
def test_ingest_csv():
    d = _ing().ingest_text("name,age\nJP,30\nAI,1", source="x.csv")
    assert d.metadata["columns"] == ["name", "age"]
    assert d.metadata["rows"] == 2 and d.metadata["n_columns"] == 2


def test_ingest_tsv():
    d = _ing().ingest_text("a\tb\n1\t2", source="x.tsv")
    assert d.doc_type is DocumentType.TSV
    assert d.metadata["columns"] == ["a", "b"]


def test_ingest_csv_empty():
    d = _ing().ingest_text("", source="x.csv")
    assert d.metadata["rows"] == 0


# -------------------- json -------------------- #
def test_ingest_json_object():
    d = _ing().ingest_text('{"x": 1, "y": 2}', source="x.json")
    assert d.metadata["keys"] == ["x", "y"]
    assert d.metadata["json_type"] == "dict"


def test_ingest_json_array():
    d = _ing().ingest_text("[1, 2, 3]", source="x.json")
    assert d.metadata["length"] == 3 and d.metadata["json_type"] == "list"


def test_ingest_json_invalid_keeps_text():
    d = _ing().ingest_text("{bad json", source="x.json")
    assert "json_error" in d.metadata


# -------------------- code / text / html -------------------- #
def test_ingest_code():
    d = _ing().ingest_text("def f():\n    pass\n", source="x.py")
    assert d.doc_type is DocumentType.CODE and d.metadata["lines"] >= 2


def test_ingest_text_plain():
    d = _ing().ingest_text("Just some text.", source="x.txt")
    assert d.doc_type is DocumentType.TEXT and d.text == "Just some text."


def test_ingest_html():
    d = _ing().ingest_text("<html><h1>Hi</h1><p>Body.</p><script>x</script></html>",
                           source="x.html")
    assert "Body" in d.text and "script" not in d.text
    assert "Hi" in d.metadata["headings"]


# -------------------- Chunker -------------------- #
def test_chunker_rejects_bad_target():
    with pytest.raises(ValueError):
        Chunker(target_chars=0)


def test_chunker_single_chunk_small_text():
    chunks = Chunker(target_chars=1000).chunk("Short text. Two sentences.")
    assert len(chunks) == 1


def test_chunker_empty():
    assert Chunker().chunk("") == []
    assert Chunker().chunk("   ") == []


def test_chunker_splits_long_text():
    text = " ".join(f"Sentence number {i} here." for i in range(20))
    chunks = Chunker(target_chars=80, overlap_chars=20).chunk(text)
    assert len(chunks) > 1


def test_chunker_offsets_exact():
    text = "One sentence. Two sentence. Three sentence. Four sentence. Five sentence."
    chunks = Chunker(target_chars=30, overlap_chars=10).chunk(text)
    for c in chunks:
        assert text[c.start:c.end] == c.text


def test_chunker_overlap():
    text = "Aaa aaa aaa. Bbb bbb bbb. Ccc ccc ccc. Ddd ddd ddd. Eee eee eee."
    chunks = Chunker(target_chars=30, overlap_chars=15).chunk(text)
    assert len(chunks) >= 2
    # consecutive chunks overlap in source range
    assert chunks[1].start < chunks[0].end


def test_chunker_zero_overlap():
    text = "Aaa. Bbb. Ccc. Ddd. Eee. Fff. Ggg. Hhh."
    chunks = Chunker(target_chars=15, overlap_chars=0).chunk(text)
    assert len(chunks) >= 2
    # no overlap: chunk starts do not go backwards into previous
    assert chunks[1].start >= chunks[0].end


def test_chunker_overlap_clamped_to_target():
    ch = Chunker(target_chars=100, overlap_chars=500)
    assert ch.overlap_chars < ch.target_chars


def test_chunk_approx_tokens():
    c = Chunk(0, "a" * 40, 0, 40)
    assert c.approx_tokens == 10


def test_chunk_to_dict():
    c = Chunk(0, "text here", 0, 9, source="s")
    assert c.to_dict()["index"] == 0 and c.to_dict()["source"] == "s"


# -------------------- Document -------------------- #
def test_document_counts():
    d = _ing().ingest_text("one two three", source="x.txt")
    assert d.word_count == 3 and d.char_count == len(d.text)


def test_document_chunks_populated():
    text = " ".join(f"S{i} word word." for i in range(30))
    d = _ing(target_chars=60, overlap_chars=15).ingest_text(text, source="x.txt")
    assert len(d.chunks) > 1
    assert all(c.source == "x.txt" for c in d.chunks)


def test_document_to_dict():
    d = _ing().ingest_text("# H\ntext", source="x.md")
    dd = d.to_dict()
    assert dd["doc_type"] == "markdown" and "n_chunks" in dd


# -------------------- file ingestion -------------------- #
def test_ingest_file_text():
    p = os.path.join(tempfile.gettempdir(), "nyx_ingest_test.txt")
    try:
        with open(p, "w", encoding="utf-8") as f:
            f.write("Hello from a file. Second sentence.")
        d = _ing().ingest_file(p)
        assert d.doc_type is DocumentType.TEXT and "Hello from a file" in d.text
    finally:
        os.remove(p)


def test_ingest_file_csv():
    p = os.path.join(tempfile.gettempdir(), "nyx_ingest_test.csv")
    try:
        with open(p, "w", encoding="utf-8") as f:
            f.write("a,b\n1,2\n3,4")
        d = _ing().ingest_file(p)
        assert d.metadata["columns"] == ["a", "b"] and d.metadata["rows"] == 2
    finally:
        os.remove(p)


def test_ingest_binary_graceful_fallback():
    # a .pdf without the optional parser must not crash — it returns an honest note
    p = os.path.join(tempfile.gettempdir(), "nyx_fake.pdf")
    try:
        with open(p, "wb") as f:
            f.write(b"%PDF-1.4 fake")
        d = _ing().ingest_file(p)
        assert d.doc_type is DocumentType.PDF
        # either text was extracted (lib present) or a note explains its absence
        assert d.text or d.note
    finally:
        os.remove(p)
