"""NYXARA · knowledge/ingest.py — documents in, clean chunks out.

The knowledge base needs *grounding material*: plain text, files, or extracted
documents, sliced into retrieval-sized chunks. This module is the front door.

* :func:`chunk_text` — a sliding-window chunker that prefers to break on paragraph
  and sentence boundaries, keeps a configurable overlap so context isn't severed at
  the seam, and **never** emits an empty chunk.
* :class:`Document` — a tiny, serialisable container (``doc_id``, ``source``,
  ``text``, ``metadata``) with a :meth:`Document.from_file` reader that prefers the
  richer :mod:`nyxara.senses.ingest` extractor (pdf/docx/markdown/…) when it is
  importable, and otherwise falls back to a plain UTF-8 read.

Everything **fails as data**, never by crashing: an unreadable file yields a
``Document`` whose ``metadata`` carries an ``error`` note rather than raising, so a
batch ingest of a thousand files is never derailed by one bad apple.

Pure standard library; the senses extractor is import-guarded and optional.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["chunk_text", "Document"]


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def _split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _hard_wrap(piece: str, max_chars: int) -> List[str]:
    """Last-resort splitter for a single oversize unit (no usable boundary)."""
    return [piece[i:i + max_chars] for i in range(0, len(piece), max_chars)]


def chunk_text(text: str, *, max_chars: int = 800, overlap: int = 150) -> List[str]:
    """Slice ``text`` into overlapping chunks of at most ``max_chars`` characters.

    The chunker accumulates whole paragraphs (then whole sentences, then — only when
    a single unit is still too big — hard character windows) into a buffer, flushing
    once adding the next unit would overflow ``max_chars``. After each flush it seeds
    the next buffer with the tail ``overlap`` characters of the one just emitted, so
    adjacent chunks share context. Empty input yields ``[]``; non-empty input always
    yields at least one non-empty chunk.
    """
    text = (text or "").strip()
    if not text:
        return []
    if max_chars <= 0:
        max_chars = 1
    overlap = max(0, min(overlap, max_chars - 1))

    # break into the smallest boundary-respecting units we can
    units: List[str] = []
    for para in _split_paragraphs(text):
        if len(para) <= max_chars:
            units.append(para)
            continue
        for sent in _split_sentences(para):
            if len(sent) <= max_chars:
                units.append(sent)
            else:
                units.extend(_hard_wrap(sent, max_chars))

    chunks: List[str] = []
    buf = ""

    def _flush() -> None:
        nonlocal buf
        piece = buf.strip()
        if piece:
            chunks.append(piece)

    for unit in units:
        if not buf:
            buf = unit
            continue
        if len(buf) + 1 + len(unit) <= max_chars:
            buf = f"{buf}\n{unit}"
        else:
            _flush()
            tail = buf[-overlap:] if overlap else ""
            # keep the overlap tail only if the new unit still fits alongside it
            if tail and len(tail) + 1 + len(unit) <= max_chars:
                buf = f"{tail}\n{unit}"
            else:
                buf = unit
    _flush()

    # defensive: never return empty for non-empty input
    if not chunks:
        chunks = [text[:max_chars]]
    return chunks


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #
@dataclass
class Document:
    """A unit of grounding material: an id, where it came from, its text, metadata."""

    source: str
    text: str
    doc_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metadata: Dict[str, Any] = field(default_factory=dict)

    def chunks(self, *, max_chars: int = 800, overlap: int = 150) -> List[str]:
        return chunk_text(self.text, max_chars=max_chars, overlap=overlap)

    def to_dict(self) -> Dict[str, Any]:
        return {"doc_id": self.doc_id, "source": self.source,
                "n_chars": len(self.text), "metadata": self.metadata}

    @staticmethod
    def from_file(path: str) -> "Document":
        """Read a file into a :class:`Document`, failing as data.

        Prefers :class:`nyxara.senses.ingest.Ingestor` (handles pdf/docx/markdown/…)
        when importable; otherwise reads UTF-8 text directly. Any failure is captured
        in ``metadata['error']`` with empty text rather than raised.
        """
        source = str(path)
        # --- preferred: the rich senses extractor (optional) --- #
        try:
            from nyxara.senses.ingest import Ingestor  # import-guarded optional
            doc = Ingestor().ingest_file(source)
            meta: Dict[str, Any] = dict(getattr(doc, "metadata", {}) or {})
            note = getattr(doc, "note", "") or ""
            if note:
                meta.setdefault("note", note)
            dt = getattr(doc, "doc_type", None)
            if dt is not None:
                meta.setdefault("doc_type", getattr(dt, "value", str(dt)))
            return Document(source=source, text=getattr(doc, "text", "") or "",
                            metadata=meta)
        except Exception:  # noqa: BLE001 — fall through to a plain read
            pass

        # --- fallback: plain UTF-8 read --- #
        try:
            with open(source, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            meta = {"bytes": len(text), "ext": os.path.splitext(source)[1].lower()}
            return Document(source=source, text=text, metadata=meta)
        except Exception as exc:  # noqa: BLE001 — fail as data
            return Document(source=source, text="",
                            metadata={"error": f"{type(exc).__name__}: {exc}"})


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile

    print("=" * 70)
    print("NYXARA knowledge-ingest self-test")
    print("=" * 70)

    # --- chunker: empty in, empty out --- #
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []

    # --- chunker: short text -> one chunk, no loss --- #
    one = chunk_text("NYXARA serves the Master.", max_chars=800)
    print(f"\nshort text          : {len(one)} chunk(s)")
    assert one == ["NYXARA serves the Master."]

    # --- chunker: boundaries + no empties + size bound --- #
    body = "\n\n".join(f"Paragraph {i} about defense in depth and the kernel." * 3
                       for i in range(8))
    chunks = chunk_text(body, max_chars=200, overlap=50)
    print(f"long text           : {len(chunks)} chunks, "
          f"max len={max(len(c) for c in chunks)}")
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)              # never empty
    assert all(len(c) <= 200 for c in chunks)          # size respected

    # --- chunker: overlap actually shares context --- #
    seq = ". ".join(f"sentence number {i} carries unique token tok{i}" for i in range(40))
    ov = chunk_text(seq, max_chars=160, overlap=60)
    shared = any(
        any(tok in ov[j + 1] for tok in ov[j][-60:].split())
        for j in range(len(ov) - 1)
    )
    print(f"overlap chunks      : {len(ov)} chunks, context shared={shared}")
    assert len(ov) > 1 and shared

    # --- chunker: an oversize single token still gets split, never empty --- #
    huge = "x" * 5000
    hw = chunk_text(huge, max_chars=300, overlap=0)
    assert len(hw) >= 16 and all(len(c) <= 300 and c for c in hw)

    # --- Document.from_file round-trips a temp .txt --- #
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as fh:
        fh.write("The owner of NYXARA is Jaypal Khoja, known as JP.")
        tmp = fh.name
    doc = Document.from_file(tmp)
    print(f"\nfrom_file           : {doc.to_dict()}")
    assert "Jaypal Khoja" in doc.text
    assert doc.source == tmp and doc.doc_id
    os.remove(tmp)

    # --- from_file fails as data on a missing path --- #
    bad = Document.from_file("/no/such/file/anywhere.txt")
    print(f"missing file        : error={bad.metadata.get('error', '')[:40]!r}")
    assert bad.text == "" and "error" in bad.metadata   # did not crash

    print("\nALL SELF-TESTS PASSED ✓")
