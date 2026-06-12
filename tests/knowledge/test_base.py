"""Tests for nyxara.knowledge.base — the RAG-grounding knowledge base."""

from __future__ import annotations

import os
import tempfile

from nyxara.knowledge.base import KnowledgeBase
from nyxara.memory.store import MemoryStore
from nyxara.mind.llm import LLM


def test_ingest_and_retrieve_in_process():
    kb = KnowledgeBase(name="profile")
    kb.ingest_text("The owner of NYXARA is Jaypal Khoja, known as JP.", source="profile")
    kb.ingest_text("Defense in depth uses many independent layers.", source="security")
    hits = kb.retrieve("who is the owner", k=3)
    assert hits
    assert any("Jaypal" in h.text for h in hits)
    assert len(kb) >= 2


def test_retrieve_returns_retrieved_chunks_with_source():
    kb = KnowledgeBase(name="kb")
    kb.ingest_text("Kernel is sovereign over all faculties.", source="architecture")
    hits = kb.retrieve("sovereign kernel", k=2)
    assert hits and hits[0].source == "architecture"


def test_store_backed_persistence():
    store = MemoryStore()
    kb = KnowledgeBase(store=store, name="kb_persist")
    n = kb.ingest_text("NYXARA proposes and the kernel disposes.", source="law")
    assert n >= 1
    hits = kb.retrieve("kernel disposes", k=3)
    assert hits and any("disposes" in h.text for h in hits)


def test_ingest_file_round_trip():
    kb = KnowledgeBase(name="files")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "note.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("The Master is Jaypal Khoja. NYXARA is loyal to the Master.")
        n = kb.ingest_file(path)
    assert n >= 1
    hits = kb.retrieve("who is loyal", k=2)
    assert hits


def test_as_rag_with_offline_llm_does_not_crash():
    kb = KnowledgeBase(name="rag")
    kb.ingest_text("The owner of NYXARA is JP.", source="profile")
    rag = kb.as_rag(LLM())
    result = rag.answer("Who is the owner?")
    # offline mock may abstain or answer, but must return a structured result
    assert hasattr(result, "answer") and isinstance(result.answer, str)


def test_empty_kb_retrieve_is_empty():
    kb = KnowledgeBase(name="empty")
    assert kb.retrieve("anything", k=3) == []
