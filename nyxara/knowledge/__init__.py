"""NYXARA · knowledge/ — persistent knowledge base for RAG grounding.

Ingest text, documents, and files; retrieve grounding chunks that plug straight into
:class:`nyxara.mind.rag.RAGPipeline`. Backed by a persistent
:class:`nyxara.memory.store.MemoryStore` when given one, or an in-process lexical
index otherwise.
"""

from __future__ import annotations

from nyxara.knowledge.base import KnowledgeBase
from nyxara.knowledge.ingest import Document, chunk_text

__all__ = ["KnowledgeBase", "Document", "chunk_text"]
