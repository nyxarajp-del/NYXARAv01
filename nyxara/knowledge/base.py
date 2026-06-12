"""NYXARA · knowledge/base.py — a persistent knowledge base for RAG grounding.

The LLM is fluent but ungrounded; :mod:`nyxara.mind.rag` keeps it honest by answering
only from retrieved evidence. That evidence has to live *somewhere* — this is that
somewhere: a :class:`KnowledgeBase` you ingest text/documents/files into, and then
retrieve grounding chunks from.

Two interchangeable backends, one API:

* **store-backed** — given a :class:`~nyxara.memory.store.MemoryStore`, every chunk is
  persisted as a SEMANTIC memory tagged with the kb's name and its source. It then
  survives the store's ``save``/``load``, benefits from the store's (hashing or
  learned) embeddings, and is recalled by associative similarity.
* **lexical** — with no store, an in-process :class:`~nyxara.mind.rag.DocumentRetriever`
  holds the chunks and retrieves by content-word overlap. Zero dependencies, fully
  offline.

Either way :meth:`KnowledgeBase.retrieve` returns
:class:`~nyxara.mind.rag.RetrievedChunk` objects, so a ``KnowledgeBase`` is a drop-in
retriever for :class:`~nyxara.mind.rag.RAGPipeline` — wire one up with :meth:`as_rag`.

Depends on :mod:`mind.rag` and :mod:`knowledge.ingest`; optionally on :mod:`memory.store`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nyxara.knowledge.ingest import Document, chunk_text
from nyxara.mind.rag import DocumentRetriever, RAGPipeline, RetrievedChunk

__all__ = ["KnowledgeBase"]

# tag prefixes so kb-owned memories are filterable inside a shared store
_KB_TAG = "kb"
_SRC_PREFIX = "src:"


@dataclass
class _Ingested:
    """Bookkeeping for one ingested source (counts + chunk ids)."""

    source: str
    n_chunks: int = 0
    mem_ids: List[str] = field(default_factory=list)


class KnowledgeBase:
    """Ingestable, retrievable grounding store — a drop-in RAG retriever."""

    def __init__(self, *, store: Optional[Any] = None, name: str = "kb",
                 max_chars: int = 800, overlap: int = 150) -> None:
        self.name = name
        self.store = store
        self.max_chars = max_chars
        self.overlap = overlap
        # lexical fallback index when there is no persistent store
        self._index: Optional[DocumentRetriever] = None if store else DocumentRetriever()
        self._sources: Dict[str, _Ingested] = {}
        self._n_chunks = 0

    # ------------------------------------------------------------------ #
    # Ingest
    # ------------------------------------------------------------------ #
    def ingest_text(self, text: str, source: str = "doc",
                    metadata: Optional[Dict[str, Any]] = None) -> int:
        """Chunk ``text`` and add it under ``source``. Returns the number of chunks."""
        chunks = chunk_text(text, max_chars=self.max_chars, overlap=self.overlap)
        if not chunks:
            return 0
        entry = self._sources.setdefault(source, _Ingested(source=source))
        meta = dict(metadata or {})
        for i, chunk in enumerate(chunks):
            if self.store is not None:
                rec = self._remember(chunk, source, {**meta, "chunk_index": i})
                entry.mem_ids.append(rec.mem_id)
            else:
                assert self._index is not None
                self._index.add(chunk, source=source, split=False)
            entry.n_chunks += 1
            self._n_chunks += 1
        return len(chunks)

    def ingest_document(self, doc: Document) -> int:
        """Ingest a :class:`Document`, carrying its metadata through."""
        meta = {"doc_id": doc.doc_id, **(doc.metadata or {})}
        return self.ingest_text(doc.text, source=doc.source, metadata=meta)

    def ingest_file(self, path: str) -> int:
        """Read ``path`` into a :class:`Document` (failing as data) and ingest it."""
        return self.ingest_document(Document.from_file(path))

    def _remember(self, text: str, source: str, metadata: Dict[str, Any]) -> Any:
        from nyxara.memory.store import MemoryType  # local: store is optional
        tags = [_KB_TAG, self.name, f"{_SRC_PREFIX}{source}"]
        meta = {"kb": self.name, "source": source, **metadata}
        return self.store.remember(
            text,
            mem_type=MemoryType.SEMANTIC,
            importance=0.6,
            tags=tags,
            metadata=meta,
        )

    # ------------------------------------------------------------------ #
    # Retrieve (the RAG-retriever contract)
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, k: int = 4) -> List[RetrievedChunk]:
        """Return up to ``k`` grounding chunks for ``query`` as ``RetrievedChunk``s."""
        if self.store is not None:
            return self._retrieve_store(query, k)
        assert self._index is not None
        return self._index.retrieve(query, k=k)

    def _retrieve_store(self, query: str, k: int) -> List[RetrievedChunk]:
        out: List[RetrievedChunk] = []
        try:
            hits = self.store.recall(query, k=k, tags=[self.name], strengthen=False)
        except Exception:  # noqa: BLE001 — fail as data: no grounding, not a crash
            return out
        for rec, score in hits:
            out.append(RetrievedChunk(
                text=rec.text(),
                source=self._source_of(rec),
                score=float(score),
                chunk_id=getattr(rec, "mem_id", "")[:8] or "rec",
            ))
        return out

    @staticmethod
    def _source_of(rec: Any) -> str:
        """Recover a chunk's origin from its metadata, then its tags."""
        meta = getattr(rec, "metadata", None) or {}
        src = meta.get("source")
        if src:
            return str(src)
        for tag in getattr(rec, "tags", ()) or ():
            if tag.startswith(_SRC_PREFIX):
                return tag[len(_SRC_PREFIX):]
        return "kb"

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self._n_chunks

    def sources(self) -> List[str]:
        return sorted(self._sources.keys())

    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "backend": "store" if self.store is not None else "lexical",
            "n_chunks": self._n_chunks,
            "n_sources": len(self._sources),
            "sources": self.sources(),
            "max_chars": self.max_chars,
            "overlap": self.overlap,
        }

    # ------------------------------------------------------------------ #
    # RAG wiring
    # ------------------------------------------------------------------ #
    def as_rag(self, llm: Any, **kwargs: Any) -> RAGPipeline:
        """Wire this knowledge base straight into a :class:`RAGPipeline`."""
        return RAGPipeline(self, llm, **kwargs)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import os
    import tempfile

    print("=" * 70)
    print("NYXARA knowledge-base self-test")
    print("=" * 70)

    PROFILE = (
        "The owner of NYXARA is Jaypal Khoja, known as JP. "
        "NYXARA was built to serve and protect JP.\n\n"
        "NYXARA runs a sovereign kernel over all faculties: "
        "the LLM proposes and the kernel disposes.\n\n"
        "Defense in depth uses at least one hundred layers of security."
    )

    # --- lexical (no store) backend --- #
    kb = KnowledgeBase(name="profile")
    n = kb.ingest_text(PROFILE, source="profile")
    print(f"\nlexical ingest      : {n} chunks, stats={kb.stats()}")
    assert n >= 1 and len(kb) == n
    assert kb.sources() == ["profile"]

    hits = kb.retrieve("who is the owner of NYXARA?", k=3)
    print(f"lexical retrieve    : top={hits[0].text[:48]!r} score={hits[0].score:.2f}")
    assert hits and isinstance(hits[0], RetrievedChunk)
    assert any("Jaypal" in h.text for h in hits)
    assert hits == sorted(hits, key=lambda c: c.score, reverse=True)  # ranked

    # --- store-backed (persistent) backend --- #
    from nyxara.memory.store import MemoryStore

    store = MemoryStore()
    kb2 = KnowledgeBase(store=store, name="kb_persist", max_chars=200, overlap=40)
    n2 = kb2.ingest_text(PROFILE, source="profile")
    print(f"\nstore ingest        : {n2} chunks, store={len(store)} memories")
    assert n2 >= 1 and len(store) == n2

    s_hits = kb2.retrieve("who is the owner of NYXARA?", k=3)
    print(f"store retrieve      : top={s_hits[0].text[:48]!r} "
          f"source={s_hits[0].source!r}")
    assert s_hits and any("Jaypal" in h.text for h in s_hits)
    assert all(h.source == "profile" for h in s_hits)  # source recovered from metadata

    # --- persistence: chunks survive save/load via the store --- #
    path = os.path.join(tempfile.gettempdir(), "nyx_kb.json")
    store.save(path)
    store2 = MemoryStore()
    loaded = store2.load(path)
    kb3 = KnowledgeBase(store=store2, name="kb_persist")
    again = kb3.retrieve("who is the owner?", k=3)
    print(f"persisted/loaded    : {loaded} memories, "
          f"recall='{again[0].text[:40] if again else None}'")
    assert again and any("Jaypal" in h.text for h in again)
    os.remove(path)

    # --- ingest_file round-trips a temp .txt --- #
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as fh:
        fh.write("The kernel is sovereign over every faculty of NYXARA.")
        tmp = fh.name
    kb.ingest_file(tmp)
    print(f"\ningest_file         : sources={kb.sources()}")
    assert tmp in kb.sources()
    fres = kb.retrieve("what is sovereign over the faculties?", k=2)
    assert any("sovereign" in h.text for h in fres)
    os.remove(tmp)

    # --- as_rag plugs into the pipeline with the offline mock LLM --- #
    from nyxara.kernel.config import NyxaraSettings, Profile
    from nyxara.mind.llm import LLM

    llm = LLM(settings=NyxaraSettings.for_profile(Profile.TEST))
    pipe = kb.as_rag(llm)
    assert isinstance(pipe, RAGPipeline)
    result = pipe.answer("Who is the owner of NYXARA?")
    print(f"\nas_rag answer       : abstained={result.abstained} "
          f"conf={result.confidence:.2f} groundedness="
          f"{result.grounding.groundedness:.2f}")
    # the offline mock may answer or abstain; either way it must not crash
    assert result.question == "Who is the owner of NYXARA?"
    assert 0.0 <= result.confidence <= 1.0

    # --- empty kb -> honest abstention through the pipeline --- #
    empty = KnowledgeBase(name="empty").as_rag(llm)
    eres = empty.answer("anything?")
    print(f"empty kb answer     : abstained={eres.abstained}")
    assert eres.abstained and eres.confidence == 0.0

    print("\nALL SELF-TESTS PASSED ✓")
