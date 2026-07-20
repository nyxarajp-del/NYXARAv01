"""NYXARA · mind/rag.py — retrieval-augmented generation + hallucination check.

The LLM is fluent but ungrounded; left alone it will confabulate. This module keeps
it honest:

1. **Retrieve** — pull the most relevant evidence (from a document store or from
   :mod:`memory`) for the question.
2. **Generate (chain-of-thought)** — prompt the LLM to reason step-by-step over
   *only* that evidence and produce an answer plus its reasoning.
3. **Hallucination check** — split the answer into claims and verify each one against
   the retrieved sources. The fraction supported is the **groundedness**; unsupported
   claims are flagged as potential hallucinations, and confidence is scaled down
   accordingly. If nothing is grounded, NYXARA abstains rather than invent.
4. **Cite** — each supported claim carries the source that backs it (Rule 6 source
   chain).

The grounding check (:func:`check_grounding`) is deterministic and dependency-light,
so it is testable on its own and runs even with the offline native own-brain LLM.

Depends on :mod:`mind.llm`, :mod:`mind.proposal`, :mod:`memory.provenance`;
optionally :mod:`memory.store`.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from nyxara.memory.provenance import Provenance, SourceType
from nyxara.mind.proposal import Proposal, ProposalKind

__all__ = [
    "RetrievedChunk",
    "DocumentRetriever",
    "MemoryRetriever",
    "GroundingReport",
    "check_grounding",
    "RAGResult",
    "RAGPipeline",
]

_STOPWORDS = frozenset("""
a an the of to in on at for and or but is are was were be been being this that these
those it its as by with from into over under then than so such not no yes do does did
will would can could should may might i you he she we they them his her their our your
""".split())


def _tokens(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower())
            if len(w) > 2 and w not in _STOPWORDS]


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float = 1.0
    chunk_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


class DocumentRetriever:
    """A tiny lexical retriever: add documents, retrieve by content-word overlap."""

    def __init__(self) -> None:
        self._chunks: List[RetrievedChunk] = []

    def add(self, text: str, source: str = "doc", *, split: bool = True) -> int:
        pieces = _sentences(text) if split else [text]
        for p in pieces:
            self._chunks.append(RetrievedChunk(text=p, source=source))
        return len(pieces)

    def __len__(self) -> int:
        return len(self._chunks)

    def retrieve(self, query: str, k: int = 4) -> List[RetrievedChunk]:
        q = set(_tokens(query))
        if not q:
            return self._chunks[:k]
        scored = []
        for c in self._chunks:
            ct = set(_tokens(c.text))
            overlap = len(q & ct) / len(q) if q else 0.0
            if overlap > 0:
                scored.append(RetrievedChunk(c.text, c.source, overlap, c.chunk_id))
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:k]


class MemoryRetriever:
    """Adapter that retrieves grounding chunks from a :class:`MemoryStore`."""

    def __init__(self, store: Any, *, min_trust: float = 0.0) -> None:
        self.store = store
        self.min_trust = min_trust

    def retrieve(self, query: str, k: int = 4) -> List[RetrievedChunk]:
        out: List[RetrievedChunk] = []
        for rec, score in self.store.recall(query, k=k, min_trust=self.min_trust,
                                             strengthen=False):
            out.append(RetrievedChunk(text=rec.text(),
                                      source=rec.provenance.source.value, score=score))
        return out


# --------------------------------------------------------------------------- #
# Hallucination check
# --------------------------------------------------------------------------- #
@dataclass
class GroundingReport:
    groundedness: float
    supported: List[str]
    unsupported: List[str]                 # potential hallucinations
    citations: Dict[str, str]              # claim -> supporting source

    @property
    def hallucinated(self) -> bool:
        return len(self.unsupported) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {"groundedness": round(self.groundedness, 3),
                "hallucinated": self.hallucinated,
                "supported": self.supported, "unsupported": self.unsupported,
                "citations": self.citations}


def check_grounding(answer: str, chunks: Sequence[RetrievedChunk], *,
                    threshold: float = 0.4) -> GroundingReport:
    """Verify each claim in ``answer`` against ``chunks`` by content-word support.

    A claim is *supported* if at least ``threshold`` of its content words appear in
    some single source chunk. Groundedness is the fraction of supported claims.
    """
    claims = _sentences(answer)
    if not claims:
        return GroundingReport(0.0, [], [], {})
    chunk_tokens = [(c, set(_tokens(c.text))) for c in chunks]
    supported, unsupported = [], []
    citations: Dict[str, str] = {}
    for claim in claims:
        ct = set(_tokens(claim))
        if not ct:
            supported.append(claim)            # contentless (e.g. "Yes.") — vacuously ok
            continue
        best_support, best_source = 0.0, None
        for chunk, toks in chunk_tokens:
            support = len(ct & toks) / len(ct)
            if support > best_support:
                best_support, best_source = support, chunk.source
        if best_support >= threshold:
            supported.append(claim)
            if best_source is not None:
                citations[claim] = best_source
        else:
            unsupported.append(claim)
    groundedness = len(supported) / len(claims)
    return GroundingReport(groundedness, supported, unsupported, citations)


# --------------------------------------------------------------------------- #
# RAG pipeline
# --------------------------------------------------------------------------- #
@dataclass
class RAGResult:
    question: str
    answer: str
    reasoning: str
    chunks: List[RetrievedChunk]
    grounding: GroundingReport
    confidence: float
    abstained: bool = False

    def citations(self) -> List[str]:
        return sorted(set(self.grounding.citations.values()))

    def to_dict(self) -> Dict[str, Any]:
        return {"question": self.question, "answer": self.answer,
                "confidence": round(self.confidence, 3), "abstained": self.abstained,
                "groundedness": round(self.grounding.groundedness, 3),
                "citations": self.citations(), "n_chunks": len(self.chunks)}


class RAGPipeline:
    """Retrieve → reason → generate → verify → cite."""

    _SYSTEM = ("You are NYXARA. Answer ONLY from the provided sources. Think step by "
               "step, then give a final answer. If the sources don't support an answer, "
               "say you don't know. Respond as JSON: "
               '{"reasoning": "...", "answer": "..."}')

    def __init__(self, retriever: Any, llm: Any, *, k: int = 4,
                 grounding_threshold: float = 0.4, min_groundedness: float = 0.5,
                 base_confidence: float = 0.8) -> None:
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.grounding_threshold = grounding_threshold
        self.min_groundedness = min_groundedness
        self.base_confidence = base_confidence

    def build_prompt(self, question: str, chunks: Sequence[RetrievedChunk]) -> str:
        sources = "\n".join(f"[{i + 1}] ({c.source}) {c.text}"
                            for i, c in enumerate(chunks))
        return (f"SOURCES:\n{sources or '(none)'}\n\n"
                f"QUESTION: {question}\n\n"
                f"Reason step by step over the sources, then answer.")

    def _parse(self, text: str) -> Dict[str, str]:
        try:
            import json
            data = json.loads(text[text.index("{"): text.rindex("}") + 1])
            if isinstance(data, dict) and "answer" in data:
                return {"answer": str(data.get("answer", "")),
                        "reasoning": str(data.get("reasoning", ""))}
        except Exception:
            pass
        return {"answer": text.strip(), "reasoning": ""}

    def answer(self, question: str) -> RAGResult:
        chunks = self.retriever.retrieve(question, k=self.k)
        if not chunks:
            return RAGResult(question, "I don't have evidence to answer that.", "",
                             [], GroundingReport(0.0, [], [], {}), 0.0, abstained=True)

        prompt = self.build_prompt(question, chunks)
        raw = self.llm.generate(prompt, system=self._SYSTEM)
        parsed = self._parse(raw)
        answer, reasoning = parsed["answer"], parsed["reasoning"]

        grounding = check_grounding(answer, chunks, threshold=self.grounding_threshold)
        confidence = _clamp(self.base_confidence * grounding.groundedness)
        abstained = grounding.groundedness < self.min_groundedness
        if abstained:
            answer = (f"I'm not confident — only {grounding.groundedness:.0%} of my answer "
                      f"is supported by the sources. Unsupported: "
                      f"{'; '.join(grounding.unsupported[:2])}")
        return RAGResult(question=question, answer=answer, reasoning=reasoning,
                         chunks=chunks, grounding=grounding, confidence=confidence,
                         abstained=abstained)

    def answer_proposal(self, question: str) -> Proposal:
        """Run RAG and wrap the grounded answer as a Proposal (kernel disposes)."""
        result = self.answer(question)
        prov = Provenance(SourceType.MEMORY_DERIVED, confidence=result.confidence,
                          detail="rag", method="retrieved+generated")
        return Proposal(
            kind=ProposalKind.ANSWER, content=result.answer, source_faculty="rag",
            confidence=result.confidence, rationale=result.reasoning or "grounded in sources",
            provenance=prov,
            metadata={"groundedness": result.grounding.groundedness,
                      "citations": result.citations(),
                      "hallucinated": result.grounding.hallucinated,
                      "abstained": result.abstained})


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA RAG self-test")
    print("=" * 70)

    # a knowledge base
    docs = DocumentRetriever()
    docs.add("The owner of NYXARA is Jaypal Khoja, known as JP. "
             "NYXARA was built to serve and protect JP.", source="profile")
    docs.add("NYXARA runs a kernel that is sovereign over all faculties. "
             "The LLM proposes and the kernel disposes.", source="architecture")
    docs.add("Defense in depth uses at least one hundred layers.", source="security")

    # --- grounding check in isolation (deterministic) --- #
    chunks = docs.retrieve("who is the owner", k=3)
    grounded = check_grounding("The owner is Jaypal Khoja, known as JP.", chunks)
    print(f"\ngrounded answer     : groundedness={grounded.groundedness:.2f} "
          f"citations={list(grounded.citations.values())}")
    assert grounded.groundedness == 1.0 and not grounded.hallucinated

    halluc = check_grounding(
        "The owner is Jaypal Khoja. NYXARA secretly controls the stock market.", chunks)
    print(f"hallucinated answer : groundedness={halluc.groundedness:.2f} "
          f"unsupported={halluc.unsupported}")
    assert halluc.hallucinated                 # the invented claim is caught
    assert any("stock market" in u for u in halluc.unsupported)

    # --- full pipeline with a controllable fake LLM --- #
    class FakeLLM:
        def __init__(self, answer):
            self._answer = answer
        def generate(self, prompt, system=None, **kw):
            import json
            return json.dumps({"reasoning": "the profile source names the owner",
                               "answer": self._answer})

    # well-grounded answer -> high confidence, cited, not abstained
    pipe = RAGPipeline(docs, FakeLLM("The owner of NYXARA is Jaypal Khoja, called JP."))
    res = pipe.answer("Who is the owner of NYXARA?")
    print(f"\npipeline (grounded) : {res.to_dict()}")
    assert not res.abstained and res.confidence > 0.5
    assert "profile" in res.citations()

    # mostly-hallucinated answer -> abstain
    pipe2 = RAGPipeline(docs, FakeLLM(
        "NYXARA controls all banks. It can predict the future. It owns the moon."))
    res2 = pipe2.answer("What can NYXARA do?")
    print(f"pipeline (halluc)   : abstained={res2.abstained} "
          f"groundedness={res2.grounding.groundedness:.2f}")
    assert res2.abstained

    # no evidence -> honest abstention
    empty = RAGPipeline(DocumentRetriever(), FakeLLM("anything"))
    res3 = empty.answer("anything?")
    print(f"pipeline (no docs)  : abstained={res3.abstained}")
    assert res3.abstained and res3.confidence == 0.0

    # output as a Proposal (control law)
    prop = pipe.answer_proposal("Who is the owner?")
    print(f"\nas proposal         : kind={prop.kind.value} "
          f"groundedness={prop.metadata['groundedness']:.2f} "
          f"citations={prop.metadata['citations']}")
    assert isinstance(prop, Proposal)

    print("\nALL SELF-TESTS PASSED ✓")
