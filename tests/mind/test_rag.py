"""Tests for nyxara.mind.rag."""

from __future__ import annotations

import json

from nyxara.mind.proposal import Proposal
from nyxara.mind.rag import (
    DocumentRetriever,
    GroundingReport,
    MemoryRetriever,
    RAGPipeline,
    RAGResult,
    RetrievedChunk,
    check_grounding,
)


class FakeLLM:
    def __init__(self, answer, reasoning="because the sources say so"):
        self._answer = answer
        self._reasoning = reasoning

    def generate(self, prompt, system=None, **kw):
        return json.dumps({"reasoning": self._reasoning, "answer": self._answer})


def _docs():
    d = DocumentRetriever()
    d.add("The owner of NYXARA is Jaypal Khoja, known as JP.", source="profile")
    d.add("NYXARA runs a sovereign kernel; the LLM proposes and the kernel disposes.",
          source="architecture")
    d.add("Defense in depth uses at least one hundred layers.", source="security")
    return d


# -------------------- DocumentRetriever -------------------- #
def test_retriever_add_and_len():
    d = DocumentRetriever()
    n = d.add("one sentence. two sentence.", source="x")
    assert n == 2
    assert len(d) == 2


def test_retriever_no_split():
    d = DocumentRetriever()
    d.add("a. b. c.", source="x", split=False)
    assert len(d) == 1


def test_retriever_retrieves_relevant():
    d = _docs()
    chunks = d.retrieve("who is the owner", k=2)
    assert chunks
    assert any("Jaypal" in c.text for c in chunks)


def test_retriever_ranks_by_overlap():
    d = _docs()
    chunks = d.retrieve("owner Jaypal Khoja", k=3)
    assert "Jaypal" in chunks[0].text  # best overlap first


def test_retriever_empty_query():
    d = _docs()
    chunks = d.retrieve("", k=2)
    assert len(chunks) <= 2


# -------------------- check_grounding -------------------- #
def test_grounding_fully_supported():
    chunks = [RetrievedChunk("The owner is Jaypal Khoja known as JP", "profile")]
    rep = check_grounding("The owner is Jaypal Khoja.", chunks)
    assert rep.groundedness == 1.0
    assert not rep.hallucinated
    assert "profile" in rep.citations.values()


def test_grounding_catches_hallucination():
    chunks = [RetrievedChunk("The owner is Jaypal Khoja", "profile")]
    rep = check_grounding(
        "The owner is Jaypal Khoja. NYXARA controls the stock market secretly.", chunks)
    assert rep.hallucinated
    assert rep.groundedness == 0.5  # one of two claims supported
    assert any("stock market" in u for u in rep.unsupported)


def test_grounding_no_claims():
    rep = check_grounding("", [RetrievedChunk("x", "s")])
    assert rep.groundedness == 0.0


def test_grounding_contentless_claim_ok():
    rep = check_grounding("Yes.", [RetrievedChunk("anything here", "s")])
    assert rep.groundedness == 1.0  # no content words -> vacuously supported


def test_grounding_threshold():
    chunks = [RetrievedChunk("apple banana cherry date elderberry", "fruit")]
    # claim shares 1 of 4 content words -> 0.25 support
    strict = check_grounding("apple rocket spaceship galaxy", chunks, threshold=0.5)
    lax = check_grounding("apple rocket spaceship galaxy", chunks, threshold=0.2)
    assert strict.hallucinated
    assert not lax.hallucinated


def test_grounding_report_to_dict():
    rep = check_grounding("The owner is JP.", [RetrievedChunk("owner is JP", "s")])
    d = rep.to_dict()
    assert "groundedness" in d and "supported" in d


# -------------------- RAGPipeline -------------------- #
def test_pipeline_grounded_answer():
    pipe = RAGPipeline(_docs(), FakeLLM("The owner of NYXARA is Jaypal Khoja, called JP."))
    res = pipe.answer("Who is the owner?")
    assert isinstance(res, RAGResult)
    assert not res.abstained
    assert res.confidence > 0.5
    assert "profile" in res.citations()


def test_pipeline_hallucinated_abstains():
    pipe = RAGPipeline(_docs(), FakeLLM(
        "NYXARA owns the moon. It controls all banks. It predicts the future precisely."))
    res = pipe.answer("What can NYXARA do?")
    assert res.abstained
    assert res.grounding.groundedness < 0.5


def test_pipeline_no_evidence_abstains():
    pipe = RAGPipeline(DocumentRetriever(), FakeLLM("anything at all"))
    res = pipe.answer("anything?")
    assert res.abstained
    assert res.confidence == 0.0
    assert res.chunks == []


def test_pipeline_parses_json_response():
    pipe = RAGPipeline(_docs(), FakeLLM("Jaypal Khoja is the owner", reasoning="the profile"))
    res = pipe.answer("owner?")
    assert res.reasoning == "the profile"


def test_pipeline_parses_plain_text_response():
    class PlainLLM:
        def generate(self, prompt, system=None, **kw):
            return "The owner is Jaypal Khoja."

    pipe = RAGPipeline(_docs(), PlainLLM())
    res = pipe.answer("owner?")
    assert "Jaypal" in res.answer


def test_pipeline_build_prompt_includes_sources():
    pipe = RAGPipeline(_docs(), FakeLLM("x"))
    chunks = _docs().retrieve("owner", k=2)
    prompt = pipe.build_prompt("Who is owner?", chunks)
    assert "SOURCES:" in prompt and "QUESTION:" in prompt
    assert "[1]" in prompt


def test_pipeline_answer_proposal():
    pipe = RAGPipeline(_docs(), FakeLLM("The owner of NYXARA is Jaypal Khoja."))
    prop = pipe.answer_proposal("Who is the owner?")
    assert isinstance(prop, Proposal)
    assert prop.source_faculty == "rag"
    assert "groundedness" in prop.metadata
    assert "citations" in prop.metadata


def test_pipeline_result_to_dict():
    pipe = RAGPipeline(_docs(), FakeLLM("The owner is Jaypal Khoja."))
    d = pipe.answer("owner?").to_dict()
    assert "answer" in d and "groundedness" in d and "citations" in d


# -------------------- MemoryRetriever -------------------- #
def test_memory_retriever():
    from nyxara.memory.provenance import Provenance, SourceType
    from nyxara.memory.store import MemoryStore, MemoryType

    store = MemoryStore()
    store.remember("The owner of NYXARA is Jaypal Khoja JP", mem_type=MemoryType.SEMANTIC,
                   provenance=Provenance(SourceType.OWNER, confidence=1.0), tags=["owner"])
    store.remember("pancake recipe with syrup", mem_type=MemoryType.SEMANTIC, tags=["food"])
    retr = MemoryRetriever(store)
    chunks = retr.retrieve("who is the owner", k=2)
    assert chunks
    assert any("Jaypal" in c.text for c in chunks)
    assert chunks[0].source == "owner"


def test_rag_over_memory_end_to_end():
    from nyxara.memory.provenance import Provenance, SourceType
    from nyxara.memory.store import MemoryStore, MemoryType

    store = MemoryStore()
    store.remember("The owner of NYXARA is Jaypal Khoja, known as JP",
                   mem_type=MemoryType.SEMANTIC,
                   provenance=Provenance(SourceType.OWNER, confidence=1.0), tags=["owner"])
    pipe = RAGPipeline(MemoryRetriever(store),
                       FakeLLM("The owner of NYXARA is Jaypal Khoja, known as JP"))
    res = pipe.answer("Who is the owner of NYXARA?")
    assert not res.abstained
    assert res.confidence > 0.5
