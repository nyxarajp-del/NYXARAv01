"""Tests for nyxara.growth.scientist — the scientist loop.

Hypothesis → experiment → compare → conclusion. Everything runs offline (no LLM,
no network), so these tests are deterministic and hermetic.
"""

from __future__ import annotations

from nyxara.growth.scientist import (
    Conclusion,
    Experiment,
    ExperimentKind,
    Hypothesis,
    InvestigationReport,
    Observation,
    Scientist,
    Verdict,
)
from nyxara.knowledge.base import KnowledgeBase
from nyxara.memory.graph import KnowledgeGraph, _configure_standard_relations
from nyxara.memory.store import MemoryStore, MemoryType


# --------------------------------------------------------------------------- #
# dataclasses
# --------------------------------------------------------------------------- #
def test_report_to_dict_roundtrip():
    rep = InvestigationReport(question="why?")
    rep.hypotheses = [Hypothesis("h", prediction=True)]
    rep.experiment = Experiment(ExperimentKind.NUMERIC, method="m", expected=4.0)
    rep.observation = Observation(value=4.0, detail="ok")
    rep.conclusion = Conclusion(Verdict.SUPPORTED, 0.9, "because")
    d = rep.to_dict()
    assert d["question"] == "why?"
    assert d["conclusion"]["verdict"] == "supported"
    assert d["experiment"]["kind"] == "numeric"
    assert d["hypotheses"][0]["prediction"] is True


def test_empty_report_to_dict_has_none_sections():
    d = InvestigationReport(question="q").to_dict()
    assert d["experiment"] is None and d["observation"] is None
    assert d["conclusion"] is None


# --------------------------------------------------------------------------- #
# the full loop, fully offline
# --------------------------------------------------------------------------- #
def test_investigate_always_returns_conclusion():
    sci = Scientist()
    rep = sci.investigate("Does anything at all happen here?")
    assert isinstance(rep, InvestigationReport)
    assert rep.hypotheses and isinstance(rep.hypotheses[0], Hypothesis)
    assert rep.experiment is not None
    assert isinstance(rep.conclusion, Conclusion)
    assert isinstance(rep.conclusion.verdict, Verdict)
    assert rep.elapsed_ms >= 0.0


def test_even_sum_claim_is_supported():
    rep = Scientist().investigate("Is the sum of two even numbers always even?")
    assert rep.experiment.kind is ExperimentKind.COMPUTATIONAL
    assert rep.conclusion.verdict is Verdict.SUPPORTED


def test_odd_sum_claim_is_supported():
    rep = Scientist().investigate("the sum of two odd numbers is even")
    assert rep.conclusion.verdict is Verdict.SUPPORTED


def test_prime_claim_true_and_false():
    sci = Scientist()
    yes = sci.investigate("Is 17 a prime number?")
    assert yes.experiment.kind is ExperimentKind.COMPUTATIONAL
    assert yes.conclusion.verdict is Verdict.SUPPORTED

    no = sci.investigate("Is 21 a prime number?")
    assert no.conclusion.verdict is Verdict.REFUTED


def test_numeric_equality_true_and_false():
    sci = Scientist()
    good = sci.investigate("is 2 + 2 = 4?")
    assert good.experiment.kind is ExperimentKind.NUMERIC
    assert good.conclusion.verdict is Verdict.SUPPORTED

    bad = sci.investigate("is 2 + 2 = 5?")
    assert bad.experiment.kind is ExperimentKind.NUMERIC
    assert bad.conclusion.verdict is Verdict.REFUTED


def test_unknown_question_without_knowledge_is_inconclusive():
    rep = Scientist().investigate("Does dark matter interact weakly?")
    assert rep.experiment.kind is ExperimentKind.KNOWLEDGE
    assert rep.conclusion.verdict is Verdict.INCONCLUSIVE


def test_conclusion_suggests_next_hypothesis():
    rep = Scientist().investigate("Is the sum of two even numbers always even?")
    assert rep.conclusion.next_hypothesis


# --------------------------------------------------------------------------- #
# computational experiments run through the real sandbox
# --------------------------------------------------------------------------- #
def test_computational_experiment_uses_sandbox():
    from nyxara.sim.sandbox import Sandbox

    sandbox = Sandbox()
    sci = Scientist(sandbox=sandbox)
    rep = sci.investigate("Is 13 a prime number?")
    assert rep.observation.success
    assert rep.conclusion.verdict is Verdict.SUPPORTED


# --------------------------------------------------------------------------- #
# recording side-effects when wired
# --------------------------------------------------------------------------- #
def test_records_into_knowledge_graph_and_memory():
    kb = KnowledgeBase(name="sci-test")
    graph = KnowledgeGraph()
    _configure_standard_relations(graph)
    mem = MemoryStore()

    sci = Scientist(knowledge=kb, knowledge_graph=graph, memory=mem)
    rep = sci.investigate("Is the sum of two even numbers always even?")

    assert rep.kb_chunks_added == 1
    assert rep.graph_triples_added == 1
    assert len(graph) >= 1
    recalled = mem.recall("experiment", k=5)
    assert any("Experiment" in rec.content for rec, _score in recalled)


def test_all_investigations_accumulates():
    sci = Scientist()
    sci.investigate("Is 5 a prime number?")
    sci.investigate("Is 6 a prime number?")
    assert len(sci.all_investigations()) == 2


# --------------------------------------------------------------------------- #
# composition with the researcher for background evidence
# --------------------------------------------------------------------------- #
def test_uses_researcher_for_background():
    class _StubResearcher:
        def __init__(self):
            self.called_with = None

        def research(self, topic):
            self.called_with = topic

            class _R:
                key_claims = ["A relevant claim about the topic."]
                summary = "summary"
            return _R()

    stub = _StubResearcher()
    rep = Scientist(researcher=stub).investigate("Tell me about widgets")
    assert stub.called_with == "Tell me about widgets"
    assert rep.background_claims == ["A relevant claim about the topic."]


def test_degrades_when_everything_is_none():
    # no researcher, sandbox, knowledge, graph, memory or llm — still concludes.
    rep = Scientist().investigate("anything")
    assert isinstance(rep.conclusion, Conclusion)
    assert rep.kb_chunks_added == 0 and rep.graph_triples_added == 0
