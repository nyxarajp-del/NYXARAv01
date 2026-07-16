"""Tests that self-discovered laws actually ground NYXARA's answers (Gap B, reasoner side).

The native reasoner must consult the law tower she built herself: a quantitative question
whose variables match a discovered law is answered FROM that law (source ``law``, verified),
not deferred to the LLM. It abstains cleanly when no discovered law applies, so the other
engines keep working. Hermetic, no LLM.
"""

from __future__ import annotations

from nyxara.growth.autonomous_scientist import BeliefModel, Belief
from nyxara.growth.law_discovery import LawDiscoveryEngine
from nyxara.mind.native_reasoner import NativeReasoner


def _engine() -> LawDiscoveryEngine:
    eng = LawDiscoveryEngine(seed=7)
    X, y = [], []
    for m in range(1, 9):
        for a in range(1, 9):
            X.append([float(m), float(a)])
            y.append(float(m * a))
    eng.discover_from_data(X, y, var_names=["m", "a"], target_name="force")
    return eng


def test_reasoner_answers_from_her_own_discovered_law():
    nr = NativeReasoner(law_discovery=_engine())
    c = nr.reason("what is force when m = 3 and a = 4?")
    assert c is not None
    assert c.source == "law"
    assert c.verified
    assert "12" in c.answer
    # the trace names it as her OWN discovery, not an LLM
    assert any("discovered" in s.summary.lower() for s in c.steps)


def test_reasoner_recalls_the_governing_law_qualitatively():
    nr = NativeReasoner(law_discovery=_engine())
    c = nr.reason("what law relates force to m and a?")
    assert c is not None
    assert c.source == "law"


def test_reasoner_abstains_when_no_discovered_law_applies():
    nr = NativeReasoner(law_discovery=_engine())
    # a factual question with no matching law and no other faculty wired → honest abstain
    assert nr.reason("what is the capital of France?") is None


def test_no_law_engine_means_the_rung_simply_defers():
    nr = NativeReasoner()  # nothing wired
    assert nr.reason("what is force when m = 3 and a = 4?") is None


def test_settled_belief_answers_a_proposition():
    bm = BeliefModel()
    bm.beliefs["photosynthesis_requires_light"] = Belief(
        variable="photosynthesis_requires_light",
        statement="photosynthesis requires sunlight energy",
        verdict="supported", confidence=0.9, evidence_count=5, alpha=9.0, beta=1.0)
    nr = NativeReasoner(belief_model=bm)
    c = nr.reason("does photosynthesis require sunlight energy?")
    assert c is not None
    assert c.source == "law"
    assert c.answer.lower().startswith("yes")
