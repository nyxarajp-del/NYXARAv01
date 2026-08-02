"""Tests for the active immune system — guard/phagocytosis.py (Part G)."""
from __future__ import annotations

from nyxara.guard.phagocytosis import DefensePlaybook, Phagocyte
from nyxara.memory.graph import KnowledgeGraph

_MALICIOUS = "import ctypes; exec(eval('payload'))"


def test_engulf_new_threat_digests_and_immunizes():
    p = Phagocyte()
    res = p.engulf(_MALICIOUS, kind="attack")
    assert res.neutralized and res.recognized is False
    ag = res.antigen
    # the mechanism was reverse-engineered (never executed in-process)
    assert "call:exec" in ag.mechanisms and "import:ctypes" in ag.mechanisms
    assert ag.severity > 0.2 and len(p.playbook) == 1


def test_second_exposure_is_recognized_fast():
    p = Phagocyte()
    p.engulf(_MALICIOUS, kind="attack")
    res2 = p.engulf(_MALICIOUS, kind="attack")
    assert res2.recognized is True and res2.neutralized is True
    assert res2.antigen.hits == 2           # exposure count bumped
    assert len(p.playbook) == 1             # no duplicate antibody


def test_unparseable_toxic_code_still_fingerprinted():
    p = Phagocyte()
    res = p.engulf("this is >>> not <<< valid python !!!", kind="toxic_code")
    assert res.neutralized and "unparseable" in res.antigen.mechanisms


def test_playbook_mirrors_to_knowledge_graph():
    kg = KnowledgeGraph()
    p = Phagocyte(playbook=DefensePlaybook(knowledge_graph=kg))
    p.engulf(_MALICIOUS, kind="attack")
    # an antibody triple lands in the graph the shields consult
    assert any(t.predicate == "is_threat" for t in kg._triples.values())


def test_benign_code_is_low_severity():
    p = Phagocyte()
    res = p.engulf("def add(a, b):\n    return a + b", kind="toxic_code")
    assert res.antigen.mechanisms == [] and res.antigen.severity <= 0.2
