"""Narrative continuum: turns weave into a causal graph; old decisions recall; abstains if nothing close."""

from __future__ import annotations

from nyxara.nyx5.continuum import NarrativeContinuum


def test_recall_old_decision():
    c = NarrativeContinuum(seed=1)
    c.weave("we will call the anchor Master JP", decision="naming: Master JP anchor")
    c.weave("some unrelated small talk about the weather")
    c.weave("the GLM-5 subordinate loop handles fallback", decision="GLM-5 subordinate loop")
    r = c.recall("what did we decide about the anchor naming")
    assert r is not None
    assert r.decision == "naming: Master JP anchor"


def test_decisions_collected():
    c = NarrativeContinuum(seed=2)
    c.weave("a", decision="d1")
    c.weave("b")
    c.weave("c", decision="d2")
    assert c.decisions() == ["d1", "d2"]


def test_causal_chain_links_backwards():
    c = NarrativeContinuum(seed=3)
    n0 = c.weave("first")
    c.weave("second")
    n2 = c.weave("third")
    chain = c.chain(n2.node_id)
    assert [n.text for n in chain] == ["third", "second", "first"]


def test_recall_abstains_when_empty():
    c = NarrativeContinuum(seed=4)
    assert c.recall("anything") is None
