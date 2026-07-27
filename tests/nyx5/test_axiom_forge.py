"""Sub-axiomatic engine: contradiction is contained (no explosion), cap respected."""

from __future__ import annotations

from nyxara.nyx5.axiom_forge import SubAxiomaticEngine, Truth


def test_contradiction_yields_both_not_explosion():
    e = SubAxiomaticEngine()
    s = e.forge("paradox")
    s.assert_("p", Truth.TRUE)
    s.assert_("p", Truth.FALSE)
    assert s.value("p") == Truth.BOTH
    assert s.is_consistent() is False


def test_unrelated_prop_survives_a_contradiction():
    e = SubAxiomaticEngine()
    s = e.forge("sys")
    s.assert_("p", Truth.TRUE)
    s.assert_("p", Truth.FALSE)      # local contradiction
    s.assert_("q", Truth.TRUE)
    # classical logic would prove q AND not-q from the contradiction; paraconsistent does not
    assert e.entails(s, "q") is True
    assert e.evaluate(s, "r") == Truth.NEITHER   # unstated stays honestly unknown


def test_boolean_evaluation():
    e = SubAxiomaticEngine()
    s = e.forge("b")
    s.assert_("a", Truth.TRUE)
    s.assert_("b", Truth.FALSE)
    assert e.evaluate(s, ("and", "a", "b")) == Truth.FALSE
    assert e.evaluate(s, ("or", "a", "b")) == Truth.TRUE
    assert e.evaluate(s, ("not", "b")) == Truth.TRUE


def test_system_cap():
    e = SubAxiomaticEngine(max_systems=2)
    assert e.forge("s1") is not None
    assert e.forge("s2") is not None
    assert e.forge("s3") is None     # cap reached
