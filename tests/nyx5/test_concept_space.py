"""Concept collapse: role query recovers filler; abstains when not confident."""

from __future__ import annotations

from nyxara.nyx5.concept_space import ConceptCollapser


def test_role_query_recovers_filler():
    c = ConceptCollapser(dim=10000, seed=1)
    record = {"capital": "washington", "currency": "dollar", "country": "usa"}
    got = c.query(record, "currency")
    assert got is not None and got[0] == "dollar"


def test_abstains_on_unknown_role():
    c = ConceptCollapser(dim=10000, seed=2, min_confidence=0.3)
    record = {"capital": "paris", "country": "france"}
    got = c.query(record, "currency")       # role not in the record
    assert got is None                       # honest abstain, no invented filler


def test_analogy_transports_mapping():
    c = ConceptCollapser(dim=10000, seed=3, min_confidence=0.05)
    source = {"country": "usa", "currency": "dollar"}
    target = {"country": "mexico", "currency": "peso"}
    got = c.analogy(source, target, "currency")
    # transported mapping should land near 'peso' among known fillers
    assert got is not None
