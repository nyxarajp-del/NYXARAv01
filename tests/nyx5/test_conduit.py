"""Symbiotic conduit: exact + fuzzy expansion, abstains (clarify) when ambiguous."""

from __future__ import annotations

from nyxara.nyx5.conduit import SymbioticConduit


def test_exact_expansion():
    c = SymbioticConduit()
    c.learn("gs", "git status")
    e = c.expand("gs")
    assert e.intent == "git status" and e.clarify is False and e.confidence == 1.0


def test_abstains_when_ambiguous():
    c = SymbioticConduit(ambiguity_gate=0.6)
    c.learn("gs", "git status")
    e = c.expand("totally unrelated phrase")
    assert e.clarify is True
    assert e.intent is None


def test_empty_phrasebook_abstains():
    c = SymbioticConduit()
    e = c.expand("anything")
    assert e.clarify is True


def test_roundtrip_preserves_phrasebook():
    c = SymbioticConduit()
    c.learn("gc", "git commit")
    c2 = SymbioticConduit()
    c2.load_dict(c.to_dict())
    assert c2.expand("gc").intent == "git commit"
