"""Tests for nyxara.mind.role_council — *computed* offline calibration.

The offline council used to stamp every role analysis with a hardcoded 0.55 confidence.
These tests prove the confidence is now genuinely computed: it varies per role and per
stimulus (driven by inter-role agreement × weight) and stays in an honest mid-band.
"""

from __future__ import annotations

from nyxara.mind.role_council import (
    RoleCouncil,
    _agreement_of,
    _offline_confidence,
    _offline_role_analyses,
)


def test_offline_confidence_is_not_a_constant():
    rs = _offline_role_analyses("Why does ice float on water?")
    confs = [round(c, 4) for _n, _t, c, _w in rs]
    # the old behaviour returned the same 0.55 for all six roles
    assert len(set(confs)) > 1, f"confidences should vary across roles, got {confs}"
    assert all(0.2 <= c <= 0.8 for c in confs), confs


def test_offline_confidence_varies_with_stimulus():
    a = tuple(round(c, 4) for _n, _t, c, _w in _offline_role_analyses("why is the sky blue"))
    b = tuple(round(c, 4) for _n, _t, c, _w in
              _offline_role_analyses("should I deploy the model to production tonight"))
    assert a != b, "different questions should yield different calibrated confidences"


def test_agreement_is_one_when_alone_and_bounded_otherwise():
    assert _agreement_of("only me here", []) == 1.0
    val = _agreement_of("the cat sat", ["the cat ran", "a dog barked"])
    assert 0.0 <= val <= 1.0


def test_higher_weight_and_agreement_lifts_confidence():
    others = ["analyse the topic carefully", "analyse the topic with care"]
    high = _offline_confidence("analyse the topic carefully now", others, weight=0.9)
    low = _offline_confidence("totally unrelated gibberish xyzzy", others, weight=0.4)
    assert high > low


def test_council_convene_offline_confidence_in_band():
    # no LLM provided -> deterministic offline path; confidence must be honest, not fixed
    council = RoleCouncil(llm=None)
    cand = council.convene("How should NYXARA prioritise the Master's requests?")
    assert 0.2 <= cand.confidence <= 0.8
    assert cand.confidence > 0
