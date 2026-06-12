"""Tests for nyxara.mind.temporal — reasoning over remembered time.

Allen's interval algebra, precedence/lag ("X before Y", "X ~N later then Y"), and
periodicity ("repeats every week").
"""

from __future__ import annotations

from nyxara.mind.temporal import (DAY, WEEK, AllenRelation, Interval, Periodicity,
                                  Precedence, TemporalReasoner, allen_relation,
                                  describe_duration)


# -------------------- Allen's interval algebra -------------------- #
def test_allen_before_and_after():
    a, b = Interval(0, 10), Interval(20, 30)
    assert allen_relation(a, b) is AllenRelation.BEFORE
    assert allen_relation(b, a) is AllenRelation.AFTER


def test_allen_meets():
    assert allen_relation(Interval(0, 10), Interval(10, 20)) is AllenRelation.MEETS
    assert allen_relation(Interval(10, 20), Interval(0, 10)) is AllenRelation.MET_BY


def test_allen_overlaps_and_during_and_equals():
    assert allen_relation(Interval(0, 15), Interval(10, 25)) is AllenRelation.OVERLAPS
    assert allen_relation(Interval(12, 18), Interval(10, 25)) is AllenRelation.DURING
    assert allen_relation(Interval(10, 25), Interval(12, 18)) is AllenRelation.CONTAINS
    assert allen_relation(Interval(5, 9), Interval(5, 9)) is AllenRelation.EQUALS


def test_allen_starts_and_finishes():
    assert allen_relation(Interval(0, 5), Interval(0, 12)) is AllenRelation.STARTS
    assert allen_relation(Interval(8, 12), Interval(0, 12)) is AllenRelation.FINISHES


def test_allen_converse_is_consistent():
    a, b = Interval(0, 15), Interval(10, 25)
    assert allen_relation(a, b).converse is allen_relation(b, a)


def test_point_events_reduce_to_before():
    assert allen_relation(Interval(3, 3), Interval(9, 9)) is AllenRelation.BEFORE
    assert allen_relation(Interval(9, 9), Interval(9, 9)) is AllenRelation.EQUALS


# -------------------- ordering -------------------- #
def test_strictly_before_and_order_fraction():
    tr = TemporalReasoner()
    tr.observe("alarm", 100.0)
    tr.observe("alarm", 200.0)
    tr.observe("response", 300.0)
    assert tr.strictly_before("alarm", "response") is True
    assert tr.strictly_before("response", "alarm") is False
    assert tr.order_fraction("alarm", "response") == 1.0


# -------------------- precedence & lag -------------------- #
def test_precedence_detects_typical_lag():
    tr = TemporalReasoner()
    # an alert, then ~3 days later an outage — three times over
    for k in range(3):
        base = k * 30 * DAY
        tr.observe("alert", base)
        tr.observe("outage", base + 3 * DAY)
    p = tr.precedence("alert", "outage")
    assert isinstance(p, Precedence)
    assert p.support == 3
    assert p.ratio == 1.0
    assert abs(p.median_lag - 3 * DAY) < 1.0
    assert p.consistency > 0.95
    assert "3 days" in p.describe()


def test_causal_candidates_ranks_precedence():
    tr = TemporalReasoner()
    for k in range(3):
        base = k * 10 * DAY
        tr.observe("cause", base)
        tr.observe("effect", base + DAY)
    cands = tr.causal_candidates(min_support=2)
    assert cands and cands[0].cause == "cause" and cands[0].effect == "effect"


def test_no_precedence_without_following_effect():
    tr = TemporalReasoner()
    tr.observe("x", 500.0)
    tr.observe("y", 100.0)        # y is before x, never after
    assert tr.precedence("x", "y") is None


# -------------------- periodicity -------------------- #
def test_periodicity_detects_weekly_rhythm():
    tr = TemporalReasoner()
    for k in range(5):
        tr.observe("review", k * WEEK)
    p = tr.periodicity("review")
    assert isinstance(p, Periodicity)
    assert abs(p.period - WEEK) < 1.0
    assert p.confidence > 0.9
    assert "week" in p.describe()


def test_irregular_events_are_not_periodic():
    tr = TemporalReasoner()
    for t in (0.0, 1 * DAY, 9 * DAY, 9.5 * DAY, 40 * DAY):
        tr.observe("noise", t)
    p = tr.periodicity("noise")
    # high variance in the gaps -> low confidence (or filtered out of rhythms())
    assert p is None or p.confidence < 0.5
    assert tr.rhythms() == [] or all(r.label != "noise" for r in tr.rhythms())


def test_too_few_occurrences_is_none():
    tr = TemporalReasoner()
    tr.observe("once", 0.0)
    tr.observe("once", WEEK)
    assert tr.periodicity("once") is None


# -------------------- helpers -------------------- #
def test_describe_duration_snaps_to_named_units():
    assert describe_duration(WEEK) == "1 week"
    assert describe_duration(3 * DAY) == "3 days"
    assert describe_duration(2 * WEEK) == "2 weeks"
