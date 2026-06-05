"""Tests for nyxara.planning.voi."""

from __future__ import annotations

import pytest

from nyxara.planning.voi import (
    ActionType,
    DecisionContext,
    InfoSource,
    Recommendation,
    ValueOfInformation,
)


def _voi():
    return ValueOfInformation()


def _sources():
    return ValueOfInformation.default_sources()


# -------------------- EVPI / EVSI -------------------- #
def test_evpi_scales_with_stakes_and_uncertainty():
    voi = _voi()
    low = voi.evpi(DecisionContext(uncertainty=0.2, stakes=0.2))
    high = voi.evpi(DecisionContext(uncertainty=0.9, stakes=0.9))
    assert high > low


def test_evpi_irreversibility_amplifies():
    voi = _voi()
    rev = voi.evpi(DecisionContext(uncertainty=0.6, stakes=0.8, reversibility=1.0))
    irr = voi.evpi(DecisionContext(uncertainty=0.6, stakes=0.8, reversibility=0.0))
    assert irr > rev


def test_evsi_is_reliability_times_evpi():
    voi = _voi()
    ctx = DecisionContext(uncertainty=0.8, stakes=0.8)
    src = InfoSource("s", reliability=0.5)
    assert voi.evsi(src, ctx) == pytest.approx(0.5 * voi.evpi(ctx))


def test_net_value():
    voi = _voi()
    ctx = DecisionContext(uncertainty=0.8, stakes=0.8)
    src = InfoSource("s", reliability=0.5, cost=0.1)
    assert voi.net_value(src, ctx) == pytest.approx(voi.evsi(src, ctx) - 0.1)


# -------------------- ACT cases -------------------- #
def test_low_uncertainty_acts():
    r = _voi().decide(uncertainty=0.03, stakes=0.9, sources=_sources())
    assert r.action is ActionType.ACT


def test_low_stakes_acts():
    r = _voi().decide(uncertainty=0.9, stakes=0.03, sources=_sources())
    assert r.action is ActionType.ACT


def test_no_worthwhile_source_acts():
    poor = [InfoSource("costly", reliability=0.2, cost=0.95)]
    r = _voi().decide(uncertainty=0.7, stakes=0.6, sources=poor)
    assert r.action is ActionType.ACT


def test_no_sources_acts():
    r = _voi().decide(uncertainty=0.8, stakes=0.8, sources=[])
    assert r.action is ActionType.ACT


# -------------------- GATHER / ASK cases -------------------- #
def test_high_uncertainty_stakes_gathers_or_asks():
    r = _voi().decide(uncertainty=0.8, stakes=0.9, sources=_sources())
    assert r.action in (ActionType.GATHER, ActionType.ASK)
    assert r.net_value > 0


def test_picks_best_net_value_source():
    voi = _voi()
    # ask-the-Master is reliable and cheap -> best net value
    r = voi.decide(uncertainty=0.7, stakes=0.8, sources=_sources())
    assert r.source == "ask the Master"


def test_gather_when_only_gather_available():
    sources = [InfoSource("investigate", kind="gather", reliability=0.8, cost=0.1)]
    r = _voi().decide(uncertainty=0.8, stakes=0.9, sources=sources)
    assert r.action is ActionType.GATHER
    assert r.source == "investigate"


def test_unavailable_source_skipped():
    sources = [InfoSource("down", kind="gather", reliability=0.9, cost=0.05,
                          available=False)]
    r = _voi().decide(uncertainty=0.8, stakes=0.9, sources=sources)
    assert r.action is ActionType.ACT  # the only source is unavailable


# -------------------- owner-intent override -------------------- #
def test_owner_intent_uncertainty_forces_ask():
    r = _voi().decide(uncertainty=0.4, stakes=0.3, about_owner_intent=True,
                      sources=_sources())
    assert r.action is ActionType.ASK
    assert "will" in r.rationale


def test_owner_intent_low_uncertainty_no_override():
    # trivial uncertainty about intent doesn't force a needless question
    r = _voi().decide(uncertainty=0.05, stakes=0.3, about_owner_intent=True,
                      sources=_sources())
    assert r.action is ActionType.ACT


def test_should_ask_owner():
    voi = _voi()
    assert voi.should_ask_owner(DecisionContext(uncertainty=0.5, about_owner_intent=True))
    assert not voi.should_ask_owner(DecisionContext(uncertainty=0.5, about_owner_intent=False))


def test_owner_intent_ask_synthesizes_source_if_missing():
    # even with no explicit ask source, the override names "the Master"
    r = _voi().decide(uncertainty=0.5, stakes=0.5, about_owner_intent=True,
                      sources=[InfoSource("investigate", kind="gather")])
    assert r.action is ActionType.ASK
    assert r.source == "the Master"


# -------------------- recommend API -------------------- #
def test_recommend_returns_recommendation():
    r = _voi().recommend(DecisionContext(uncertainty=0.8, stakes=0.8), _sources())
    assert isinstance(r, Recommendation)


def test_recommendation_to_dict():
    r = _voi().decide(uncertainty=0.8, stakes=0.8, sources=_sources())
    d = r.to_dict()
    assert "action" in d and "evpi" in d and "net_value" in d


def test_default_sources():
    srcs = ValueOfInformation.default_sources()
    assert any(s.kind == "ask" for s in srcs)
    assert any(s.kind == "gather" for s in srcs)


def test_info_source_to_dict():
    d = InfoSource("x", reliability=0.5, cost=0.2).to_dict()
    assert d["reliability"] == 0.5
