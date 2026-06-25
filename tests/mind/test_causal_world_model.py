"""Tests for nyxara.mind.causal_world_model — causation vs. correlation."""

from __future__ import annotations

import random

import pytest

from nyxara.mind.causal_world_model import (
    CAUSAL,
    COINCIDENTAL,
    CONFOUNDED,
    REVERSE,
    CausalLink,
    CausalWorldModel,
    Counterfactual,
    Event,
)


# --------------------------------------------------------------------------- #
# data builders — well-separated episodes so within-episode order is clean
# --------------------------------------------------------------------------- #
def _chain_model(seed: int = 1, n: int = 300) -> CausalWorldModel:
    """rain → wet_ground → slippery; birds_singing is independent background noise."""
    rng = random.Random(seed)
    m = CausalWorldModel(window=10.0)
    for k in range(n):
        base = k * 100.0
        if rng.random() < 0.5:
            m.observe("rain", at=base)
            if rng.random() < 0.9:
                m.observe("wet_ground", at=base + 1)
                if rng.random() < 0.85:
                    m.observe("slippery", at=base + 2)
        if rng.random() < 0.3:
            m.observe("birds_singing", at=base + rng.uniform(0, 90))
    m.discover()
    return m


def _confounded_model(seed: int = 2, n: int = 400) -> CausalWorldModel:
    """summer → {ice_cream, drowning}: ice_cream & drowning correlate but neither causes
    the other (independent effects of the common cause summer)."""
    rng = random.Random(seed)
    m = CausalWorldModel(window=10.0)
    for k in range(n):
        base = k * 100.0
        if rng.random() < 0.5:
            m.observe("summer", at=base)
            if rng.random() < 0.85:
                m.observe("ice_cream", at=base + 1)
            if rng.random() < 0.8:
                m.observe("drowning", at=base + 2)
        else:
            if rng.random() < 0.1:
                m.observe("ice_cream", at=base + 1)
            if rng.random() < 0.05:
                m.observe("drowning", at=base + 2)
    m.discover()
    return m


def _intervention_model(seed: int = 3, n: int = 400) -> CausalWorldModel:
    """A & B are driven by a *hidden* (unrecorded) common cause; intervening do(A) does not
    bring B, so A→B is exposed as non-causal."""
    rng = random.Random(seed)
    m = CausalWorldModel(window=10.0)
    for k in range(n):
        base = k * 100.0
        if rng.random() < 0.5:                       # hidden common cause fires (not recorded)
            if rng.random() < 0.9:
                m.observe("A", at=base + 1)
            if rng.random() < 0.9:
                m.observe("B", at=base + 2)
        if rng.random() < 0.4:                        # she forces A — B should not follow
            m.observe("A", at=base + 5, intervention=True)
            if rng.random() < 0.1:
                m.observe("B", at=base + 6)
    m.discover()
    return m


# --------------------------------------------------------------------------- #
# the core distinction: causation, not correlation
# --------------------------------------------------------------------------- #
def test_genuine_cause_is_found():
    m = _chain_model()
    assert m.is_causal("rain", "slippery").verdict == CAUSAL
    assert m.is_causal("wet_ground", "slippery").verdict == CAUSAL


def test_independent_noise_is_not_causal():
    m = _chain_model()
    v = m.is_causal("birds_singing", "slippery")
    assert v.verdict != CAUSAL
    assert not v.is_causal


def test_why_returns_ranked_causes():
    m = _chain_model()
    causes = m.why("slippery")
    assert causes, "slippery should have learned causes"
    labels = [l.cause for l in causes]
    assert "rain" in labels and "wet_ground" in labels
    assert "birds_singing" not in labels
    # confidence-sorted, all genuinely causal
    assert all(l.is_causal for l in causes)
    assert causes == sorted(causes, key=lambda l: l.confidence * abs(l.strength), reverse=True)


def test_confounded_pair_is_not_called_causal():
    m = _confounded_model()
    v = m.is_causal("ice_cream", "drowning")
    assert v.verdict == CONFOUNDED
    assert v.evidence.get("confounder") == "summer"
    assert not v.is_causal


def test_common_cause_is_causal():
    m = _confounded_model()
    assert m.is_causal("summer", "drowning").verdict == CAUSAL
    # and the only learned cause of drowning is summer, never ice_cream
    why = [l.cause for l in m.why("drowning")]
    assert "summer" in why
    assert "ice_cream" not in why


def test_intervention_overrides_spurious_correlation():
    m = _intervention_model()
    v = m.is_causal("A", "B")
    assert v.verdict != CAUSAL
    assert "interventional_lift" in v.evidence


def test_wrong_direction_is_not_causal():
    # cause cleanly precedes effect; the wrong-direction query must never be called causal
    m = CausalWorldModel(window=5.0)
    for k in range(60):
        base = k * 100.0
        m.observe("press_switch", at=base)
        m.observe("light_on", at=base + 1)
    m.discover()
    assert m.is_causal("press_switch", "light_on").verdict == CAUSAL
    assert m.is_causal("light_on", "press_switch").verdict != CAUSAL


def test_reverse_direction_detected():
    # B→A (A follows B), with the occasional B trailing A so the A→B window has weak signal
    # while B→A is strong: the is_causal(A, B) query must report REVERSE.
    rng = random.Random(5)
    m = CausalWorldModel(window=15.0)
    for k in range(120):
        base = k * 100.0
        m.observe("B", at=base)
        m.observe("A", at=base + 1)          # B reliably precedes A
        if rng.random() < 0.3:               # sometimes B trails A → weak forward A→B signal
            m.observe("B", at=base + 2)
    m.discover()
    assert m.is_causal("A", "B").verdict == REVERSE


def test_too_little_data_is_humble():
    m = CausalWorldModel()
    m.observe("x", at=0.0)
    m.observe("y", at=1.0)
    v = m.is_causal("x", "y")
    assert v.verdict == COINCIDENTAL
    assert v.confidence < 0.3


# --------------------------------------------------------------------------- #
# the SCM surface — reused from mind.strategies
# --------------------------------------------------------------------------- #
def test_counterfactual_contrast():
    m = _chain_model()
    cf = m.counterfactual("rain", "slippery", factual=1.0, counter=0.0)
    assert isinstance(cf, Counterfactual)
    # removing the cause should reduce the effect
    assert cf.delta < 0.0


def test_predict_effects_forward():
    m = _chain_model()
    effects = m.predict_effects("rain")
    assert isinstance(effects, dict)
    assert "slippery" in effects
    assert m.predict_effects("rain", target="slippery") != 0.0


def test_as_causal_graph_only_causal_edges():
    m = _confounded_model()
    edges = m.as_causal_graph()
    pairs = {(c, e) for c, e, _ in edges}
    assert ("summer", "drowning") in pairs
    assert ("ice_cream", "drowning") not in pairs


def test_explain_is_honest_when_unknown():
    m = CausalWorldModel()
    assert "no established cause" in m.explain("nonexistent").lower()


# --------------------------------------------------------------------------- #
# bookkeeping & persistence
# --------------------------------------------------------------------------- #
def test_discover_returns_causal_links_sorted():
    m = _chain_model()
    causal = m.discover()
    assert causal and all(isinstance(l, CausalLink) and l.is_causal for l in causal)
    assert causal == sorted(causal, key=lambda l: l.confidence * abs(l.strength), reverse=True)


def test_persistence_round_trip(tmp_path):
    m = _chain_model()
    before = {(l.cause, l.effect): l.verdict for l in m.links()}
    path = tmp_path / "cwm.json"
    m.save(str(path))

    reloaded = CausalWorldModel.load(str(path), window=10.0)
    assert reloaded.stats()["events"] == m.stats()["events"]
    reloaded.discover()
    after = {(l.cause, l.effect): l.verdict for l in reloaded.links()}
    assert after == before


def test_intervention_flag_recorded():
    m = CausalWorldModel()
    ev = m.observe("act", at=0.0, intervention=True)
    assert isinstance(ev, Event) and ev.intervention
    assert m._int_by_label.get("act") == [0.0]


def test_observe_many_shares_timestamp():
    m = CausalWorldModel()
    m.observe_many(["a", "b", "c"], at=5.0, interventions=["a"])
    assert m._by_label["a"] == [5.0] and m._by_label["b"] == [5.0]
    assert m._int_by_label.get("a") == [5.0]
    assert "b" not in m._int_by_label


def test_max_vars_bounded():
    m = CausalWorldModel(max_vars=10)
    for i in range(40):
        m.observe(f"label_{i}", at=float(i))
    assert len(m._by_label) <= 10


def test_confounder_screening_can_be_disabled():
    rng = random.Random(2)
    m = CausalWorldModel(window=10.0, confounder_screening=False)
    for k in range(400):
        base = k * 100.0
        if rng.random() < 0.5:
            m.observe("summer", at=base)
            if rng.random() < 0.85:
                m.observe("ice_cream", at=base + 1)
            if rng.random() < 0.8:
                m.observe("drowning", at=base + 2)
    m.discover()
    # without screening, the confounded pair is no longer demoted to CONFOUNDED
    assert m.is_causal("ice_cream", "drowning").verdict != CONFOUNDED


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
