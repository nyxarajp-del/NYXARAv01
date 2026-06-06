"""Tests for nyxara.senses.predictive."""

from __future__ import annotations

import pytest

from nyxara.senses.binding import Modality, Percept, Provenance
from nyxara.senses.predictive import (PerceptSurprise, PerceptualPredictor, SensoryPredictor,
                                      SurpriseResult, SymbolPredictor, SymbolSurprise,
                                      percept_features)


# -------------------- SensoryPredictor -------------------- #
def test_first_observation_zero_surprise():
    sp = SensoryPredictor()
    r = sp.observe([1.0, 2.0])
    assert r.surprise == 0.0 and r.n == 1 and not r.anomaly


def test_predict_none_before_training():
    assert SensoryPredictor().predict() is None


def test_predict_after_training():
    sp = SensoryPredictor()
    sp.observe([1.0, 1.0])
    sp.observe([2.0, 2.0])
    pred = sp.predict()
    assert pred is not None and len(pred) == 2


def test_predictable_trend_low_surprise():
    sp = SensoryPredictor(warmup=5)
    r = None
    for i in range(20):
        r = sp.observe([float(i)])
    assert r.surprise < 1.0


def test_shock_high_surprise_and_anomaly():
    sp = SensoryPredictor(warmup=5)
    for i in range(20):
        sp.observe([float(i)])
    shock = sp.observe([1000.0])
    assert shock.surprise > 5.0
    assert shock.attention > 0.9
    assert shock.anomaly or shock.novelty


def test_attention_monotonic_in_surprise():
    sp = SensoryPredictor()
    sp.observe([0.0])
    a = sp.observe([0.1]).attention
    sp2 = SensoryPredictor()
    sp2.observe([0.0])
    b = sp2.observe([5.0]).attention
    assert b > a


def test_dim_mismatch_raises():
    sp = SensoryPredictor()
    sp.observe([1.0, 2.0])
    with pytest.raises(ValueError):
        sp.observe([1.0])


def test_novelty_needs_warmup():
    sp = SensoryPredictor(warmup=10)
    # within warmup, novelty cannot fire yet
    results = [sp.observe([float(i)]) for i in range(5)]
    assert not any(r.novelty for r in results)


def test_reset():
    sp = SensoryPredictor()
    sp.observe([1.0])
    sp.observe([2.0])
    sp.reset()
    assert sp.predict() is None and sp.n == 0


def test_surprise_result_to_dict():
    sp = SensoryPredictor()
    sp.observe([1.0])
    d = sp.observe([2.0]).to_dict()
    assert "surprise" in d and "attention" in d and "novelty" in d


def test_steady_state_low_surprise():
    sp = SensoryPredictor(warmup=3)
    r = None
    for _ in range(15):
        r = sp.observe([5.0, 5.0])  # constant
    assert r.surprise < 0.5


# -------------------- SymbolPredictor -------------------- #
def test_symbol_first_observation():
    sy = SymbolPredictor()
    r = sy.observe("A")
    assert isinstance(r, SymbolSurprise) and r.n == 1


def test_repeated_symbol_low_surprise():
    sy = SymbolPredictor()
    r = None
    for _ in range(10):
        r = sy.observe("A")
    assert r.surprise < 1.0  # highly predictable


def test_rare_symbol_high_surprise():
    sy = SymbolPredictor()
    for _ in range(20):
        sy.observe("A")
    common = sy.observe("A")
    rare = sy.observe("Z")
    assert rare.surprise > common.surprise


def test_novel_symbol_flagged():
    sy = SymbolPredictor()
    for s in "AAAAAAAAAA":
        sy.observe(s)
    assert sy.observe("Q").novelty


def test_predict_most_likely():
    sy = SymbolPredictor()
    for _ in range(5):
        sy.observe("A")
    sy.observe("B")
    assert sy.predict() in ("A", "B")


def test_bigram_prediction():
    sy = SymbolPredictor()
    # A is always followed by B
    for _ in range(10):
        sy.observe("A")
        sy.observe("B")
    sy.observe("A")
    assert sy.predict() == "B"


def test_probability_sums_reasonably():
    sy = SymbolPredictor()
    for s in "AABAB":
        sy.observe(s)
    p_a = sy.probability("A")
    p_unseen = sy.probability("ZZZ")
    assert 0 < p_unseen < p_a <= 1.0


def test_symbol_reset():
    sy = SymbolPredictor()
    sy.observe("A")
    sy.reset()
    assert sy.predict() == "" and sy._n == 0


def test_symbol_surprise_to_dict():
    sy = SymbolPredictor()
    d = sy.observe("A").to_dict()
    assert "surprise" in d and "probability" in d


# -------------------- percept_features -------------------- #
def test_percept_features_shape():
    p = Percept(Modality.TEXT, "x", Provenance("owner", 0.9), fingerprint=1,
                salience=0.5, entities=["A"], tags=["t"])
    f = percept_features(p)
    assert len(f) == 6 and f[0] == 0.5 and f[1] == 0.9


def test_percept_features_bounded():
    p = Percept(Modality.IMAGE, "x", Provenance("s", 1.0), fingerprint=1,
                salience=1.0, entities=["a"] * 20, tags=["t"] * 20, count=100)
    f = percept_features(p)
    assert all(0.0 <= v <= 1.0 for v in f)


# -------------------- PerceptualPredictor -------------------- #
def _routine(i):
    return Percept(Modality.TEXT, f"r{i}", Provenance("owner", 0.9), fingerprint=i + 1,
                   salience=0.3)


def test_perceptual_observe_returns_result():
    pp = PerceptualPredictor()
    out = pp.observe(_routine(0))
    assert isinstance(out, PerceptSurprise)


def test_perceptual_off_pattern_high_attention():
    pp = PerceptualPredictor(warmup=4)
    for i in range(10):
        pp.observe(_routine(i))
    odd = Percept(Modality.IMAGE, "?!", Provenance("https://evil.com", 0.05),
                  fingerprint=999, salience=0.3, tags=["threat", "image"],
                  entities=["X", "Y"])
    out = pp.observe(odd)
    routine_out = pp.observe(_routine(11))
    assert out.attention > routine_out.attention


def test_perceptual_boost_salience():
    pp = PerceptualPredictor(warmup=4)
    for i in range(10):
        pp.observe(_routine(i))
    odd = Percept(Modality.AUDIO, "?", Provenance("x", 0.1), fingerprint=999,
                  salience=0.2, tags=["threat"])
    out = pp.observe(odd, boost_salience=True)
    assert odd.salience >= out.attention


def test_perceptual_no_boost_by_default():
    pp = PerceptualPredictor(warmup=4)
    for i in range(10):
        pp.observe(_routine(i))
    odd = Percept(Modality.AUDIO, "?", Provenance("x", 0.1), fingerprint=999, salience=0.2)
    pp.observe(odd)
    assert odd.salience == 0.2  # unchanged without boost_salience


def test_perceptual_modality_surprise():
    pp = PerceptualPredictor(warmup=4)
    for i in range(10):
        pp.observe(_routine(i))  # all TEXT
    out = pp.observe(Percept(Modality.AUDIO, "x", Provenance("s", 0.5), fingerprint=1))
    # a new modality is surprising on the symbolic channel
    assert out.modality_surprise > 0


def test_perceptual_reset():
    pp = PerceptualPredictor()
    pp.observe(_routine(0))
    pp.reset()
    assert pp.sensory.predict() is None and pp.symbols.predict() == ""


def test_percept_surprise_to_dict():
    pp = PerceptualPredictor()
    d = pp.observe(_routine(0)).to_dict()
    assert "surprise" in d and "attention" in d and "percept_id" in d
