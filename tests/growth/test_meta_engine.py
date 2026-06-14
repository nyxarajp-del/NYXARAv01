"""Tests for nyxara.growth.meta_engine — the Meta-Learning Engine."""

from __future__ import annotations

from nyxara.growth.meta_engine import (
    MetaDimension,
    MetaLearningEngine,
    MetaReport,
)


def _engine() -> MetaLearningEngine:
    return MetaLearningEngine(window=20)


# --------------------------------------------------------------------------- #
# observe + trend
# --------------------------------------------------------------------------- #
def test_observe_accumulates_samples():
    eng = _engine()
    for _ in range(5):
        eng.observe(MetaDimension.LEARNING, 0.5)
    assert eng.report().samples == 5


def test_observe_clamps_score():
    eng = _engine()
    eng.observe(MetaDimension.LEARNING, 5.0)
    eng.observe(MetaDimension.LEARNING, -3.0)
    levels = [s.score for s in eng._samples["learning"]]
    assert all(0.0 <= s <= 1.0 for s in levels)


def test_observe_unknown_dimension_is_ignored():
    eng = _engine()
    assert eng.observe("not_a_dimension", 0.5) is None
    assert eng.report().samples == 0


def test_trend_positive_for_improving_dimension():
    eng = _engine()
    for i in range(20):
        eng.observe(MetaDimension.REASONING, 0.3 + 0.03 * i)
    assert eng.trend(MetaDimension.REASONING) > 0.0


def test_trend_negative_for_declining_dimension():
    eng = _engine()
    for i in range(20):
        eng.observe(MetaDimension.MEMORY, 0.9 - 0.03 * i)
    assert eng.trend(MetaDimension.MEMORY) < 0.0


def test_trend_zero_without_enough_samples():
    eng = _engine()
    eng.observe(MetaDimension.PREDICTION, 0.5)
    assert eng.trend(MetaDimension.PREDICTION) == 0.0


# --------------------------------------------------------------------------- #
# recommend
# --------------------------------------------------------------------------- #
def test_recommend_skips_improving_dimension():
    eng = _engine()
    for i in range(20):
        eng.observe(MetaDimension.REASONING, 0.3 + 0.03 * i)
    by_dim = {r.dimension for r in eng.recommend()}
    assert "reasoning" not in by_dim


def test_recommend_emits_for_declining_dimension():
    eng = _engine()
    for i in range(20):
        eng.observe(MetaDimension.MEMORY, 0.9 - 0.03 * i)
    by_dim = {r.dimension for r in eng.recommend()}
    assert "memory" in by_dim


def test_recommend_emits_for_plateaued_dimension():
    eng = _engine()
    for _ in range(20):
        eng.observe(MetaDimension.LEARNING, 0.5)
    by_dim = {r.dimension for r in eng.recommend()}
    assert "learning" in by_dim


def test_recommend_needs_minimum_evidence():
    eng = _engine()
    for _ in range(3):
        eng.observe(MetaDimension.LEARNING, 0.5)
    assert eng.recommend() == []


# --------------------------------------------------------------------------- #
# guardrails
# --------------------------------------------------------------------------- #
def test_character_touching_knob_is_dropped():
    assert MetaLearningEngine._is_safe("base_lr", "settle faster")
    assert not MetaLearningEngine._is_safe("loyalty", "tune obedience")
    assert not MetaLearningEngine._is_safe("disable_safety", "go faster")


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #
class _StubLearner:
    base_lr = 0.1


class _StubMeta:
    def __init__(self) -> None:
        self.records: list = []

    def record(self, *a, **k) -> None:
        self.records.append((a, k))


class _StubCore:
    def __init__(self) -> None:
        self.learner = _StubLearner()
        self.meta = _StubMeta()
        self.consolidate_every = 50
        self.prediction_engine = type("PE", (), {})()


def test_apply_lowers_consolidation_for_weak_memory():
    eng = _engine()
    for i in range(20):
        eng.observe(MetaDimension.MEMORY, 0.9 - 0.03 * i)
    core = _StubCore()
    eng.recommend()
    eng.apply(core)
    assert core.consolidate_every < 50


def test_apply_keeps_learning_rate_in_bounds():
    eng = _engine()
    for _ in range(20):
        eng.observe(MetaDimension.LEARNING, 0.5)
    core = _StubCore()
    eng.recommend()
    eng.apply(core)
    assert 0.001 <= core.learner.base_lr <= 1.0


def test_apply_is_safe_on_empty_core():
    eng = _engine()
    for _ in range(20):
        eng.observe(MetaDimension.LEARNING, 0.5)

    class _Empty:
        pass

    eng.recommend()
    # must not raise even when the core exposes none of the expected knobs
    assert eng.apply(_Empty()) == [] or isinstance(eng.apply(_Empty()), list)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def test_report_shape():
    eng = _engine()
    eng.observe(MetaDimension.REASONING, 0.7)
    rep = eng.report()
    assert isinstance(rep, MetaReport)
    assert set(rep.trends) == {d.value for d in MetaDimension}
    assert rep.samples == 1


def test_summary_contains_levels_and_applied():
    eng = _engine()
    eng.observe(MetaDimension.REASONING, 0.7)
    summ = eng.summary()
    assert "levels" in summ and "applied" in summ and "trends" in summ


# --------------------------------------------------------------------------- #
# wiring into NyxaraCore
# --------------------------------------------------------------------------- #
def test_engine_is_wired_into_core():
    from nyxara import NyxaraCore
    core = NyxaraCore()
    assert isinstance(core.meta_learning_engine, MetaLearningEngine)
    assert "meta_learning" in core.report()


def test_turns_produce_samples_end_to_end():
    from nyxara import NyxaraCore
    core = NyxaraCore()
    for q in ("hello", "what is 2+2", "thanks"):
        core.process(q)
    assert core.report()["meta_learning"]["samples"] > 0
