"""Complementary Learning Systems — the fast+slow architecture, tested for real.

These tests prove the mechanism does what the neuroscience claims: pattern separation decorrelates
overlapping episodes, the hippocampus encodes in one shot while the cortex lags, sleep consolidates
fast→slow, congruent experience consolidates faster, synaptic homeostasis renormalises without
flipping decisions, REM generative rehearsal protects a task even after its episodes are evicted, and
the whole thing forgets far less than a single online learner — while staying a drop-in Learner and
keeping the loyalty core fail-closed.
"""

from __future__ import annotations

import json

import pytest

from nyxara.eval.continual import evaluate_cls
from nyxara.growth.cls import (
    ComplementaryLearningSystem,
    Neocortex,
    SparseProjector,
    _cosine_sparse,
)
from nyxara.growth.learn import Experience
from nyxara.guard.value_learning import IMMUTABLE_VALUES
from nyxara.kernel.errors import ValidationError


# --------------------------------------------------------------------------- #
# Pattern separation (dentate gyrus)
# --------------------------------------------------------------------------- #
def test_pattern_separation_is_deterministic_and_decorrelates():
    proj = SparseProjector(dim=256, k=16, seed=1)
    a = {"shared_a": 1.0, "shared_b": 1.0, "ctx_0": 1.0}
    b = {"shared_a": 1.0, "shared_b": 1.0, "ctx_1": 1.0}   # 2/3 features overlap
    ca, cb = proj.code(a), proj.code(b)
    assert proj.code(a) == ca, "codes must be deterministic (same input → same code)"
    assert len(ca) <= 16, "k-WTA keeps at most k active dimensions"
    # heavily overlapping inputs still map to *decorrelated* codes (that is the separation)
    assert _cosine_sparse(ca, cb) < 0.9
    # a totally different cue is more separated than a mostly-overlapping one
    far = proj.code({"totally": 1.0, "different": 1.0})
    assert _cosine_sparse(ca, far) <= _cosine_sparse(ca, cb) + 1e-9


# --------------------------------------------------------------------------- #
# One-shot fast encoding, slow cortex lags
# --------------------------------------------------------------------------- #
def test_hippocampus_encodes_one_shot_while_cortex_lags():
    cls = ComplementaryLearningSystem(fast_lr=0.6, slow_lr=0.05, seed=2)
    cls.record("act", {"novel": 1.0}, reward=1.0, task="A")
    assert cls.hippocampus.predict("act", {"novel": 1.0}) > 0.3   # learned in one shot
    assert abs(cls.neocortex.predict("act", {"novel": 1.0})) < 0.05  # cortex has not learned yet


def test_pattern_completion_confidence_tracks_familiarity():
    cls = ComplementaryLearningSystem(seed=3)
    for _ in range(3):
        cls.record("act", {"a": 1.0, "b": 1.0}, reward=1.0, task="A")
    assert cls.hippocampus.confidence({"a": 1.0, "b": 1.0}) > 0.8   # a seen cue completes strongly
    assert cls.hippocampus.confidence({"z": 1.0}) < 0.5            # an unseen cue does not


# --------------------------------------------------------------------------- #
# Sleep consolidates hippocampus → cortex
# --------------------------------------------------------------------------- #
def test_sleep_consolidates_fast_into_slow():
    cls = ComplementaryLearningSystem(fast_lr=0.6, slow_lr=0.1, seed=4)
    for _ in range(30):
        cls.record("act", {"novel": 1.0}, reward=1.0, task="A")
    before = cls.neocortex.predict("act", {"novel": 1.0})
    rep = None
    for _ in range(15):
        rep = cls.sleep(deep=False)
    after = cls.neocortex.predict("act", {"novel": 1.0})
    assert after > before + 0.2, "NREM replay must move knowledge into the cortex"
    assert rep is not None and rep.replayed_to_cortex > 0


def test_blended_value_uses_cortex_once_consolidated():
    cls = ComplementaryLearningSystem(fast_lr=0.6, slow_lr=0.1, seed=6)
    for _ in range(30):
        cls.record("act", {"f": 1.0}, reward=1.0, task="A")
    for _ in range(15):
        cls.sleep(deep=True)
    cls.consolidate()
    assert cls.value("act", {"f": 1.0}) > 0.5
    assert cls.best_action({"f": 1.0}, ["act", "other"]) == "act"


# --------------------------------------------------------------------------- #
# Schema-gated consolidation (Tse et al.)
# --------------------------------------------------------------------------- #
def test_schema_congruence_gates_consolidation():
    ctx = Neocortex(base_lr=0.1, congruence_gain=3.0, seed=7)
    base = [Experience("act", {"x": 1.0, "y": 1.0}, 1.0, task="A") for _ in range(5)]
    ctx.consolidate_batch(base)            # forms a schema over {x, y, act}
    congruent = Experience("act", {"x": 1.0, "y": 1.0}, 1.0, task="A")
    novel = Experience("other", {"p": 1.0, "q": 1.0}, 1.0, task="B")
    assert ctx.congruence(congruent) >= 0.5
    assert ctx.congruence(congruent) > ctx.congruence(novel)


# --------------------------------------------------------------------------- #
# Synaptic homeostasis (SHY)
# --------------------------------------------------------------------------- #
def test_homeostatic_downscale_preserves_ranking_and_anchors():
    ctx = Neocortex(base_lr=0.2, ewc_lambda=1.0, seed=8)
    for _ in range(20):
        ctx.consolidate_batch([Experience("act", {"f1": 1.0}, 1.0, task="A"),
                               Experience("act", {"f2": 1.0}, 0.4, task="A")])
    ctx.learner.consolidate()                     # freeze EWC anchors
    anchors_before = json.loads(json.dumps(dict(ctx.learner.model._w_star["act"])))
    w = ctx.learner.model._w["act"]
    hi_before, lo_before = w["f1"], w["f2"]
    assert hi_before > lo_before
    ctx.homeostatic_downscale(0.9)
    # magnitudes shrank, but the ranking (and therefore every decision) is preserved
    assert w["f1"] == pytest.approx(hi_before * 0.9, rel=1e-9)
    assert w["f1"] > w["f2"]
    # the frozen anchors are untouched by homeostasis
    assert dict(ctx.learner.model._w_star["act"]) == anchors_before


# --------------------------------------------------------------------------- #
# Generative pseudo-rehearsal (REM) protects an EVICTED task
# --------------------------------------------------------------------------- #
def test_rem_pseudo_rehearsal_protects_evicted_task():
    cls = ComplementaryLearningSystem(fast_lr=0.6, slow_lr=0.1, rem_pseudo_batch=24, seed=5)
    for _ in range(40):
        cls.record("act", {"shared": 1.0, "ctx_a": 1.0}, reward=1.0, task="A")
    for _ in range(15):
        cls.sleep(deep=True)
    cls.consolidate()
    a_before = cls.neocortex.predict("act", {"shared": 1.0, "ctx_a": 1.0})
    assert a_before > 0.3

    # wipe every hippocampal trace of task A — only the cortex + its prototype remain
    cls.hippocampus._traces.clear()
    cls.hippocampus.learner.buffer._buf.clear()
    cls.hippocampus.learner.buffer._reserve.clear()

    # learn an interfering task B; REM regenerates A from its prototype each night
    for _ in range(60):
        cls.record("act", {"shared": 1.0, "ctx_b": 1.0}, reward=-1.0, task="B")
        cls.sleep(deep=True)
    a_after = cls.neocortex.predict("act", {"shared": 1.0, "ctx_a": 1.0})
    assert a_after > 0.0, "the evicted task must not sign-flip — REM kept it alive"


# --------------------------------------------------------------------------- #
# The headline claim: two systems forget less than one (measured)
# --------------------------------------------------------------------------- #
def test_cls_forgets_less_than_single_system():
    report = evaluate_cls(n_tasks=4, n_train=60, seed=7)
    assert report.protected.forgetting < report.baseline.forgetting
    assert report.protected.average_accuracy > report.baseline.average_accuracy
    assert report.forgetting_reduction > 0.05


def test_cls_survives_eviction():
    report = evaluate_cls(n_tasks=4, n_train=60, seed=7, evict=True)
    assert report.protected.forgetting < report.baseline.forgetting


# --------------------------------------------------------------------------- #
# Drop-in Learner parity + persistence + character safety
# --------------------------------------------------------------------------- #
def test_cls_is_a_drop_in_learner():
    cls = ComplementaryLearningSystem(seed=1)
    # the surface the orchestrator relies on
    for attr in ("record", "update", "value", "best_action", "replay", "consolidate",
                 "to_dict", "load_dict", "attach_synapses", "report"):
        assert callable(getattr(cls, attr))
    cls.record("act", {"f": 1.0}, reward=1.0, task="A")
    # .model exposes the cortex's per-action weights the way _learner_weight_vector reads them
    assert isinstance(cls.model._w, dict)
    cls.replay(balanced=True)
    assert isinstance(cls.model._w["act"], dict)
    # .buffer is the episodic store used by _dominant_task_tag / ContinuousLearner
    assert "A" in cls.buffer.tasks()
    assert isinstance(cls.report(), dict)


def test_cls_persistence_round_trips_losslessly():
    cls = ComplementaryLearningSystem(fast_lr=0.5, slow_lr=0.1, seed=5)
    for _ in range(30):
        cls.record("act", {"shared": 1.0, "ctx_a": 1.0}, reward=1.0, task="A")
    for _ in range(10):
        cls.sleep(deep=True)
    cls.consolidate()
    blob = json.loads(json.dumps(cls.to_dict()))     # prove it is JSON-safe
    restored = ComplementaryLearningSystem(seed=99)   # deliberately different config
    assert restored.load_dict(blob)
    for feats in ({"shared": 1.0, "ctx_a": 1.0}, {"ctx_a": 1.0}):
        assert restored.neocortex.predict("act", feats) == pytest.approx(
            cls.neocortex.predict("act", feats), abs=1e-9)
    assert restored.neocortex._prototypes.keys() == cls.neocortex._prototypes.keys()
    assert set(IMMUTABLE_VALUES).issubset(restored.protected)


def test_cls_refuses_to_learn_over_the_loyalty_core():
    cls = ComplementaryLearningSystem(seed=1)
    for name in list(IMMUTABLE_VALUES)[:2]:
        with pytest.raises(ValidationError):
            cls.record(name, {"x": 1.0}, reward=-1.0)
    # a tampered checkpoint can never unprotect a character value (fail-closed via sub-learners)
    tampered = cls.to_dict()
    tampered["slow"]["protected"] = []
    tampered["fast"]["protected"] = []
    fresh = ComplementaryLearningSystem(seed=2)
    assert fresh.load_dict(tampered)
    assert set(IMMUTABLE_VALUES).issubset(fresh.protected)
