"""Tests for lifelong persistence of the reward learner (nyxara.growth.learn).

These lock in the invariant that makes continuous learning *lifelong* rather than
per-session: the learned weights, the frozen EWC anti-forgetting anchors, and the replay
buffer all round-trip losslessly through ``to_dict``/``load_dict``, and the loyalty core can
never be un-protected by a tampered checkpoint.
"""

from __future__ import annotations

import json

from nyxara.growth.learn import Experience, Learner, LinearValueModel, ReplayBuffer
from nyxara.guard.value_learning import IMMUTABLE_VALUES


def _train_two_interfering_tasks() -> Learner:
    """Two tasks that share a feature by construction, so they provably interfere."""
    L = Learner(base_lr=0.2, ewc_lambda=2.0, task_reserve=8, der_alpha=0.5, seed=3)
    for _ in range(60):
        L.record("act_a", {"shared": 1.0, "only_a": 1.0}, reward=1.0, task="A")
    L.consolidate()
    for _ in range(60):
        L.record("act_b", {"shared": 1.0, "only_b": 1.0}, reward=-1.0, task="B")
    return L


def test_linear_value_model_round_trip():
    m = LinearValueModel(ewc_lambda=1.5)
    for _ in range(30):
        m.update("greet", {"with_master": 1.0}, reward=1.0, lr=0.2)
    m.consolidate()
    blob = json.loads(json.dumps(m.to_dict()))     # JSON-safe
    m2 = LinearValueModel()
    assert m2.load_dict(blob)
    assert abs(m.predict("greet", {"with_master": 1.0})
               - m2.predict("greet", {"with_master": 1.0})) < 1e-12
    assert abs(m.importance("greet", "with_master")
               - m2.importance("greet", "with_master")) < 1e-12
    assert m2.ewc_lambda == m.ewc_lambda


def test_replay_buffer_round_trip():
    b = ReplayBuffer(capacity=100, task_reserve=5, seed=1)
    for i in range(40):
        b.add(Experience(f"a{i%3}", {"f": float(i)}, reward=float(i % 2), task=f"t{i%3}"))
    blob = json.loads(json.dumps(b.to_dict()))
    b2 = ReplayBuffer()
    assert b2.load_dict(blob)
    assert len(b2) == len(b)
    assert b2.counts_by_task() == b.counts_by_task()
    assert b2.tasks() == b.tasks()


def test_learner_round_trip_preserves_predictions_and_anchors():
    L = _train_two_interfering_tasks()
    blob = json.loads(json.dumps(L.to_dict()))
    L2 = Learner(base_lr=0.99, seed=99)            # deliberately different config
    assert L2.load_dict(blob)
    for act, feats in (("act_a", {"shared": 1.0, "only_a": 1.0}),
                       ("act_b", {"shared": 1.0, "only_b": 1.0})):
        assert abs(L.value(act, feats) - L2.value(act, feats)) < 1e-12
    assert L2._steps == L._steps
    assert len(L2.buffer) == len(L.buffer)
    assert L2.base_lr == L.base_lr                 # config restored from the checkpoint
    assert L2.der_alpha == L.der_alpha
    assert abs(L2.model.importance("act_a", "shared")
               - L.model.importance("act_a", "shared")) < 1e-12


def test_tampered_checkpoint_cannot_unprotect_the_core():
    L = Learner(seed=1)
    tampered = L.to_dict()
    tampered["protected"] = []                     # attacker tries to strip protection
    L2 = Learner(seed=2)
    L2.load_dict(tampered)
    assert set(IMMUTABLE_VALUES).issubset(L2.protected)
    # and the guard still fires after a tampered load
    import pytest
    from nyxara.kernel.errors import ValidationError
    with pytest.raises(ValidationError):
        L2.update("loyalty_to_master", {"x": 1.0}, reward=-1.0)


def test_continued_learning_after_restore():
    """After a restore, learning simply continues — the step counter and weights carry on."""
    L = _train_two_interfering_tasks()
    steps_before = L._steps
    blob = json.loads(json.dumps(L.to_dict()))
    L2 = Learner(base_lr=0.2, ewc_lambda=2.0, seed=3)
    L2.load_dict(blob)
    for _ in range(10):
        L2.record("act_a", {"shared": 1.0, "only_a": 1.0}, reward=1.0, task="A")
    assert L2._steps == steps_before + 10
