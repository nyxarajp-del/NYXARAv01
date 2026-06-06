"""Tests for nyxara.growth.learn."""

from __future__ import annotations

import random

import pytest

from nyxara.growth.learn import Experience, Learner, LinearValueModel, ReplayBuffer
from nyxara.kernel.errors import ValidationError


# -------------------- Experience / ReplayBuffer -------------------- #
def test_experience_to_dict():
    e = Experience("a", {"f": 1.0}, 0.5, context="c")
    assert e.to_dict()["action"] == "a" and e.to_dict()["reward"] == 0.5


def test_buffer_add_len():
    b = ReplayBuffer()
    b.add(Experience("a", {"f": 1.0}, 1.0))
    assert len(b) == 1


def test_buffer_capacity():
    b = ReplayBuffer(capacity=3)
    for i in range(5):
        b.add(Experience("a", {"f": float(i)}, 1.0))
    assert len(b) == 3


def test_buffer_recent():
    b = ReplayBuffer()
    for i in range(5):
        b.add(Experience(f"a{i}", {"f": 1.0}, 1.0))
    assert [e.action for e in b.recent(2)] == ["a3", "a4"]


def test_buffer_sample_empty():
    assert ReplayBuffer().sample(5) == []


def test_buffer_sample_size():
    b = ReplayBuffer()
    for i in range(10):
        b.add(Experience("a", {"f": 1.0}, 1.0))
    assert len(b.sample(3, rng=random.Random(0))) == 3


def test_buffer_sample_more_than_available():
    b = ReplayBuffer()
    b.add(Experience("a", {"f": 1.0}, 1.0))
    assert len(b.sample(10, rng=random.Random(0))) == 1


def test_buffer_prioritized_favours_high_reward():
    b = ReplayBuffer()
    for i in range(50):
        b.add(Experience("a", {"f": 1.0}, reward=(5.0 if i == 0 else 0.0)))
    rng = random.Random(0)
    hits = sum(1 for _ in range(300) if b.sample(1, rng=rng)[0].reward == 5.0)
    assert hits > 6  # far above the ~6 expected under uniform sampling


def test_buffer_uniform_sampling():
    b = ReplayBuffer()
    for i in range(10):
        b.add(Experience("a", {"f": 1.0}, 1.0))
    s = b.sample(5, rng=random.Random(0), prioritized=False)
    assert len(s) == 5


# -------------------- LinearValueModel -------------------- #
def test_model_predict_zero_initial():
    assert LinearValueModel().predict("a", {"f": 1.0}) == 0.0


def test_model_update_moves_toward_reward():
    m = LinearValueModel()
    for _ in range(100):
        m.update("a", {"f": 1.0}, 1.0, lr=0.2)
    assert abs(m.predict("a", {"f": 1.0}) - 1.0) < 0.05


def test_model_update_returns_error():
    m = LinearValueModel()
    err = m.update("a", {"f": 1.0}, 1.0, lr=0.1)
    assert err == 1.0  # initial prediction 0


def test_model_importance_accumulates():
    m = LinearValueModel()
    for _ in range(5):
        m.update("a", {"f": 1.0}, 1.0, lr=0.1)
    assert m.importance("a", "f") > 0


def test_model_consolidate_anchors():
    m = LinearValueModel(ewc_lambda=3.0)
    for _ in range(50):
        m.update("a", {"f": 1.0}, 1.0, lr=0.2)
    m.consolidate()
    before = m.predict("a", {"f": 1.0})
    # try to push it the other way; the anchor resists
    for _ in range(50):
        m.update("a", {"f": 1.0}, -1.0, lr=0.2)
    after = m.predict("a", {"f": 1.0})
    assert abs(after - before) < abs(before - (-1.0))  # didn't fully flip


def test_model_no_ewc_flips_freely():
    m = LinearValueModel(ewc_lambda=0.0)
    for _ in range(50):
        m.update("a", {"f": 1.0}, 1.0, lr=0.2)
    m.consolidate()
    for _ in range(50):
        m.update("a", {"f": 1.0}, -1.0, lr=0.2)
    assert m.predict("a", {"f": 1.0}) < 0  # flipped


# -------------------- Learner: basic learning -------------------- #
def test_learner_value_converges():
    L = Learner(base_lr=0.2)
    for _ in range(100):
        L.record("act", {"f": 1.0}, 1.0)
    assert abs(L.value("act", {"f": 1.0}) - 1.0) < 0.1


def test_learner_best_action():
    L = Learner(base_lr=0.2)
    for _ in range(60):
        L.record("good", {"ctx": 1.0}, 1.0)
        L.record("bad", {"ctx": 1.0}, -1.0)
    assert L.best_action({"ctx": 1.0}, ["good", "bad"]) == "good"


def test_learner_records_to_buffer():
    L = Learner()
    L.record("a", {"f": 1.0}, 1.0)
    assert len(L.buffer) == 1


def test_learner_update_no_buffer():
    L = Learner()
    L.update("a", {"f": 1.0}, 1.0)
    assert len(L.buffer) == 0  # update() alone doesn't store


def test_learner_returns_error():
    L = Learner()
    assert L.update("a", {"f": 1.0}, 1.0) == 1.0


# -------------------- protected core -------------------- #
def test_cannot_learn_protected_action():
    L = Learner()
    with pytest.raises(ValidationError):
        L.update("loyalty_to_master", {"f": 1.0}, -1.0)


def test_cannot_use_protected_feature():
    L = Learner()
    with pytest.raises(ValidationError):
        L.update("act", {"corrigibility": 1.0}, -1.0)


def test_custom_protected_set():
    L = Learner(protected=["secret_skill"])
    with pytest.raises(ValidationError):
        L.update("secret_skill", {"f": 1.0}, 1.0)
    L.update("loyalty_to_master", {"f": 1.0}, 1.0)  # not protected in this learner


# -------------------- learning-rate schedule -------------------- #
def test_lr_constant_by_default():
    L = Learner(base_lr=0.5)
    lr0 = L.lr()
    for _ in range(10):
        L.update("a", {"f": 1.0}, 1.0)
    assert L.lr() == lr0


def test_lr_decays():
    L = Learner(base_lr=1.0, lr_decay=0.1)
    lr0 = L.lr()
    for _ in range(20):
        L.update("a", {"f": 1.0}, 1.0)
    assert L.lr() < lr0


# -------------------- catastrophic forgetting protection -------------------- #
def _train(L, feats, reward, steps):
    for _ in range(steps):
        L.record("act", feats, reward)


def test_plain_learner_forgets():
    A = {"shared": 1.0, "a": 1.0}
    B = {"shared": 1.0, "b": 1.0}
    L = Learner(base_lr=0.1, seed=1)
    _train(L, A, 1.0, 120)
    _train(L, B, -1.0, 200)
    assert abs(L.value("act", A) - 1.0) > 0.2  # forgot A


def test_ewc_protects():
    A = {"shared": 1.0, "a": 1.0}
    B = {"shared": 1.0, "b": 1.0}
    plain = Learner(base_lr=0.1, seed=1)
    _train(plain, A, 1.0, 120)
    _train(plain, B, -1.0, 200)

    ewc = Learner(base_lr=0.1, ewc_lambda=3.0, seed=1)
    _train(ewc, A, 1.0, 120)
    ewc.consolidate()
    _train(ewc, B, -1.0, 200)

    assert abs(ewc.value("act", A) - 1.0) < abs(plain.value("act", A) - 1.0)


def test_replay_protects():
    A = {"shared": 1.0, "a": 1.0}
    B = {"shared": 1.0, "b": 1.0}
    plain = Learner(base_lr=0.1, seed=2)
    _train(plain, A, 1.0, 120)
    _train(plain, B, -1.0, 200)

    rep = Learner(base_lr=0.1, seed=2)
    _train(rep, A, 1.0, 120)
    for _ in range(200):
        rep.record("act", B, -1.0)
        rep.replay(8)

    assert abs(rep.value("act", A) - 1.0) < abs(plain.value("act", A) - 1.0)


def test_replay_returns_count():
    L = Learner()
    for _ in range(10):
        L.record("a", {"f": 1.0}, 1.0)
    assert L.replay(5) == 5


def test_replay_empty_buffer():
    assert Learner().replay(5) == 0


# -------------------- reporting -------------------- #
def test_report():
    L = Learner()
    L.record("a", {"f": 1.0}, 1.0)
    r = L.report()
    assert r["steps"] == 1 and r["buffer"] == 1 and "protected" in r
