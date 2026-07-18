"""Tests: ModelBasedReasoner now scores by the real single objective (EFE).

The exact inversion of the old bug: `epistemic` used to be the model's
confidence (which repels the agent from anything unknown); it is now computed
expected information gain (which attracts it). Reward is absorbed into the
preference prior C — not a separate objective term.
"""

from __future__ import annotations

import random

from nyxara.mind.reasoner import ModelBasedReasoner, ReasoningQuery
from nyxara.mind.world_model import WorldModel

LEFT = ["left"] * 6
RIGHT = ["right"] * 6


def _step(x: float, a: str) -> float:
    return {"left": x - 1.0, "right": x + 1.0, "stay": x}[a]


def _half_world(*, reward: float = 0.0) -> WorldModel:
    wm = WorldModel(k=3, distance_scale=2.0)
    rng = random.Random(0)
    for _ in range(400):
        x = rng.uniform(-10.0, 0.0)
        a = rng.choice(["left", "right", "stay"])
        wm.observe((x,), a, (_step(x, a),), reward=reward)
    return wm


def test_epistemic_is_info_gain_not_confidence():
    mbr = ModelBasedReasoner(_half_world())
    q = ReasoningQuery(start=(-1.0,), horizon=6, candidates=[LEFT, RIGHT])
    scores = {tuple(s.policy): s for s in mbr.evaluate(q)}
    # the unknown region carries MORE epistemic value (old code: less "confidence")
    assert scores[tuple(RIGHT)].epistemic > scores[tuple(LEFT)].epistemic


def test_zero_reward_still_explores():
    """No reward_fn, no preference — curiosity through the reasoner path too."""
    mbr = ModelBasedReasoner(_half_world())
    q = ReasoningQuery(start=(-1.0,), horizon=6, candidates=[LEFT, RIGHT])
    c = mbr.reason(q)
    assert c.answer == RIGHT


def test_reward_folds_into_preference():
    mbr = ModelBasedReasoner(_half_world(), epistemic_weight=0.0)
    q = ReasoningQuery(start=(-5.0,), horizon=3,
                       candidates=[LEFT[:3], RIGHT[:3]],
                       reward_fn=lambda s, a, ns: 1.0 if a == "right" else 0.0)
    assert mbr.reason(q).answer == RIGHT[:3]
    # conflicting reward vs preference target: raising the C precision
    # (preference_weight) shifts the winner — both live in ONE channel (ln C)
    q2 = ReasoningQuery(start=(-5.0,), horizon=3, preference=(-8.0,),
                        candidates=[LEFT[:3], RIGHT[:3]],
                        reward_fn=lambda s, a, ns: 1.0 if a == "right" else 0.0)
    weak_pref = ModelBasedReasoner(_half_world(), epistemic_weight=0.0,
                                   preference_weight=0.01)
    strong_pref = ModelBasedReasoner(_half_world(), epistemic_weight=0.0,
                                     preference_weight=5.0)
    assert weak_pref.reason(q2).answer == RIGHT[:3]    # reward dominates
    assert strong_pref.reason(q2).answer == LEFT[:3]   # preference dominates


def test_conclusion_confidence_still_model_coverage():
    dense = ModelBasedReasoner(_half_world())
    q_dense = ReasoningQuery(start=(-5.0,), preference=(-8.0,), horizon=4,
                             candidates=[LEFT[:4], RIGHT[:4]])
    c_dense = dense.reason(q_dense)
    empty = ModelBasedReasoner(WorldModel())
    q_empty = ReasoningQuery(start=(-5.0,), preference=(-8.0,), horizon=4,
                             candidates=[LEFT[:4], RIGHT[:4]])
    c_empty = empty.reason(q_empty)
    assert c_dense.confidence > c_empty.confidence
    assert c_empty.confidence <= 0.05                  # an empty model claims nothing


def test_posterior_over_candidates():
    mbr = ModelBasedReasoner(_half_world())
    q = ReasoningQuery(start=(-1.0,), horizon=6, candidates=[LEFT, RIGHT])
    post = mbr.posterior(q)
    assert len(post) == 2
    assert sum(p for _, p in post) > 0.99


def test_grows_candidates_when_none_given():
    """Sophisticated planning: start + world model but NO candidates — the engine
    grows policies by EFE-pruned tree search instead of returning nothing."""
    mbr = ModelBasedReasoner(_half_world())
    q = ReasoningQuery(start=(-1.0,), horizon=3)       # candidates=None
    assert mbr.applicability(q) > 0.0
    c = mbr.reason(q)
    assert c.answer is not None
    assert len(c.answer) >= 1
