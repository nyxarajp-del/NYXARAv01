"""Tests for nyxara.mind.world_model.TransferWorldModel — cross-domain transfer.

The transfer model trains ONE shared encoder over ``[state ⊕ learned action embedding]``
with a concept hierarchy, so knowledge moves between actions and domains instead of every
action owning a private, from-scratch model.
"""

from __future__ import annotations

import random

import pytest

np = pytest.importorskip("numpy")  # the transfer model requires numpy

from nyxara.mind.world_model import (  # noqa: E402
    TransferWorldModel,
    WorldModel,
    build_world_model,
)

_MOVES = {"left": -1.0, "right": 1.0, "stay": 0.0}


def _master(seed=0, n=600, **kw):
    """A model that has mastered the three base actions."""
    wm = TransferWorldModel(seed=seed, **kw)
    rng = random.Random(seed)
    for _ in range(n):
        a = rng.choice(list(_MOVES))
        x = rng.uniform(-10, 10)
        wm.observe((x,), a, (x + _MOVES[a],), reward=-abs(x + _MOVES[a]))
    return wm


def _add_action(wm, action, delta, n=8, seed=99):
    rng = random.Random(seed)
    for _ in range(n):
        x = rng.uniform(-10, 10)
        wm.observe((x,), action, (x + delta,), reward=-abs(x + delta))
    return wm


# -------------------- drop-in contract -------------------- #
def test_transfer_model_is_a_world_model():
    assert isinstance(TransferWorldModel(), WorldModel)     # inherits rollout/cf/intervene


def test_learns_base_dynamics():
    wm = _master()
    assert abs(wm.predict((4.0,), "left").next_state[0] - 3.0) < 0.3
    assert abs(wm.predict((2.0,), "right").next_state[0] - 3.0) < 0.3
    assert abs(wm.predict((2.0,), "stay").next_state[0] - 2.0) < 0.3
    # states stay native python floats (no numpy leakage into trajectories)
    assert type(wm.predict((4.0,), "left").next_state[0]) is float


# -------------------- the headline: cross-domain transfer -------------------- #
def test_new_action_transfers_from_a_known_sibling():
    # 'descend' is brand new and behaves exactly like 'left', with only 8 samples
    wm = _add_action(_master(), "descend", -1.0, n=8)
    p = wm.predict((4.0,), "descend")
    assert abs(p.next_state[0] - 3.0) < 0.4                 # inherited 'left''s −1 dynamics
    assert p.confidence > 0.5                               # and is confident via transfer
    # it landed in the same concept as 'left'
    assert wm.concepts.concept_of("descend") == wm.concepts.concept_of("left")


def test_transfer_beats_learning_from_scratch_at_equal_data():
    # same 8 samples of a new action, with vs without prior sibling knowledge to transfer from
    transferred = _add_action(_master(), "descend", -1.0, n=8).predict((4.0,), "descend")
    scratch = _add_action(TransferWorldModel(seed=0), "descend", -1.0, n=8).predict((4.0,),
                                                                                     "descend")
    # transfer is far more confident about the (correct) borrowed dynamics than 8 raw samples
    assert transferred.confidence > 0.5
    assert transferred.confidence > scratch.confidence + 0.3


def test_cross_domain_two_new_actions_at_once():
    wm = _master()
    _add_action(wm, "descend", -1.0, n=8, seed=1)           # ≡ left
    _add_action(wm, "ascend", 1.0, n=8, seed=2)             # ≡ right
    assert abs(wm.predict((4.0,), "descend").next_state[0] - 3.0) < 0.5
    assert abs(wm.predict((4.0,), "ascend").next_state[0] - 5.0) < 0.5
    assert wm.concepts.concept_of("descend") == wm.concepts.concept_of("left")
    assert wm.concepts.concept_of("ascend") == wm.concepts.concept_of("right")


# -------------------- honest uncertainty -------------------- #
def test_unknown_action_is_zero_confidence_noop():
    wm = _master()
    p = wm.predict((0.0,), "teleport")
    assert p.confidence == 0.0 and p.next_state == (0.0,)


def test_confidence_is_honest_out_of_distribution():
    wm = _master()
    near = wm.predict((4.0,), "left").confidence
    far = wm.predict((1000.0,), "left").confidence
    assert far < near and far < 0.1


def test_confidence_rises_as_an_action_gathers_its_own_data():
    few = _add_action(_master(), "descend", -1.0, n=2, seed=5).predict((4.0,), "descend")
    many = _add_action(_master(), "descend", -1.0, n=40, seed=5).predict((4.0,), "descend")
    assert many.confidence >= few.confidence


# -------------------- inherited imagination still works -------------------- #
def test_rollout_plans_a_path_home():
    wm = _master()

    def go_home(state):
        x = state[0]
        return "left" if x > 0.5 else ("right" if x < -0.5 else "stay")

    traj = wm.rollout((8.0,), go_home, steps=14)
    assert abs(traj.final_state[0]) < 1.5


def test_counterfactual_and_intervene():
    wm = _master()
    cf = wm.counterfactual((5.0,), ["left"] * 6, ["right"] * 6, steps=6)
    assert cf.better() == "a" and cf.reward_difference > 0
    base = wm.rollout((4.0,), ["left"] * 5, steps=5)
    iv = wm.intervene((4.0,), ["left"] * 5, at_step=2, action="right", steps=5)
    assert iv.final_state[0] > base.final_state[0]


# -------------------- symbolic fallback (strict superset) -------------------- #
def test_symbolic_states_fall_back_to_exact_memory():
    wm = _master()
    wm.observe(("here",), "warp", ("there",), reward=2.0)
    p = wm.predict(("here",), "warp")
    assert p.next_state == ("there",) and p.confidence == 1.0
    assert "warp" in wm.actions()


def test_len_and_stats():
    wm = _add_action(_master(), "descend", -1.0, n=8)
    s = wm.stats()
    assert s["backend"] == "shared-encoder+concept-hierarchy"
    assert s["concepts"] >= 1 and "descend" in s["numeric_actions"]
    assert len(wm) == 608
    assert "concept#" in wm.concept_report()


# -------------------- factory wiring -------------------- #
def test_factory_auto_prefers_transfer_with_numpy():
    wm = build_world_model("auto")
    assert isinstance(wm, TransferWorldModel)
    assert isinstance(build_world_model("transfer"), TransferWorldModel)
    # unknown kwargs are filtered per backend (callers may pass a superset safely)
    assert isinstance(build_world_model("transfer", k=2, ensemble=9), TransferWorldModel)
