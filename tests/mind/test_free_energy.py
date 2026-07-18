"""Tests for nyxara.mind.free_energy — the single active-inference objective.

The headline proofs:
* perception and action minimise the SAME free energy (one shared instance),
* curiosity is EMERGENT: an agent with zero reward and zero preference still
  explores the unknown region, because the epistemic term of EFE drives it,
* preferences alone exploit; both together balance.
"""

from __future__ import annotations

import random

import pytest

from nyxara.mind.free_energy import (
    FreeEnergyEngine,
    PolicyEvaluation,
    PreferenceModel,
    _policy_key,
    _softmax,
)
from nyxara.mind.predictive_core import PredictiveCore
from nyxara.mind.world_model import WorldModel

LEFT = ["left"] * 6
RIGHT = ["right"] * 6
STAY = ["stay"] * 6


def _step(x: float, a: str) -> float:
    return {"left": x - 1.0, "right": x + 1.0, "stay": x}[a]


def _half_world(*, reward: float = 0.0) -> WorldModel:
    """±1 walk world trained ONLY on x ∈ [−10, 0] — x > 0 is unknown territory."""
    wm = WorldModel(k=3, distance_scale=2.0)
    rng = random.Random(0)
    for _ in range(400):
        x = rng.uniform(-10.0, 0.0)
        a = rng.choice(["left", "right", "stay"])
        wm.observe((x,), a, (_step(x, a),), reward=reward)
    return wm


# -------------------- the single objective -------------------- #
def test_perception_and_action_share_free_energy():
    core = PredictiveCore(belief=[0.0], preference=[-5.0])
    eng = FreeEnergyEngine(predictive=core, world_model=_half_world(),
                          preferences=PreferenceModel(target=[-5.0]))
    assert eng.predictive is core                      # literally one instance
    # perception minimises variational free energy on that instance
    F0 = eng.variational_free_energy([-5.0])
    for _ in range(20):
        eng.perceive([-5.0])
    assert eng.variational_free_energy([-5.0]) < F0
    # action reads the SAME preference target through the SAME engine
    choice = eng.select((-1.0,), [LEFT, RIGHT])
    assert list(choice.policy) == LEFT                 # toward the shared preference


def test_curiosity_emerges_with_zero_preference():
    """The bolted-on-RL-free exploration proof: no reward, no preference target —
    the agent still seeks the unknown region because information gain lowers EFE."""
    eng = FreeEnergyEngine(world_model=_half_world(), preferences=PreferenceModel())
    evals = {_policy_key(e.policy): e
             for e in eng.evaluate_policies((-1.0,), [LEFT, RIGHT])}
    assert evals[repr(RIGHT)].epistemic > evals[repr(LEFT)].epistemic
    assert evals[repr(RIGHT)].expected_free_energy \
        < evals[repr(LEFT)].expected_free_energy
    choice = eng.select((-1.0,), [LEFT, RIGHT])
    assert list(choice.policy) == RIGHT


def test_preference_only_exploits():
    eng = FreeEnergyEngine(world_model=_half_world(), epistemic_weight=0.0,
                           preferences=PreferenceModel(target=[-5.0]))
    choice = eng.select((-1.0,), [LEFT, RIGHT])
    assert list(choice.policy) == LEFT


def test_combined_balances():
    """With both channels live, EFE ordering matches the recomputed arithmetic and
    the posterior spreads mass over both candidates."""
    eng = FreeEnergyEngine(world_model=_half_world(),
                           preferences=PreferenceModel(target=[-3.0], precision=0.05))
    evals = eng.evaluate_policies((-1.0,), [LEFT, RIGHT])
    for e in evals:
        assert e.expected_free_energy == pytest.approx(
            e.risk + e.ambiguity - e.epistemic)
    assert all(e.posterior > 0.0 for e in evals)


# -------------------- posterior & precision -------------------- #
def test_policy_posterior_softmax():
    eng = FreeEnergyEngine(world_model=_half_world())
    evals = eng.evaluate_policies((-5.0,), [LEFT, RIGHT, STAY])
    assert sum(e.posterior for e in evals) == pytest.approx(1.0)
    # monotone: lower EFE never gets lower posterior (habits are empty here)
    for a, b in zip(evals, evals[1:]):
        assert a.expected_free_energy <= b.expected_free_energy
        assert a.posterior >= b.posterior


def test_gamma_adapts():
    eng = FreeEnergyEngine(world_model=_half_world(), gamma=4.0)
    assert eng.update_gamma() == 4.0                   # no evidence yet — unchanged
    eng.evaluate_policies((-5.0,), [LEFT, STAY])       # dense, confident region
    assert eng.update_gamma() > eng.gamma_min


def test_softmax_numpy_matches_pure_python():
    logits = [-1.0, 0.5, 2.0, -3.3, 0.0]
    p_py = _softmax(logits, use_numpy=False)
    p_np = _softmax(logits, use_numpy=True)
    assert max(abs(a - b) for a, b in zip(p_py, p_np)) < 1e-9


# -------------------- epistemic value is computed -------------------- #
def test_epistemic_is_computed_not_supplied():
    wm = _half_world()
    eng = FreeEnergyEngine(world_model=wm)
    unseen = eng.epistemic_value((5.0,), "right")      # unknown region
    dense = eng.epistemic_value((-5.0,), "right")      # well-modelled region
    assert unseen > dense
    assert unseen == pytest.approx(wm.learning_progress((5.0,), "right"))


def test_reward_absorbed_as_log_preference():
    prefs = PreferenceModel(reward_fn=lambda s, a, ns: 1.0 if a == "right" else 0.0)
    eng = FreeEnergyEngine(world_model=_half_world(), epistemic_weight=0.0,
                           preferences=prefs)
    choice = eng.select((-5.0,), [LEFT, RIGHT])
    assert list(choice.policy) == RIGHT                # reward ranks through ln C only


# -------------------- degradation -------------------- #
def test_degrades_without_world_model():
    eng = FreeEnergyEngine(world_model=None)
    choice = eng.select((0.0,), [LEFT, RIGHT])
    assert choice is not None
    assert choice.epistemic == 0.0
    assert eng.epistemic_value((0.0,), "left") == 0.0


def test_select_empty_returns_none():
    eng = FreeEnergyEngine(world_model=_half_world())
    assert eng.select((0.0,), []) is None


# -------------------- habits -------------------- #
def test_habits_emerge():
    eng = FreeEnergyEngine(world_model=_half_world())
    before = {_policy_key(e.policy): e.posterior
              for e in eng.evaluate_policies((-5.0,), [LEFT, STAY])}
    for _ in range(12):
        eng.select((-5.0,), [LEFT, STAY])
    winner = _policy_key(eng.last_selection.policy)
    after = {_policy_key(e.policy): e.posterior
             for e in eng.evaluate_policies((-5.0,), [LEFT, STAY])}
    assert after[winner] >= before[winner]             # habit raised its mass
    # ...without touching the EFE itself
    evals = eng.evaluate_policies((-5.0,), [LEFT, STAY])
    for e in evals:
        assert e.expected_free_energy == pytest.approx(
            e.risk + e.ambiguity - e.epistemic)


def test_habit_never_overrides_strong_efe():
    """κ is bounded: a strongly better (lower-EFE) policy beats an ingrained habit."""
    eng = FreeEnergyEngine(world_model=_half_world(), gamma=8.0, habit_weight=0.3,
                           preferences=PreferenceModel(target=[-7.0], precision=2.0))
    for _ in range(20):
        eng.select((-1.0,), [STAY])                    # ingrain a stay habit
    choice = eng.select((-1.0,), [LEFT, STAY])
    assert list(choice.policy) == LEFT                 # strong preference still wins


# -------------------- sophisticated planning -------------------- #
def test_plan_generates_policies_without_candidates():
    eng = FreeEnergyEngine(world_model=_half_world())
    plans = eng.plan((-1.0,), depth=3, beam=4)
    assert plans                                       # grew its own policies
    assert all(isinstance(e, PolicyEvaluation) for e in plans)
    assert plans[0].epistemic > 0.0                    # curiosity intact in search
    assert "right" in list(plans[0].policy)            # drawn toward the unknown


def test_plan_beam_prunes():
    eng = FreeEnergyEngine(world_model=_half_world())
    plans = eng.plan((-5.0,), depth=2, beam=3)
    assert 0 < len(plans) <= 3


def test_plan_without_world_model_is_empty():
    assert FreeEnergyEngine(world_model=None).plan((0.0,)) == []


# -------------------- preference learning -------------------- #
def test_preference_learning_moves_C():
    prefs = PreferenceModel(target=[0.0], precision=1.0)
    for _ in range(10):
        prefs.learn([2.0], 1.0)                        # good outcomes at 2.0
    assert prefs.target[0] > 0.0                       # C moved toward the good
    moved = prefs.target[0]
    prefs.learn([-9.0], -1.0)                          # a bad outcome
    assert prefs.target[0] == pytest.approx(moved)     # C never dragged toward bad
    # negative valence only softens precision
    assert prefs.precision < 1.1


def test_preference_learning_adopts_target_when_none():
    prefs = PreferenceModel()
    prefs.learn([3.0, 4.0], 0.8)
    assert prefs.target == [3.0, 4.0]


# -------------------- persistence -------------------- #
def test_engine_roundtrip():
    eng = FreeEnergyEngine(world_model=_half_world(),
                           preferences=PreferenceModel(target=[1.0]))
    eng.select((-5.0,), [LEFT, STAY])
    snap = eng.to_dict()
    fresh = FreeEnergyEngine()
    fresh.load_dict(snap)
    assert fresh._habit_counts == eng._habit_counts
    assert fresh.preferences.target == [1.0]


# -------------------- bridge into PredictiveCore -------------------- #
def test_actions_for_core_computes_info_gain():
    core = PredictiveCore(belief=[0.0], preference=[5.0])
    eng = FreeEnergyEngine(predictive=core, world_model=_half_world())
    actions = eng.actions_for_core((-5.0,), ["right", "left"])
    assert len(actions) == 2
    for a in actions:
        assert a.info_gain == pytest.approx(
            eng.epistemic_value((-5.0,), a.name))      # computed, not defaulted
    choice = core.act(actions, gamma=eng.gamma)        # the live production path
    assert choice.action.name in ("right", "left")
