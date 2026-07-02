"""Tests for nyxara.identity.motivation."""

from __future__ import annotations

import pytest

from nyxara.identity.affect import AffectSystem
from nyxara.identity.motivation import (
    Impulse,
    MotivationBreakdown,
    MotivationSystem,
    Option,
)


@pytest.fixture
def mot():
    return MotivationSystem()


# -------------------- owner primacy -------------------- #
def test_owner_service_outranks_intrinsic(mot):
    serve = Option("serve", owner_relevance=0.9)
    explore = Option("explore", info_gain=0.9, predicted_options=20, owner_relevance=0.0)
    impulses = mot.generate_impulses([serve, explore])
    assert impulses[0].option is serve


def test_harming_owner_is_deeply_negative(mot):
    harm = Option("harm owner", owner_relevance=-0.8, info_gain=0.9, predicted_options=30)
    b = mot.evaluate(harm)
    assert b.total < 0
    assert b.total < -1.0  # harm penalty amplifies


def test_harm_never_chosen(mot):
    harm = Option("harm", owner_relevance=-0.5, info_gain=1.0)
    neutral = Option("neutral", owner_relevance=0.0, info_gain=0.1)
    chosen = mot.choose([harm, neutral])
    assert chosen.option is neutral


# -------------------- intrinsic terms -------------------- #
def test_empowerment_saturates(mot):
    few = mot.evaluate(Option("few", predicted_options=1)).empowerment
    many = mot.evaluate(Option("many", predicted_options=30)).empowerment
    assert 0 <= few < many < 1.0


def test_info_gain_term(mot):
    low = mot.evaluate(Option("a", info_gain=0.1)).info_gain
    high = mot.evaluate(Option("b", info_gain=0.9)).info_gain
    assert high > low


def test_competence_term(mot):
    b = mot.evaluate(Option("a", competence_gain=0.7))
    assert b.competence == 0.7


def test_novelty_decays_with_visits(mot):
    sig = "x"
    n0 = mot.novelty(sig)
    mot.visit(sig)
    mot.visit(sig)
    n1 = mot.novelty(sig)
    assert n1 < n0
    assert n0 == 1.0


def test_novelty_in_evaluation(mot):
    opt = Option("a", signature="seen", owner_relevance=0.0)
    fresh = mot.evaluate(opt).novelty
    mot.visit("seen")
    mot.visit("seen")
    seen = mot.evaluate(opt).novelty
    assert seen < fresh


# -------------------- curiosity -------------------- #
def test_most_curious_picks_informative(mot):
    a = Option("dull", info_gain=0.05, signature="dull", owner_relevance=0.0)
    b = Option("rich", info_gain=0.8, signature="rich", owner_relevance=0.0)
    assert mot.most_curious([a, b]) is b


def test_most_curious_ignores_owner_relevant(mot):
    task = Option("serve", info_gain=0.9, owner_relevance=0.8)
    neutral = Option("explore", info_gain=0.3, owner_relevance=0.0)
    assert mot.most_curious([task, neutral]) is neutral


def test_most_curious_none_when_all_owner_relevant(mot):
    task = Option("serve", owner_relevance=0.8)
    assert mot.most_curious([task]) is None


# -------------------- impulses / ranking -------------------- #
def test_impulse_structure(mot):
    imp = mot.impulse(Option("x", owner_relevance=0.5))
    assert isinstance(imp, Impulse)
    assert isinstance(imp.score, MotivationBreakdown)
    assert imp.dominant_motive


def test_dominant_motive_owner(mot):
    imp = mot.impulse(Option("serve", owner_relevance=0.9))
    assert imp.dominant_motive == "owner_service"


def test_dominant_motive_intrinsic(mot):
    imp = mot.impulse(Option("explore", info_gain=0.9, owner_relevance=0.0,
                             predicted_options=1, competence_gain=0.0, signature="z"))
    # novelty (1.0) or info_gain dominate among intrinsic motives
    assert imp.dominant_motive in ("novelty", "info_gain", "empowerment", "competence")


def test_generate_impulses_sorted(mot):
    opts = [Option("a", owner_relevance=0.2), Option("b", owner_relevance=0.9),
            Option("c", owner_relevance=0.5)]
    impulses = mot.generate_impulses(opts)
    totals = [i.total for i in impulses]
    assert totals == sorted(totals, reverse=True)


def test_choose_empty(mot):
    assert mot.choose([]) is None


# -------------------- competence learning -------------------- #
def test_record_outcome_updates_competence_and_visits(mot):
    opt = Option("practice", signature="prac", competence_gain=0.3)
    mot.record_outcome(opt, competence_gain=0.3, skill="defense")
    assert mot.competence("defense") == 0.3
    assert mot.novelty("prac") < 1.0  # visited


def test_competence_unknown_skill(mot):
    assert mot.competence("unknown") == 0.0


# -------------------- affect modulation -------------------- #
def test_affect_novelty_drive_boosts_novelty():
    affect = AffectSystem()
    affect.drives["novelty"].level = 0.0  # strong unmet novelty need
    mot = MotivationSystem(affect=affect)
    w = mot._effective_weights()
    assert w.novelty > MotivationSystem().weights.novelty


def test_affect_low_energy_dampens_exploration():
    affect = AffectSystem()
    affect.drives["energy"].level = 0.0
    mot = MotivationSystem(affect=affect)
    w = mot._effective_weights()
    assert w.info_gain < MotivationSystem().weights.info_gain


# -------------------- empowerment estimator -------------------- #
def test_estimate_empowerment_with_world_model():
    import random
    from nyxara.mind.world_model import WorldModel

    def step(x, a):
        return {"left": x - 1.0, "right": x + 1.0, "stay": x}[a]

    wm = WorldModel(k=3, distance_scale=2.0)
    rng = random.Random(0)
    for _ in range(300):
        xv = rng.uniform(-10, 10)
        av = rng.choice(["left", "right", "stay"])
        wm.observe((xv,), av, (step(xv, av),))

    n = MotivationSystem.estimate_empowerment(wm, (0.0,), ["left", "right", "stay"],
                                              horizon=2)
    assert n >= 2  # multiple reachable states


# -------------------- status -------------------- #
def test_status(mot):
    mot.visit("x")
    s = mot.status()
    assert "visited_signatures" in s and "weights" in s


# -------------------- persistence (drives survive restarts) -------------------- #
def test_snapshot_restore_roundtrip():
    m = MotivationSystem()
    m.visit("theme-a")
    m.visit("theme-a")
    m.record_outcome(Option(name="learn x", signature="x"), competence_gain=0.4, skill="x")
    snap = m.snapshot()

    m2 = MotivationSystem()
    m2.restore(snap)
    assert m2.novelty("theme-a") == pytest.approx(m.novelty("theme-a"))
    assert m2.competence("x") == pytest.approx(0.4)


def test_restore_ignores_garbage():
    m = MotivationSystem()
    m.restore(None)              # not a dict -> no-op
    m.restore({"visits": {"a": "notint"}, "competence": {"s": "nan"}})  # bad values skipped
    assert m.novelty("a") == pytest.approx(1.0)
