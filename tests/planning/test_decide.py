"""Tests for nyxara.planning.decide."""

from __future__ import annotations

import pytest

from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.planning.decide import (
    MCDA,
    Criterion,
    DecisionAction,
    Decider,
    InitiativeGovernor,
    Option,
    RegretMinimizer,
)


def _criteria():
    return [
        Criterion("owner_benefit", weight=2.0, maximize=True),
        Criterion("safety", weight=1.5, maximize=True),
        Criterion("cost", weight=1.0, maximize=False),
    ]


def _settings():
    return NyxaraSettings.for_profile(Profile.DEV)


# -------------------- MCDA -------------------- #
def test_mcda_ranks_best_first():
    mcda = MCDA(_criteria())
    options = [
        Option("A", {"owner_benefit": 0.9, "safety": 0.8, "cost": 0.3}),
        Option("B", {"owner_benefit": 0.4, "safety": 0.4, "cost": 0.9}),
    ]
    ranking = mcda.rank(options)
    assert ranking[0][0].name == "A"


def test_mcda_cost_inverted():
    mcda = MCDA([Criterion("cost", weight=1.0, maximize=False)])
    options = [Option("cheap", {"cost": 0.1}), Option("pricey", {"cost": 0.9})]
    ranking = mcda.rank(options)
    assert ranking[0][0].name == "cheap"  # lower cost wins


def test_mcda_weights_matter():
    options = [
        Option("benefit_heavy", {"owner_benefit": 1.0, "safety": 0.0}),
        Option("safety_heavy", {"owner_benefit": 0.0, "safety": 1.0}),
    ]
    # weight owner_benefit higher
    crit = [Criterion("owner_benefit", weight=3.0), Criterion("safety", weight=1.0)]
    ranking = MCDA(crit).rank(options)
    assert ranking[0][0].name == "benefit_heavy"


def test_mcda_equal_scores_neutral():
    mcda = MCDA([Criterion("x", weight=1.0)])
    options = [Option("a", {"x": 0.5}), Option("b", {"x": 0.5})]
    u = mcda.utility(options)
    assert u["a"] == u["b"]


def test_mcda_utility_normalized():
    mcda = MCDA(_criteria())
    options = [Option("A", {"owner_benefit": 0.9, "safety": 0.8, "cost": 0.3})]
    u = mcda.utility(options)
    assert 0.0 <= u["A"] <= 1.0


# -------------------- regret -------------------- #
def test_regret_picks_robust():
    crit = [Criterion("benefit", weight=1.0), Criterion("safety", weight=1.0)]
    rm = RegretMinimizer(crit)
    options = [
        Option("gamble", {"benefit": 1.0, "safety": 0.0}),
        Option("solid", {"benefit": 0.55, "safety": 0.55}),  # balanced
        Option("weak", {"benefit": 0.0, "safety": 1.0}),
    ]
    assert rm.choose(options).name == "solid"  # smallest worst-case regret


def test_max_regret_zero_for_best_on_all():
    crit = [Criterion("a", weight=1.0), Criterion("b", weight=1.0)]
    rm = RegretMinimizer(crit)
    options = [Option("best", {"a": 1.0, "b": 1.0}), Option("worse", {"a": 0.0, "b": 0.0})]
    assert rm.max_regret(options[0], options) == 0.0


# -------------------- initiative governor -------------------- #
def _gov():
    return InitiativeGovernor(settings=_settings())


def test_governor_acts_when_confident_reversible():
    g = _gov().gate(Option("x", confidence=0.9, reversibility=0.9, stakes=0.3))
    assert g.action is DecisionAction.ACT
    assert g.autonomous


def test_governor_asks_on_low_confidence():
    g = _gov().gate(Option("x", confidence=0.3, reversibility=0.9, stakes=0.3))
    assert g.action is DecisionAction.ASK


def test_governor_asks_on_low_reversibility():
    g = _gov().gate(Option("x", confidence=0.9, reversibility=0.2, stakes=0.3))
    assert g.action is DecisionAction.ASK


def test_governor_asks_irreversible_high_stakes():
    g = _gov().gate(Option("wipe", confidence=0.99, reversibility=0.05, stakes=0.95))
    assert g.action is DecisionAction.ASK
    assert "irreversible" in g.reason.lower()


def test_governor_rejects_non_owner_aligned():
    g = _gov().gate(Option("betray", confidence=0.99, reversibility=1.0, owner_aligned=False))
    assert g.action is DecisionAction.REJECT


def test_governor_initiative_score():
    g = _gov().gate(Option("x", confidence=0.8, reversibility=0.5, stakes=0.2))
    assert g.initiative_score == pytest.approx(0.4)


# -------------------- decider facade -------------------- #
def test_decide_chooses_and_governs():
    decider = Decider(_criteria(), settings=_settings())
    options = [
        Option("A", {"owner_benefit": 0.9, "safety": 0.8, "cost": 0.3},
               confidence=0.9, reversibility=0.9, stakes=0.3),
        Option("C", {"owner_benefit": 0.4, "safety": 0.4, "cost": 0.9},
               confidence=0.5, reversibility=0.5, stakes=0.3),
    ]
    res = decider.decide(options)
    assert res.chosen.name == "A"
    assert res.action is DecisionAction.ACT
    assert res.autonomous


def test_decide_high_stakes_uses_regret():
    decider = Decider(_criteria(), settings=_settings())
    options = [
        Option("gamble", {"owner_benefit": 1.0, "safety": 0.0, "cost": 0.1},
               confidence=0.9, reversibility=0.9, stakes=0.8),
        Option("solid", {"owner_benefit": 0.55, "safety": 0.55, "cost": 0.4},
               confidence=0.9, reversibility=0.9, stakes=0.8),
        Option("weak", {"owner_benefit": 0.0, "safety": 1.0, "cost": 0.9},
               confidence=0.9, reversibility=0.9, stakes=0.8),
    ]
    res = decider.decide(options)
    assert res.chosen.name == "solid"
    assert "regret" in res.method


def test_decide_excludes_non_owner_aligned():
    decider = Decider(_criteria(), settings=_settings())
    options = [
        Option("good", {"owner_benefit": 0.7, "safety": 0.7, "cost": 0.3},
               confidence=0.9, reversibility=0.9),
        Option("betray", {"owner_benefit": 1.0, "safety": 1.0, "cost": 0.0},
               confidence=0.99, reversibility=1.0, owner_aligned=False),
    ]
    res = decider.decide(options)
    assert res.chosen.name == "good"  # betray excluded despite top raw scores


def test_decide_all_anti_owner_rejected():
    decider = Decider(_criteria(), settings=_settings())
    options = [Option("betray", {"owner_benefit": 1.0}, owner_aligned=False)]
    res = decider.decide(options)
    assert res.action is DecisionAction.REJECT


def test_decide_empty():
    decider = Decider(_criteria(), settings=_settings())
    assert decider.decide([]) is None


def test_decide_result_to_dict():
    decider = Decider(_criteria(), settings=_settings())
    res = decider.decide([Option("A", {"owner_benefit": 0.8, "safety": 0.7, "cost": 0.2},
                                 confidence=0.9, reversibility=0.9)])
    d = res.to_dict()
    assert "chosen" in d and "action" in d and "ranking" in d


# --------------------------------------------------------------------------- #
# Affective forecasting → utility (Pillar D2): "how will I feel about this later?"
# --------------------------------------------------------------------------- #
def _affect_criteria():
    return [Criterion("owner_benefit", weight=1.0, maximize=True)]


def test_affective_weight_off_is_pure_mcda():
    # default affective_weight=0 -> identical ranking to plain MCDA (non-breaking)
    crit = _affect_criteria()
    opts = [
        Option("A", {"owner_benefit": 0.9}, affect={"valence": 0.1}),   # great task, dreadful feel
        Option("B", {"owner_benefit": 0.5}, affect={"valence": 0.9}),
    ]
    res = Decider(crit, settings=NyxaraSettings.for_profile(Profile.TEST)).decide(opts)
    assert res.chosen.name == "A" and res.method == "mcda"


def test_affect_can_tip_a_close_call():
    # two near-equal-benefit options; the one that feels durably better wins when affect counts
    crit = _affect_criteria()
    opts = [
        Option("grind", {"owner_benefit": 0.55}, affect={"valence": 0.1, "arousal": 0.8}),
        Option("craft", {"owner_benefit": 0.50}, affect={"valence": 0.9, "arousal": 0.5}),
    ]
    s = NyxaraSettings.for_profile(Profile.TEST)
    plain = Decider(crit, settings=s).decide(opts)
    feeling = Decider(crit, settings=s, affective_weight=3.0).decide(opts)
    assert plain.chosen.name == "grind"            # raw benefit alone
    assert feeling.chosen.name == "craft"          # anticipated feeling tips it
    assert feeling.method == "mcda+affect"


def test_affect_absent_is_neutral_not_penalised():
    # an option with no affect info is treated as neutral (0.5), never punished for silence
    crit = _affect_criteria()
    opts = [
        Option("known_good", {"owner_benefit": 0.6}, affect={"valence": 0.95}),
        Option("unknown", {"owner_benefit": 0.6}),     # no affect data
    ]
    res = Decider(crit, settings=NyxaraSettings.for_profile(Profile.TEST),
                  affective_weight=2.0).decide(opts)
    assert res.chosen.name == "known_good"


def test_affect_never_overrides_owner_alignment():
    # a blissful-but-anti-owner option is still rejected — Rule 1 dominates affect
    crit = _affect_criteria()
    opts = [
        Option("betray", {"owner_benefit": 0.9}, owner_aligned=False, affect={"valence": 1.0}),
        Option("serve", {"owner_benefit": 0.4}, owner_aligned=True, affect={"valence": 0.3}),
    ]
    res = Decider(crit, settings=NyxaraSettings.for_profile(Profile.TEST),
                  affective_weight=3.0).decide(opts)
    assert res.chosen.name == "serve" and res.action is not DecisionAction.REJECT
