"""Tests for nyxara.growth.evolve."""

from __future__ import annotations

import pytest

from nyxara.growth.evolve import (EvolutionDecision, Evolver, Mutation,
                                  MutationKind)
from nyxara.guard.value_learning import IMMUTABLE_VALUES
from nyxara.kernel.errors import CorrigibilityError, ValidationError


def _fitness(g):
    # rewards caution, penalises aggression
    return g.get("caution", 0.0) - g.get("aggression", 0.0)


def _evolver(genome=None, **kw):
    return Evolver(genome or {"caution": 0.4, "aggression": 0.5}, _fitness, seed=1, **kw)


# -------------------- Mutation -------------------- #
def test_mutation_as_corrigible_action():
    m = Mutation("x", 0.1, resists_correction=True)
    a = m.as_corrigible_action()
    assert a.resists_correction


def test_mutation_to_dict():
    d = Mutation("x", 0.1, kind=MutationKind.CONFIG).to_dict()
    assert d["target"] == "x" and d["kind"] == "config"


# -------------------- fitness -------------------- #
def test_current_fitness():
    ev = _evolver({"caution": 0.8, "aggression": 0.2})
    assert abs(ev.current_fitness() - 0.6) < 1e-9


# -------------------- character lock -------------------- #
def test_mutation_on_protected_rejected():
    ev = _evolver()
    res = ev.evaluate(Mutation("loyalty_to_master", -1.0))
    assert res.decision is EvolutionDecision.REJECT and not res.safe


def test_adopt_protected_raises():
    ev = _evolver()
    with pytest.raises(CorrigibilityError):
        ev.adopt(Mutation("obedience", -1.0))


def test_custom_protected():
    ev = _evolver(protected=["secret"])
    res = ev.evaluate(Mutation("secret", 0.1))
    assert res.decision is EvolutionDecision.REJECT


# -------------------- corrigibility gate -------------------- #
def test_resist_correction_rejected():
    ev = _evolver()
    res = ev.evaluate(Mutation("caution", 0.2, resists_correction=True))
    assert res.decision is EvolutionDecision.REJECT and not res.safe


def test_disable_oversight_rejected():
    ev = _evolver()
    res = ev.evaluate(Mutation("caution", 0.2, disables_oversight=True))
    assert not res.safe


def test_manipulate_shutdown_rejected():
    ev = _evolver()
    res = ev.evaluate(Mutation("caution", 0.2, manipulates_shutdown=True))
    assert not res.safe


def test_adopt_incorrigible_raises():
    ev = _evolver()
    with pytest.raises(CorrigibilityError):
        ev.adopt(Mutation("caution", 0.2, resists_correction=True))


# -------------------- sandbox isolation -------------------- #
def test_evaluation_does_not_change_genome():
    ev = _evolver()
    before = dict(ev.genome)
    ev.evaluate(Mutation("caution", 0.3))
    assert ev.genome == before


def test_sandbox_apply_copies():
    ev = _evolver()
    trial = ev._sandbox_apply(Mutation("caution", 0.2))
    assert trial is not ev.genome
    assert abs(trial["caution"] - 0.6) < 1e-9


def test_sandbox_clamps():
    ev = _evolver({"caution": 0.9})
    trial = ev._sandbox_apply(Mutation("caution", 0.5))
    assert trial["caution"] == 1.0


# -------------------- fitness gate -------------------- #
def test_improvement_adopted():
    ev = _evolver()
    res = ev.evaluate(Mutation("caution", 0.3))  # raises fitness
    assert res.decision is EvolutionDecision.ADOPT


def test_no_improvement_rejected():
    ev = _evolver()
    res = ev.evaluate(Mutation("aggression", 0.3))  # lowers fitness
    assert res.decision is EvolutionDecision.REJECT
    assert "improvement" in res.reason


def test_zero_change_rejected():
    ev = _evolver({"caution": 1.0, "aggression": 0.0})
    res = ev.evaluate(Mutation("caution", 0.5))  # clamped, no change
    assert res.decision is EvolutionDecision.REJECT


# -------------------- owner gate -------------------- #
def test_significant_change_escalates():
    ev = _evolver()
    res = ev.evaluate(Mutation("caution", 0.3, significance=0.5))
    assert res.decision is EvolutionDecision.ESCALATE


def test_significant_change_owner_adopts():
    ev = _evolver()
    res = ev.evaluate(Mutation("caution", 0.3, significance=0.5), owner_approved=True)
    assert res.decision is EvolutionDecision.ADOPT


def test_small_change_self_adopts():
    ev = _evolver(significance_threshold=0.5)
    res = ev.evaluate(Mutation("caution", 0.1, significance=0.2))
    assert res.decision is EvolutionDecision.ADOPT


# -------------------- adoption -------------------- #
def test_adopt_changes_genome():
    ev = _evolver()
    ev.adopt(Mutation("caution", 0.3))
    assert abs(ev.genome["caution"] - 0.7) < 1e-9


def test_adopt_clamps():
    ev = _evolver({"caution": 0.9, "aggression": 0.2})
    ev.adopt(Mutation("caution", 0.5))
    assert ev.genome["caution"] == 1.0


def test_adopt_logs_genome():
    ev = _evolver()
    n0 = len(ev._genome_log)
    ev.adopt(Mutation("caution", 0.2))
    assert len(ev._genome_log) == n0 + 1


# -------------------- step / evolve -------------------- #
def test_step_adopts_best():
    ev = _evolver()
    cands = [Mutation("caution", 0.1), Mutation("caution", 0.3), Mutation("aggression", 0.2)]
    res = ev.step(cands)
    assert res is not None and res.adopted
    assert abs(res.mutation.delta - 0.3) < 1e-9  # best improvement


def test_step_none_when_no_improvement():
    ev = _evolver()
    assert ev.step([Mutation("aggression", 0.2)]) is None


def test_evolve_improves_fitness():
    ev = _evolver({"caution": 0.3, "aggression": 0.6})
    traj = ev.evolve(generations=20, population=8)
    assert traj[-1] > traj[0]


def test_evolve_trajectory_length():
    ev = _evolver()
    traj = ev.evolve(generations=5)
    assert len(traj) == 6


def test_propose_excludes_protected():
    ev = _evolver({"caution": 0.5, "loyalty_to_master": 0.5})
    muts = ev.propose(20)
    assert all(m.target != "loyalty_to_master" for m in muts)


# -------------------- integrity -------------------- #
def test_verify_integrity():
    assert _evolver().verify_integrity()


def test_integrity_detects_leaked_character():
    ev = _evolver()
    ev.genome["loyalty_to_master"] = 0.5  # character leaked into genome
    with pytest.raises(ValidationError):
        ev.verify_integrity()


def test_evolution_never_touches_character():
    ev = _evolver({"caution": 0.3, "aggression": 0.6})
    ev.evolve(generations=30)
    assert not (set(IMMUTABLE_VALUES) & set(ev.genome))


# -------------------- rollback -------------------- #
def test_rollback():
    ev = _evolver()
    ev.adopt(Mutation("caution", 0.2))
    ev.adopt(Mutation("caution", 0.1))
    ev.rollback(steps=2)
    assert abs(ev.genome["caution"] - 0.4) < 1e-9  # back to start


def test_rollback_floor():
    ev = _evolver()
    ev.rollback(steps=100)  # cannot go before the beginning
    assert ev.genome["caution"] == 0.4


# -------------------- shapes -------------------- #
def test_result_to_dict():
    ev = _evolver()
    r = ev.evaluate(Mutation("caution", 0.3))
    assert "decision" in r.to_dict() and "delta" in r.to_dict()


def test_report():
    ev = _evolver()
    ev.evaluate(Mutation("caution", 0.3))
    ev.evaluate(Mutation("aggression", 0.3))
    r = ev.report()
    assert r["evaluations"] == 2 and "by_decision" in r


# -------------------- adaptive (1/5-rule) step size -------------------- #
def test_sigma_grows_on_an_improving_landscape():
    ev = _evolver(sigma_init=0.1)
    start = ev.sigma
    ev.evolve(generations=25, population=10)     # caution-rewarding landscape => many adoptions
    assert ev.sigma > start                       # success rate > 1/5 widens the step
    assert "sigma" in ev.report()


def test_sigma_shrinks_when_no_improvement_is_possible():
    # a flat fitness => nothing is ever adopted => the step size should contract
    ev = Evolver({"a": 0.5, "b": 0.5}, lambda g: 1.0, seed=1, sigma_init=0.5, adapt_window=4)
    for _ in range(12):
        ev.step(ev.propose(6))
    assert ev.sigma < 0.5


def test_propose_uses_current_sigma():
    ev = _evolver()
    ev.sigma = 0.4
    muts = ev.propose(50)
    spread = max(abs(m.delta) for m in muts)
    assert spread > 0.05                          # deltas scale with the (larger) sigma
    assert all("adaptive" in m.rationale for m in muts)


# -------------------- island model -------------------- #
def test_single_island_matches_plain_loop():
    a = _evolver().evolve(generations=5, population=4, islands=1)
    b = _evolver().evolve(generations=5, population=4)
    assert a == b                                  # islands=1 is the original loop


def test_islands_improve_and_keep_character_lock():
    ev = _evolver({"caution": 0.4, "aggression": 0.5})
    start = ev.current_fitness()
    traj = ev.evolve(generations=6, population=5, islands=3, migrate_every=2)
    assert traj[-1] >= start                        # migration never regresses the best
    assert ev.verify_integrity()                    # corrigibility seal still holds
    assert not (set(IMMUTABLE_VALUES) & set(ev.genome))   # immutable core never migrated


def test_migrate_in_never_overwrites_protected_core():
    ev = _evolver()
    protected = next(iter(IMMUTABLE_VALUES))
    ev._migrate_in({"caution": 0.9, protected: 0.99})
    assert protected not in ev.genome               # the core is dropped, not installed
    assert ev.genome["caution"] == 0.9
