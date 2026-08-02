"""Tests for nyxara.causal.self_simulation."""

from __future__ import annotations

from nyxara.causal.self_simulation import SelfSimulationRace


def test_lowest_free_energy_branch_wins():
    race = SelfSimulationRace(max_branches=8)
    energies = {"a": 10.0, "b": 2.0, "c": 5.0}
    verdict = race.race(list(energies), lambda h: (h, energies[h]), verify=lambda o: 1.0)
    assert verdict.collapsed is True
    assert verdict.winner.hypothesis == "b"          # least surprising
    assert verdict.evaluated == 3


def test_verification_can_outweigh_a_slightly_higher_energy():
    race = SelfSimulationRace(w_verification=1.0, w_free_energy=1.0)
    # 'good' has higher energy but perfect verification; 'bad' is low energy but unverified
    def rollout(h):
        return (h, {"good": 1.0, "bad": 0.0}[h])

    verdict = race.race(["good", "bad"], rollout,
                        verify=lambda o: 1.0 if o == "good" else 0.0)
    assert verdict.winner.hypothesis == "good"


def test_a_crashing_branch_is_sandboxed_not_fatal():
    race = SelfSimulationRace()

    def rollout(h):
        if h == "poison":
            raise RuntimeError("branch blew up")
        return (h, 1.0)

    verdict = race.race(["poison", "safe"], rollout, verify=lambda o: 1.0)
    assert verdict.winner.hypothesis == "safe"
    poison = next(b for b in verdict.branches if b.hypothesis == "poison")
    assert poison.ok is False and "RuntimeError" in poison.error


def test_branch_count_is_bounded():
    race = SelfSimulationRace(max_branches=4)
    verdict = race.race(list(range(100)), lambda h: (h, float(h)), verify=lambda o: 1.0)
    assert verdict.evaluated == 4                     # N capped
    assert verdict.winner.hypothesis == 0            # lowest energy among the first 4


def test_empty_field_collapses_to_nothing():
    race = SelfSimulationRace()
    verdict = race.race([], lambda h: (h, 1.0))
    assert verdict.winner is None
    assert verdict.collapsed is False


def test_all_branches_failing_yields_no_winner():
    race = SelfSimulationRace()

    def boom(h):
        raise ValueError("nope")

    verdict = race.race(["x", "y"], boom)
    assert verdict.winner is None
    assert all(not b.ok for b in verdict.branches)
