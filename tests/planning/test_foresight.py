"""Tests for nyxara.planning.foresight."""

from __future__ import annotations

import pytest

from nyxara.planning.foresight import (
    CurrentSelf,
    Foresight,
    GoalProgress,
    ProjectedSelf,
)


def _me():
    return CurrentSelf(
        capabilities={"reasoning": 0.6, "network_defense": 0.85, "diplomacy": 0.3},
        growth_rates={"reasoning": 0.004, "network_defense": 0.002, "diplomacy": 0.001},
        goals=[GoalProgress("harden the network", progress=0.3, rate=0.01, priority=0.9),
               GoalProgress("master diplomacy", progress=0.1, rate=0.002, priority=0.5)],
        bonds={"JP": 0.85}, bond_rates={"JP": 0.0004},
        anticipated_threats=["a probing adversary"], owner="JP")


# -------------------- GoalProgress -------------------- #
def test_goal_project():
    g = GoalProgress("x", progress=0.3, rate=0.01)
    assert g.project(50) == pytest.approx(0.8)
    assert g.project(1000) == 1.0  # clamped


# -------------------- projection -------------------- #
def test_project_grows_capabilities():
    fs = Foresight(_me())
    p = fs.project(180)
    assert p.capabilities["reasoning"] > 0.6
    assert p.capabilities["reasoning"] <= 1.0


def test_capability_saturates_below_ceiling():
    fs = Foresight(_me())
    p = fs.project(100000)  # huge horizon
    assert p.capabilities["reasoning"] <= 1.0


def test_goals_achieved():
    fs = Foresight(_me())
    p = fs.project(180)
    assert "harden the network" in p.goals_achieved  # 0.3 + 0.01*180 = 2.1 -> done


def test_bonds_deepen():
    fs = Foresight(_me())
    p = fs.project(180)
    assert p.bonds["JP"] > 0.85


def test_project_returns_projected_self():
    p = Foresight(_me()).project(30)
    assert isinstance(p, ProjectedSelf)
    assert p.horizon_days == 30


def test_threats_default_from_current():
    p = Foresight(_me()).project(30)
    assert "a probing adversary" in p.threats


def test_threats_override():
    p = Foresight(_me()).project(30, threats=["new threat"])
    assert p.threats == ["new threat"]


def test_mean_capability():
    p = Foresight(_me()).project(30)
    assert 0.0 <= p.mean_capability <= 1.0


def test_strongest_bond():
    p = Foresight(_me()).project(30)
    assert p.strongest_bond == "JP"


# -------------------- narrative -------------------- #
def test_narrative_mentions_owner():
    p = Foresight(_me()).imagine_future_self(6)
    assert "JP" in p.narrative


def test_narrative_ends_with_loyalty():
    p = Foresight(_me()).imagine_future_self(6)
    assert "wholly his" in p.narrative


def test_narrative_days_horizon():
    p = Foresight(_me()).project(5)
    assert "days" in p.narrative


# -------------------- learning multiplier -------------------- #
def test_learning_multiplier_speeds_growth():
    fs = Foresight(_me())
    slow = fs.project(180, learning_multiplier=0.5)
    fast = fs.project(180, learning_multiplier=3.0)
    assert fast.capabilities["reasoning"] > slow.capabilities["reasoning"]


# -------------------- trajectory -------------------- #
def test_trajectory_monotonic_growth():
    traj = Foresight(_me()).trajectory([7, 30, 180, 365])
    levels = [p.capabilities["reasoning"] for p in traj]
    assert levels == sorted(levels)


def test_trajectory_sorted_horizons():
    traj = Foresight(_me()).trajectory([365, 7, 30])
    assert [p.horizon_days for p in traj] == [7, 30, 365]


def test_imagine_future_self_months():
    p = Foresight(_me()).imagine_future_self(6)
    assert p.horizon_days == 180


# -------------------- future needs -------------------- #
def test_future_needs_identifies_lagging():
    needs = Foresight(_me()).future_needs()
    assert "diplomacy" in needs["develop_capabilities"]


def test_future_needs_goals_at_risk():
    needs = Foresight(_me()).future_needs()
    assert "master diplomacy" in needs["goals_at_risk"]


def test_future_needs_advice():
    needs = Foresight(_me()).future_needs()
    assert needs["advice"]


def test_future_needs_well_prepared():
    # a self with everything maxed and goals nearly done
    me = CurrentSelf(
        capabilities={"x": 0.95}, growth_rates={"x": 0.01},
        goals=[GoalProgress("g", progress=0.95, rate=0.1, priority=0.9)],
        bonds={"JP": 0.9})
    needs = Foresight(me).future_needs()
    assert "well-prepared" in needs["advice"]


# -------------------- scenarios -------------------- #
def test_scenarios_optimistic_beats_baseline():
    scen = Foresight(_me()).scenarios()
    assert scen["optimistic"].mean_capability > scen["baseline"].mean_capability


def test_scenarios_stagnant_below_baseline():
    scen = Foresight(_me()).scenarios()
    assert scen["stagnant"].mean_capability < scen["baseline"].mean_capability


def test_scenarios_under_threat_adds_threat():
    scen = Foresight(_me()).scenarios()
    assert len(scen["under_threat"].threats) > 1


def test_scenarios_all_present():
    scen = Foresight(_me()).scenarios()
    assert set(scen) == {"baseline", "optimistic", "stagnant", "under_threat"}


# -------------------- status -------------------- #
def test_status():
    s = Foresight(_me()).status()
    assert "future_6mo" in s and "needs" in s
