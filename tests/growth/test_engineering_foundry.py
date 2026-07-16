"""Tests for the Engineering Foundry — invent a formula, then DESIGN the machine (no LLM).

These prove the engine does *real* engineering, not theatre:

* the portfolio optimiser genuinely converges on a physics target (beating a random baseline),
* multi-objective design yields a real non-dominated Pareto front,
* an invented :class:`~nyxara.growth.law_discovery.Law` actually drives a design (formula→device),
* the self-upgrade loop never degrades a design and persists across sessions,
* and the honest feasibility gate proves impossible "magic" targets INFEASIBLE with the
  conservation law they break — never faking them — and logs them to the impossibility ledger.

Every path runs with **no LLM** (the engine takes no LLM collaborator at all).
"""

from __future__ import annotations

import random

from nyxara.growth.engineering_foundry import (
    Constraint,
    DesignSpace,
    DeviceSpec,
    EngineeringFoundry,
    FeasibilityGate,
    MultiPhysicsEvaluator,
    Objective,
    Parameter,
    _pareto_front,
    build_archetype,
)
from nyxara.growth.law_discovery import LawDiscoveryEngine


# --------------------------------------------------------------------------------------------------
# Real optimisation against real physics
# --------------------------------------------------------------------------------------------------
def test_optimizer_hits_rc_time_constant_and_beats_random():
    """The foundry tunes R, C so the *measured* RC time-constant hits the target — and does so far
    better than a random search of the same budget (proof the search is real, not luck)."""
    foundry = EngineeringFoundry(seed=1, default_budget=140)
    design = foundry.engineer_device(archetype="rc_filter", target=0.5)
    assert design.status == "DESIGNED"
    err = abs(design.metrics["tau"] - 0.5)
    assert err < 0.06, f"optimizer failed to converge: tau={design.metrics['tau']}"

    # equal-budget random baseline over the same design space
    from nyxara.sim.circuit_world import RCCircuit
    circuit = RCCircuit()
    rng = random.Random(99)
    best = float("inf")
    for _ in range(140):
        R = rng.uniform(0.1, 10.0); C = rng.uniform(0.01, 2.0)
        tau = circuit.measure_time_constant(R, C, resolution=900) or 0.0
        best = min(best, abs(tau - 0.5))
    assert err <= best + 1e-9, f"design ({err}) should beat random baseline ({best})"


def test_resonator_hits_wave_speed():
    """Tuning tension and density to a target wave speed v=√(T/μ), measured by the real FD string."""
    foundry = EngineeringFoundry(seed=2, default_budget=120)
    design = foundry.engineer_device(archetype="resonator", target=2.0)
    assert design.status == "DESIGNED"
    assert abs(design.metrics["wave_speed"] - 2.0) < 0.1


# --------------------------------------------------------------------------------------------------
# Multi-objective → a genuine Pareto front
# --------------------------------------------------------------------------------------------------
def test_multi_objective_yields_non_dominated_front():
    foundry = EngineeringFoundry(seed=3, default_budget=140)
    design = foundry.engineer_device(archetype="rc_filter", target=0.5)
    assert design.pareto_front, "expected a non-empty Pareto front for a 2-objective device"
    # every reported front point must be mutually non-dominated
    pts = [(p["parameters"], p["objectives"], p["metrics"]) for p in design.pareto_front]
    front = _pareto_front(pts)
    assert len(front) == len(pts), "reported front contains dominated points"


def test_pareto_front_helper_is_correct():
    pts = [
        ({"a": 0}, [1.0, 0.0], {}),
        ({"a": 1}, [0.0, 1.0], {}),
        ({"a": 2}, [0.5, 0.5], {}),
        ({"a": 3}, [0.4, 0.4], {}),   # dominated by (0.5, 0.5)
    ]
    front = _pareto_front(pts)
    vecs = {tuple(v) for _p, v, _m in front}
    assert (0.4, 0.4) not in vecs
    assert (1.0, 0.0) in vecs and (0.0, 1.0) in vecs and (0.5, 0.5) in vecs


# --------------------------------------------------------------------------------------------------
# Formula → device (an invented Law drives the design), and no-LLM
# --------------------------------------------------------------------------------------------------
def test_invented_law_drives_a_device():
    """Discover a law from data, then design a device whose behaviour model IS that formula."""
    eng = LawDiscoveryEngine(seed=1)
    eng.discover_laws(rounds=1)
    laws = eng.known_laws()
    assert laws, "law discovery produced no laws to engineer from"
    law = laws[-1]

    foundry = EngineeringFoundry(law_discovery=eng, seed=5, default_budget=90)
    design = foundry.design_from_law(law, target=None)
    assert design.status == "DESIGNED"
    assert design.archetype == "law_device"
    # the reported output must be reproducible by the law's own predictor at the chosen parameters
    names = list(law.var_names)
    cols = [[float(design.parameters[n])] for n in names]
    predicted = law.predict(cols)
    assert predicted is not None
    assert abs(predicted[0] - design.metrics["law_output"]) < 1e-6


def test_no_arg_engineer_invents_or_uses_a_law():
    eng = LawDiscoveryEngine(seed=2)
    foundry = EngineeringFoundry(law_discovery=eng, seed=7, default_budget=80)
    # no law known yet → she invents one on demand, then designs from it (formula↔device feedback)
    design = foundry.engineer_device()
    assert design.status in ("DESIGNED", "SPECULATIVE")
    assert eng.total_laws >= 1


def test_runs_with_no_llm_and_no_collaborators():
    """The engine has no LLM parameter at all; it must design with zero collaborators."""
    foundry = EngineeringFoundry(seed=9, default_budget=80)
    assert foundry.law_discovery is None and foundry.first_principles is None
    design = foundry.engineer_device(archetype="electrostatic_trap", target=5.0)
    assert design.status == "DESIGNED"
    assert abs(design.metrics["force"] - 5.0) < 1.0


# --------------------------------------------------------------------------------------------------
# The honest feasibility gate — impossible "magic" is proven infeasible, never faked
# --------------------------------------------------------------------------------------------------
def test_over_unity_is_infeasible_by_energy_conservation():
    foundry = EngineeringFoundry(seed=1, default_budget=40)
    design = foundry.engineer_device("a zero-point energy over-unity reactor")
    assert design.status == "INFEASIBLE"
    assert "energy" in design.feasibility["law"].lower()
    assert foundry.infeasible_logged == 1


def test_anti_gravity_and_time_travel_are_infeasible():
    gate = FeasibilityGate()
    ag = gate.classify("an anti-gravity reactionless propulsion engine")
    assert ag.status == "INFEASIBLE" and "momentum" in ag.law.lower()
    tt = gate.classify("a time-travel machine to rewind time")
    assert tt.status == "INFEASIBLE" and ("causal" in tt.law.lower() or "relativ" in tt.law.lower())


def test_structured_over_unity_claim_is_rejected():
    gate = FeasibilityGate()
    v = gate.classify("power cell", claims={"energy_in": 10.0, "energy_out": 25.0})
    assert v.status == "INFEASIBLE" and "energy" in v.law.lower()
    ok = gate.classify("power cell", claims={"energy_in": 10.0, "energy_out": 7.0})
    assert ok.status == "FEASIBLE"


def test_immortality_is_speculative_not_faked():
    gate = FeasibilityGate()
    v = gate.classify("a device for biological immortality to live forever")
    assert v.status == "SPECULATIVE"       # honest: not designed, not claimed, not a hard violation


def test_feasible_engineering_target_passes():
    gate = FeasibilityGate()
    v = gate.classify("an RC low-pass filter with a 0.5s time constant")
    assert v.status == "FEASIBLE"


# --------------------------------------------------------------------------------------------------
# Self-upgrade loop + persistence
# --------------------------------------------------------------------------------------------------
def test_upgrade_never_degrades_and_can_improve(tmp_path):
    path = str(tmp_path / "foundry.json")
    foundry = EngineeringFoundry(seed=4, default_budget=30, path=path)
    d1 = foundry.engineer_device(archetype="rc_filter", target=0.5)
    s1 = d1.score
    d2 = foundry.upgrade_device("rc-filter")            # widens space, 2× budget
    assert d2.score >= s1 - 1e-9, "an upgrade must never degrade the incumbent"
    # the kept design is either the improved challenger (version bumped) or the incumbent (logged)
    if d2.score > s1 + 1e-9:
        assert d2.version == d1.version + 1
    else:
        assert any("upgrade-rejected" in tag for tag in d2.lineage)


def test_device_tower_persists_across_sessions(tmp_path):
    path = str(tmp_path / "foundry.json")
    f1 = EngineeringFoundry(seed=1, default_budget=90, path=path)
    d = f1.engineer_device(archetype="rc_filter", target=0.5)
    assert d.status == "DESIGNED"
    n = f1.total_devices

    # a brand-new foundry pointed at the same tower reloads her designs (compounding across sessions)
    f2 = EngineeringFoundry(seed=2, default_budget=90, path=path)
    assert f2.total_devices == n
    names = [dev.name for dev in f2.known_devices()]
    assert "rc-filter" in names


def test_impossibility_ledger_persists(tmp_path):
    path = str(tmp_path / "foundry.json")
    f1 = EngineeringFoundry(seed=1, default_budget=40, path=path)
    f1.engineer_device("an over-unity free energy machine")
    assert f1.infeasible_logged == 1
    assert f1.impossibility_ledger(), "the ledger should record the impossible target"

    f2 = EngineeringFoundry(seed=1, default_budget=40, path=path)
    assert f2.impossibility_ledger(), "the ledger must persist across sessions"


# --------------------------------------------------------------------------------------------------
# Multi-physics evaluator
# --------------------------------------------------------------------------------------------------
def test_multiphysics_evaluator_merges_sources():
    ev = (MultiPhysicsEvaluator()
          .add("a", lambda p: {"x": p["k"] * 2.0})
          .add("b", lambda p: {"y": p["k"] + 1.0}))
    out = ev({"k": 3.0})
    assert out == {"x": 6.0, "y": 4.0}
    assert set(ev.sources()) == {"a", "b"}


def test_custom_spec_with_constraint():
    """A hand-built spec: maximise output subject to a hard constraint — constraint must hold."""
    space = DesignSpace([Parameter("x", 0.0, 10.0)])
    spec = DeviceSpec(
        name="capped", space=space,
        objectives=[Objective("out", lambda m: m["v"], maximize=True)],
        evaluator=lambda p: {"v": p["x"]},
        constraints=[Constraint("cap", lambda m: m["v"], kind="<=", bound=4.0)],
        archetype="custom", target="a bounded amplifier")
    foundry = EngineeringFoundry(seed=1, default_budget=120)
    design = foundry.design(spec)
    assert design.status == "DESIGNED"
    assert design.metrics["v"] <= 4.0 + 1e-3, "the hard constraint must be respected"
    assert design.constraints_ok


def test_all_archetypes_build_and_design():
    from nyxara.growth.engineering_foundry import archetype_names
    foundry = EngineeringFoundry(seed=1, default_budget=60)
    for name in archetype_names():
        spec = build_archetype(name)
        assert isinstance(spec, DeviceSpec)
        design = foundry.design(spec)
        assert design.status == "DESIGNED", f"{name} failed to design"
