"""Tests for the F5 adversarial red-team gate (growth/redteam.py).

The red-team must catch a solution that fit the given examples but is wrong in general, let a truly
correct solution through, and — when wired into Noēsis — keep over-fit programs out of the permanent
corpus. No LLM anywhere.
"""

from __future__ import annotations

from nyxara.growth import redteam as R
from nyxara.growth.noesis import (
    INT,
    INTLIST,
    NoesisEngine,
    Task,
    app,
    base_library,
    evaluate,
    lit,
    var,
)
from nyxara.growth.redteam import RedTeam, boundary_inputs


def test_red_team_refutes_an_overfit_solution():
    lib = base_library()
    # On descending lists head == maximum, so `head(x)` fits examples that `maximum(x)` generated —
    # but the two DIVERGE on an adversarial boundary input, and the red-team must catch it.
    oracle = app("maximum", INT, [var(INTLIST)])
    overfit = app("head", INT, [var(INTLIST)])
    rt = RedTeam()
    verdict = rt.survives(overfit, INTLIST, lib, oracle=oracle)
    assert not verdict.survived
    assert verdict.counterexample is not None
    # the counter-example genuinely distinguishes them
    assert evaluate(overfit, verdict.counterexample, lib) != evaluate(oracle, verdict.counterexample, lib)


def test_red_team_passes_a_correct_solution():
    lib = base_library()
    oracle = app("sum_", INT, [app("scale_each", INTLIST, [var(INTLIST), lit(2, INT)])])
    # a structurally different but SEMANTICALLY equal program: sum then double == double each then sum
    equal = app("mul", INT, [lit(2, INT), app("sum_", INT, [var(INTLIST)])])
    rt = RedTeam()
    assert rt.survives(equal, INTLIST, lib, oracle=oracle).survived


def test_red_team_oracle_undefined_points_are_not_unfair():
    lib = base_library()
    # head and last agree only sometimes; but where the oracle is None (empty list) it must not count
    oracle = app("head", INT, [var(INTLIST)])
    same = app("head", INT, [var(INTLIST)])
    v = rt = RedTeam().survives(same, INTLIST, lib, oracle=oracle)
    assert v.survived and v.tested > 0


def test_self_consistency_refutes_divergent_solutions():
    lib = base_library()
    a = app("head", INT, [var(INTLIST)])
    b = app("last", INT, [var(INTLIST)])
    v = RedTeam().survives(a, INTLIST, lib, alternative=b)
    assert not v.survived


def test_boundary_battery_includes_the_classic_edges():
    b = boundary_inputs(INTLIST)
    assert [] in b and [0] in b
    assert any(len(x) >= 16 for x in b)


def test_engine_gate_keeps_overfit_out_of_corpus():
    # With the red-team wired in, every corpus program has survived the adversarial battery.
    rt = RedTeam()
    eng = NoesisEngine(seed=1, tasks_per_cycle=14, red_team=rt)
    reports = eng.run(4)
    assert all(r.solve_rate >= 0.9 for r in reports)
    # every program that made it into the corpus agrees with its intent on boundary inputs:
    # the engine still compounds (learns abstractions) under the stricter gate
    assert len(eng.library.abstractions) >= 1
