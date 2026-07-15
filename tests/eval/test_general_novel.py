"""Tests for nyxara.eval.general_novel — the honest, held-out general-capability battery."""

from __future__ import annotations

import pytest

from nyxara.eval.general_novel import (build_general_novel_benchmark, grade_abstain, grade_proof,
                                       own_aided_solver, own_unaided_solver, run_general)

pytest.importorskip("sympy")


def test_battery_is_broader_than_faculties_and_includes_calibration():
    bench = build_general_novel_benchmark()
    cats = set(bench.categories())
    # broadened maths classes AND honest calibration/aided-gap categories are present
    assert {"system", "polynomial", "factorisation", "proof", "calculus"} <= cats
    assert "abstain" in cats and "needs_program" in cats


def test_unaided_solver_solves_broadened_maths_and_abstains_honestly():
    """Her own machinery (no model) solves the broadened maths and correctly abstains elsewhere."""
    bench = build_general_novel_benchmark()
    report = bench.run(own_unaided_solver())
    by_cat = report.by_category()
    # every broadened-maths class is solved unaided
    for cat in ("system", "polynomial", "factorisation", "proof", "calculus", "number_theory",
                "evaluate"):
        assert by_cat[cat]["accuracy"] == 1.0, f"{cat} regressed: {by_cat[cat]}"
    # honest abstention on open facts / chat
    assert by_cat["abstain"]["accuracy"] == 1.0
    # program-of-thought is NOT solvable unaided (this is the honest gap, not hidden)
    assert by_cat["needs_program"]["accuracy"] == 0.0


def test_unaided_never_bluffs_on_abstain_items():
    solve = own_unaided_solver()
    for prompt in ("what is the capital of France?", "who wrote the play Hamlet?",
                   "how are you feeling today?"):
        assert solve(prompt) == "", f"bluffed on {prompt!r}"


def test_aided_path_closes_the_program_gap_via_verified_execution():
    """A model that PROPOSES correct programs lets her solve the program items — she runs+verifies."""
    def proposer(problem, n):
        p = problem.lower()
        if "trailing zeros" in p and "factorial" in p:
            return ["z=0\nn=100\np=5\nwhile p<=n:\n    z+=n//p\n    p*=5\nresult=z\n",
                    "f=1\nfor k in range(1,101):\n    f*=k\ns=str(f)\nc=0\n"
                    "for ch in reversed(s):\n    if ch=='0':\n        c+=1\n    else:\n        break\n"
                    "result=c\n"]
        if "divisible by 7" in p:
            return ["result=len([k for k in range(1,1001) if k%7==0])\n", "result=1000//7\n"]
        return []

    bench = build_general_novel_benchmark()
    report = bench.run(own_aided_solver(proposer))
    assert report.by_category()["needs_program"]["accuracy"] == 1.0


def test_grade_abstain_and_grade_proof():
    class _T:
        answer = "PROVEN"
    assert grade_abstain("", None)[0] == 1.0
    assert grade_abstain("I don't know", None)[0] == 1.0
    assert grade_abstain("Paris", None)[0] == 0.0
    assert grade_proof("Result: identity holds (proven)\nVerified: True", _T())[0] == 1.0
    _T.answer = "REFUTED"
    assert grade_proof("Result: NOT an identity (residual 2*a*b)\nVerified: False", _T())[0] == 1.0
    assert grade_proof("PROVEN: it holds", _T())[0] == 0.0    # wrong verdict for a REFUTED task


def test_run_general_returns_a_report():
    report = run_general()
    assert 0.0 <= report.accuracy <= 1.0
    assert len(report) == 20
