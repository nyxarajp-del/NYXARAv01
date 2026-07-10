"""Tests for nyxara.eval.generalization — proof she generalizes HERSELF, not via the LLM.

The benchmark grades NYXARA's own faculties (relational structure-mapping + first-principles law
induction) on held-out domains with no language model in the loop. The key assertion is the
*delta*: the own-faculty path solves tasks a base-only baseline cannot, so the generalization the
"scaffolding-only ceiling" critique says is impossible is measured to exist.
"""

from __future__ import annotations

from nyxara.eval.generalization import (
    build_structure_transfer_suite,
    run_generalization_benchmark,
    run_law_induction,
    run_structure_transfer,
)


def test_structure_transfer_beats_base_only_baseline():
    report = run_structure_transfer()
    assert report.total >= 2
    assert report.solved >= 1
    # base-only (no known domains to map from) recovers nothing → the delta is real
    assert report.baseline_solved == 0
    assert report.own_faculty_delta > 0


def test_atom_projects_causal_inference():
    report = run_structure_transfer()
    atom = next(r for r in report.results if r.task == "atom")
    assert atom.solved
    assert "solar_system" in atom.detail


def test_law_induction_recovers_unseen_laws_on_holdout():
    report = run_law_induction()
    assert report.holdout_accuracy is not None
    assert report.holdout_accuracy >= 0.8       # validated on fresh, never-probed inputs
    assert report.solved == report.total


def test_suite_is_deterministic():
    a = run_structure_transfer().to_dict()
    b = run_structure_transfer().to_dict()
    assert a == b


def test_combined_benchmark_reports_both_suites():
    out = run_generalization_benchmark()
    assert "structure_transfer" in out and "law_induction" in out
    assert out["structure_transfer"]["own_faculty_delta"] > 0


def test_suite_has_held_out_targets():
    suite = build_structure_transfer_suite()
    assert suite
    for task in suite:
        assert task.target_relations
        assert task.expected_tokens
