"""Tests for F3 — neuromodulated plasticity + REM pruning (growth/neuromod.py).

Novel/surprising solves must be written in stronger than habitual ones; stale abstractions must be
pruned but ARCHIVED (reversibly) and never at the cost of a base primitive or a still-used dependency.
No LLM anywhere.
"""

from __future__ import annotations

from nyxara.growth.neuromod import Neuromodulators, REMPruner, UtilityLedger
from nyxara.growth.noesis import (
    INT,
    Abstraction,
    NoesisEngine,
    app,
    base_library,
    hole,
)
from nyxara.growth.redteam import RedTeam


def test_novel_surprising_solve_learns_faster_than_habitual():
    nm = Neuromodulators()
    novel_hard = nm.weight(novelty=1.0, expansions=3000)
    habitual_easy = nm.weight(novelty=0.0, expansions=10)
    assert novel_hard > habitual_easy
    assert habitual_easy >= nm.min_lr
    assert novel_hard <= nm.max_lr


def test_utility_ledger_flags_unused_as_stale():
    ledger = UtilityLedger(window=3, holding_cost=0.5, grace=1)
    # an abstraction that is never used accrues negative utility and becomes stale after grace
    for _ in range(4):
        ledger.record_cycle({})           # zero usage each cycle
    assert ledger.is_stale("absX") is False  # unknown name → no history
    ledger.record_cycle({"absX": 0})
    for _ in range(4):
        ledger.record_cycle({"absX": 0})
    assert ledger.is_stale("absX") is True


def test_used_abstraction_is_not_stale():
    ledger = UtilityLedger(window=3, holding_cost=0.5, grace=1)
    for _ in range(5):
        ledger.record_cycle({"absU": 3})  # steadily used
    assert ledger.is_stale("absU") is False


def test_rem_pruner_archives_reversibly_and_protects_dependencies():
    lib = base_library()
    # a base abstraction and one that DEPENDS on it
    lib.add_abstraction(Abstraction("abs0", (INT,), INT, app("add", INT, [hole(0, INT), hole(0, INT)])))
    dependent = Abstraction("abs1", (INT,), INT, app("abs0", INT, [hole(0, INT)]))
    lib.add_abstraction(dependent)

    ledger = UtilityLedger(window=2, holding_cost=0.5, grace=0)
    for _ in range(3):
        ledger.record_cycle({"abs0": 0, "abs1": 0})  # both look unused…

    report = REMPruner().prune(lib, ledger)
    # abs0 is depended on by abs1 → protected; abs1 (a leaf) may be pruned
    assert "abs0" not in report.pruned
    assert "abs1" in report.pruned
    # pruning is reversible — nothing is truly lost
    assert "abs1" in lib.archived
    assert lib.restore_abstraction("abs1")
    assert "abs1" in lib.abstractions


def test_engine_with_full_stack_still_compounds_and_persists(tmp_path):
    eng = NoesisEngine(
        seed=1, tasks_per_cycle=12,
        red_team=RedTeam(),
        neuromod=Neuromodulators(),
        pruner=REMPruner(),
        ledger=UtilityLedger(),
        prune_every=3,
    )
    reports = eng.run(6)
    assert reports[-1].avg_solution_size < reports[0].avg_solution_size
    assert len(eng.library.abstractions) >= 1
    # archived abstractions round-trip through persistence
    path = str(tmp_path / "n.json")
    eng.save(path)
    fresh = NoesisEngine(seed=0)
    assert fresh.load(path)
    assert len(fresh.library.archived) == len(eng.library.archived)
