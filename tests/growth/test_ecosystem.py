"""Tests for F11 — the symbiotic internal ecosystem (growth/ecosystem.py).

Explorer proposes a broad candidate pool (incl. sub-program fragments), the Skeptic breaks unsound /
degenerate candidates, and the Synthesizer folds survivors in on the strict MDL gate. The result must
still compound, and every adopted abstraction must pass the Skeptic. No LLM anywhere.
"""

from __future__ import annotations

from nyxara.growth.ecosystem import Ecosystem, Explorer, Skeptic
from nyxara.growth.formal_proof import typecheck
from nyxara.growth.noesis import (
    INT,
    INTLIST,
    NoesisEngine,
    app,
    base_library,
    hole,
    lit,
    var,
)


def _sum_scaled(k):
    return app("sum_", INT, [app("scale_each", INTLIST, [var(INTLIST), lit(k, INT)])])


def test_explorer_mines_subprogram_fragments():
    exp = Explorer()
    corpus = [_sum_scaled(2), _sum_scaled(3)]
    bodies = [b.pretty() for b in exp.propose(corpus)]
    # it finds not just the whole-program skeleton but the INNER fragment scale_each(x, $0)
    assert any("scale_each(x, $0)" == b for b in bodies)


def test_skeptic_rejects_a_constant_fold_but_keeps_a_data_abstraction():
    lib = base_library()
    sk = Skeptic()
    # references the data → accepted
    good = app("sum_", INT, [app("scale_each", INTLIST, [var(INTLIST), hole(0, INT)])])
    assert sk.accept(good, lib)
    # a constant fold that never touches x → rejected (degenerate)
    const = app("mul", INT, [hole(0, INT), app("add", INT, [lit(1, INT), lit(3, INT)])])
    assert not sk.accept(const, lib)


def test_ecosystem_adopts_and_reports_pressure():
    lib = base_library()
    corpus = [_sum_scaled(2), _sum_scaled(3), _sum_scaled(4), _sum_scaled(5)]
    eco = Ecosystem()
    adopted = eco.evolve(corpus, lib, next_id=0, seed=0)
    assert adopted                          # at least one concept survived
    assert eco.last.proposed >= eco.last.adopted
    # every adopted abstraction type-checks (the Skeptic did its job)
    for a in adopted:
        assert typecheck(a.body, lib, a.arg_types)


def test_engine_with_ecosystem_still_compounds():
    eng = NoesisEngine(seed=1, tasks_per_cycle=14, ecosystem=Ecosystem())
    reports = eng.run(6)
    assert reports[-1].avg_solution_size < reports[0].avg_solution_size
    assert len(eng.library.abstractions) >= 1
