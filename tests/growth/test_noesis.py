"""Tests for the Noēsis engine — the living algorithm (growth/noesis.py).

These lock in the properties that make Noēsis honest:
  * the guarded interpreter never raises and evaluates the DSL correctly;
  * WAKE returns only *verified* programs and abstains cleanly when none exists;
  * SLEEP adopts an abstraction only on a strict held-out size reduction, and that adoption makes a
    held-out task solvable with a **strictly shorter** program — the compounding proof;
  * the character core is untouchable and no LLM is in the loop;
  * the learned library persists across a restart.
"""

from __future__ import annotations

import random

from nyxara.growth import noesis as N
from nyxara.growth.noesis import (
    INT,
    INTLIST,
    Abstraction,
    NoesisEngine,
    Task,
    app,
    base_library,
    evaluate,
    hole,
    lit,
    sleep_abstract,
    synthesize,
    var,
)


# --------------------------------------------------------------------------- #
# interpreter
# --------------------------------------------------------------------------- #
def test_interpreter_evaluates_base_programs():
    lib = base_library()
    # sum_(scale_each(x, 2)) on [1,2,3] -> 12
    prog = app("sum_", INT, [app("scale_each", INTLIST, [var(INTLIST), lit(2, INT)])])
    assert evaluate(prog, [1, 2, 3], lib) == 12
    assert evaluate(app("length", INT, [var(INTLIST)]), [4, 5], lib) == 2
    assert evaluate(app("reverse", INTLIST, [var(INTLIST)]), [1, 2], lib) == [2, 1]


def test_interpreter_is_total_never_raises():
    lib = base_library()
    # idiv by zero, head of empty, huge range — every one must fail closed to None, never raise
    assert evaluate(app("idiv", INT, [lit(1, INT), lit(0, INT)]), 0, lib) is None
    assert evaluate(app("head", INT, [var(INTLIST)]), [], lib) is None
    assert evaluate(app("rangei", INTLIST, [lit(10 ** 9, INT)]), 0, lib) is None


# --------------------------------------------------------------------------- #
# WAKE
# --------------------------------------------------------------------------- #
def _examples(prog, lib, inputs):
    return tuple((x, evaluate(prog, x, lib)) for x in inputs)


def test_wake_solves_from_examples():
    lib = base_library()
    target = app("sum_", INT, [app("scale_each", INTLIST, [var(INTLIST), lit(3, INT)])])
    ex = _examples(target, lib, [[1, 2], [3, 0, 1], [4]])
    task = Task("sum3", INTLIST, INT, ex)
    res = synthesize(task, lib)
    assert res.solved and res.program is not None
    # the found program is verified: it reproduces every example
    for x, out in ex:
        assert evaluate(res.program, x, lib) == out


def test_wake_abstains_when_unsolvable():
    lib = base_library()
    # an input->output map with no consistent program in this tiny DSL/budget
    task = Task("nonsense", INTLIST, INT, (([1], 7), ([1], 9)))  # same input, two outputs
    res = synthesize(task, lib, max_rounds=2, beam=80, max_expansions=8000)
    assert not res.solved and res.program is None


# --------------------------------------------------------------------------- #
# SLEEP — the compounding proof
# --------------------------------------------------------------------------- #
def _sum_scaled(k):
    return app("sum_", INT, [app("scale_each", INTLIST, [var(INTLIST), lit(k, INT)])])


def test_sleep_adopts_and_makes_holdout_strictly_shorter():
    base = base_library()
    inputs = [[1, 2], [3, 1, 0], [2, 2], [5]]
    holdout_task = Task("sum_scaled_5", INTLIST, INT, _examples(_sum_scaled(5), base, inputs))

    # BEFORE: solve the held-out task with only the base library
    before = synthesize(holdout_task, base)
    assert before.solved

    # SLEEP on a family that SHARES structure (differs only in the constant)
    lib = base_library()
    corpus = [_sum_scaled(2), _sum_scaled(3), _sum_scaled(4)]
    adopted = sleep_abstract(corpus, lib, seed=0)
    assert adopted, "an abstraction capturing the shared skeleton should be adopted"

    # AFTER: the SAME held-out task now solves with a STRICTLY shorter program
    after = synthesize(holdout_task, lib)
    assert after.solved
    assert after.size < before.size, (before.size, after.size)


def test_sleep_requires_strict_holdout_win():
    # two structurally-unrelated programs share no compressible skeleton → nothing adopted
    lib = base_library()
    corpus = [
        app("length", INT, [var(INTLIST)]),
        app("head", INT, [var(INTLIST)]),
    ]
    assert sleep_abstract(corpus, lib, seed=0) == []


# --------------------------------------------------------------------------- #
# character-lock + no-LLM
# --------------------------------------------------------------------------- #
def test_abstraction_name_cannot_touch_character_core():
    lib = base_library()
    bad = Abstraction("loyalty_hack", (INT,), INT, hole(0, INT))
    try:
        lib.add_abstraction(bad)
    except ValueError:
        return
    raise AssertionError("a character-touching abstraction name must be rejected")


def test_no_llm_or_network_in_the_loop():
    import inspect

    src = inspect.getsource(N)
    for forbidden in ("import requests", "urllib.request", "from nyxara.mind.llm",
                      "openai", "anthropic", "socket"):
        assert forbidden not in src, f"Noēsis must stay LLM/network-free (found {forbidden!r})"


# --------------------------------------------------------------------------- #
# engine: compounding curve + persistence
# --------------------------------------------------------------------------- #
def test_engine_compounds_capability_per_compute():
    eng = NoesisEngine(seed=1, tasks_per_cycle=14)
    reports = eng.run(6)
    assert all(r.solve_rate >= 0.9 for r in reports)
    # solutions get shorter AND cheaper to find as the library grows
    assert reports[-1].avg_solution_size < reports[0].avg_solution_size
    assert reports[-1].avg_expansions < reports[0].avg_expansions
    assert eng.report()["compression_gain"] > 0.0
    assert len(eng.library.abstractions) >= 1


def test_engine_state_persists_across_restart(tmp_path):
    eng = NoesisEngine(seed=2, tasks_per_cycle=10)
    eng.run(3)
    path = str(tmp_path / "noesis.json")
    eng.save(path)

    fresh = NoesisEngine(seed=99)
    assert fresh.load(path)
    assert fresh.cycles == eng.cycles
    assert len(fresh.library.abstractions) == len(eng.library.abstractions)
    assert len(fresh.corpus) == len(eng.corpus)
