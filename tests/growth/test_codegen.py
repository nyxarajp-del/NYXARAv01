"""Tests for F15 — recursive meta-invention / codegen (growth/codegen.py).

A synthesised kernel must be behaviourally IDENTICAL to the interpreter, is promoted only on a measured
correctness+speed win, never claims a win it didn't measure, and never resorts to string exec. No LLM.
"""

from __future__ import annotations

import nyxara.growth.codegen as cg_mod
from nyxara.growth.codegen import KernelForge, compile_prog
from nyxara.growth.noesis import (
    INT,
    INTLIST,
    Abstraction,
    app,
    base_library,
    evaluate,
    hole,
    lit,
    var,
)


def _prog():
    return app("sum_", INT, [app("inc_each", INTLIST,
               [app("scale_each", INTLIST, [var(INTLIST), lit(2, INT)]), lit(3, INT)])])


_INPUTS = [[1, 2, 3], [4, 5], [0, -1, 2, 7], [9], []]


def test_kernel_is_behaviourally_identical():
    lib = base_library()
    prog = _prog()
    k = compile_prog(prog, lib)
    for x in _INPUTS:
        assert k(x) == evaluate(prog, x, lib)


def test_compiles_an_abstraction():
    lib = base_library()
    lib.add_abstraction(Abstraction("absS", (INT,), INT,
        app("sum_", INT, [app("scale_each", INTLIST, [var(INTLIST), hole(0, INT)])])))
    prog = app("absS", INT, [lit(4, INT)])
    k = compile_prog(prog, lib)
    for x in _INPUTS:
        assert k(x) == evaluate(prog, x, lib)


def test_correct_kernel_can_be_promoted_and_runs():
    res = KernelForge(repeats=200, min_speedup=0.0).optimize(_prog(), base_library(), _INPUTS)
    assert res.correct and res.promoted and res.kernel is not None
    assert res.kernel([1, 2, 3]) == evaluate(_prog(), [1, 2, 3], base_library())


def test_no_fake_win_when_threshold_impossible():
    # correct, but an impossible speed bar → NOT promoted, and no kernel handed back (honest)
    res = KernelForge(repeats=200, min_speedup=1000.0).optimize(_prog(), base_library(), _INPUTS)
    assert res.correct is True
    assert res.promoted is False
    assert res.kernel is None


def test_wrong_kernel_is_never_promoted(monkeypatch):
    # force the compiled kernel to disagree → the correctness gate must refuse promotion
    monkeypatch.setattr(cg_mod, "compile_prog", lambda prog, lib: (lambda x: "WRONG"))
    res = KernelForge(repeats=50, min_speedup=0.0).optimize(_prog(), base_library(), _INPUTS)
    assert res.correct is False and res.promoted is False and res.kernel is None


def test_codegen_uses_no_string_exec():
    import inspect
    src = inspect.getsource(cg_mod)
    assert "exec(" not in src and "eval(" not in src   # closures only — never arbitrary code
