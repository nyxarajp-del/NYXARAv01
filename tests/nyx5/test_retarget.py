"""Ontological compiler: AST -> ISA bytecode, emulator-validated; unknown opcode refused; emulator-only."""

from __future__ import annotations

import pytest

from nyxara.nyx5.retarget import OntologicalCompiler


def test_compiles_and_emulates():
    c = OntologicalCompiler()
    r = c.build(("*", ("+", 2, 3), 4))
    assert r.validated is True
    assert r.emulated == 20.0


def test_isa_missing_opcode_refused():
    c = OntologicalCompiler(isa=("push", "add"))   # no 'mul'
    with pytest.raises(ValueError):
        c.compile(("*", 2, 3))


def test_negation():
    c = OntologicalCompiler()
    r = c.build(("neg", 5))
    assert r.emulated == -5.0


def test_require_emulation_gate():
    c = OntologicalCompiler(require_emulation=False)
    r = c.build(("+", 1, 2))
    assert r.validated is False       # not emulator-validated -> honestly flagged unvalidated
