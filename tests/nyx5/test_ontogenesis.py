"""Ontological bytecode genesis: custom DSL runs, halts on budget, no hardware path exists."""

from __future__ import annotations

import pytest

from nyxara.nyx5.ontogenesis import OntologicalBytecodeGenesis, StackVM, VMError


def test_dsl_computes():
    g = OntologicalBytecodeGenesis()
    assert g.execute("PUSH 2\nPUSH 3\nADD\nPUSH 4\nMUL") == 20.0   # (2+3)*4


def test_step_budget_halts():
    vm = StackVM(max_steps=3)
    with pytest.raises(VMError):
        vm.run([("push", 1.0)] * 100)


def test_unknown_opcode_rejected():
    g = OntologicalBytecodeGenesis()
    with pytest.raises(VMError):
        g.assemble("FROB 3")


def test_vm_is_software_only():
    # the VM's opcodes touch only its own stack — no import/exec/hardware surface exists
    assert "system" not in StackVM.OPS
    assert "exec" not in StackVM.OPS
    assert StackVM.OPS <= {"push", "add", "sub", "mul", "div", "dup", "swap", "neg", "halt"}
