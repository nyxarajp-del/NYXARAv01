"""NYXARA · nyx5/retarget.py — the ontological compiler (🧬, pillar 17).

The hardware cousin of pillar 16: given the *description* of a target instruction-set, NYX-5 acts as a
**retargetable code-generation backend** — she lowers an abstract syntax tree into that target's bytecode
and validates the emitted code by running it in a **software emulator**. This is how a mind could target a
brand-new chip for which no compiler yet exists: describe the ISA, and she writes the backend.

Honest ceiling (non-negotiable): she generates and verifies code for a *described / simulated* target.
Running on real new silicon still needs that hardware's genuine toolchain and driver — there is **no
silicon-direct or register-level execution here**, only the emulator. "Reads the transistors and instantly
runs" is a pitch; the reality is honest cross-compilation validated in simulation.

Reuses the VM/assembler of :mod:`nyxara.nyx5.ontogenesis` as the default emulator. Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from nyxara.nyx5.ontogenesis import Instr, StackVM

__all__ = ["CompileResult", "OntologicalCompiler"]

AST = Any  # ('+', a, b) | ('-', a, b) | ('*', a, b) | ('/', a, b) | ('neg', a) | number


@dataclass
class CompileResult:
    """Emitted code plus its emulator-validated output (or the reason validation was refused)."""

    program: List[Instr]
    emulated: Any
    validated: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"n_instr": len(self.program), "emulated": self.emulated, "validated": self.validated}


class OntologicalCompiler:
    """Lower an AST to a described ISA's bytecode and validate it in a software emulator."""

    # a described target: opcode names it understands. Default mirrors the ontogenesis StackVM.
    def __init__(self, *, isa: Tuple[str, ...] = ("push", "add", "sub", "mul", "div", "neg"),
                 require_emulation: bool = True, max_steps: int = 100_000) -> None:
        self.isa = tuple(isa)
        self.require_emulation = bool(require_emulation)
        self.max_steps = int(max_steps)

    _BINOP = {"+": "add", "-": "sub", "*": "mul", "/": "div"}

    def compile(self, ast: AST) -> List[Instr]:
        """Post-order lowering of the AST into stack bytecode for the described ISA."""
        program: List[Instr] = []
        self._emit(ast, program)
        for op, _ in program:
            if op not in self.isa:
                raise ValueError(f"target ISA does not implement opcode {op!r}")
        return program

    def _emit(self, node: AST, out: List[Instr]) -> None:
        if isinstance(node, (int, float)):
            out.append(("push", float(node)))
            return
        op = node[0]
        if op == "neg":
            self._emit(node[1], out)
            out.append(("neg", None))
            return
        if op in self._BINOP:
            self._emit(node[1], out)
            self._emit(node[2], out)
            out.append((self._BINOP[op], None))
            return
        raise ValueError(f"unsupported AST node: {op!r}")

    def build(self, ast: AST) -> CompileResult:
        """Compile + (if required) validate in the emulator. Refuses to accept unvalidated code."""
        program = self.compile(ast)
        if not self.require_emulation:
            return CompileResult(program=program, emulated=None, validated=False)
        emulated = StackVM(max_steps=self.max_steps).run(program)
        return CompileResult(program=program, emulated=emulated, validated=True)
