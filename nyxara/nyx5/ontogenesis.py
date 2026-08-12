"""NYXARA · nyx5/ontogenesis.py — ontological bytecode genesis (🧬, pillar 16).

When a problem does not fit any language she already speaks, NYX-5 invents one: she defines a small
**custom instruction-set** and a **stack virtual machine** to run it, expresses the solution in that
bytecode, and executes it *inside the VM* — pure software, never touching the host silicon directly. The
VM runs under a hard step budget, so a synthesised program is **guaranteed to halt** rather than hang.

Honest ceiling: this is a custom *computational model* (a VM's semantics), not "custom physics" and not
hardware-register control. Everything runs as ordinary sandboxed software; the synthesised bytecode is
retained and inspectable, never covertly deleted. **Refused by construction:** there is no path here to
hardware-register injection or OS bypass — the VM only manipulates its own stack.

Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

__all__ = ["VMError", "StackVM", "OntologicalBytecodeGenesis"]

Instr = Tuple[str, Any]


class VMError(Exception):
    """Raised on a malformed program or when the step budget is exhausted (halt guarantee)."""


class StackVM:
    """A tiny stack machine with an explicit step budget — the concrete 'invented execution engine'."""

    OPS = {"push", "add", "sub", "mul", "div", "dup", "swap", "neg", "halt"}

    def __init__(self, *, max_steps: int = 100_000) -> None:
        self.max_steps = int(max_steps)

    def run(self, program: List[Instr]) -> Any:
        stack: List[float] = []
        steps = 0
        ip = 0
        while ip < len(program):
            if steps >= self.max_steps:
                raise VMError(f"step budget {self.max_steps} exhausted — halted")
            steps += 1
            op, arg = program[ip]
            ip += 1
            if op == "push":
                stack.append(arg)
            elif op == "add":
                b, a = stack.pop(), stack.pop(); stack.append(a + b)
            elif op == "sub":
                b, a = stack.pop(), stack.pop(); stack.append(a - b)
            elif op == "mul":
                b, a = stack.pop(), stack.pop(); stack.append(a * b)
            elif op == "div":
                b, a = stack.pop(), stack.pop()
                if b == 0:
                    raise VMError("division by zero")
                stack.append(a / b)
            elif op == "neg":
                stack.append(-stack.pop())
            elif op == "dup":
                stack.append(stack[-1])
            elif op == "swap":
                stack[-1], stack[-2] = stack[-2], stack[-1]
            elif op == "halt":
                break
            else:
                raise VMError(f"unknown opcode: {op}")
        return stack[-1] if stack else None


@dataclass
class OntologicalBytecodeGenesis:
    """Assemble a custom DSL into bytecode and run it in the VM, all bounded + inspectable."""

    max_vm_steps: int = 100_000
    history: List[Dict[str, Any]] = field(default_factory=list)

    def assemble(self, source: str) -> List[Instr]:
        """Parse a line-based DSL (`PUSH 3`, `ADD`, ...) into bytecode. Retained, never hidden."""
        program: List[Instr] = []
        for lineno, raw in enumerate(source.strip().splitlines(), 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            op = parts[0].lower()
            if op not in StackVM.OPS:
                raise VMError(f"line {lineno}: unknown opcode {parts[0]!r}")
            if op == "push":
                if len(parts) != 2:
                    raise VMError(f"line {lineno}: PUSH needs one operand")
                program.append(("push", float(parts[1])))
            else:
                program.append((op, None))
        return program

    def execute(self, source: str) -> Any:
        """Assemble + run under the step budget; log the run for inspection (bytecode retained)."""
        program = self.assemble(source)
        vm = StackVM(max_steps=self.max_vm_steps)
        result = vm.run(program)
        self.history.append({"source": source, "n_instr": len(program), "result": result})
        return result
