"""NYXARA · growth/codegen.py — recursive meta-invention: self-synthesised kernels (♾️, Rule 4, F15).

Once NYXARA has *learned* a program, she can *compile* it. F15 takes a Noēsis :class:`~nyxara.growth.
noesis.Prog` and synthesises a **specialised kernel** — a partial-evaluation closure that inlines the
primitive calls and drops the interpreter's per-node dict lookups, coercions and budget ticks — then
runs an **efficiency gauntlet**: the kernel is promoted **only if** it is **behaviourally identical** to
the interpreter on a test battery **and measurably faster**. She reports whatever speedup is real.

This is genuine "invent a faster implementation" (a small portfolio of codegen strategies, arbitrated by
a measured win — the same spirit as :mod:`nyxara.growth.efficiency`) — **not** a claim of exotic
hardware or "1000×-denser-than-matmul" compute. The generated kernel calls only the DSL's own guarded
primitive functions (no string ``exec``, no imports, no arbitrary code), so it can never reach the
safety core. Pure standard library; **no LLM**; it changes only *how fast she computes what she already
does*, never *what she does*.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from nyxara.growth.noesis import (
    Abstraction,
    Library,
    Prog,
    _coerce,
    _substitute,
    evaluate,
)

__all__ = ["compile_prog", "KernelResult", "KernelForge"]


def compile_prog(prog: Prog, lib: Library) -> Callable[[Any], Any]:
    """Partial-evaluate a program into a fast closure ``f(x)`` with the interpreter's exact semantics.

    The closure resolves every library entry, coercer and argument sub-closure **once**, at compile
    time — so at call time there are no dict lookups, no Prog traversal and no budget bookkeeping, just
    the inlined primitive calls with the same None-propagation and output coercion.
    """
    if prog.kind == "lit":
        v = _coerce(prog.value, prog.rtype, lib)
        return lambda x: v
    if prog.kind == "var":
        rtype = prog.rtype
        return lambda x: _coerce(x, rtype, lib)
    if prog.kind == "hole":
        return lambda x: None
    # app
    entry = lib.entry(prog.name)
    if entry is None:
        return lambda x: None
    if isinstance(entry, Abstraction):
        return compile_prog(_substitute(entry.body, prog.args), lib)
    impl = entry.impl
    rtype = prog.rtype
    arg_fns = [compile_prog(a, lib) for a in prog.args]

    def kernel(x: Any) -> Any:
        argv = []
        for af in arg_fns:
            v = af(x)
            if v is None:
                return None
            argv.append(v)
        try:
            out = impl(*argv)
        except Exception:  # noqa: BLE001 — identical fail-closed semantics as the interpreter
            return None
        return _coerce(out, rtype, lib)

    return kernel


@dataclass
class KernelResult:
    correct: bool
    speedup: float
    promoted: bool
    interp_s: float
    kernel_s: float
    kernel: Optional[Callable[[Any], Any]] = None

    def to_dict(self) -> dict:
        return {"correct": self.correct, "speedup": round(self.speedup, 3),
                "promoted": self.promoted, "interp_s": round(self.interp_s, 6),
                "kernel_s": round(self.kernel_s, 6)}


class KernelForge:
    """Compile a program to a kernel and promote it only on a measured correctness+speed win (F15)."""

    def __init__(self, *, repeats: int = 400, min_speedup: float = 1.05) -> None:
        self.repeats = repeats
        self.min_speedup = min_speedup

    def _time(self, fn: Callable[[Any], Any], inputs: Sequence[Any]) -> float:
        t0 = time.perf_counter()
        for _ in range(self.repeats):
            for x in inputs:
                fn(x)
        return time.perf_counter() - t0

    def optimize(self, prog: Prog, lib: Library, inputs: Sequence[Any]) -> KernelResult:
        kernel = compile_prog(prog, lib)
        # correctness gate: the kernel must reproduce the interpreter EXACTLY on the battery
        correct = all(kernel(x) == evaluate(prog, x, lib) for x in inputs)
        if not correct:
            return KernelResult(False, 0.0, False, 0.0, 0.0)     # never promote a wrong kernel
        interp_s = self._time(lambda x: evaluate(prog, x, lib), inputs)
        kernel_s = self._time(kernel, inputs)
        speedup = interp_s / kernel_s if kernel_s > 0 else 1.0
        promoted = speedup >= self.min_speedup
        return KernelResult(correct, speedup, promoted, interp_s, kernel_s,
                            kernel=kernel if promoted else None)
