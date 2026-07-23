"""NYXARA · growth/superoptimize.py — equivalence-checked super-optimization (🧬, F).

Beyond tidying code (:mod:`nyxara.growth.mdl_refactor`), NYXARA can *re-synthesize* a hot function into a
faster one — but a faster function that computes the wrong answer is worthless. This module keeps a
candidate implementation only if it is **equivalent to the original over a large adversarial input
battery** (Popperian falsification, the same discipline as :mod:`nyxara.growth.redteam`) **and measurably
faster**. Anything that disagrees on a single input is rejected, however fast it is.

Honest boundary: equivalence is established **empirically over a boundary + sampled battery**, not by a
universal formal proof, and speedups are ordinary **pure-Python / numpy** algorithmic wins — this module
does **not** emit assembly, C-extensions, or custom bytecode, and makes no "1000×, provably leak-free"
claim. It selects a verified-equivalent, benchmarked-faster candidate, or honestly keeps the original when
none qualifies. Candidate *generation* (GP / grammar search) plugs in from the caller; this module owns
the verify-and-select decision. Pure standard library; no LLM.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["EquivResult", "SuperOptResult", "SuperOptimizer"]

Case = Tuple[tuple, dict]   # (args, kwargs)


@dataclass
class EquivResult:
    equivalent: bool
    tested: int
    counterexample: Optional[Any] = None
    reason: str = ""


@dataclass
class SuperOptResult:
    chosen: str                     # "original" when nothing beat it
    speedup: float                  # original_time / chosen_time (>1 means faster)
    timings: Dict[str, float] = field(default_factory=dict)
    rejected: Dict[str, str] = field(default_factory=dict)   # name -> why (non-equivalent)
    improved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"chosen": self.chosen, "speedup": round(self.speedup, 3),
                "improved": self.improved,
                "timings": {k: round(v, 9) for k, v in self.timings.items()},
                "rejected": dict(self.rejected)}


def _normalise(cases: Sequence[Any]) -> List[Case]:
    out: List[Case] = []
    for c in cases:
        if isinstance(c, tuple) and len(c) == 2 and isinstance(c[0], tuple) and isinstance(c[1], dict):
            out.append(c)              # already (args, kwargs)
        elif isinstance(c, tuple):
            out.append((c, {}))        # a bare args tuple
        else:
            out.append(((c,), {}))     # a single positional value
    return out


class SuperOptimizer:
    """Verify-then-benchmark: keep only a proven-equivalent, faster implementation."""

    def __init__(self, *, repeat: int = 5, min_speedup: float = 1.05) -> None:
        self.repeat = max(1, int(repeat))
        self.min_speedup = float(min_speedup)

    def check_equivalence(self, reference: Callable, candidate: Callable,
                          cases: Sequence[Any]) -> EquivResult:
        """Refute ``candidate`` on the first case where it disagrees with ``reference``."""
        norm = _normalise(cases)
        tested = 0
        for args, kwargs in norm:
            try:
                want = reference(*args, **kwargs)
            except Exception:  # noqa: BLE001 — reference undefined here is not a fair discriminator
                continue
            tested += 1
            try:
                got = candidate(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — a crash where the reference is defined = not equivalent
                return EquivResult(False, tested, counterexample=(args, kwargs),
                                   reason=f"candidate raised {exc.__class__.__name__} on {args!r}")
            if got != want:
                return EquivResult(False, tested, counterexample=(args, kwargs),
                                   reason=f"disagrees on {args!r}: {got!r} != {want!r}")
        return EquivResult(True, tested)

    def _time(self, fn: Callable, cases: List[Case]) -> float:
        best = float("inf")
        for _ in range(self.repeat):
            t0 = time.perf_counter()
            for args, kwargs in cases:
                try:
                    fn(*args, **kwargs)
                except Exception:  # noqa: BLE001 — timing a defined-elsewhere case; skip
                    pass
            best = min(best, time.perf_counter() - t0)
        return best

    def optimize(self, reference: Callable, candidates: Dict[str, Callable],
                 cases: Sequence[Any]) -> SuperOptResult:
        """Return the fastest candidate that is equivalent to ``reference`` and beats ``min_speedup``,
        else keep the original. Non-equivalent candidates are rejected with a reason."""
        norm = _normalise(cases)
        rejected: Dict[str, str] = {}
        verified: Dict[str, Callable] = {}
        for name, fn in candidates.items():
            eq = self.check_equivalence(reference, fn, cases)
            if eq.equivalent:
                verified[name] = fn
            else:
                rejected[name] = eq.reason
        timings = {"original": self._time(reference, norm)}
        for name, fn in verified.items():
            timings[name] = self._time(fn, norm)
        ref_t = timings["original"]
        best_name, best_t = "original", ref_t
        for name in verified:
            if timings[name] < best_t:
                best_name, best_t = name, timings[name]
        speedup = (ref_t / best_t) if best_t > 0 else 1.0
        # only adopt a challenger that clears the minimum-speedup bar
        if best_name != "original" and speedup < self.min_speedup:
            best_name, speedup = "original", 1.0
        return SuperOptResult(chosen=best_name, speedup=speedup, timings=timings,
                              rejected=rejected, improved=best_name != "original")
