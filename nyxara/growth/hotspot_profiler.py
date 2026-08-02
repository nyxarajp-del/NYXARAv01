"""NYXARA · growth/hotspot_profiler.py — find her own hot, native-compilable functions (Part C1).

Before NYXARA can make herself faster, she has to know *where* she is slow. This is the one genuinely
missing input to her existing self-optimization pipeline: a runtime profiler. It runs a representative
workload under ``cProfile`` (stdlib), ranks functions by self-time, and emits the hottest **eligible**
ones — the narrow, pure, native-compilable kind — as candidates for ``growth/native_forge.NativeForge``.

Eligibility is conservative: a candidate must live in NYXARA's own source (not stdlib / site-packages),
not in a constitutional ``PROTECTED_RELPATHS`` file, and — since the profiler cannot prove purity — it is
flagged for the forge's own equivalence gauntlet to confirm. Deterministic timing; no side effects.
"""
from __future__ import annotations

import cProfile
import pstats
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

__all__ = ["Hotspot", "HotspotProfiler"]

# Files that must never be touched by self-optimization (mirrors self_optimize.PROTECTED_RELPATHS).
_PROTECTED = ("kernel/rules.py", "kernel/invariants.py", "kernel/config.py", "identity/values.py",
              "identity/soul.py", "guard/value_learning.py", "guard/corrigibility.py",
              "guard/auth.py", "growth/loyalty.py")


@dataclass
class Hotspot:
    """One hot function ranked by self-time — a candidate for native compilation."""

    func: str            # function name
    filename: str        # source file
    lineno: int
    tottime: float       # cumulative self-time (seconds) across the workload
    ncalls: int
    eligible: bool       # passes the conservative source/location filter

    @property
    def locus(self) -> str:
        return f"{self.filename}:{self.lineno}"

    def to_dict(self) -> dict:
        return {"func": self.func, "locus": self.locus, "tottime": round(self.tottime, 6),
                "ncalls": self.ncalls, "eligible": self.eligible}


class HotspotProfiler:
    """Profile a workload and rank its hottest (eligible) functions (Part C1)."""

    def __init__(self, *, package_marker: str = "nyxara") -> None:
        self.package_marker = package_marker

    def profile(self, workload: Callable[[], Any], *, top: int = 10) -> List[Hotspot]:
        """Run ``workload`` under cProfile and return the hottest functions, self-time descending."""
        pr = cProfile.Profile()
        pr.enable()
        try:
            workload()
        finally:
            pr.disable()
        stats = pstats.Stats(pr)
        rows: List[Hotspot] = []
        for (filename, lineno, func), (ncalls, _nc, tottime, _ct, _cs) in stats.stats.items():  # type: ignore[attr-defined]
            rows.append(Hotspot(func=func, filename=filename, lineno=lineno, tottime=tottime,
                                ncalls=ncalls, eligible=self._eligible(filename, func)))
        rows.sort(key=lambda h: h.tottime, reverse=True)
        return rows[:top]

    def eligible_hotspots(self, workload: Callable[[], Any], *, top: int = 10) -> List[Hotspot]:
        """Only the hottest functions that pass the eligibility filter — the forge's candidate list."""
        return [h for h in self.profile(workload, top=top * 3) if h.eligible][:top]

    def _eligible(self, filename: str, func: str) -> bool:
        fn = (filename or "").replace("\\", "/")
        if self.package_marker not in fn:
            return False                                  # not NYXARA's own code (stdlib/3rd-party)
        if any(p in fn for p in _PROTECTED):
            return False                                  # constitutional file — never
        if func.startswith("<"):
            return False                                  # <listcomp>, <module>, <genexpr>, …
        return True
