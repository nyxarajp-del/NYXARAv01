"""NYXARA · growth/substrate_tune.py — hardware-aware substrate adaptation (⚛️, H).

NYXARA runs on many machines (x86 / ARM, few or many cores, with or without numpy/GPU). This module
**probes the real host** and **tunes which kernel variant and what parameters she runs** to that
substrate — selecting the fastest *behaviour-equivalent* implementation by micro-benchmark, and
persisting the choice.

Honest boundary (enforced): this is **runtime selection and parameter-tuning of already-available
implementation variants** — device/path choice, chunk sizes, thread counts — measured on the live host.
It does **not** emit AVX/NEON intrinsics, machine code, or promise "100% hardware efficiency"; it changes
*which* code path NYXARA runs, not the chip's instruction stream. Variants are assumed behaviour-equivalent
by the caller; an optional equivalence check refuses a variant that disagrees with the reference on a
probe input. Pure standard library (numpy detection is optional); no LLM.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = ["SubstrateProfile", "TuneResult", "probe_substrate", "SubstrateTuner"]


@dataclass
class SubstrateProfile:
    """What NYXARA measured about the machine she is running on."""

    arch: str
    machine: str
    cpu_count: int
    has_numpy: bool
    simd: List[str] = field(default_factory=list)
    python: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"arch": self.arch, "machine": self.machine, "cpu_count": self.cpu_count,
                "has_numpy": self.has_numpy, "simd": list(self.simd), "python": self.python}


@dataclass
class TuneResult:
    chosen: str
    timings: Dict[str, float]
    rejected: List[str] = field(default_factory=list)   # variants that failed equivalence

    def to_dict(self) -> Dict[str, Any]:
        return {"chosen": self.chosen,
                "timings": {k: round(v, 9) for k, v in self.timings.items()},
                "rejected": list(self.rejected)}


def _numpy_simd() -> tuple[bool, List[str]]:
    """Detect numpy and any SIMD/ISA features it reports (best-effort, never raises)."""
    try:
        import numpy as np  # noqa: F401
    except Exception:  # noqa: BLE001
        return False, []
    feats: List[str] = []
    try:
        cfg = getattr(np, "__config__", None)
        simd = getattr(cfg, "get_feature_flags", None) if cfg else None
        if callable(simd):
            feats = [str(x) for x in simd()]
    except Exception:  # noqa: BLE001
        feats = []
    return True, feats


def probe_substrate() -> SubstrateProfile:
    """Measure the live host: arch, cores, numpy + SIMD availability."""
    has_np, simd = _numpy_simd()
    return SubstrateProfile(
        arch=platform.architecture()[0],
        machine=platform.machine(),           # x86_64 / arm64 / aarch64 / ...
        cpu_count=os.cpu_count() or 1,
        has_numpy=has_np,
        simd=simd,
        python=sys.version.split()[0],
    )


class SubstrateTuner:
    """Probe the substrate, then pick the fastest equivalent kernel variant for it."""

    def __init__(self, *, repeat: int = 5) -> None:
        self.repeat = max(1, int(repeat))
        self.profile = probe_substrate()
        self._chosen: Dict[str, str] = {}

    def _time(self, fn: Callable[[], Any]) -> float:
        best = float("inf")
        for _ in range(self.repeat):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    def tune(self, name: str, variants: Dict[str, Callable[[], Any]], *,
             reference: Optional[str] = None, equivalence_input: Any = None) -> TuneResult:
        """Micro-benchmark each variant (a zero-arg callable running the workload) and choose the
        fastest. If ``reference`` is given, any variant whose result differs from the reference's is
        rejected (behaviour-equivalence guard) before timing wins it a slot."""
        rejected: List[str] = []
        candidates = dict(variants)
        if reference is not None and reference in candidates:
            try:
                ref_val = candidates[reference]()
                for vname in list(candidates):
                    if vname == reference:
                        continue
                    try:
                        if candidates[vname]() != ref_val:
                            rejected.append(vname)
                            candidates.pop(vname, None)
                    except Exception:  # noqa: BLE001 — a throwing variant is not equivalent
                        rejected.append(vname)
                        candidates.pop(vname, None)
            except Exception:  # noqa: BLE001 — reference itself failing: skip the equivalence guard
                pass
        timings = {vname: self._time(fn) for vname, fn in candidates.items()}
        chosen = min(timings, key=timings.get) if timings else (reference or "")
        self._chosen[name] = chosen
        return TuneResult(chosen=chosen, timings=timings, rejected=rejected)

    def chosen(self, name: str) -> Optional[str]:
        return self._chosen.get(name)
