"""NYXARA · growth/intelligence.py — the Intelligence Index (📈, the RSIE equation).

The Recursive Self-Improvement Engine grows NYXARA's intelligence each cycle. This module
makes that growth an explicit, measured, *persisted* quantity rather than an implicit hope —
the literal flywheel equation the Master asked for::

    I_(t+1) = f(I_t, C_available)

where ``I`` is a bounded intelligence index in ``[0, 1]`` and ``C_available`` is the compute
NYXARA actually has (CPU / RAM / GPU, read honestly by :mod:`nyxara.kernel.compute`). Each
recursive-self-improvement cycle measures REAL signals — benchmark accuracy, the
own-model handoff rate, how many weaknesses she resolved, and how much she now knows — blends
them into a target, scales the realized gain by available compute, and folds it into the prior
index with momentum (so progress is smoothed, not jittery). The new index is then used to
**scale her improvement effort** (how many source edits she'll attempt, how deep she benchmarks)
by what the machine can carry.

The index survives restarts by riding NYXARA's existing long-term memory (one protected,
high-importance SEMANTIC record tagged ``deep-synapse``) — no new persistence format, no
schema migration. It degrades gracefully: with no memory it lives in-process; with no torch /
``/proc`` the compute read is still a valid ``[0, 1]`` capacity (compute.py guarantees it never
raises). Nothing here modifies source or touches a gate — it only *measures* and *advises*.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

__all__ = ["IntelligenceState", "IntelligenceIndex"]

_INDEX_TAG = "intelligence-index"


# --------------------------------------------------------------------------- #
# The thing that grows
# --------------------------------------------------------------------------- #
@dataclass
class IntelligenceState:
    """One settled reading of NYXARA's intelligence — the ``I_t`` in ``I_(t+1)=f(I_t,C)``."""

    index: float = 0.0
    t: int = 0
    last_inputs: Dict[str, float] = field(default_factory=dict)
    compute: Dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"index": round(float(self.index), 6), "t": int(self.t),
                "last_inputs": {k: round(float(v), 6) for k, v in self.last_inputs.items()},
                "compute": dict(self.compute), "at": self.at}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IntelligenceState":
        if not isinstance(d, dict):
            return cls()
        return cls(index=float(d.get("index", 0.0) or 0.0), t=int(d.get("t", 0) or 0),
                   last_inputs=dict(d.get("last_inputs", {}) or {}),
                   compute=dict(d.get("compute", {}) or {}),
                   at=float(d.get("at", time.time()) or time.time()))

    def summary(self) -> str:
        return (f"I_{self.t} = {self.index:.4f}  "
                f"(inputs={ {k: round(v, 3) for k, v in self.last_inputs.items()} }, "
                f"device={self.compute.get('recommended_device', '?')})")


# --------------------------------------------------------------------------- #
# The function f(I_t, C_available)
# --------------------------------------------------------------------------- #
class IntelligenceIndex:
    """Measure, update, persist and *act on* NYXARA's intelligence index.

    Parameters
    ----------
    memory:    a :class:`~nyxara.memory.store.MemoryStore` for cross-restart persistence
               (optional — falls back to in-process state when absent).
    settings:  :class:`~nyxara.kernel.config.NyxaraSettings` (pulled from ``get_settings`` if
               not supplied); reads ``self_improvement`` weights/momentum/bounds.
    """

    def __init__(self, *, memory: Any = None, settings: Any = None) -> None:
        from nyxara.kernel.config import get_settings
        self.settings = settings or get_settings()
        self.memory = memory
        self._cached: Optional[IntelligenceState] = None

    # ---------------------------------------------------------------------- #
    # Signals — read REAL evidence out of one RSI report (+ memory size)
    # ---------------------------------------------------------------------- #
    def compute_signals(self, report: Any) -> Dict[str, float]:
        """Extract bounded ``[0, 1]`` learning signals from a SelfImprovementReport."""
        bench = getattr(report, "benchmark", None) or {}
        weaknesses = getattr(report, "weaknesses", None) or {}

        accuracy = _clamp01(float(bench.get("accuracy", 0.0) or 0.0))

        # own-model handoff rate: how often her OWN model answered unaided (wrapper -> own AI)
        handoff = bench.get("handoff", {}) or {}
        total = sum(int(v) for v in handoff.values()) if handoff else 0
        handoff_rate = (float(handoff.get("self", 0)) / total) if total else 0.0

        # weaknesses resolved this cycle vs the backlog she still carries (Beta-style estimate)
        kept = float(getattr(report, "kept", 0) or 0)
        n_weak = float(weaknesses.get("n_weaknesses", 0) or 0)
        weaknesses_resolved = (kept + 0.5) / (kept + n_weak + 1.0)

        # how much she now knows — normalized memory footprint + lessons just stored
        mem_size = 0
        try:
            mem_size = len(self.memory) if self.memory is not None else 0
        except Exception:  # noqa: BLE001 — sizing memory is best-effort
            mem_size = 0
        lessons = float(getattr(report, "lessons_stored", 0) or 0)
        knowledge = _clamp01((mem_size + lessons) / 500.0)

        return {"accuracy": accuracy, "handoff": _clamp01(handoff_rate),
                "weaknesses": _clamp01(weaknesses_resolved), "knowledge": knowledge}

    # ---------------------------------------------------------------------- #
    # Compute capacity C_available -> [0, 1]
    # ---------------------------------------------------------------------- #
    @staticmethod
    def compute_capacity(compute: Any) -> float:
        """Map a ComputeReport (CPU / RAM / GPU) to a single ``[0, 1]`` capacity. Never raises."""
        try:
            cpu = int(getattr(compute, "cpu_count", 1) or 1)
            mem = getattr(compute, "memory", {}) or {}
            avail = mem.get("available_mb") or mem.get("total_mb") or 0
            gpu = getattr(compute, "gpu", None)
            gpu_ok = bool(getattr(gpu, "available", False))
        except Exception:  # noqa: BLE001
            return 0.0
        cpu_score = min(1.0, math.log2(cpu + 1) / math.log2(33))   # ~32 cores -> 1.0
        mem_score = min(1.0, float(avail) / 16384.0)               # ~16 GB available -> 1.0
        gpu_score = 1.0 if gpu_ok else 0.0
        return _clamp01(0.4 * cpu_score + 0.3 * mem_score + 0.3 * gpu_score)

    # ---------------------------------------------------------------------- #
    # The equation: I_(t+1) = f(I_t, C_available)
    # ---------------------------------------------------------------------- #
    def update(self, prior: IntelligenceState, signals: Dict[str, float],
               compute: Any) -> IntelligenceState:
        """Fold this cycle's signals + available compute into the prior index, with momentum."""
        weights = dict(getattr(self.settings.self_improvement, "intelligence_weights", {}) or {})
        if not weights:
            weights = {"accuracy": 0.4, "knowledge": 0.2, "weaknesses": 0.2, "handoff": 0.2}
        wsum = sum(weights.values()) or 1.0
        base = sum(weights.get(k, 0.0) * float(signals.get(k, 0.0)) for k in weights) / wsum

        capacity = self.compute_capacity(compute)
        # compute modulates the realized gain (a stronger machine converts evidence into more
        # capability), but never zeroes it — a smart small machine still grows, just slower.
        realized = base * (0.5 + 0.5 * capacity)

        momentum = float(getattr(self.settings.self_improvement, "intelligence_momentum", 0.7))
        momentum = min(1.0, max(0.0, momentum))
        index = _clamp01(momentum * float(prior.index) + (1.0 - momentum) * realized)

        compute_d = compute.to_dict() if hasattr(compute, "to_dict") else dict(compute or {})
        inputs = dict(signals)
        inputs["base"] = round(base, 6)
        inputs["capacity"] = round(capacity, 6)
        return IntelligenceState(index=index, t=prior.t + 1, last_inputs=inputs,
                                 compute=compute_d, at=time.time())

    # ---------------------------------------------------------------------- #
    # Acting on the index — scale improvement effort by I_t and compute
    # ---------------------------------------------------------------------- #
    def effort_budget(self, state: IntelligenceState, compute: Any) -> Dict[str, Any]:
        """Translate the index + compute into how hard NYXARA should push this cycle.

        Returns a bounded budget (never exceeds config limits). When
        ``self_improvement.scale_effort_by_compute`` is off, returns the unscaled config max.
        """
        cfg = self.settings.self_improvement
        cap_max = int(getattr(cfg, "max_edits_per_cycle", 3))
        if not bool(getattr(cfg, "scale_effort_by_compute", True)):
            return {"max_edits_per_cycle": cap_max, "recursion_depth": None,
                    "benchmark_full": True, "intensity": 1.0}
        capacity = self.compute_capacity(compute)
        intensity = _clamp01(0.5 * float(state.index) + 0.5 * capacity)
        max_edits = int(round(intensity * cap_max))
        recursion_depth = max(1, min(20, 1 + int(round(intensity * 19))))
        return {"max_edits_per_cycle": max_edits, "recursion_depth": recursion_depth,
                "benchmark_full": capacity >= 0.6, "intensity": round(intensity, 4)}

    # ---------------------------------------------------------------------- #
    # Persistence (rides the existing long-term memory, protected from forgetting)
    # ---------------------------------------------------------------------- #
    def load(self) -> IntelligenceState:
        """Load the last persisted state, or a zeroed ``I_0`` on a fresh store."""
        if self._cached is not None:
            return self._cached
        rec = self._find_record()
        if rec is not None:
            state = IntelligenceState.from_dict(rec.metadata.get("state", {}))
        else:
            state = IntelligenceState()
        self._cached = state
        return state

    def save(self, state: IntelligenceState) -> bool:
        """Persist ``state`` in place (reconsolidate the single record, or create it once)."""
        self._cached = state
        if self.memory is None:
            return False
        try:
            from nyxara.memory.provenance import Provenance, SourceType
            from nyxara.memory.store import MemoryType
        except Exception:  # noqa: BLE001
            return False
        tag = str(getattr(self.settings.memory, "deep_synapse_tag", "deep-synapse"))
        content = f"[intelligence-index] I_{state.t} = {state.index:.4f}"
        meta = {"state": state.to_dict()}
        try:
            existing = self._find_record()
            if existing is not None:
                # forget the old reading and write the fresh one (keeps exactly one record)
                self.memory.forget(existing.mem_id)
            self.memory.remember(
                content, mem_type=MemoryType.SEMANTIC,
                provenance=Provenance(SourceType.SELF_REFLECTION, confidence=0.9),
                importance=1.0, tags=[_INDEX_TAG, tag], metadata=meta)
            return True
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            return False

    def _find_record(self) -> Any:
        if self.memory is None:
            return None
        try:
            for rec in self.memory._kv.values():
                if _INDEX_TAG in rec.tags and "state" in rec.metadata:
                    return rec
        except Exception:  # noqa: BLE001
            return None
        return None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:  # noqa: BLE001
        return 0.0


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.kernel.compute import ComputeReport, GPUInfo
    from nyxara.memory.store import MemoryStore

    print("=" * 70)
    print("NYXARA intelligence-index self-test")
    print("=" * 70)

    idx = IntelligenceIndex(memory=MemoryStore())

    # a synthetic "good cycle" report
    class _Rep:
        benchmark = {"accuracy": 0.8, "handoff": {"self": 6, "teacher": 4, "none": 0}}
        weaknesses = {"n_weaknesses": 2}
        kept = 1
        lessons_stored = 10

    big = ComputeReport(cpu_count=16, memory={"available_mb": 32000, "total_mb": 64000},
                        gpu=GPUInfo(available=True, backend="cuda", count=1, names=["A100"]),
                        torch_available=True)
    small = ComputeReport(cpu_count=1, memory={"available_mb": 512, "total_mb": 1024},
                          gpu=GPUInfo(), torch_available=False)

    signals = idx.compute_signals(_Rep())
    print(f"\nsignals             : {signals}")
    assert all(0.0 <= v <= 1.0 for v in signals.values())

    cap_big = IntelligenceIndex.compute_capacity(big)
    cap_small = IntelligenceIndex.compute_capacity(small)
    print(f"capacity big/small  : {cap_big:.3f} / {cap_small:.3f}")
    assert 0.0 <= cap_small < cap_big <= 1.0

    s0 = IntelligenceState()
    s1 = idx.update(s0, signals, big)
    print(f"I_0 -> I_1          : {s0.index:.4f} -> {s1.index:.4f}  (t={s1.t})")
    assert 0.0 <= s1.index <= 1.0 and s1.t == 1 and s1.index > s0.index

    # higher accuracy must not lower the index
    hi = idx.update(s0, {**signals, "accuracy": 1.0}, big)
    lo = idx.update(s0, {**signals, "accuracy": 0.0}, big)
    assert hi.index >= lo.index

    # effort scales with compute: a big box earns a larger edit budget than a tiny one
    budget_big = idx.effort_budget(s1, big)
    budget_small = idx.effort_budget(s1, small)
    print(f"effort big/small    : {budget_big} / {budget_small}")
    assert budget_big["max_edits_per_cycle"] >= budget_small["max_edits_per_cycle"]
    assert 1 <= budget_big["recursion_depth"] <= 20
    assert budget_big["max_edits_per_cycle"] <= idx.settings.self_improvement.max_edits_per_cycle

    # persistence round-trips through MemoryStore as ONE protected record
    idx.save(s1)
    idx2 = IntelligenceIndex(memory=idx.memory)
    loaded = idx2.load()
    print(f"persisted/loaded    : {loaded.summary()}")
    assert loaded.t == s1.t and abs(loaded.index - s1.index) < 1e-6
    rec = idx2._find_record()
    assert rec is not None and rec.importance >= 0.9
    assert idx.settings.memory.deep_synapse_tag in rec.tags

    # a fresh store starts at a zeroed I_0
    assert IntelligenceIndex(memory=MemoryStore()).load().t == 0

    print("\nALL SELF-TESTS PASSED ✓")
