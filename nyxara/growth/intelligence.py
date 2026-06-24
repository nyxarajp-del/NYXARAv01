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

# the four capability dimensions the index tracks, and the growth action that invests in each
_DIMENSIONS = ("accuracy", "handoff", "weaknesses", "knowledge")
_ACTION_FOR = {
    "accuracy":   "deepen_reasoning",      # weak reasoning → think harder / longer
    "handoff":    "train_self_model",      # over-reliant on the teacher → forge own brain
    "weaknesses": "resolve_weaknesses",    # backlog of unfixed weaknesses → edit source
    "knowledge":  "acquire_knowledge",     # thin knowledge → research / distil more
}
# relative compute cost of each action — the planner discounts an expensive action on a weak
# machine (you cannot afford to forge a brain on a Raspberry Pi), so the chosen investment is one
# she can actually pay for, not merely the one with the largest nominal deficit.
_ACTION_COST = {"deepen_reasoning": 0.2, "resolve_weaknesses": 0.4,
                "acquire_knowledge": 0.5, "train_self_model": 1.0}


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
    history: list = field(default_factory=list)   # recent index readings — for trend/plateau
    # recent EXTERNAL held-out / adversarial scores — the transfer signal whose trend (over the
    # config transfer-window) tells whether proxy gains are *real* or merely Goodharted overfitting.
    validation_history: list = field(default_factory=list)
    # per-dimension Beta(α, β) posteriors over "this dimension is healthy" — the uncertainty the
    # planner samples (a never-tried dimension is wide; a consistently-weak one is confidently low)
    dim_alpha: Dict[str, float] = field(default_factory=dict)
    dim_beta: Dict[str, float] = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"index": round(float(self.index), 6), "t": int(self.t),
                "last_inputs": {k: round(float(v), 6) for k, v in self.last_inputs.items()},
                "compute": dict(self.compute),
                "history": [round(float(v), 6) for v in self.history],
                "validation_history": [round(float(v), 6) for v in self.validation_history],
                "dim_alpha": {k: round(float(v), 6) for k, v in self.dim_alpha.items()},
                "dim_beta": {k: round(float(v), 6) for k, v in self.dim_beta.items()},
                "at": self.at}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IntelligenceState":
        if not isinstance(d, dict):
            return cls()
        return cls(index=float(d.get("index", 0.0) or 0.0), t=int(d.get("t", 0) or 0),
                   last_inputs=dict(d.get("last_inputs", {}) or {}),
                   compute=dict(d.get("compute", {}) or {}),
                   history=list(d.get("history", []) or []),
                   validation_history=list(d.get("validation_history", []) or []),
                   dim_alpha={k: float(v) for k, v in (d.get("dim_alpha", {}) or {}).items()},
                   dim_beta={k: float(v) for k, v in (d.get("dim_beta", {}) or {}).items()},
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

        # stability: of the self-edits she attempted, what fraction *stuck* (kept vs rolled back).
        # A diagnostic signal (not folded into the weighted index) the growth directive reads.
        rolled = float(getattr(report, "rolled_back", 0) or 0)
        stability = _clamp01((kept + 1.0) / (kept + rolled + 1.0))

        # Pillar F diagnostics (read opportunistically; not folded into the weighted index unless
        # the Master adds weights for them). `frontier` = open-ended coverage growth; `rigor` =
        # fraction of conclusions she could certify with a proof. Default 0.0 when absent.
        frontier = _clamp01(float(getattr(report, "frontier", 0.0) or 0.0))
        rigor = _clamp01(float(getattr(report, "rigor", 0.0) or 0.0))

        # EXTERNAL ground truth: the combined held-out + adversarial score the RSI loop measured on
        # tasks it never edits against (recursive_improvement.run_validation). This is the only
        # signal NYXARA cannot Goodhart — it rises only when capability genuinely transfers. Default
        # 0.0 when validation was not run, so the index degrades exactly as before.
        validation = getattr(report, "validation", None) or {}
        transfer = _clamp01(float(validation.get("transfer_score", 0.0) or 0.0)) \
            if isinstance(validation, dict) else 0.0

        return {"accuracy": accuracy, "handoff": _clamp01(handoff_rate),
                "weaknesses": _clamp01(weaknesses_resolved), "knowledge": knowledge,
                "transfer": transfer,
                "stability": stability, "frontier": frontier, "rigor": rigor}

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
               compute: Any, *, validated: Optional[bool] = None) -> IntelligenceState:
        """Fold this cycle's signals + available compute into the prior index, with momentum.

        ``validated`` declares whether an EXTERNAL held-out / adversarial score was measured this
        cycle (the ``transfer`` signal is real, not a default 0.0). When it is, the held-out score
        is tracked and the **Goodhart guard** runs: if the self-referential proxy rose but transfer
        did not improve across the configured window, the index gain is discounted so capability is
        only ever *claimed* when it genuinely transfers. When ``None`` it is inferred from whether a
        ``transfer`` signal was supplied — so callers that never validate behave exactly as before.
        """
        cfg = self.settings.self_improvement
        weights = dict(getattr(cfg, "intelligence_weights", {}) or {})
        if not weights:
            weights = {"accuracy": 0.4, "knowledge": 0.2, "weaknesses": 0.2, "handoff": 0.2}
        wsum = sum(weights.values()) or 1.0
        base = sum(weights.get(k, 0.0) * float(signals.get(k, 0.0)) for k in weights) / wsum

        capacity = self.compute_capacity(compute)
        # compute modulates the realized gain (a stronger machine converts evidence into more
        # capability), but never zeroes it — a smart small machine still grows, just slower.
        realized = base * (0.5 + 0.5 * capacity)

        if validated is None:
            validated = "transfer" in signals
        # ---- external validation: track held-out transfer and guard against Goodharting ---- #
        transfer = _clamp01(float(signals.get("transfer", 0.0)))
        val_history = list(prior.validation_history)
        if validated:
            val_history = (val_history[-11:] + [transfer])   # keep the last 12 held-out readings
        window = max(2, int(getattr(cfg, "transfer_window", 3)))
        recent = val_history[-window:]
        transfer_gain = _slope(recent) if len(recent) >= 2 else 0.0   # per-cycle held-out change
        prior_base = float(prior.last_inputs.get("base", prior.index))
        proxy_gain = base - prior_base
        # Goodhart: the proxy climbed (> the plateau epsilon) but held-out transfer did not. The
        # number is rising without real, transferable capability — exactly the failure mode the
        # Master flagged. Only meaningful once the window has filled (≥2 real readings).
        eps = 0.005
        goodhart = bool(validated and getattr(cfg, "goodhart_guard", True)
                        and len(val_history) >= 2
                        and proxy_gain > eps and transfer_gain <= eps)
        if goodhart:
            # refuse to climb on un-transferred gains: keep only a transfer_weight fraction of the
            # gain over the prior index, so the index barely moves until capability actually transfers
            keep = _clamp01(float(getattr(cfg, "transfer_weight", 0.3)))
            realized = float(prior.index) + (realized - float(prior.index)) * keep

        momentum = float(getattr(cfg, "intelligence_momentum", 0.7))
        momentum = min(1.0, max(0.0, momentum))
        index = _clamp01(momentum * float(prior.index) + (1.0 - momentum) * realized)

        compute_d = compute.to_dict() if hasattr(compute, "to_dict") else dict(compute or {})
        history = (list(prior.history)[-11:] + [index])     # keep the last 12 readings
        inputs = dict(signals)
        inputs["base"] = round(base, 6)
        inputs["capacity"] = round(capacity, 6)
        inputs["trend"] = round(_slope(history), 6)         # rising / plateau / declining
        inputs["transfer"] = round(transfer, 6)             # this cycle's external held-out score
        inputs["transfer_gain"] = round(transfer_gain, 6)   # held-out trend over the window
        inputs["goodhart"] = 1.0 if goodhart else 0.0       # was the proxy diverging from transfer?
        # fold each dimension's signal into its Beta posterior (fractional update): a healthy
        # signal credits α, a weak one credits β. This is what the uncertainty-aware planner reads.
        dim_alpha = dict(prior.dim_alpha)
        dim_beta = dict(prior.dim_beta)
        for k in _DIMENSIONS:
            s = _clamp01(float(signals.get(k, 0.0)))
            dim_alpha[k] = float(dim_alpha.get(k, 1.0)) + s
            dim_beta[k] = float(dim_beta.get(k, 1.0)) + (1.0 - s)
        return IntelligenceState(index=index, t=prior.t + 1, last_inputs=inputs,
                                 compute=compute_d, history=history,
                                 validation_history=val_history,
                                 dim_alpha=dim_alpha, dim_beta=dim_beta, at=time.time())

    # ---------------------------------------------------------------------- #
    # Acting on the index — diagnose the bottleneck and DIRECT the next investment
    # ---------------------------------------------------------------------- #
    def growth_directive(self, state: IntelligenceState, compute: Any = None) -> Dict[str, Any]:
        """Turn the index into a *decision*: which capability dimension is the bottleneck, what
        growth action to take next, and whether progress has plateaued (so effort should
        escalate). This is what makes the index a closed-loop *driver* rather than a readout —
        the weakest real signal steers where the next cycle invests.

        Reads the most recent signals off ``state.last_inputs``, so it is computable up front
        (before a cycle) from the persisted prior state alone."""
        sig = {k: float(state.last_inputs.get(k, 0.0))
               for k in _DIMENSIONS if k in state.last_inputs}
        action_for = _ACTION_FOR
        if not sig:
            return {"focus": "accuracy", "action": "deepen_reasoning", "weakest_value": 0.0,
                    "trend": 0.0, "plateaued": False, "escalate": False,
                    "rationale": "no signals yet — default to deepening reasoning"}
        focus = min(sig, key=lambda k: sig[k])               # the weakest dimension leads
        trend = float(state.last_inputs.get("trend", _slope(state.history)))
        plateaued = bool(state.t >= 3 and abs(trend) < 0.005 and state.index < 0.95)
        capacity = self.compute_capacity(compute) if compute is not None else 0.0
        escalate = bool(plateaued and capacity >= 0.4)       # stalled + room to push → push harder
        # external-validation health: did the last cycle's proxy diverge from held-out transfer?
        transfer_gain = float(state.last_inputs.get("transfer_gain", 0.0))
        goodhart = bool(state.last_inputs.get("goodhart", 0.0) >= 0.5)
        return {
            "focus": focus, "action": action_for[focus], "weakest_value": round(sig[focus], 4),
            "trend": round(trend, 6), "plateaued": plateaued, "escalate": escalate,
            "transfer_gain": round(transfer_gain, 6), "goodhart": goodhart,
            "rationale": (f"weakest dimension '{focus}'={sig[focus]:.2f}; trend={trend:+.4f}"
                          + (f"; transfer={transfer_gain:+.4f}" if "transfer_gain"
                             in state.last_inputs else "")
                          + ("; ⚠ GOODHART: proxy rose but held-out did not — gain discounted"
                             if goodhart else "")
                          + ("; plateaued — escalate effort" if escalate else "")),
        }

    # ---------------------------------------------------------------------- #
    # Acting on the index — PLAN the next action under uncertainty (lookahead)
    # ---------------------------------------------------------------------- #
    def plan_action(self, state: IntelligenceState, compute: Any = None, *,
                    rng: Any = None, action_scores: Optional[Dict[str, float]] = None
                    ) -> Dict[str, Any]:
        """Choose the next growth action by Thompson-sampling each dimension's *deficit*,
        discounted by what the machine can afford and (optionally) by the learned payoff of each
        action from the improvement ledger.

        Unlike :meth:`growth_directive` (which greedily picks the single weakest point estimate),
        this samples each dimension's Beta posterior — so an under-measured dimension is explored,
        a confidently-healthy one is left alone, and the *expected value* (deficit × affordability ×
        learned-payoff) decides where the next cycle invests. This is the planning step that turns
        a readout into a decision under uncertainty.

        Falls back to :meth:`growth_directive` when no per-dimension posteriors exist yet (a fresh
        state), so early cycles behave exactly as before."""
        import random as _random
        r = rng or _random.Random()
        if not state.dim_alpha:
            return self.growth_directive(state, compute)
        capacity = self.compute_capacity(compute) if compute is not None else 0.5
        evs: Dict[str, float] = {}
        deficits: Dict[str, float] = {}
        for dim in _DIMENSIONS:
            a = float(state.dim_alpha.get(dim, 1.0))
            b = float(state.dim_beta.get(dim, 1.0))
            # sample the deficit posterior Beta(β, α): a high draw means "likely unhealthy" —
            # exactly the dimension worth investing in this cycle
            try:
                deficit = r.betavariate(max(1e-3, b), max(1e-3, a))
            except Exception:  # noqa: BLE001
                deficit = b / (a + b) if (a + b) > 0 else 0.5
            deficits[dim] = deficit
            action = _ACTION_FOR[dim]
            cost = _ACTION_COST.get(action, 0.5)
            # affordability: a costly action is discounted on a weak machine, never on a strong one
            afford = 1.0 - cost * (1.0 - capacity)
            learned = float((action_scores or {}).get(action, 1.0))
            evs[action] = deficit * max(0.05, afford) * max(0.05, learned)
        best_dim = max(_DIMENSIONS, key=lambda d: evs[_ACTION_FOR[d]])
        trend = float(state.last_inputs.get("trend", _slope(state.history)))
        plateaued = bool(state.t >= 3 and abs(trend) < 0.005 and state.index < 0.95)
        escalate = bool(plateaued and capacity >= 0.4)
        transfer_gain = float(state.last_inputs.get("transfer_gain", 0.0))
        goodhart = bool(state.last_inputs.get("goodhart", 0.0) >= 0.5)
        return {
            "focus": best_dim, "action": _ACTION_FOR[best_dim],
            "weakest_value": round(1.0 - deficits[best_dim], 4),
            "expected_values": {k: round(v, 4) for k, v in evs.items()},
            "trend": round(trend, 6), "plateaued": plateaued, "escalate": escalate,
            "transfer_gain": round(transfer_gain, 6), "goodhart": goodhart,
            "planner": "thompson",
            "rationale": (f"planned action '{_ACTION_FOR[best_dim]}' for '{best_dim}' "
                          f"(EV={evs[_ACTION_FOR[best_dim]]:.3f}, capacity={capacity:.2f})"
                          + ("; ⚠ GOODHART: held-out flat while proxy rose" if goodhart else "")
                          + ("; plateaued — escalate effort" if escalate else "")),
        }

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


def _slope(values: Any) -> float:
    """Least-squares slope of a short index history (per-cycle change). 0.0 for <2 points."""
    try:
        ys = [float(v) for v in values]
    except Exception:  # noqa: BLE001
        return 0.0
    n = len(ys)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 0:
        return 0.0
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / denom


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

    # the index DIRECTS growth: it names the weakest dimension and the action to take
    weak_handoff = idx.update(s0, {**signals, "handoff": 0.0, "accuracy": 0.9,
                                   "weaknesses": 0.9, "knowledge": 0.9}, big)
    d = idx.growth_directive(weak_handoff, big)
    print(f"\ndirective           : {d['action']} (focus={d['focus']}, "
          f"escalate={d['escalate']})")
    assert d["focus"] == "handoff" and d["action"] == "train_self_model"

    # a plateau (flat history) on a capable machine escalates effort
    flat = IntelligenceState(index=0.5, t=5, history=[0.5] * 6,
                             last_inputs={"accuracy": 0.4, "handoff": 0.9,
                                          "weaknesses": 0.9, "knowledge": 0.9, "trend": 0.0})
    dp = idx.growth_directive(flat, big)
    assert dp["plateaued"] and dp["escalate"] and dp["focus"] == "accuracy"
    print(f"plateau directive   : {dp['rationale']}")

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

    # --- anti-Goodhart: a rising proxy with FLAT held-out transfer must not inflate the index --- #
    # Two histories from the same starting state: one where transfer rises with the proxy (honest),
    # one where the proxy is forced up while held-out stays flat (gaming). The honest run must end
    # higher, and the gamed run must raise the Goodhart flag.
    idx3 = IntelligenceIndex()
    honest = gamed = IntelligenceState()
    base_sig = {"accuracy": 0.6, "handoff": 0.6, "weaknesses": 0.6, "knowledge": 0.6}
    for k in range(1, 5):
        proxy = min(1.0, 0.6 + 0.1 * k)               # proxy climbs every cycle in BOTH runs
        honest = idx3.update(honest, {**base_sig, "accuracy": proxy, "transfer": proxy}, big)
        gamed = idx3.update(gamed, {**base_sig, "accuracy": proxy, "transfer": 0.3}, big)
    print(f"\nhonest vs gamed     : I={honest.index:.4f} vs {gamed.index:.4f} "
          f"(gamed goodhart={bool(gamed.last_inputs.get('goodhart'))})")
    assert honest.index > gamed.index, "transferred gains must beat Goodharted ones"
    assert gamed.last_inputs.get("goodhart") == 1.0, "flat held-out + rising proxy must flag Goodhart"
    assert honest.last_inputs.get("goodhart") == 0.0, "honest transfer must not flag Goodhart"
    assert len(gamed.validation_history) >= 4 and gamed.last_inputs.get("transfer") == 0.3

    print("\nALL SELF-TESTS PASSED ✓")
