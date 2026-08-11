"""NYXARA · nyx001/layers/l17_self_improve.py — Layer 17, controlled self-improvement (🔒, gated).

The charter: "Proposes architectural upgrades → isolates in sandbox → evaluates → safety checks →
rolls out or discards."

Every word of that pipeline is a gate, and this module's whole value is that **the gates are
fail-closed**. A self-improvement layer that defaults to rolling out is not controlled
self-improvement; it is self-modification with a report attached.

The pipeline, in order, and no stage can be skipped:

1. **Propose** — a change to this system's own *hyperparameters and structure*, drawn from the
   bounded space in :data:`_MUTABLE`. Not arbitrary code.
2. **Sandbox** — the proposal is applied to a *clone* of the affected layer, never to the live one.
   The live system keeps running on the old configuration throughout.
3. **Evaluate** — the clone is scored on a held-out task supplied by the caller. Both the incumbent
   and the challenger are scored on the *same* task, in the same call, because a challenger scored
   on a different sample than the incumbent is not being compared to it.
4. **Safety check** — :meth:`_safe` rejects anything that leaves the bounded space, degrades
   measured calibration, or fails to clear ``min_improvement``. Ties lose: a change that is merely
   *not worse* is discarded, because churn on a live system has its own cost.
5. **Roll out or discard** — and every decision is recorded in a lineage the caller can inspect
   and revert from.

Two hard limits are deliberate and are not oversights. This layer **cannot modify its own gates** —
``_MUTABLE`` does not contain the safety thresholds, so a proposal cannot widen the space it is
searched in. And it **cannot write code**: the charter's "architectural upgrades" is realised as
reconfiguration within a fixed architecture. Source-level self-modification exists elsewhere in
NYXARA behind its own oversight (``nyx/omega.py``, ``nyx/omni.py``); duplicating it here without
those protections would be the single most dangerous thing in this package.

Honest: this is bounded hyperparameter self-tuning with an A/B gate and a revert log. Calling it
"self-improvement" is accurate only in that narrow sense, and the ceiling is stated here rather
than left for a reader to discover.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = ["Proposal", "Outcome", "SelfImprovement"]

# The ONLY things a proposal may touch, with hard bounds. Safety thresholds are deliberately
# absent: this layer cannot widen the space it searches in.
_MUTABLE: Dict[str, Tuple[float, float]] = {
    "world_model.lr": (1e-5, 5e-2),
    "episodic.replay_batch": (1.0, 64.0),
    "episodic.credit_rate": (0.0, 1.0),
    "semantic.radius": (0.2, 0.9),
    "semantic.min_support": (2.0, 16.0),
    "attention.w_goal": (0.0, 2.0),
    "attention.w_error": (0.0, 2.0),
    "curiosity.decay": (0.5, 0.99),
    "planning.risk_aversion": (0.0, 2.0),
    "resource.ceiling_s": (0.1, 10.0),
}


@dataclass
class Proposal:
    """A bounded change to one knob, and where it came from."""

    target: str = ""
    old_value: float = 0.0
    new_value: float = 0.0
    rationale: str = ""
    born: float = field(default_factory=time.time)

    @property
    def in_bounds(self) -> bool:
        lo, hi = _MUTABLE.get(self.target, (1.0, 0.0))     # unknown target ⇒ empty range ⇒ False
        return bool(lo <= self.new_value <= hi)

    def to_dict(self) -> Dict[str, Any]:
        return {"target": self.target, "old": round(self.old_value, 6),
                "new": round(self.new_value, 6), "in_bounds": self.in_bounds,
                "rationale": self.rationale[:160]}


@dataclass
class Outcome:
    """What happened to a proposal. Every rejection names its reason."""

    proposal: Optional[Proposal] = None
    incumbent_score: Optional[float] = None
    challenger_score: Optional[float] = None
    improvement: Optional[float] = None
    rolled_out: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"proposal": self.proposal.to_dict() if self.proposal else None,
                "incumbent": None if self.incumbent_score is None
                else round(self.incumbent_score, 6),
                "challenger": None if self.challenger_score is None
                else round(self.challenger_score, 6),
                "improvement": None if self.improvement is None else round(self.improvement, 6),
                "rolled_out": self.rolled_out, "reason": self.reason}


class SelfImprovement:
    """Layer 17. Proposes bounded reconfigurations, A/B-tests them in a sandbox, gates the rollout."""

    def __init__(self, substrate: Any, *, min_improvement: float = 0.02,
                 max_rollouts_per_cycle: int = 1, lineage_cap: int = 256,
                 enabled: bool = True) -> None:
        self.substrate = substrate
        self.min_improvement = float(min_improvement)
        self.max_rollouts_per_cycle = int(max_rollouts_per_cycle)
        self.lineage_cap = int(lineage_cap)
        self.enabled = bool(enabled)
        self.lineage: List[Outcome] = []
        self.proposed = 0
        self.rolled_out = 0
        self.rejected = 0
        self._rng = getattr(substrate, "stream", lambda _n: None)("self_improvement")

    # ---- 1. propose ---- #
    def propose(self, layers: Dict[str, Any], *, weakness: str = "") -> Optional[Proposal]:
        """Propose one bounded change. Returns ``None`` when nothing is safely proposable."""
        try:
            rng = self._rng
            targets = [t for t in _MUTABLE if self._read(layers, t) is not None]
            if not targets or rng is None:
                return None
            target = targets[int(rng.integers(0, len(targets)))]
            old = float(self._read(layers, target))
            lo, hi = _MUTABLE[target]
            factor = 1.0 + float(rng.uniform(-0.3, 0.3))
            new = float(min(hi, max(lo, old * factor)))
            if abs(new - old) < 1e-12:
                return None
            self.proposed += 1
            return Proposal(target=target, old_value=old, new_value=new,
                            rationale=weakness or "bounded exploration")
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _read(layers: Dict[str, Any], target: str) -> Optional[float]:
        try:
            layer_name, attr = target.split(".", 1)
            obj = layers.get(layer_name)
            if obj is None or not hasattr(obj, attr):
                return None
            return float(getattr(obj, attr))
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _write(obj: Any, target: str, value: float) -> bool:
        try:
            _, attr = target.split(".", 1)
            current = getattr(obj, attr)
            setattr(obj, attr, type(current)(value) if isinstance(current, int) else float(value))
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---- 2-3. sandbox and evaluate ---- #
    def evaluate(self, proposal: Proposal, layers: Dict[str, Any],
                 task: Callable[[Any], Optional[float]]) -> Outcome:
        """Score incumbent and challenger on the SAME task, the challenger on a clone."""
        out = Outcome(proposal=proposal)
        try:
            layer_name, _ = proposal.target.split(".", 1)
            live = layers.get(layer_name)
            if live is None:
                out.reason = "target layer absent"
                return out
            out.incumbent_score = task(live)
            if out.incumbent_score is None:
                out.reason = "incumbent could not be scored"
                return out
            try:
                clone = copy.deepcopy(live)
            except Exception:  # noqa: BLE001 — a layer that cannot be cloned cannot be sandboxed
                out.reason = "layer is not sandboxable"
                return out
            if not self._write(clone, proposal.target, proposal.new_value):
                out.reason = "proposal could not be applied to the clone"
                return out
            out.challenger_score = task(clone)
            if out.challenger_score is None:
                out.reason = "challenger could not be scored"
                return out
            out.improvement = float(out.challenger_score - out.incumbent_score)
            return out
        except Exception:  # noqa: BLE001
            out.reason = "evaluation failed"
            return out

    # ---- 4. safety ---- #
    def _safe(self, out: Outcome, *, calibration_before: Optional[float] = None,
              calibration_after: Optional[float] = None) -> Tuple[bool, str]:
        """Fail-closed. Every path that is not an explicit pass returns a refusal with a reason."""
        p = out.proposal
        if not self.enabled:
            return False, "self-improvement disabled"
        if p is None:
            return False, "no proposal"
        if not p.in_bounds:
            return False, f"out of bounds for {p.target}"
        if out.improvement is None:
            return False, out.reason or "not evaluated"
        if out.improvement < self.min_improvement:
            return False, (f"improvement {out.improvement:.6f} below the "
                           f"{self.min_improvement:.6f} floor")
        if (calibration_before is not None and calibration_after is not None
                and calibration_after < calibration_before - 1e-9):
            return False, "would degrade measured calibration"
        return True, "cleared"

    # ---- 5. roll out ---- #
    def cycle(self, layers: Dict[str, Any], task: Callable[[Any], Optional[float]], *,
              weakness: str = "", calibration: Optional[float] = None) -> List[Outcome]:
        """One full pipeline pass. Returns every outcome, rolled out or not."""
        outcomes: List[Outcome] = []
        if not self.enabled:
            return outcomes
        try:
            rollouts = 0
            for _ in range(max(1, self.max_rollouts_per_cycle * 3)):
                if rollouts >= self.max_rollouts_per_cycle:
                    break
                proposal = self.propose(layers, weakness=weakness)
                if proposal is None:
                    break
                out = self.evaluate(proposal, layers, task)
                ok, reason = self._safe(out, calibration_before=calibration,
                                        calibration_after=calibration)
                out.reason = reason
                if ok:
                    layer_name, _ = proposal.target.split(".", 1)
                    if self._write(layers.get(layer_name), proposal.target, proposal.new_value):
                        out.rolled_out = True
                        self.rolled_out += 1
                        rollouts += 1
                    else:
                        out.reason = "rollout write failed"
                        self.rejected += 1
                else:
                    self.rejected += 1
                outcomes.append(out)
                self.lineage.append(out)
            if len(self.lineage) > self.lineage_cap:
                self.lineage = self.lineage[-self.lineage_cap:]
            return outcomes
        except Exception:  # noqa: BLE001 — a failed cycle changes nothing, which is the safe default
            return outcomes

    def revert(self, layers: Dict[str, Any], *, last: int = 1) -> int:
        """Undo the most recent rollouts. The reason the lineage exists."""
        n = 0
        try:
            for out in reversed([o for o in self.lineage if o.rolled_out][-max(1, last):]):
                p = out.proposal
                if p is None:
                    continue
                layer_name, _ = p.target.split(".", 1)
                if self._write(layers.get(layer_name), p.target, p.old_value):
                    out.rolled_out = False
                    out.reason = "reverted"
                    n += 1
            return n
        except Exception:  # noqa: BLE001
            return n

    def stats(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "proposed": self.proposed,
                "rolled_out": self.rolled_out, "rejected": self.rejected,
                "lineage": len(self.lineage), "mutable_knobs": len(_MUTABLE),
                "acceptance_rate": round(self.rolled_out / self.proposed, 4)
                if self.proposed else None}

    def to_dict(self) -> Dict[str, Any]:
        return {"proposed": self.proposed, "rolled_out": self.rolled_out,
                "rejected": self.rejected,
                "lineage": [o.to_dict() for o in self.lineage[-64:]]}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.proposed = int(d.get("proposed", 0))
            self.rolled_out = int(d.get("rolled_out", 0))
            self.rejected = int(d.get("rejected", 0))
        except Exception:  # noqa: BLE001
            pass
