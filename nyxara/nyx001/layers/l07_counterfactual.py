"""NYXARA · nyx001/layers/l07_counterfactual.py — Layer 7, counterfactuals (🔀, "what if not X?").

The charter: "Internal simulation ('What if event X didn't happen?') to develop causal reasoning."

The mechanism is intervention, not correlation. To ask what X contributed, the engine takes the
state *before* X, applies a modified action sequence with X removed or replaced, rolls the world
model forward over both, and reports the divergence. That is a ``do()`` operation on a learned
model — the thing that distinguishes a causal question from an associational one. Layer 4 can
observe that A precedes B; only this layer can ask whether B still happens when A does not.

Two guards keep the answers honest, and both are load-bearing:

* **The model's own uncertainty bounds the claim.** A rollout from a world model with a 0.4 mean
  error is not evidence about causation. :attr:`Counterfactual.trustworthy` is False whenever the
  rollout's terminal uncertainty exceeds ``trust_ceiling``, and every consumer is expected to read
  it. Fabricated foresight is worse than none.
* **Divergence is not effect size.** Two rollouts that differ tell you the *model* thinks X
  matters. The report says exactly that. Whether the world agrees is what Layer 16 tests by
  actually intervening, and until it has, this is a hypothesis with a number attached.

:meth:`attribute` runs the counterfactual over each of several candidate causes and ranks them by
divergence — the closest this package comes to "why did that happen", and it is careful to call
the result an *attribution under the current model* rather than a cause.

Honest: this reasons about a learned approximation, so its counterfactuals inherit every bias in
what NYX V.001 has seen. On a system it has not modelled well, the answers are confidently wrong —
which is precisely what ``trustworthy`` is for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

__all__ = ["Counterfactual", "CounterfactualEngine"]


@dataclass
class Counterfactual:
    """One "what if not X" — the two futures, their divergence, and whether to believe it."""

    intervened_on: str = ""
    divergence: float = 0.0
    horizon: int = 0
    uncertainty: float = 1.0
    trustworthy: bool = False
    factual: List[Any] = field(default_factory=list)
    counterfactual: List[Any] = field(default_factory=list)

    @property
    def mattered(self) -> bool:
        """Did removing it change the modelled future at all — and can we believe that?"""
        return self.trustworthy and self.divergence > 1e-6

    def to_dict(self) -> Dict[str, Any]:
        return {"intervened_on": self.intervened_on, "divergence": round(self.divergence, 6),
                "horizon": self.horizon, "uncertainty": round(self.uncertainty, 4),
                "trustworthy": self.trustworthy, "mattered": self.mattered,
                "note": "divergence under the current world model, not a measured effect"}


class CounterfactualEngine:
    """Layer 7. Simulates alternatives by intervening on a learned world model."""

    def __init__(self, substrate: Any, world_model: Any = None, *, horizon: int = 4,
                 trust_ceiling: float = 0.5) -> None:
        self.substrate = substrate
        self.world_model = world_model
        self.horizon = int(horizon)
        self.trust_ceiling = float(trust_ceiling)
        self.runs = 0

    def attach(self, world_model: Any) -> None:
        self.world_model = world_model

    def _rollout(self, state: Any, actions: Sequence[Any]) -> List[Any]:
        wm = self.world_model
        if wm is None or state is None:
            return []
        try:
            preds = wm.rollout(state, list(actions), horizon=self.horizon)
            return preds or []
        except Exception:  # noqa: BLE001
            return []

    def what_if_not(self, state: Any, actions: Sequence[Any], *, remove: int = 0,
                    replacement: Any = None) -> Counterfactual:
        """Roll forward with and without action ``remove``; report how far the futures separate."""
        out = Counterfactual(intervened_on=str(actions[remove]) if 0 <= remove < len(actions)
                             else f"index {remove}")
        try:
            if np is None or self.world_model is None or state is None:
                return out
            factual = list(actions)
            counter = list(actions)
            if 0 <= remove < len(counter):
                if replacement is None:
                    counter.pop(remove)
                else:
                    counter[remove] = replacement
            f_preds = self._rollout(state, factual)
            c_preds = self._rollout(state, counter)
            if not f_preds or not c_preds:
                return out
            n = min(len(f_preds), len(c_preds))
            div = 0.0
            for i in range(n):
                a = np.asarray(f_preds[i].state, dtype=np.float64).ravel()
                b = np.asarray(c_preds[i].state, dtype=np.float64).ravel()
                m = min(a.size, b.size)
                div += float(np.mean((a[:m] - b[:m]) ** 2))
            out.divergence = div / max(1, n)
            out.horizon = n
            out.uncertainty = float(f_preds[n - 1].uncertainty)
            out.trustworthy = out.uncertainty <= self.trust_ceiling
            out.factual = [p.state for p in f_preds[:n]]
            out.counterfactual = [p.state for p in c_preds[:n]]
            self.runs += 1
            return out
        except Exception:  # noqa: BLE001
            return out

    def attribute(self, state: Any, actions: Sequence[Any]) -> List[Counterfactual]:
        """Which of these actions the model thinks mattered, ranked. An attribution, not a cause."""
        out: List[Counterfactual] = []
        for i in range(len(actions)):
            cf = self.what_if_not(state, actions, remove=i)
            out.append(cf)
        out.sort(key=lambda c: (c.trustworthy, c.divergence), reverse=True)
        return out

    def explain(self, cf: Counterfactual) -> str:
        """One line that never overstates what a rollout divergence establishes."""
        if not cf.trustworthy:
            return (f"cannot say whether {cf.intervened_on} mattered — the world model's own "
                    f"uncertainty at horizon {cf.horizon} is {cf.uncertainty:.2f}")
        if not cf.mattered:
            return f"removing {cf.intervened_on} changes nothing the model can see"
        return (f"under the current model, removing {cf.intervened_on} diverges the next "
                f"{cf.horizon} steps by {cf.divergence:.4f} — a modelled attribution, untested")

    def stats(self) -> Dict[str, Any]:
        return {"runs": self.runs, "horizon": self.horizon,
                "trust_ceiling": self.trust_ceiling,
                "attached": self.world_model is not None}

    def to_dict(self) -> Dict[str, Any]:
        return {"runs": self.runs}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.runs = int(d.get("runs", 0))
        except Exception:  # noqa: BLE001
            pass
