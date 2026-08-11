"""NYXARA · nyx001/layers/l08_planning.py — Layer 8, planning (🗺, goal → sub-goals → act).

The charter: "Decomposing goals into sub-goals, simulating outcomes, estimating risks, and acting."

The planner searches over action sequences by simulating each in the world model and scoring the
imagined outcome. It is a bounded best-first search, not an optimiser: ``beam`` sequences are kept
at each depth and ``max_expansions`` caps the total work, because an unbounded planner over a
learned model is a very expensive way to produce confident nonsense.

Risk is estimated as **CVaR** — the mean of the worst ``tail_fraction`` of simulated outcomes —
rather than as variance or as a worst case. Variance punishes upside as much as downside; a single
worst case is dominated by whichever rollout happened to be most wrong. CVaR asks the question a
planner actually needs answered: *when this goes badly, how badly does it go?*

The honesty rules here are the ones that matter most in the whole package, because this is the
layer whose output becomes action:

* **A plan inherits the model's uncertainty.** ``feasible`` goes False when the rollout's terminal
  uncertainty exceeds ``trust_ceiling``. A plan built on a world model that cannot see is not a
  plan, and it is reported as infeasible rather than as a low-confidence plan.
* **Simulated value is labelled simulated.** Nothing here has been tried. :meth:`explain` never
  says "this will work"; it says what the model expects and how much that expectation is worth.
* **It proposes; it does not act.** :class:`Plan` is returned upward. The decision to execute
  belongs to the kernel and its unchanged fail-closed gate, exactly as everywhere else in NYXARA.

Honest: this searches over a small discrete action set against a learned model. It is not a
general planner, it has no notion of the world beyond what Layer 2 encodes, and its horizon is
short by construction because rollout confidence decays geometrically.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.nyx001.layers.types import Action, Plan

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

__all__ = ["Planner"]


class Planner:
    """Layer 8. Decomposes a goal, simulates candidate action sequences, ranks by value and risk."""

    def __init__(self, substrate: Any, world_model: Any = None, *, horizon: int = 4,
                 beam: int = 6, max_expansions: int = 256, tail_fraction: float = 0.2,
                 trust_ceiling: float = 0.6, risk_aversion: float = 0.5) -> None:
        self.substrate = substrate
        self.world_model = world_model
        self.horizon = int(horizon)
        self.beam = int(beam)
        self.max_expansions = int(max_expansions)
        self.tail_fraction = float(tail_fraction)
        self.trust_ceiling = float(trust_ceiling)
        self.risk_aversion = float(risk_aversion)
        self.plans = 0
        self.expansions = 0

    def attach(self, world_model: Any) -> None:
        self.world_model = world_model

    # ---- decomposition ---- #
    def decompose(self, goal: str, *, concepts: Optional[List[Any]] = None) -> List[str]:
        """Sub-goals, drawn from the concepts actually in play — not from a template.

        With no concepts available this returns the goal itself as a single sub-goal. That is the
        honest degenerate case: a system that has formed no concepts has nothing to decompose
        *with*, and inventing plausible-sounding sub-goals would be manufacturing structure.
        """
        g = str(goal).strip()
        if not g:
            return []
        subs = [c.name for c in (concepts or []) if getattr(c, "support", 0) >= 3][:4]
        if not subs:
            return [g]
        return [f"{g} · via {s}" for s in subs]

    # ---- simulation ---- #
    def _score(self, state: Any, actions: Sequence[Any],
               target: Optional[Any]) -> Tuple[float, float, float, int]:
        """Simulate one sequence. Returns (value, cvar_risk, terminal_uncertainty, horizon)."""
        wm = self.world_model
        if wm is None or state is None or np is None:
            return 0.0, 1.0, 1.0, 0
        try:
            preds = wm.rollout(state, list(actions), horizon=self.horizon)
            self.expansions += 1
            if not preds:
                return 0.0, 1.0, 1.0, 0
            steps: List[float] = []
            for p in preds:
                s = np.asarray(p.state, dtype=np.float64).ravel()
                if target is not None:
                    t = np.asarray(target, dtype=np.float64).ravel()
                    n = min(s.size, t.size)
                    # closer to the target is better; negated distance so higher is better
                    steps.append(-float(np.mean((s[:n] - t[:n]) ** 2)))
                else:
                    # with no target, prefer futures the model finds *informative* — high change
                    steps.append(float(np.mean(np.abs(s))))
            value = float(sum(steps) / len(steps))
            arr = np.sort(np.asarray(steps, dtype=np.float64))
            k = max(1, int(len(arr) * self.tail_fraction))
            cvar = float(arr[:k].mean())            # mean of the worst tail
            risk = float(max(0.0, value - cvar))    # how far the bad tail sits below the mean
            return value, risk, float(preds[-1].uncertainty), len(preds)
        except Exception:  # noqa: BLE001
            return 0.0, 1.0, 1.0, 0

    # ---- the search ---- #
    def plan(self, goal: str, state: Any, *, action_space: Optional[Sequence[str]] = None,
             target: Optional[Any] = None, concepts: Optional[List[Any]] = None) -> Plan:
        """Bounded best-first search over action sequences. Always returns a Plan, possibly empty."""
        out = Plan(goal=str(goal), horizon=self.horizon)
        space = list(action_space or [])
        if not space or self.world_model is None or state is None:
            out.feasible = False
            return out
        try:
            self.expansions = 0
            beams: List[Tuple[float, List[str]]] = [(0.0, [])]
            best: Tuple[float, float, float, int, List[str]] = (-1e18, 1.0, 1.0, 0, [])
            for _depth in range(self.horizon):
                scored: List[Tuple[float, List[str]]] = []
                for _, prefix in beams:
                    for a in space:
                        if self.expansions >= self.max_expansions:
                            break
                        seq = prefix + [a]
                        value, risk, unc, hz = self._score(state, seq, target)
                        adjusted = value - self.risk_aversion * risk
                        scored.append((adjusted, seq))
                        if adjusted > best[0]:
                            best = (adjusted, risk, unc, hz, seq)
                    if self.expansions >= self.max_expansions:
                        break
                if not scored:
                    break
                scored.sort(key=lambda t: t[0], reverse=True)
                beams = scored[: self.beam]
                if self.expansions >= self.max_expansions:
                    break

            adjusted, risk, unc, hz, seq = best
            out.steps = [Action(name=a, expected_gain=round(adjusted, 4), risk=round(risk, 4))
                         for a in seq]
            out.simulated_value = round(adjusted, 6)
            out.risk = round(risk, 6)
            out.horizon = hz
            out.feasible = bool(seq) and unc <= self.trust_ceiling
            self.plans += 1
            self._last_uncertainty = unc
            return out
        except Exception:  # noqa: BLE001 — a failed search yields no plan, never a broken turn
            out.feasible = False
            return out

    def explain(self, plan: Plan) -> str:
        """Never says a plan will work. Says what the model expects, and what that is worth."""
        if not plan.steps:
            return f"no plan for '{plan.goal[:60]}' — nothing in the action space was simulable"
        if not plan.feasible:
            unc = getattr(self, "_last_uncertainty", 1.0)
            return (f"plan for '{plan.goal[:60]}' is not trustworthy: the world model's terminal "
                    f"uncertainty is {unc:.2f} over {plan.horizon} steps — this is not foresight")
        names = " → ".join(a.name for a in plan.steps)
        return (f"simulated plan for '{plan.goal[:60]}': {names} "
                f"(modelled value {plan.simulated_value:.4f}, tail risk {plan.risk:.4f}, "
                f"untried)")

    def stats(self) -> Dict[str, Any]:
        return {"plans": self.plans, "horizon": self.horizon, "beam": self.beam,
                "max_expansions": self.max_expansions,
                "attached": self.world_model is not None}

    def to_dict(self) -> Dict[str, Any]:
        return {"plans": self.plans}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.plans = int(d.get("plans", 0))
        except Exception:  # noqa: BLE001
            pass
