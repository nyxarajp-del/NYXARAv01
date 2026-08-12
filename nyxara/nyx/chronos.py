"""NYXARA · nyx/chronos.py — deciding by looking at how it turns out (⏳, L-CHRONOS).

Committing to a plan because it sounds best is guessing. This module lets NYX branch a decision
into many simulated futures, score each option by what actually happens across them — including
the bad tail, not just the average — and feed that ranking back into the superposition as
evidence. The option that survives the most futures wins, and it wins on measured outcomes.

Two honest boundaries, both enforced in code rather than described and ignored:

**It only applies to decisions.** Simulating futures for "what pulls an apple down" is
meaningless — there is no future to roll out, only a fact to recall. :meth:`applies` requires the
turn to be about *what to do*. Everything else is left to the rest of the cycle.

**It needs a world model that has actually learned something.** With an empty model the
underlying simulator returns zeros for every option — a confident-looking table of nothing.
Rather than pass that off as foresight, :meth:`explore` reports ``blind=True`` with the reason,
and contributes **no** evidence. As she learns real dynamics (statements like "the temperature
rises when the heater is on" become transitions via ``cognition/language_grounding``), the model
fills and foresight starts working. Until then she says she cannot see ahead.

Honest framing of the scale, too: "thousands of futures" is exactly as many as ``max_branches``
allows, run sequentially on ordinary silicon. These are Monte Carlo rollouts over an
*approximate* model — not parallel universes, and not a guarantee. A wrong model yields
confidently wrong futures, which is why the report carries the model's coverage and the
simulator's own confidence rather than only a winner.

Composes :class:`~nyxara.abyss.timeline_simulator.TimelineSimulator` (risk-aware CVaR ranking)
over :mod:`nyxara.mind.world_model`. Fail-soft throughout.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["Future", "Futures", "TemporalCausalMatrix"]

#: Mirrors :data:`nyxara.abyss.timeline_simulator.DEFAULT_NOISE_SCALE`. Copied rather than
#: imported so this module keeps its lazy, fail-soft relationship with the simulator — nothing
#: in the nyx import chain should pull the world model in just to read one float.
DEFAULT_NOISE_SCALE = 0.01

# A decision asks what to *do*; a question asks what *is*. Only the first has futures.
_DECISION = re.compile(
    r"\b(should|shall|ought|which (?:one|option|way)|better to|worth it|what do i do|"
    r"what should|plan|choose|decide|pick|plan for|risk|plan to|do i|shall i)\b", re.I)


@dataclass
class Future:
    """How one option turned out across the futures it was rolled through."""

    option: str
    expected: float = 0.0
    tail_risk: float = 0.0           # CVaR of losses — the mean of its worst outcomes
    goal_probability: float = 0.0
    score: float = 0.0               # risk-aware: expected − risk_aversion · tail_risk
    branches: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"option": self.option, "expected": round(self.expected, 6),
                "tail_risk": round(self.tail_risk, 6),
                "goal_probability": round(self.goal_probability, 6),
                "score": round(self.score, 6), "branches": self.branches}


@dataclass
class Futures:
    """What the simulation saw — or why it could not see anything."""

    ran: bool = False
    blind: bool = False              # no world model worth simulating with
    reason: str = ""
    best: str = ""
    confidence: float = 0.0          # the simulator's own, not ours
    coverage: int = 0                # how much the world model has actually learned
    branches: int = 0                # futures actually rolled, not futures asked for
    clipped: bool = False            # the time budget cut the pass short of what was asked
    horizon: int = 0
    options: List[Future] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"ran": self.ran, "blind": self.blind, "reason": self.reason,
                "best": self.best, "confidence": round(self.confidence, 4),
                "coverage": self.coverage, "branches": self.branches,
                "clipped": self.clipped,
                "horizon": self.horizon, "elapsed_ms": round(self.elapsed_ms, 2),
                "options": [o.to_dict() for o in self.options]}


class TemporalCausalMatrix:
    """Branches a decision into futures and ranks the options by how they actually turn out."""

    def __init__(self, *, world_model: Any = None, max_branches: int = 256, horizon: int = 4,
                 budget_ms: float = 400.0, risk_aversion: float = 0.5,
                 min_coverage: int = 8, seed: int = 42,
                 noise_scale: float = DEFAULT_NOISE_SCALE) -> None:
        self._world_model = world_model
        self._tried = world_model is not None
        # Whether the model she sees ahead with came from outside — the kernel's shared one — or
        # is the private one she builds for herself. Only the first can ever fill.
        self._adopted = world_model is not None
        self.max_branches = max(1, int(max_branches))
        self.horizon = max(1, int(horizon))
        self.budget_ms = max(1.0, float(budget_ms))
        self.risk_aversion = float(risk_aversion)
        # Divergence between the branches. At zero the rollouts of one option are the same
        # trajectory repeated, and "many futures" is one future counted many times.
        self.noise_scale = max(0.0, float(noise_scale))
        # Below this many learned transitions the model cannot say anything about the future,
        # and the simulator's zeros would look like findings.
        self.min_coverage = max(1, int(min_coverage))
        self.seed = int(seed)

    # ---- when it applies --------------------------------------------------- #
    @staticmethod
    def applies(stimulus: str) -> bool:
        """Is this a decision — something with futures — rather than a question of fact?

        NYX V.02: an **imperative** has futures too, and the marker list below never saw one.
        "delete build/ and commit" contains no word from ``_DECISION``, yet it is exactly the
        kind of turn that should be rolled forward before anything happens. Asking
        :mod:`nyxara.nyx.intent` for the mood catches it, in three registers; the marker list
        stays as the floor so a reader failure degrades to V.01 behaviour.
        """
        try:
            if _DECISION.search(str(stimulus or "")):
                return True
        except Exception:  # noqa: BLE001
            return False
        try:
            from nyxara.nyx.intent import is_command
            return is_command(stimulus)
        except Exception:  # noqa: BLE001 — the reader is a capability, never a dependency
            return False

    def adopt_world_model(self, model: Any) -> bool:
        """Take the kernel's **shared** world model as the one to see ahead with.

        Left to itself this class builds a private model with :func:`build_world_model`, which is
        born empty and — because nothing in the think-cycle calls :meth:`learn` — stays empty for
        the life of the process. Foresight was therefore permanently ``blind``: honest, but never
        able to become anything else. The kernel's model is the one that actually fills, from the
        embodied loop's real transitions, so adopting it is what turns L-CHRONOS on. Called by
        :meth:`~nyxara.nyx.brain.NyxBrain.attach_kernel`; returns whether the model was taken.
        """
        if model is None:
            return False
        try:
            len(model)                     # must expose the surface `coverage()` reads
        except Exception:  # noqa: BLE001 — an unusable model is not an improvement
            return False
        self._world_model = model
        self._tried = True
        self._adopted = True
        return True

    def world_model(self) -> Any:
        if not self._tried:
            self._tried = True
            try:
                from nyxara.mind.world_model import build_world_model
                self._world_model = build_world_model()
            except Exception:  # noqa: BLE001 — without a model she simply cannot see ahead
                self._world_model = None
        return self._world_model

    def coverage(self) -> int:
        """How many transitions the model has actually learned. Zero means blind."""
        model = self.world_model()
        try:
            return int(len(model)) if model is not None else 0
        except Exception:  # noqa: BLE001
            return 0

    def available(self) -> bool:
        """Honest capability report: a model exists *and* has learned enough to simulate."""
        return self.coverage() >= self.min_coverage

    def learn(self, state: Sequence[Any], action: Any, next_state: Sequence[Any],
              reward: float = 0.0) -> bool:
        """Teach the model one real transition — this is what makes foresight possible."""
        model = self.world_model()
        if model is None:
            return False
        try:
            model.observe(state, action, next_state, reward=float(reward))
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---- the simulation ----------------------------------------------------- #
    def explore(self, options: Sequence[Any], *, state: Optional[Sequence[float]] = None,
                branches: Optional[int] = None) -> Futures:
        """Roll each option through many futures and rank them by risk-aware outcome."""
        out = Futures(horizon=self.horizon)
        started = time.monotonic()
        try:
            labels = [str(getattr(o, "source", o)) for o in options if o is not None]
            if len(labels) < 2:
                out.reason = "fewer than two options — nothing to compare futures between"
                return out

            out.coverage = self.coverage()
            if out.coverage < self.min_coverage:
                out.blind = True
                out.reason = (f"the world model has learned {out.coverage} transitions "
                              f"(needs {self.min_coverage}); simulating would return zeros "
                              f"dressed as foresight")
                return out

            from nyxara.abyss.timeline_simulator import TimelineSimulator
            simulator = TimelineSimulator(self.world_model(), seed=self.seed)
            asked = min(self.max_branches, int(branches or self.max_branches))
            # ``budget_ms`` was configured, stored and never consulted — a knob an operator could
            # turn with nothing on the other end of it. It is now the simulator's wall-clock cap,
            # so a slow model costs bounded time instead of unbounded time.
            report = simulator.simulate(list(state or [0.0]), labels,
                                        branches=asked, horizon=self.horizon,
                                        noise_scale=self.noise_scale,
                                        risk_aversion=self.risk_aversion,
                                        budget_ms=self.budget_ms)
            # report what she actually rolled, not what she asked for: a clipped pass that still
            # claimed the full budget would be a number that is not true.
            out.branches = int(getattr(report, "total_branches", asked) or asked)
            out.clipped = out.branches < asked
            out.ran = True
            out.best = str(getattr(report, "best_action", "") or "")
            out.confidence = float(getattr(report, "confidence", 0.0))
            out.options = [
                Future(option=str(o.action), expected=float(o.expected_reward),
                       tail_risk=float(o.tail_risk),
                       goal_probability=float(o.goal_probability),
                       score=float(o.score), branches=int(o.branches))
                for o in (getattr(report, "action_rankings", []) or [])]
            return out
        except Exception:  # noqa: BLE001 — a failed simulation sees nothing, it does not crash
            out.reason = "the simulation failed and was abandoned"
            return out
        finally:
            out.elapsed_ms = (time.monotonic() - started) * 1000.0

    # ---- feeding the decision ------------------------------------------------ #
    @staticmethod
    def evidence(futures: Futures) -> Dict[str, float]:
        """Turn the ranking into likelihoods for :meth:`SolutionSuperposition.observe`.

        Returns ``{}`` when the simulation was blind or did not run — contributing nothing is
        the correct behaviour, and much better than contributing zeros.
        """
        try:
            if not futures.ran or futures.blind or not futures.options:
                return {}
            scores = [o.score for o in futures.options]
            low, high = min(scores), max(scores)
            span = high - low
            if span <= 1e-12:
                return {}                 # every option fared identically — that is not evidence
            # Map to a likelihood ratio in [0.5, 2.0]: real signal, bounded influence. Foresight
            # informs the decision; it does not get to override everything else in one step.
            return {o.option: 0.5 + 1.5 * ((o.score - low) / span) for o in futures.options}
        except Exception:  # noqa: BLE001
            return {}

    def stats(self) -> Dict[str, Any]:
        try:
            coverage = self.coverage()
            return {"available": coverage >= self.min_coverage, "coverage": coverage,
                    "min_coverage": self.min_coverage, "max_branches": self.max_branches,
                    "horizon": self.horizon, "risk_aversion": self.risk_aversion,
                    "budget_ms": self.budget_ms, "noise_scale": self.noise_scale,
                    # False means she is simulating over her own private model, which nothing
                    # teaches — so `coverage` will stay where it is and she will stay blind.
                    "shared_world_model": self._adopted}
        except Exception:  # noqa: BLE001
            return {"available": False}
