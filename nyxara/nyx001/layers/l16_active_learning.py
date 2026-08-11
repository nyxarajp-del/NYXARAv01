"""NYXARA · nyx001/layers/l16_active_learning.py — Layer 16, the loop that closes (🔄, consequence).

The charter: "Continuous loop: Observe → Model → Act → Consequence → Error → Learn."

Every other layer can run without ever touching anything. This is the layer where the system
*acts* and then has to live with what happened, and that changes the epistemics completely: an
action is an **intervention**, and an intervention is the only thing that can turn Layer 7's
modelled attributions into evidence about actual causation.

The loop is explicit in :meth:`step`:

1. **Observe** — a percept enters.
2. **Model** — predict what follows.
3. **Act** — choose from the action space, biased by curiosity's frontier and by expected
   information gain, not by reward.
4. **Consequence** — the environment answers.
5. **Error** — measure the prediction against it.
6. **Learn** — one gradient step, plus a replayed batch from episodic memory.

Two properties are load-bearing:

* **It proposes actions; it does not execute them.** The environment is an injected callable. In a
  live NYXARA the kernel supplies one that routes through the unchanged fail-closed gate. Nothing
  here can reach past that, by construction: this layer has no tools, no shell, no filesystem.
* **Interventions are recorded as interventions.** :meth:`intervene` marks the episode so Layer 7
  can distinguish "I observed A then B" from "I *did* A and then B happened". Only the second
  supports a causal claim, and conflating them is how a system convinces itself of superstitions.

Honest: exploration here is information-seeking within a discrete action set someone else defined.
The system does not invent new actions — that is Layer 17's territory, and only for its own code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from nyxara.nyx001.layers.types import Observation, Percept

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

__all__ = ["LoopStep", "ActiveLearning"]


@dataclass
class LoopStep:
    """One turn of the loop, with every stage's outcome recorded."""

    observed: bool = False
    predicted: bool = False
    action: Optional[str] = None
    intervened: bool = False
    consequence: bool = False
    error: Optional[float] = None
    learned: bool = False
    replayed: int = 0
    info_gain: float = 0.0

    @property
    def closed(self) -> bool:
        """Did the loop actually close — act, consequence, error, learn?"""
        return bool(self.action and self.consequence and self.error is not None and self.learned)

    def to_dict(self) -> Dict[str, Any]:
        return {"observed": self.observed, "predicted": self.predicted, "action": self.action,
                "intervened": self.intervened, "consequence": self.consequence,
                "error": None if self.error is None else round(self.error, 6),
                "learned": self.learned, "replayed": self.replayed,
                "info_gain": round(self.info_gain, 6), "closed": self.closed}


class ActiveLearning:
    """Layer 16. Runs observe → model → act → consequence → error → learn, and closes it."""

    def __init__(self, substrate: Any, *, action_space: Optional[Sequence[str]] = None,
                 explore_rate: float = 0.25, replay_every: int = 4,
                 max_history: int = 512) -> None:
        self.substrate = substrate
        self.action_space: List[str] = list(action_space or [])
        self.explore_rate = float(explore_rate)
        self.replay_every = int(replay_every)
        self.max_history = int(max_history)
        self._history: List[LoopStep] = []
        self.steps = 0
        self.interventions = 0
        self.closed_loops = 0
        self._rng = getattr(substrate, "stream", lambda _n: None)("active_learning")

    def set_action_space(self, actions: Sequence[str]) -> None:
        self.action_space = list(actions or [])

    # ---- act ---- #
    def choose(self, *, curiosity: Any = None, planner: Any = None, state: Any = None,
               goal: str = "") -> Optional[str]:
        """Pick an action. Information gain leads; a plan is consulted; exploration breaks ties.

        Deliberately *not* reward-driven — the charter's whole point about Layer 9 is that the
        system seeks useful unknowns. Where curiosity has a live frontier, that wins.
        """
        if not self.action_space:
            return None
        try:
            rng = self._rng
            if rng is not None and float(rng.random()) < self.explore_rate:
                return str(self.action_space[int(rng.integers(0, len(self.action_space)))])
            if planner is not None and state is not None and goal:
                plan = planner.plan(goal, state, action_space=self.action_space)
                if plan.feasible and plan.steps:
                    return plan.steps[0].name
            if curiosity is not None:
                frontier = curiosity.frontier(k=1)
                if frontier:
                    # map the frontier region onto an action deterministically, so the same
                    # unknown is probed the same way and the result is attributable
                    idx = abs(hash(frontier[0].key)) % len(self.action_space)
                    return str(self.action_space[idx])
            return str(self.action_space[0])
        except Exception:  # noqa: BLE001
            return self.action_space[0] if self.action_space else None

    # ---- the loop ---- #
    def step(self, observation: Any, environment: Optional[Callable[[str], Any]] = None, *,
             perception: Any = None, world_model: Any = None, memory: Any = None,
             semantic: Any = None, curiosity: Any = None, planner: Any = None,
             goal: str = "", cognitive: Any = None) -> LoopStep:
        """One full turn. Every stage is optional; the step reports which ones actually ran."""
        out = LoopStep()
        try:
            self.steps += 1
            # 1. observe
            obs = observation if isinstance(observation, Observation) \
                else Observation(payload=observation)
            percept: Optional[Percept] = None
            if perception is not None:
                percept = perception.perceive(obs)
                out.observed = percept is not None
            state = percept.vec if percept is not None else None

            # 2. model
            action = self.choose(curiosity=curiosity, planner=planner, state=state, goal=goal)
            out.action = action
            if world_model is not None and state is not None:
                world_model.predict(state, action)
                out.predicted = True

            # 3-4. act, consequence
            result = None
            if environment is not None and action is not None:
                result = environment(action)
                out.consequence = result is not None
                out.intervened = True
                self.interventions += 1

            # 5. error
            actual = None
            if result is not None and perception is not None:
                nxt = result if isinstance(result, Observation) else Observation(payload=result)
                actual = perception.perceive(nxt).vec
            if actual is not None and world_model is not None:
                out.error = world_model.observe(actual, cognitive=cognitive)
                out.learned = True

            # 6. learn — remember, form concepts, feed curiosity, replay
            if memory is not None and state is not None:
                ep = memory.remember(str(getattr(obs, "payload", ""))[:120], context=state,
                                     prediction_error=out.error or 0.0,
                                     action=action,
                                     outcome="intervened" if out.intervened else "observed")
                if ep is not None and semantic is not None:
                    concept = semantic.absorb(ep)
                    if curiosity is not None:
                        key = curiosity.region_key(concept, state)
                        r = curiosity.observe(key, out.error or 0.0)
                        out.info_gain = float(getattr(r, "progress", 0.0)) if r else 0.0
                if self.steps % max(1, self.replay_every) == 0:
                    out.replayed = self._replay(memory, world_model, cognitive)

            if out.closed:
                self.closed_loops += 1
            self._history.append(out)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]
            return out
        except Exception:  # noqa: BLE001 — a broken loop turn is a lost lesson, never a crash
            return out

    def _replay(self, memory: Any, world_model: Any, cognitive: Any) -> int:
        """Rehearse prioritised episodes. This is what protects against forgetting."""
        try:
            if memory is None or world_model is None:
                return 0
            batch = memory.replay()
            n = 0
            for ep in batch:
                if ep.context is None:
                    continue
                world_model.predict(ep.context, ep.action)
                world_model.observe(ep.context, cognitive=cognitive)
                n += 1
            return n
        except Exception:  # noqa: BLE001
            return 0

    def intervene(self, action: str, environment: Callable[[str], Any], *,
                  memory: Any = None, state: Any = None) -> Optional[Any]:
        """A deliberate ``do(action)``, marked as an intervention so Layer 7 can tell the difference."""
        try:
            result = environment(action)
            self.interventions += 1
            if memory is not None:
                memory.remember(f"do({action})", context=state, action=action,
                                outcome="intervention")
            return result
        except Exception:  # noqa: BLE001
            return None

    def close_rate(self) -> Optional[float]:
        """Fraction of turns where the loop actually closed. ``None`` before any step."""
        if not self._history:
            return None
        return float(sum(1 for s in self._history if s.closed) / len(self._history))

    def stats(self) -> Dict[str, Any]:
        errs = [s.error for s in self._history if s.error is not None]
        return {"steps": self.steps, "interventions": self.interventions,
                "closed_loops": self.closed_loops, "close_rate": self.close_rate(),
                "action_space": len(self.action_space),
                "mean_error": round(float(sum(errs) / len(errs)), 6) if errs else None}

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": self.steps, "interventions": self.interventions,
                "closed_loops": self.closed_loops,
                "history": [s.to_dict() for s in self._history[-64:]]}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.steps = int(d.get("steps", 0))
            self.interventions = int(d.get("interventions", 0))
            self.closed_loops = int(d.get("closed_loops", 0))
        except Exception:  # noqa: BLE001
            pass
