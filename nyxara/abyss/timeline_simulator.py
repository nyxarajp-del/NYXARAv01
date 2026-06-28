"""NYXARA · abyss/timeline_simulator.py — the Parallel-Futures Engine (Abyss · 1).

Before NYXARA commits to *any* action, she does not reason about a single imagined
future — she branches the present into **thousands of parallel timelines**, rolls each
one forward through her learned world model, injects realistic noise so the futures
genuinely diverge, and then looks at the *whole distribution* of outcomes. Only then
does she choose — so every decision is calculated against what could happen, not a
single optimistic guess.

This is the macro counterpart to :mod:`nyxara.mind.world_simulator` (which scores one
candidate action's side-effects). The timeline simulator compares *many* candidate
actions across *many* sampled futures and ranks them under a **risk-aware** score that
protects the Master even in the bad tail (Rule 3 — guard the downside).

It reuses, rather than reinvents, the engines already in the codebase:

* :mod:`nyxara.mind.world_model` — ``WorldModel.predict`` gives the per-step dynamics
  (with **honest confidence** that decays out of the region it has actually seen);
* :mod:`nyxara.sim.montecarlo` — :class:`Estimate` aggregates each action's reward
  distribution (mean, 95% CI, percentiles, and **CVaR tail risk**).

Honest uncertainty is preserved end-to-end: with little or no world-model experience
the reported confidence is low and the intervals are wide — she never fakes certainty
about futures she has not learned. Pure standard library (numpy only optionally, via
the world-model factory, with a graceful fallback).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from nyxara.mind.world_model import WorldModel, build_world_model
from nyxara.sim.montecarlo import Estimate

__all__ = ["ActionOutcome", "TimelineReport", "TimelineSimulator"]

State = Sequence[Any]
RewardFn = Callable[[Any, Any, Any], float]
GoalFn = Callable[[Any], bool]


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class ActionOutcome:
    """The distribution of futures that follow from committing to one first action."""

    action: Any
    expected_reward: float          # mean total reward across the sampled futures
    ci95: tuple                     # 95% confidence interval for the mean
    p05: float                      # 5th percentile (a pessimistic future)
    p95: float                      # 95th percentile (an optimistic future)
    tail_risk: float                # CVaR of losses — mean of the worst ~5% of futures
    goal_probability: float         # fraction of futures that reached the goal
    mean_confidence: float          # mean world-model confidence along the rollouts
    branches: int                   # how many futures were sampled for this action
    score: float                    # risk-aware score used for ranking

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "expected_reward": round(self.expected_reward, 4),
            "ci95": [round(self.ci95[0], 4), round(self.ci95[1], 4)],
            "p05": round(self.p05, 4),
            "p95": round(self.p95, 4),
            "tail_risk": round(self.tail_risk, 4),
            "goal_probability": round(self.goal_probability, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "branches": self.branches,
            "score": round(self.score, 4),
        }


@dataclass
class TimelineReport:
    """The outcome of a full parallel-futures pass over a set of candidate actions."""

    start_state: tuple
    total_branches: int
    horizon: int
    best_action: Any
    action_rankings: List[ActionOutcome]
    confidence: float               # honest overall confidence in this pass [0,1]
    expected_outcome: str           # human-readable summary
    goal_probability: Optional[float] = None  # best action's probability of reaching goal
    risk_aversion: float = 0.5

    def recommend(self) -> Any:
        """The single best first action under the risk-aware ranking (or None)."""
        return self.best_action

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_state": list(self.start_state),
            "total_branches": self.total_branches,
            "horizon": self.horizon,
            "best_action": self.best_action,
            "confidence": round(self.confidence, 4),
            "expected_outcome": self.expected_outcome,
            "goal_probability": (round(self.goal_probability, 4)
                                 if self.goal_probability is not None else None),
            "risk_aversion": self.risk_aversion,
            "action_rankings": [o.to_dict() for o in self.action_rankings],
        }


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
class TimelineSimulator:
    """Branch the present into thousands of futures and rank candidate actions.

    Parameters
    ----------
    world_model:    a :class:`~nyxara.mind.world_model.WorldModel` (or any object with the
                    same ``predict`` / ``__len__`` surface). When ``None`` a fresh model is
                    built via :func:`~nyxara.mind.world_model.build_world_model` so the
                    caller can ``observe`` experience into ``self.world_model`` directly.
    seed:           base RNG seed — the whole pass is reproducible for a fixed seed.
    """

    def __init__(self, world_model: Any = None, *, seed: int = 0) -> None:
        self.world_model = world_model if world_model is not None else build_world_model("auto")
        self.seed = seed

    # ---------------------------------------------------------------------- #
    @staticmethod
    def _numeric(state: Sequence[Any]) -> bool:
        return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in state)

    def _rollout(self, start: State, first_action: Any, actions: List[Any],
                 horizon: int, reward_fn: Optional[RewardFn], goal_fn: Optional[GoalFn],
                 rng: random.Random, noise_scale: float) -> tuple:
        """One imagined future: a single stochastic trajectory through the world model.

        Returns ``(total_reward, mean_confidence, reached_goal)``.
        """
        cur = tuple(start)
        total = 0.0
        confs: List[float] = []
        reached = False
        for t in range(horizon):
            action = first_action if (t == 0 and first_action is not None) else rng.choice(actions)
            pred = self.world_model.predict(cur, action)
            nxt = tuple(pred.next_state)
            # inject realistic divergence so the parallel futures genuinely differ
            if noise_scale > 0.0 and self._numeric(nxt):
                nxt = tuple(v + rng.gauss(0.0, noise_scale) for v in nxt)
            reward = (reward_fn(cur, action, nxt) if reward_fn is not None else pred.reward)
            total += float(reward)
            confs.append(pred.confidence)
            cur = nxt
            if goal_fn is not None and goal_fn(cur):
                reached = True
                break
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        return total, mean_conf, reached

    def simulate(self, start_state: State, candidate_actions: Sequence[Any], *,
                 branches: int = 1000, horizon: int = 8,
                 reward_fn: Optional[RewardFn] = None, goal_fn: Optional[GoalFn] = None,
                 noise_scale: float = 0.0, risk_aversion: float = 0.5) -> TimelineReport:
        """Branch ``start_state`` into ``branches`` futures and rank ``candidate_actions``.

        ``branches`` is the *total* number of futures, split evenly across the candidate
        actions (so "a thousand futures" means a thousand rollouts in all). Each action is
        scored by a **risk-aware** value ``mean_reward − risk_aversion · tail_risk``, where
        ``tail_risk`` is the CVaR of losses (the mean of the worst ~5% of its futures).
        """
        actions = list(candidate_actions)
        start = tuple(start_state)
        if not actions:
            return TimelineReport(start, 0, horizon, None, [], 0.0,
                                  "no candidate actions to simulate",
                                  risk_aversion=risk_aversion)

        per_action = max(1, branches // len(actions))
        outcomes: List[ActionOutcome] = []
        for i, a in enumerate(actions):
            rng = random.Random((self.seed * 1_000_003) ^ (i + 1))
            rewards: List[float] = []
            confs: List[float] = []
            hits = 0
            for _ in range(per_action):
                total, mean_conf, reached = self._rollout(
                    start, a, actions, horizon, reward_fn, goal_fn, rng, noise_scale)
                rewards.append(total)
                confs.append(mean_conf)
                hits += 1 if reached else 0
            est = Estimate(rewards)
            # treat negative reward as loss; CVaR of losses = expected worst-tail damage
            loss_tail = Estimate([-r for r in rewards]).cvar(0.95)
            mean_conf = sum(confs) / len(confs) if confs else 0.0
            goal_prob = hits / per_action
            score = est.mean - risk_aversion * loss_tail
            outcomes.append(ActionOutcome(
                action=a, expected_reward=est.mean, ci95=est.ci95(),
                p05=est.percentile(5), p95=est.percentile(95), tail_risk=loss_tail,
                goal_probability=goal_prob, mean_confidence=mean_conf,
                branches=per_action, score=score))

        outcomes.sort(key=lambda o: o.score, reverse=True)
        best = outcomes[0]

        # honest overall confidence: blends learned-experience volume with how confident
        # the world model actually was along the best action's rollouts.
        try:
            wm_size = len(self.world_model)
        except Exception:  # noqa: BLE001
            wm_size = 0
        data_conf = min(0.9, wm_size / 100.0) if wm_size else 0.0
        confidence = round(0.5 * data_conf + 0.5 * best.mean_confidence, 3)

        outcome = (
            f"Simulated {per_action * len(actions)} futures over {horizon} steps across "
            f"{len(actions)} actions. Best: '{best.action}' "
            f"(E[reward]={best.expected_reward:.2f}, tail_risk={best.tail_risk:.2f}, "
            f"P(goal)={best.goal_probability:.2f}). Confidence {confidence:.2f}.")

        return TimelineReport(
            start_state=start, total_branches=per_action * len(actions), horizon=horizon,
            best_action=best.action, action_rankings=outcomes, confidence=confidence,
            expected_outcome=outcome,
            goal_probability=(best.goal_probability if goal_fn is not None else None),
            risk_aversion=risk_aversion)

    def recommend(self, start_state: State, candidate_actions: Sequence[Any],
                  **kwargs: Any) -> Any:
        """Convenience: run :meth:`simulate` and return only the best first action."""
        return self.simulate(start_state, candidate_actions, **kwargs).best_action


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA timeline-simulator self-test")
    print("=" * 70)

    # A tiny 1-D world: position x; actions move it; reward = -|x| (home is 0).
    def true_step(x: float, a: str) -> float:
        return {"left": x - 1.0, "right": x + 1.0, "stay": x}[a]

    wm = WorldModel(k=3, distance_scale=2.0)
    rng = random.Random(0)
    for _ in range(400):
        x = rng.uniform(-10, 10)
        a = rng.choice(["left", "right", "stay"])
        wm.observe((x,), a, (true_step(x, a),), reward=-abs(true_step(x, a)))

    sim = TimelineSimulator(world_model=wm, seed=0)

    # From x = +5, going LEFT (toward home at 0) must beat going RIGHT.
    rfn = lambda s, a, ns: -abs(ns[0])  # noqa: E731 — reward = -distance from home
    report = sim.simulate((5.0,), ["left", "right", "stay"],
                          branches=900, horizon=6, reward_fn=rfn, noise_scale=0.05)
    print("\nfutures from x=+5:")
    for o in report.action_rankings:
        print(f"  {o.action:>5}: E[reward]={o.expected_reward:7.2f}  "
              f"tail_risk={o.tail_risk:6.2f}  score={o.score:7.2f}")
    print(f"\n  best_action : {report.best_action}")
    print(f"  confidence  : {report.confidence}")
    assert report.best_action == "left", report.best_action

    # Goal probability: reaching x within [-1, 1].
    goal = lambda st: abs(st[0]) <= 1.0  # noqa: E731
    r2 = sim.simulate((4.0,), ["left", "right", "stay"],
                      branches=900, horizon=10, reward_fn=rfn, goal_fn=goal, noise_scale=0.02)
    gp = {o.action: o.goal_probability for o in r2.action_rankings}
    print(f"\n  P(reach home | left/right) : {gp['left']:.2f} / {gp['right']:.2f}")
    assert r2.best_action == "left"
    # honest, relative claim: going toward home reaches it more often than going away
    assert gp["left"] > gp["right"] and gp["left"] > 0.0

    # Honest uncertainty: an untrained model yields low confidence.
    blank = TimelineSimulator(world_model=WorldModel(), seed=0)
    r3 = blank.simulate((0.0,), ["left", "right"], branches=200, horizon=4)
    print(f"\n  untrained confidence  : {r3.confidence} (honest about the unknown)")
    assert r3.confidence < 0.3

    print("\nALL SELF-TESTS PASSED ✓")
