"""NYXARA · abyss/butterfly_effect.py — the Butterfly-Effect Engine (Abyss · 2).

A wing-beat in the present can become a storm in the future. NYXARA takes the smallest
imaginable detail — a few milliseconds of network-latency drop, a 0.01-second change in
the Master's typing rhythm, a rounding error in a sensor reading — and asks: *if this one
tiny thing were different, how different would the far future be?*

The engine isolates the consequence of a **perturbation to the present state** (not a
change of action): it rolls the *same* policy forward from both the baseline state and a
minutely perturbed copy, then measures how the two futures **diverge step by step**. From
that it reports:

* the **cascade curve** — state-distance between the two futures at each step;
* the **amplification factor** — final divergence ÷ the initial nudge. Greater than 1 means
  the system is *chaotic* here (the butterfly's wing matters); less than 1 means it is
  *stable* (small errors die out). This is a discrete Lyapunov-style sensitivity signal;
* the **reward shift** the tiny change ultimately caused.

And, given a state, it **ranks every input dimension by sensitivity** — telling NYXARA
*which* small detail of the present most controls the future, so she watches the variables
that actually matter.

Reuses the codebase's engines rather than reinventing them:

* :class:`~nyxara.mind.world_model.WorldModel` — ``rollout`` gives both imagined
  trajectories (with **honest confidence** that decays out of seen regions);
* :func:`~nyxara.mind.world_model._dist` — the same state metric used by the world model's
  own counterfactual divergence detection.

Honest uncertainty is preserved: with little world-model experience the reported
confidence is low. Pure standard library (numpy only optionally, via the model factory).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from nyxara.mind.world_model import WorldModel, _dist, build_world_model

__all__ = ["Perturbation", "CascadeReport", "ButterflyEffect"]

State = Sequence[Any]
Policy = Any  # a sequence of actions or a callable(state) -> action (WorldModel.rollout)
RewardFn = Callable[[Any, Any, Any], float]


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class Perturbation:
    """A minute change to one dimension of the present state."""

    dimension: int
    delta: float
    label: str = ""

    def describe(self) -> str:
        name = self.label or f"dim[{self.dimension}]"
        return f"{name} {'+' if self.delta >= 0 else ''}{self.delta:g}"


@dataclass
class CascadeReport:
    """How a tiny perturbation to the present cascades through the imagined future."""

    perturbation: Perturbation
    divergence_curve: List[float]   # state-distance between baseline & perturbed per step
    initial_divergence: float       # the size of the nudge itself
    final_divergence: float         # how far apart the two futures end up
    amplification: float            # final ÷ initial (>1 chaotic, <1 stable)
    reward_shift: float             # perturbed total reward − baseline total reward
    horizon: int
    confidence: float               # honest confidence in this projection [0,1]

    @property
    def is_chaotic(self) -> bool:
        """True when the tiny change grows — the butterfly's wing matters here."""
        return self.amplification > 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perturbation": self.perturbation.describe(),
            "dimension": self.perturbation.dimension,
            "delta": self.perturbation.delta,
            "divergence_curve": [round(d, 6) for d in self.divergence_curve],
            "initial_divergence": round(self.initial_divergence, 6),
            "final_divergence": round(self.final_divergence, 6),
            "amplification": round(self.amplification, 4),
            "is_chaotic": self.is_chaotic,
            "reward_shift": round(self.reward_shift, 4),
            "horizon": self.horizon,
            "confidence": round(self.confidence, 4),
        }


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class ButterflyEffect:
    """Propagate minute perturbations through the world model and rank sensitivities.

    Parameters
    ----------
    world_model:    a :class:`~nyxara.mind.world_model.WorldModel` (or any object with the
                    same ``predict`` / ``rollout`` / ``__len__`` surface). When ``None`` a
                    fresh model is built via
                    :func:`~nyxara.mind.world_model.build_world_model`.
    seed:           base RNG seed (only used if a stochastic policy is supplied).
    """

    def __init__(self, world_model: Any = None, *, seed: int = 0) -> None:
        self.world_model = world_model if world_model is not None else build_world_model("auto")
        self.seed = seed

    # ---------------------------------------------------------------------- #
    def propagate(self, start: State, policy: Policy, perturbation: Perturbation, *,
                  horizon: int = 12, reward_fn: Optional[RewardFn] = None) -> CascadeReport:
        """Roll ``policy`` from ``start`` and from a copy nudged by ``perturbation``; measure
        how the two imagined futures diverge over ``horizon`` steps."""
        base_start = [float(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
                      else v for v in start]
        pert_start = list(base_start)
        d = perturbation.dimension
        if not (0 <= d < len(pert_start)):
            raise IndexError(f"perturbation dimension {d} out of range for state of "
                             f"length {len(pert_start)}")
        pert_start[d] = pert_start[d] + perturbation.delta

        base = self.world_model.rollout(base_start, policy, steps=horizon, reward_fn=reward_fn)
        pert = self.world_model.rollout(pert_start, policy, steps=horizon, reward_fn=reward_fn)

        n = min(len(base.states), len(pert.states))
        curve = [_dist(base.states[i], pert.states[i]) for i in range(n)]
        initial = abs(float(perturbation.delta))
        final = curve[-1] if curve else 0.0
        eps = 1e-12
        amplification = final / max(eps, initial)
        reward_shift = pert.total_reward - base.total_reward
        confidence = 0.5 * (base.mean_confidence + pert.mean_confidence)

        return CascadeReport(
            perturbation=perturbation, divergence_curve=curve,
            initial_divergence=initial, final_divergence=final,
            amplification=amplification, reward_shift=reward_shift,
            horizon=horizon, confidence=round(confidence, 4))

    def sensitivity(self, start: State, policy: Policy, *, horizon: int = 12,
                    delta: float = 0.01, dims: Optional[Sequence[int]] = None,
                    labels: Optional[Sequence[str]] = None,
                    reward_fn: Optional[RewardFn] = None) -> List[CascadeReport]:
        """Perturb each numeric dimension of ``start`` by ``delta`` in turn and return the
        per-dimension :class:`CascadeReport`\\ s, **sorted most-sensitive first**.

        ``delta`` may be a single float (applied to every dimension) — this answers "which
        tiny detail of the present most controls the future?"
        """
        if dims is None:
            dims = [i for i, v in enumerate(start)
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
        reports: List[CascadeReport] = []
        for i in dims:
            label = ""
            if labels is not None and 0 <= i < len(labels):
                label = labels[i]
            p = Perturbation(dimension=i, delta=delta, label=label)
            reports.append(self.propagate(start, policy, p, horizon=horizon,
                                          reward_fn=reward_fn))
        reports.sort(key=lambda r: r.amplification, reverse=True)
        return reports

    def most_sensitive(self, start: State, policy: Policy, **kwargs: Any) -> Optional[CascadeReport]:
        """Convenience: the single most sensitive input dimension (or None)."""
        ranking = self.sensitivity(start, policy, **kwargs)
        return ranking[0] if ranking else None


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA butterfly-effect self-test")
    print("=" * 70)

    rng = random.Random(0)

    # --- 1-D world with a chaotic 'grow' map and a stable 'decay' map --- #
    #   grow:  x -> 1.5 x   (tiny errors amplify)
    #   decay: x -> 0.5 x   (tiny errors die out)
    wm = WorldModel(k=3, distance_scale=2.0)
    for _ in range(1200):
        x = rng.uniform(-5, 5)
        wm.observe((x,), "grow", (1.5 * x,))
        wm.observe((x,), "decay", (0.5 * x,))

    bfe = ButterflyEffect(world_model=wm)

    grow = bfe.propagate((2.0,), ["grow"] * 6, Perturbation(0, 0.05, "seed"), horizon=6)
    decay = bfe.propagate((2.0,), ["decay"] * 6, Perturbation(0, 0.05, "seed"), horizon=6)
    print(f"\ngrow  : amplification={grow.amplification:.2f}  chaotic={grow.is_chaotic}")
    print(f"        curve={[round(d, 3) for d in grow.divergence_curve]}")
    print(f"decay : amplification={decay.amplification:.2f}  chaotic={decay.is_chaotic}")
    print(f"        curve={[round(d, 3) for d in decay.divergence_curve]}")
    assert grow.is_chaotic and grow.amplification > 2.0      # the wing-beat became a storm
    assert not decay.is_chaotic and decay.amplification < 1.0  # the nudge died out

    # --- 2-D world: dim0 grows, dim1 stays — dim0 must rank as more sensitive --- #
    wm2 = WorldModel(k=3, distance_scale=2.0)
    for _ in range(1500):
        a = rng.uniform(-5, 5)
        b = rng.uniform(-5, 5)
        wm2.observe((a, b), "step", (1.5 * a, b))   # only the first dimension amplifies
    bfe2 = ButterflyEffect(world_model=wm2)
    ranking = bfe2.sensitivity((2.0, 2.0), ["step"] * 6, horizon=6, delta=0.05,
                               labels=["latency", "typing_speed"])
    print("\nsensitivity ranking (2-D world):")
    for r in ranking:
        print(f"  {r.perturbation.describe():<18} amplification={r.amplification:.2f}")
    assert ranking[0].perturbation.dimension == 0   # the amplifying dimension dominates
    assert ranking[0].amplification > ranking[1].amplification

    top = bfe2.most_sensitive((2.0, 2.0), ["step"] * 6, horizon=6, delta=0.05,
                              labels=["latency", "typing_speed"])
    print(f"\nmost sensitive detail : {top.perturbation.describe()} "
          f"(×{top.amplification:.1f})")
    assert top is not None and top.perturbation.label == "latency"

    # --- honest uncertainty: an untrained model is not confident --- #
    blank = ButterflyEffect(world_model=WorldModel())
    r = blank.propagate((1.0,), ["grow"] * 4, Perturbation(0, 0.01), horizon=4)
    print(f"\nuntrained confidence  : {r.confidence} (honest about the unknown)")
    assert r.confidence < 0.3

    print("\nALL SELF-TESTS PASSED ✓")
