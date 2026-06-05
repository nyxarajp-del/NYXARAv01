"""NYXARA · sim/montecarlo.py — probabilistic rollout & risk estimation (✦).

The world is not deterministic. An action that *usually* works can fail; a cost that is
*nominally* small can spike; a plan that looks safe in the best case can endanger the
Master in the tail. Before NYXARA commits, it does not reason about a single imagined
future — it **samples thousands of them** and looks at the whole distribution.

This module is NYXARA's Monte Carlo engine:

* :class:`Distribution` — a small, seedable family of samplers (constant, Bernoulli,
  uniform, normal, triangular, categorical) used to describe uncertain costs/outcomes.
* :class:`MonteCarlo` — a generic estimator: run a scalar experiment ``fn(rng) -> float``
  many times, returning an :class:`Estimate` with mean, standard error, confidence
  interval, percentiles, and tail risk (Value-at-Risk / Conditional VaR). It can also
  **run until converged** — sampling in batches until the relative standard error is
  small enough or a trial budget is exhausted.
* :class:`StochasticAction` — a planner :class:`~nyxara.planning.planner.Action` made
  uncertain: a success probability, a cost distribution, and per-effect realisation
  probabilities.
* :class:`MonteCarloPlanner` — rolls a *plan* forward thousands of times against a
  fact-state, estimating probability of reaching the goal, the cost distribution, tail
  cost (CVaR — what the worst few percent look like), and robustness; and **compares**
  candidate plans under a risk-aware score so NYXARA picks the one that protects the
  Master even in the bad tail (Rule 3).

Deterministic given a seed (reproducible, auditable). Pure standard library; integrates
with :mod:`planning.planner` and, optionally, :mod:`sim.sandbox`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, FrozenSet, List, Optional, Sequence,
                    Set, Tuple, Union)

__all__ = [
    "Distribution",
    "Constant",
    "Bernoulli",
    "Uniform",
    "Normal",
    "Triangular",
    "Categorical",
    "Estimate",
    "MonteCarlo",
    "StochasticAction",
    "Rollout",
    "PlanEstimate",
    "MonteCarloPlanner",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #
class Distribution:
    """A seedable scalar sampler. Subclasses implement :meth:`sample`."""

    def sample(self, rng: random.Random) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def mean(self) -> float:
        """Analytic mean where known; subclasses may override."""
        raise NotImplementedError

    def __call__(self, rng: random.Random) -> float:
        return self.sample(rng)


@dataclass
class Constant(Distribution):
    value: float = 0.0

    def sample(self, rng: random.Random) -> float:
        return self.value

    def mean(self) -> float:
        return self.value


@dataclass
class Bernoulli(Distribution):
    p: float = 0.5

    def sample(self, rng: random.Random) -> float:
        return 1.0 if rng.random() < self.p else 0.0

    def mean(self) -> float:
        return _clamp(self.p)


@dataclass
class Uniform(Distribution):
    lo: float = 0.0
    hi: float = 1.0

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.lo, self.hi)

    def mean(self) -> float:
        return (self.lo + self.hi) / 2.0


@dataclass
class Normal(Distribution):
    mu: float = 0.0
    sigma: float = 1.0
    floor: Optional[float] = None     # clamp lower bound (e.g. cost >= 0)
    ceil: Optional[float] = None

    def sample(self, rng: random.Random) -> float:
        x = rng.gauss(self.mu, self.sigma)
        if self.floor is not None:
            x = max(self.floor, x)
        if self.ceil is not None:
            x = min(self.ceil, x)
        return x

    def mean(self) -> float:
        return self.mu


@dataclass
class Triangular(Distribution):
    lo: float = 0.0
    mode: float = 0.5
    hi: float = 1.0

    def sample(self, rng: random.Random) -> float:
        return rng.triangular(self.lo, self.hi, self.mode)

    def mean(self) -> float:
        return (self.lo + self.mode + self.hi) / 3.0


@dataclass
class Categorical(Distribution):
    """Weighted choice over numeric outcomes."""
    outcomes: Sequence[float] = field(default_factory=tuple)
    weights: Sequence[float] = field(default_factory=tuple)

    def sample(self, rng: random.Random) -> float:
        return float(rng.choices(list(self.outcomes),
                                 weights=list(self.weights) or None)[0])

    def mean(self) -> float:
        w = list(self.weights) or [1.0] * len(self.outcomes)
        tot = sum(w) or 1.0
        return sum(o * wi for o, wi in zip(self.outcomes, w)) / tot


def _as_dist(x: Union[Distribution, float, int]) -> Distribution:
    return x if isinstance(x, Distribution) else Constant(float(x))


# --------------------------------------------------------------------------- #
# Estimate — the distribution of an experiment's outcomes
# --------------------------------------------------------------------------- #
@dataclass
class Estimate:
    samples: List[float]

    # ---- central tendency ---- #
    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        return sum(self.samples) / self.n if self.samples else 0.0

    @property
    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        m = self.mean
        return sum((x - m) ** 2 for x in self.samples) / (self.n - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def sem(self) -> float:
        """Standard error of the mean."""
        return self.std / math.sqrt(self.n) if self.n else 0.0

    @property
    def rel_sem(self) -> float:
        """Relative standard error (|sem / mean|); 0 if mean is ~0."""
        m = abs(self.mean)
        return self.sem / m if m > 1e-12 else 0.0

    def ci95(self) -> Tuple[float, float]:
        """Normal-approx 95% confidence interval for the mean."""
        h = 1.96 * self.sem
        return (self.mean - h, self.mean + h)

    # ---- order statistics ---- #
    def percentile(self, p: float) -> float:
        """The p-th percentile (p in [0,100]) by linear interpolation."""
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        if len(s) == 1:
            return s[0]
        rank = _clamp(p / 100.0, 0.0, 1.0) * (len(s) - 1)
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        if lo == hi:
            return s[lo]
        return s[lo] + (s[hi] - s[lo]) * (rank - lo)

    @property
    def median(self) -> float:
        return self.percentile(50.0)

    @property
    def minimum(self) -> float:
        return min(self.samples) if self.samples else 0.0

    @property
    def maximum(self) -> float:
        return max(self.samples) if self.samples else 0.0

    # ---- tail risk (treating samples as losses: higher = worse) ---- #
    def value_at_risk(self, alpha: float = 0.95) -> float:
        """VaR_alpha: the loss threshold not exceeded with probability ``alpha``."""
        return self.percentile(alpha * 100.0)

    def cvar(self, alpha: float = 0.95) -> float:
        """Conditional VaR (expected shortfall): mean of the worst ``1-alpha`` tail."""
        if not self.samples:
            return 0.0
        threshold = self.value_at_risk(alpha)
        tail = [x for x in self.samples if x >= threshold]
        return sum(tail) / len(tail) if tail else threshold

    # ---- predicate probability ---- #
    def prob(self, predicate: Callable[[float], bool]) -> float:
        if not self.samples:
            return 0.0
        return sum(1 for x in self.samples if predicate(x)) / self.n

    # ---- presentation ---- #
    def histogram(self, bins: int = 10) -> List[Tuple[float, float, int]]:
        """Return ``(lo, hi, count)`` buckets spanning [min, max]."""
        if not self.samples:
            return []
        lo, hi = self.minimum, self.maximum
        if hi <= lo:
            return [(lo, hi, self.n)]
        width = (hi - lo) / bins
        counts = [0] * bins
        for x in self.samples:
            idx = min(bins - 1, int((x - lo) / width))
            counts[idx] += 1
        return [(lo + i * width, lo + (i + 1) * width, counts[i]) for i in range(bins)]

    def to_dict(self) -> Dict[str, Any]:
        lo, hi = self.ci95()
        return {"n": self.n, "mean": round(self.mean, 4), "std": round(self.std, 4),
                "sem": round(self.sem, 4), "ci95": [round(lo, 4), round(hi, 4)],
                "median": round(self.median, 4),
                "p05": round(self.percentile(5), 4), "p95": round(self.percentile(95), 4),
                "min": round(self.minimum, 4), "max": round(self.maximum, 4)}


# --------------------------------------------------------------------------- #
# Generic Monte Carlo engine
# --------------------------------------------------------------------------- #
class MonteCarlo:
    """Estimate the distribution of a scalar experiment by repeated sampling."""

    def __init__(self, seed: Optional[int] = 0) -> None:
        self.seed = seed

    def run(self, fn: Callable[[random.Random], float], trials: int = 1000,
            *, seed: Optional[int] = None) -> Estimate:
        """Run ``fn(rng)`` ``trials`` times; return an :class:`Estimate`."""
        rng = random.Random(self.seed if seed is None else seed)
        return Estimate([float(fn(rng)) for _ in range(max(0, trials))])

    def run_until(self, fn: Callable[[random.Random], float], *,
                  rel_tol: float = 0.02, batch: int = 500,
                  min_trials: int = 500, max_trials: int = 50_000,
                  seed: Optional[int] = None) -> Estimate:
        """Sample in batches until the relative standard error is small enough.

        Stops when ``rel_sem <= rel_tol`` (and at least ``min_trials`` were run) or
        when ``max_trials`` is reached. Reproducible for a fixed seed.
        """
        rng = random.Random(self.seed if seed is None else seed)
        samples: List[float] = []
        while len(samples) < max_trials:
            for _ in range(batch):
                samples.append(float(fn(rng)))
            est = Estimate(samples)
            if len(samples) >= min_trials and 0.0 < est.rel_sem <= rel_tol:
                return est
            if est.rel_sem == 0.0 and len(samples) >= min_trials:
                # zero variance: already converged
                return est
        return Estimate(samples)


# --------------------------------------------------------------------------- #
# Stochastic actions & plan rollout
# --------------------------------------------------------------------------- #
@dataclass
class StochasticAction:
    """A planner action with uncertainty: success chance, cost, per-effect realisation."""
    name: str
    preconditions: FrozenSet[str] = frozenset()
    add_effects: FrozenSet[str] = frozenset()
    del_effects: FrozenSet[str] = frozenset()
    success_prob: float = 1.0
    cost: Distribution = field(default_factory=lambda: Constant(1.0))
    # optional: probability a *specific* add-effect actually realises (default 1.0 each)
    effect_probs: Dict[str, float] = field(default_factory=dict)
    failure_cost_fraction: float = 1.0   # fraction of sampled cost still paid on failure

    def applicable(self, state: Set[str]) -> bool:
        return self.preconditions <= state

    def simulate(self, state: Set[str], rng: random.Random) -> Tuple[Set[str], float, bool]:
        """Apply this action stochastically; return (new_state, cost_incurred, succeeded)."""
        cost = max(0.0, self.cost.sample(rng))
        if rng.random() >= _clamp(self.success_prob):
            # failed: pay (a fraction of) cost, apply no effects
            return set(state), cost * _clamp(self.failure_cost_fraction), False
        ns = set(state) - self.del_effects
        for e in self.add_effects:
            if rng.random() < _clamp(self.effect_probs.get(e, 1.0)):
                ns.add(e)
        return ns, cost, True

    @classmethod
    def from_action(cls, action: Any, *, success_prob: float = 1.0,
                    cost: Optional[Distribution] = None,
                    effect_probs: Optional[Dict[str, float]] = None,
                    failure_cost_fraction: float = 1.0) -> "StochasticAction":
        """Wrap a deterministic :class:`planning.planner.Action` with uncertainty."""
        return cls(
            name=action.name,
            preconditions=frozenset(action.preconditions),
            add_effects=frozenset(action.add_effects),
            del_effects=frozenset(action.del_effects),
            success_prob=success_prob,
            cost=cost if cost is not None else Constant(float(getattr(action, "cost", 1.0))),
            effect_probs=dict(effect_probs or {}),
            failure_cost_fraction=failure_cost_fraction)


@dataclass
class Rollout:
    """The result of one simulated execution of a plan."""
    goal_reached: bool
    total_cost: float
    steps_completed: int
    aborted_at: Optional[str]            # action name where it blocked/failed, or None
    final_state: FrozenSet[str]

    def to_dict(self) -> Dict[str, Any]:
        return {"goal_reached": self.goal_reached, "cost": round(self.total_cost, 3),
                "steps": self.steps_completed, "aborted_at": self.aborted_at}


def _rollout(plan: Sequence[StochasticAction], initial: Set[str], goal: Set[str],
             rng: random.Random, *, stop_on_failure: bool = True) -> Rollout:
    state = set(initial)
    total_cost = 0.0
    completed = 0
    aborted_at: Optional[str] = None
    for act in plan:
        if not act.applicable(state):
            aborted_at = act.name
            break
        state, cost, ok = act.simulate(state, rng)
        total_cost += cost
        if ok:
            completed += 1
        else:
            aborted_at = act.name
            if stop_on_failure:
                break
    return Rollout(goal_reached=goal <= state, total_cost=total_cost,
                   steps_completed=completed, aborted_at=aborted_at,
                   final_state=frozenset(state))


# --------------------------------------------------------------------------- #
# Plan estimate & Monte Carlo planner
# --------------------------------------------------------------------------- #
@dataclass
class PlanEstimate:
    label: str
    trials: int
    success_prob: float
    cost: Estimate                       # cost over ALL rollouts
    cost_given_success: Estimate         # cost over successful rollouts only
    steps: Estimate
    abort_counts: Dict[str, int]         # action name -> times it blocked/failed

    @property
    def expected_cost(self) -> float:
        return self.cost.mean

    @property
    def tail_cost(self) -> float:
        """CVaR(95%) of cost — the mean of the worst 5% of runs."""
        return self.cost.cvar(0.95)

    @property
    def robustness(self) -> float:
        """A [0,1] verdict: high success and a contained tail."""
        # success weighted, penalised by tail-vs-median cost blow-up
        med = max(1e-9, self.cost.median)
        blowup = _clamp((self.tail_cost - med) / (med * 3.0), 0.0, 1.0)
        return _clamp(self.success_prob * (1.0 - 0.5 * blowup))

    def risk_score(self, *, cost_weight: float = 0.1, risk_aversion: float = 0.5) -> float:
        """Risk-aware score (higher is better): reward success, penalise expected and
        tail cost. ``risk_aversion`` blends expected cost with the CVaR tail."""
        blended_cost = ((1.0 - risk_aversion) * self.expected_cost
                        + risk_aversion * self.tail_cost)
        return self.success_prob - cost_weight * blended_cost

    @property
    def worst_blocker(self) -> Optional[str]:
        return max(self.abort_counts, key=self.abort_counts.get) if self.abort_counts else None

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "trials": self.trials,
                "success_prob": round(self.success_prob, 4),
                "expected_cost": round(self.expected_cost, 3),
                "tail_cost_cvar95": round(self.tail_cost, 3),
                "robustness": round(self.robustness, 4),
                "risk_score": round(self.risk_score(), 4),
                "worst_blocker": self.worst_blocker,
                "cost": self.cost.to_dict()}


class MonteCarloPlanner:
    """Roll plans forward many times to estimate success, cost and tail risk."""

    def __init__(self, seed: Optional[int] = 0) -> None:
        self.seed = seed

    def evaluate(self, plan: Sequence[StochasticAction],
                 initial_facts: Sequence[str], goal: Sequence[str], *,
                 trials: int = 2000, stop_on_failure: bool = True,
                 label: str = "plan", seed: Optional[int] = None) -> PlanEstimate:
        rng = random.Random(self.seed if seed is None else seed)
        init = set(initial_facts)
        goalset = set(goal)
        costs: List[float] = []
        success_costs: List[float] = []
        steps: List[float] = []
        successes = 0
        aborts: Dict[str, int] = {}
        for _ in range(max(1, trials)):
            r = _rollout(plan, init, goalset, rng, stop_on_failure=stop_on_failure)
            costs.append(r.total_cost)
            steps.append(float(r.steps_completed))
            if r.goal_reached:
                successes += 1
                success_costs.append(r.total_cost)
            if r.aborted_at is not None:
                aborts[r.aborted_at] = aborts.get(r.aborted_at, 0) + 1
        return PlanEstimate(
            label=label, trials=trials, success_prob=successes / max(1, trials),
            cost=Estimate(costs), cost_given_success=Estimate(success_costs),
            steps=Estimate(steps), abort_counts=aborts)

    def compare(self, plans: Dict[str, Sequence[StochasticAction]],
                initial_facts: Sequence[str], goal: Sequence[str], *,
                trials: int = 2000, risk_aversion: float = 0.5,
                cost_weight: float = 0.1, seed: Optional[int] = None
                ) -> List[PlanEstimate]:
        """Evaluate several plans and return them ranked best→worst by risk score."""
        results = [self.evaluate(p, initial_facts, goal, trials=trials,
                                 label=name, seed=seed)
                   for name, p in plans.items()]
        results.sort(key=lambda e: e.risk_score(cost_weight=cost_weight,
                                                 risk_aversion=risk_aversion),
                     reverse=True)
        return results


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA Monte Carlo self-test")
    print("=" * 70)

    # ---- generic estimator: estimate pi by sampling the unit square ---- #
    def in_circle(rng: random.Random) -> float:
        x, y = rng.random(), rng.random()
        return 4.0 if x * x + y * y <= 1.0 else 0.0

    mc = MonteCarlo(seed=7)
    pi_est = mc.run(in_circle, trials=20000)
    print(f"\nestimate of pi      : {pi_est.mean:.3f}  ci95={tuple(round(v,3) for v in pi_est.ci95())}")
    assert abs(pi_est.mean - math.pi) < 0.1

    # ---- run-until-converged ---- #
    conv = mc.run_until(lambda r: r.gauss(10.0, 2.0), rel_tol=0.01, seed=3)
    print(f"converged mean      : {conv.mean:.3f} (n={conv.n}, rel_sem={conv.rel_sem:.4f})")
    assert abs(conv.mean - 10.0) < 0.3
    assert conv.rel_sem <= 0.01 or conv.n >= 50_000

    # ---- tail risk on a heavy-tailed cost ---- #
    loss = mc.run(lambda r: max(0.0, r.gauss(5.0, 2.0)) + (50.0 if r.random() < 0.02 else 0.0),
                  trials=10000)
    print(f"cost mean / VaR95   : {loss.mean:.2f} / {loss.value_at_risk(0.95):.2f}"
          f"  CVaR95={loss.cvar(0.95):.2f}")
    assert loss.cvar(0.95) > loss.mean   # the tail is worse than the average

    # ---- plan rollout: two plans to the same goal ---- #
    # Plan A: one flaky-but-cheap action that often fails.
    plan_a = [StochasticAction("hack_fast", add_effects=frozenset({"done"}),
                               success_prob=0.6, cost=Constant(1.0))]
    # Plan B: a reliable two-step path that costs a bit more.
    plan_b = [
        StochasticAction("prepare", add_effects=frozenset({"ready"}),
                         success_prob=0.99, cost=Constant(1.0)),
        StochasticAction("execute", preconditions=frozenset({"ready"}),
                         add_effects=frozenset({"done"}),
                         success_prob=0.97, cost=Normal(2.0, 0.5, floor=0.0)),
    ]

    planner = MonteCarloPlanner(seed=42)
    est_a = planner.evaluate(plan_a, [], ["done"], trials=4000, label="A:fast")
    est_b = planner.evaluate(plan_b, [], ["done"], trials=4000, label="B:reliable")
    print(f"\nplan A success      : {est_a.success_prob:.3f}  E[cost]={est_a.expected_cost:.2f}")
    print(f"plan B success      : {est_b.success_prob:.3f}  E[cost]={est_b.expected_cost:.2f}")
    assert 0.5 < est_a.success_prob < 0.7      # ~0.6 as configured
    assert est_b.success_prob > 0.9            # reliable path
    assert est_b.robustness > est_a.robustness

    ranked = planner.compare({"A:fast": plan_a, "B:reliable": plan_b}, [], ["done"],
                             trials=4000, risk_aversion=0.6)
    print(f"\nranked (risk-aware) : {[e.label for e in ranked]}")
    assert ranked[0].label == "B:reliable"     # NYXARA prefers the robust plan

    # ---- wrap a deterministic planner Action ---- #
    from nyxara.planning.planner import Action
    det = Action.make("open_door", pre=["have_key"], add=["door_open"], cost=2.0)
    sto = StochasticAction.from_action(det, success_prob=0.9)
    assert sto.name == "open_door" and sto.preconditions == frozenset({"have_key"})
    assert sto.cost.mean() == 2.0
    print(f"\nwrapped action      : {sto.name} p={sto.success_prob} cost~{sto.cost.mean()}")

    print(f"\nplan B report       : {est_b.to_dict()}")
    print("\nALL SELF-TESTS PASSED ✓")
