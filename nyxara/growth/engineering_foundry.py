"""NYXARA · growth/engineering_foundry.py — invent a formula, then DESIGN the machine (🛠️, no LLM).

The second half of "magic engineering". :mod:`nyxara.growth.law_discovery` lets NYXARA *invent* new
empirical/physical laws from data she gathers herself; this engine *uses* those formulas — and the
real physics sandboxes in :mod:`nyxara.sim` — to **design, validate, and iteratively upgrade real
device concepts in simulation**. It closes the loop::

    invent a law  →  design a device from it  →  need a better law  →  invent it  →  upgrade the device

Everything here is her own compute — **no LLM anywhere in the loop** (a test enforces it). The moving
parts:

* **Feasibility gate + impossibility ledger.** Before optimising anything, a target is judged against
  the fundamental conservation laws (energy/1st law, entropy/2nd law, momentum, causality). "Magic"
  targets that violate them — over-unity / zero-point energy, reaction-less / anti-gravity thrust,
  time reversal, faster-than-light — are returned as an honest ``INFEASIBLE`` verdict *with the law
  they break*, and logged to a persisted ledger. She never fakes a machine physics forbids. This is
  the capability working correctly, not a limitation.
* **Multi-physics evaluator.** One device is scored across several coupled ``nyxara.sim`` engines at
  once (a discovered :class:`~nyxara.growth.law_discovery.Law` via ``Law.predict`` is just another
  evaluator), so a design is judged on combined physics, not one narrow model.
* **Device archetype library.** A registry of parametric, real-physics device templates (RC/RL
  filters, resonators, electrostatic traps, kinetic pressure vessels, and law-driven devices) she
  parameterises and composes.
* **Portfolio multi-objective optimiser.** A portfolio of real optimisers — random search (baseline),
  compass/pattern search, a CMA-ES-style evolution strategy, and ``scipy.optimize`` when present —
  arbitrated by a persisted UCB1 meta-gate that learns which optimiser wins per problem class.
  Multi-objective via a genuine non-dominated (Pareto) front.
* **Self-upgrade + device tower.** ``upgrade_device`` re-opens a prior design, widens its space,
  re-optimises, and keeps the result **only if it is measurably better** — then persists it to a
  device tower so designs compound across sessions and she genuinely grows more powerful over time.

Optional ``numpy``/``scipy`` accelerate the search; without them a pure-stdlib fallback runs the same
algorithms. Deterministic given a seed.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Optional accelerators — import-guarded, with a real stdlib fallback (repo convention).
try:  # numpy is a core dep, but stay honest if it is absent
    import numpy as _np  # type: ignore
    _HAS_NUMPY = True
except Exception:  # noqa: BLE001
    _np = None  # type: ignore
    _HAS_NUMPY = False

try:
    from scipy import optimize as _scipy_opt  # type: ignore
    _HAS_SCIPY = True
except Exception:  # noqa: BLE001
    _scipy_opt = None  # type: ignore
    _HAS_SCIPY = False

__all__ = [
    "Parameter",
    "DesignSpace",
    "Objective",
    "Constraint",
    "DeviceSpec",
    "FeasibilityVerdict",
    "FeasibilityGate",
    "MultiPhysicsEvaluator",
    "DeviceDesign",
    "EngineeringFoundry",
]


# --------------------------------------------------------------------------------------------------
# Design space — the tunable parameters of a device
# --------------------------------------------------------------------------------------------------
@dataclass
class Parameter:
    """A single tunable design parameter, bounded to a physically sensible range."""
    name: str
    low: float
    high: float

    def clamp(self, value: float) -> float:
        return min(self.high, max(self.low, float(value)))

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.low, self.high)


@dataclass
class DesignSpace:
    """An ordered set of :class:`Parameter` — the box the optimiser searches in."""
    params: List[Parameter] = field(default_factory=list)

    def names(self) -> List[str]:
        return [p.name for p in self.params]

    def sample(self, rng: random.Random) -> Dict[str, float]:
        return {p.name: p.sample(rng) for p in self.params}

    def to_vector(self, point: Dict[str, float]) -> List[float]:
        return [float(point[p.name]) for p in self.params]

    def from_vector(self, vec: Sequence[float]) -> Dict[str, float]:
        return {p.name: p.clamp(vec[i]) for i, p in enumerate(self.params)}

    def clamp_vector(self, vec: Sequence[float]) -> List[float]:
        return [p.clamp(vec[i]) for i, p in enumerate(self.params)]

    def widened(self, factor: float = 0.5) -> "DesignSpace":
        """A copy with every bound stretched outward by ``factor`` of its span — used on upgrade to
        let a design escape a local optimum without abandoning the incumbent's region."""
        out: List[Parameter] = []
        for p in self.params:
            span = (p.high - p.low) or abs(p.high) or 1.0
            pad = span * float(factor)
            out.append(Parameter(p.name, p.low - pad, p.high + pad))
        return DesignSpace(out)


# --------------------------------------------------------------------------------------------------
# Objectives & constraints (multi-objective)
# --------------------------------------------------------------------------------------------------
@dataclass
class Objective:
    """One optimisation objective: ``fn(metrics) -> float``, maximised or minimised."""
    name: str
    fn: Callable[[Dict[str, float]], float]
    maximize: bool = True
    weight: float = 1.0

    def signed(self, metrics: Dict[str, float]) -> float:
        """Value oriented so that *larger is always better* (the optimiser only maximises)."""
        v = float(self.fn(metrics))
        return v if self.maximize else -v


@dataclass
class Constraint:
    """A hard constraint ``fn(metrics)`` compared against ``bound``. Violations are penalised."""
    name: str
    fn: Callable[[Dict[str, float]], float]
    kind: str = "<="            # "<=" | ">=" | "=="
    bound: float = 0.0
    tol: float = 1e-6

    def violation(self, metrics: Dict[str, float]) -> float:
        """Non-negative amount by which the constraint is violated (0.0 ⇒ satisfied)."""
        v = float(self.fn(metrics))
        if self.kind == "<=":
            return max(0.0, v - self.bound)
        if self.kind == ">=":
            return max(0.0, self.bound - v)
        return max(0.0, abs(v - self.bound) - self.tol)   # "=="


# --------------------------------------------------------------------------------------------------
# Feasibility — the first-principles gate + impossibility ledger
# --------------------------------------------------------------------------------------------------
@dataclass
class FeasibilityVerdict:
    """An honest, physics-grounded ruling on whether a target *can* be engineered at all."""
    status: str                 # FEASIBLE | INFEASIBLE | SPECULATIVE
    law: str = ""               # the conservation law / principle invoked
    reason: str = ""

    @property
    def feasible(self) -> bool:
        return self.status == "FEASIBLE"

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "law": self.law, "reason": self.reason}


class FeasibilityGate:
    """Judge a target against the fundamental conservation laws — deterministic, auditable, no LLM.

    The classifier is intentionally conservative: only targets that provably break a conservation
    law are ``INFEASIBLE``; targets merely *beyond current science* are ``SPECULATIVE`` (honest, not
    faked, not claimed); everything else is ``FEASIBLE`` and proceeds to real optimisation. A
    :class:`~nyxara.mind.first_principles.FirstPrinciplesEngine`, if supplied, is consulted for a
    corroborating derivation but never overrides a hard conservation ruling.
    """

    # (any-substring trigger, law, reason) — hard violations of a conservation law
    _INFEASIBLE = (
        (("over-unity", "over unity", "overunity", "perpetual motion", "perpetual-motion",
          "free energy", "zero-point energy", "zero point energy", "zeropoint", "cop > 1",
          "efficiency > 1", "efficiency>1", "more energy out than in", "infinite energy"),
         "Conservation of energy (1st law of thermodynamics)",
         "Net energy output cannot exceed input in a closed system; no device can create energy."),
        (("perpetual", "entropy decrease of an isolated", "reverse entropy", "maxwell's demon",
          "maxwell demon", "unlimited cooling", "100% efficient heat engine",
          "second-law violation"),
         "2nd law of thermodynamics (entropy)",
         "Total entropy of an isolated system cannot decrease; no cyclic engine reaches 100%."),
        (("anti-gravity", "antigravity", "anti gravity", "reactionless", "reaction-less",
          "reactionless drive", "inertialess", "gravity shield", "gravity shielding",
          "thrust without propellant", "propellantless"),
         "Conservation of momentum (Newton's 3rd law / equivalence principle)",
         "Net thrust requires expelled reaction mass; mass cannot be shielded from gravity."),
        (("time travel", "time-travel", "time manipulation", "time reversal", "reverse time",
          "travel to the past", "rewind time", "faster than light", "faster-than-light", "ftl",
          "warp drive", "superluminal"),
         "Causality / special relativity",
         "No signal or object can precede its cause or exceed light speed in vacuum."),
    )
    # beyond current science, but not a hard conservation violation — honest 'we cannot claim this'
    _SPECULATIVE = (
        (("immortality", "immortal", "live forever", "eternal life", "halt aging forever",
          "cure death", "biological immortality", "never die"),
         "Thermodynamics of open living systems",
         "Aging can be slowed in principle, but indefinite entropy defiance in a living body is "
         "beyond established science; NYXARA will not claim a device she cannot ground."),
        (("consciousness upload", "mind upload", "teleportation of matter", "psychic", "telepathy",
          "telekinesis"),
         "No established physical mechanism",
         "No known physical law supports this; treated as speculative rather than designed."),
    )

    def __init__(self, first_principles: Any = None) -> None:
        self.first_principles = first_principles

    def classify(self, target: str, *, claims: Optional[Dict[str, Any]] = None) -> FeasibilityVerdict:
        low = (target or "").lower()
        # Structured over-unity claim (energy_out > energy_in) — a hard, numeric 1st-law violation.
        if claims:
            e_in = claims.get("energy_in")
            e_out = claims.get("energy_out")
            if isinstance(e_in, (int, float)) and isinstance(e_out, (int, float)):
                if e_out > e_in * (1.0 + 1e-9):
                    return FeasibilityVerdict(
                        "INFEASIBLE", "Conservation of energy (1st law of thermodynamics)",
                        f"Claimed output {e_out} exceeds input {e_in}; energy cannot be created.")
        for triggers, law, reason in self._INFEASIBLE:
            if any(t in low for t in triggers):
                return FeasibilityVerdict("INFEASIBLE", law, reason)
        for triggers, law, reason in self._SPECULATIVE:
            if any(t in low for t in triggers):
                return FeasibilityVerdict("SPECULATIVE", law, reason)
        # Feasible: optionally attach a first-principles corroboration (never blocks).
        note = ""
        if self.first_principles is not None:
            try:
                d = self.first_principles.derive(target)
                if d is not None:
                    note = "First-principles derivation available."
            except Exception:  # noqa: BLE001
                note = ""
        return FeasibilityVerdict("FEASIBLE", "", note)


# --------------------------------------------------------------------------------------------------
# Device specification & multi-physics evaluation
# --------------------------------------------------------------------------------------------------
Evaluator = Callable[[Dict[str, float]], Dict[str, float]]


class MultiPhysicsEvaluator:
    """Combine several physics evaluators into one metrics dict for a candidate design.

    Each sub-evaluator maps a parameter dict to a metrics dict; their outputs are merged (later keys
    win on collision, so name them distinctly). A discovered :class:`Law` is wrapped by
    :meth:`from_law` into just another evaluator — that is how an invented formula drives a design.
    """

    def __init__(self) -> None:
        self._subs: List[Tuple[str, Evaluator]] = []

    def add(self, name: str, evaluator: Evaluator) -> "MultiPhysicsEvaluator":
        self._subs.append((name, evaluator))
        return self

    def sources(self) -> List[str]:
        return [n for n, _ in self._subs]

    def __call__(self, point: Dict[str, float]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for name, ev in self._subs:
            try:
                out = ev(point)
            except Exception:  # noqa: BLE001 — a failed sub-model contributes nothing, never crashes
                continue
            if isinstance(out, dict):
                for k, v in out.items():
                    try:
                        metrics[k] = float(v)
                    except (TypeError, ValueError):
                        continue
        return metrics

    @staticmethod
    def from_law(law: Any, output_key: str = "law_output") -> Evaluator:
        """Wrap a discovered :class:`~nyxara.growth.law_discovery.Law` as an evaluator.

        The law's ``var_names`` are read from the parameter dict; ``Law.predict`` (already executable
        for both sparse and GP laws) yields the predicted target, exposed as ``output_key``."""
        names = list(getattr(law, "var_names", []) or [])

        def _ev(point: Dict[str, float]) -> Dict[str, float]:
            if not names:
                return {}
            cols = [[float(point.get(n, 0.0))] for n in names]
            pred = law.predict(cols)
            if not pred:
                return {}
            return {output_key: float(pred[0])}

        return _ev


@dataclass
class DeviceSpec:
    """A complete, physics-grounded device design problem."""
    name: str
    space: DesignSpace
    objectives: List[Objective]
    evaluator: Evaluator
    constraints: List[Constraint] = field(default_factory=list)
    archetype: str = "custom"
    target: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------------------
# The optimiser portfolio (all real, all LLM-free)
# --------------------------------------------------------------------------------------------------
ScalarObjective = Callable[[List[float]], float]   # maximise; operates on a design vector


def _random_search(space: DesignSpace, objective: ScalarObjective, budget: int,
                   rng: random.Random) -> Tuple[List[float], float]:
    best_vec = space.to_vector(space.sample(rng))
    best_val = objective(best_vec)
    for _ in range(max(0, budget - 1)):
        vec = space.to_vector(space.sample(rng))
        val = objective(vec)
        if val > best_val:
            best_vec, best_val = vec, val
    return best_vec, best_val


def _pattern_search(space: DesignSpace, objective: ScalarObjective, budget: int,
                    rng: random.Random) -> Tuple[List[float], float]:
    """Compass / coordinate search: probe ±step along each axis, shrink the step when stuck.

    A genuine derivative-free local optimiser (Hooke–Jeeves flavour) — deterministic given the seed
    of its random restart point."""
    dim = len(space.params)
    x = space.clamp_vector(space.to_vector(space.sample(rng)))
    fx = objective(x)
    steps = [0.25 * ((p.high - p.low) or 1.0) for p in space.params]
    evals = 1
    while evals < budget and any(s > 1e-9 for s in steps):
        improved = False
        for d in range(dim):
            for sign in (1.0, -1.0):
                if evals >= budget:
                    break
                trial = list(x)
                trial[d] = space.params[d].clamp(trial[d] + sign * steps[d])
                ft = objective(trial)
                evals += 1
                if ft > fx:
                    x, fx = trial, ft
                    improved = True
                    break
        if not improved:
            steps = [s * 0.5 for s in steps]
    return x, fx


def _evolution_strategy(space: DesignSpace, objective: ScalarObjective, budget: int,
                        rng: random.Random) -> Tuple[List[float], float]:
    """A CMA-ES-style (μ/λ) evolution strategy with a self-adapting isotropic step size.

    Covariance-free (rank-μ mean update + 1/5-success step control) so it needs no numpy, yet it is
    a real, competitive black-box optimiser — not a toy."""
    dim = len(space.params)
    spans = [((p.high - p.low) or abs(p.high) or 1.0) for p in space.params]
    mean = space.clamp_vector(space.to_vector(space.sample(rng)))
    sigma = 0.3                                   # step size as a fraction of each span
    lam = max(6, 4 + int(3 * math.log(max(2, dim))))
    mu = max(1, lam // 2)
    best_vec, best_val = list(mean), objective(mean)
    evals = 1
    while evals < budget:
        pop: List[Tuple[float, List[float]]] = []
        for _ in range(lam):
            if evals >= budget:
                break
            child = [space.params[d].clamp(mean[d] + sigma * spans[d] * rng.gauss(0.0, 1.0))
                     for d in range(dim)]
            val = objective(child)
            evals += 1
            pop.append((val, child))
            if val > best_val:
                best_vec, best_val = child, val
        if not pop:
            break
        pop.sort(key=lambda t: t[0], reverse=True)
        parents = [c for _, c in pop[:mu]]
        mean = [sum(p[d] for p in parents) / len(parents) for d in range(dim)]
        # 1/5-success rule: expand the step when many children beat the old best, shrink otherwise.
        success = sum(1 for v, _ in pop if v > best_val) / len(pop)
        sigma *= math.exp((success - 0.2) / 0.8) if success != 0.2 else 1.0
        sigma = min(1.0, max(1e-4, sigma))
    return best_vec, best_val


def _scipy_de(space: DesignSpace, objective: ScalarObjective, budget: int,
              rng: random.Random) -> Tuple[List[float], float]:
    """SciPy differential evolution when available — a strong global optimiser."""
    bounds = [(p.low, p.high) for p in space.params]
    holder: Dict[str, Tuple[List[float], float]] = {}

    def _neg(v: Any) -> float:
        vec = [float(x) for x in v]
        val = objective(vec)
        cur = holder.get("best")
        if cur is None or val > cur[1]:
            holder["best"] = (vec, val)
        return -val

    popsize = max(5, min(15, budget // (2 * max(1, len(space.params)))))
    maxiter = max(1, budget // (popsize * max(1, len(space.params))))
    _scipy_opt.differential_evolution(              # type: ignore[union-attr]
        _neg, bounds, maxiter=maxiter, popsize=popsize, tol=1e-8, seed=rng.randint(0, 2**31 - 1),
        polish=False, init="sobol" if len(space.params) <= 8 else "latinhypercube")
    if "best" in holder:
        return holder["best"]
    v = space.to_vector(space.sample(rng))
    return v, objective(v)


# name → (fn, always_available)
_OPTIMIZERS: Dict[str, Callable[..., Tuple[List[float], float]]] = {
    "random": _random_search,
    "pattern": _pattern_search,
    "evolution": _evolution_strategy,
}
if _HAS_SCIPY:
    _OPTIMIZERS["scipy_de"] = _scipy_de


class OptimizerPortfolio:
    """Run several optimisers under a shared budget, arbitrated by a persisted UCB1 meta-gate.

    The gate learns, across sessions and per problem-class, which optimiser tends to win, and skews
    the budget toward it while always leaving an exploration floor — the same portfolio+meta-gate
    shape used by :mod:`nyxara.mind.intuition`."""

    def __init__(self, stats: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        self.stats: Dict[str, Dict[str, float]] = stats or {}

    def _ucb(self, name: str, total_n: int) -> float:
        s = self.stats.get(name)
        if not s or s.get("n", 0) < 1:
            return float("inf")                     # try every optimiser at least once
        mean = s["reward"] / s["n"]
        return mean + math.sqrt(2.0 * math.log(max(1, total_n)) / s["n"])

    def _record(self, name: str, reward: float) -> None:
        s = self.stats.setdefault(name, {"n": 0.0, "reward": 0.0})
        s["n"] += 1.0
        s["reward"] += float(reward)

    def optimize(self, space: DesignSpace, objective: ScalarObjective, *, budget: int,
                 rng: random.Random) -> Tuple[List[float], float, str, Dict[str, float]]:
        names = list(_OPTIMIZERS.keys())
        total_n = int(sum(s.get("n", 0) for s in self.stats.values())) or 1
        scores = {n: self._ucb(n, total_n) for n in names}
        finite = [v for v in scores.values() if math.isfinite(v)]
        base = (max(finite) if finite else 1.0) or 1.0
        weights = {n: (base * 1.5 if math.isinf(scores[n]) else max(1e-3, scores[n]))
                   for n in names}
        wsum = sum(weights.values()) or 1.0
        floor = max(8, budget // (len(names) * 3))   # exploration floor per optimiser
        results: List[Tuple[str, List[float], float]] = []
        per_optimizer: Dict[str, float] = {}
        for n in names:
            share = int(budget * weights[n] / wsum)
            b = max(floor, share)
            vec, val = _OPTIMIZERS[n](space, objective, b, rng)
            results.append((n, vec, val))
            per_optimizer[n] = val
        results.sort(key=lambda t: t[2], reverse=True)
        best_name, best_vec, best_val = results[0]
        # Reward = normalised rank within this batch (winner 1.0), folded into the persisted gate.
        vals = [v for _, _, v in results]
        lo, hi = min(vals), max(vals)
        for n, _v, val in results:
            reward = 1.0 if hi <= lo else (val - lo) / (hi - lo)
            self._record(n, reward)
        return best_vec, best_val, best_name, per_optimizer


# --------------------------------------------------------------------------------------------------
# Pareto (multi-objective) utilities
# --------------------------------------------------------------------------------------------------
def _dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """True if objective-vector ``a`` dominates ``b`` (all ≥, at least one >). Larger is better."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def _pareto_front(points: List[Tuple[Dict[str, float], List[float], Dict[str, float]]]
                  ) -> List[Tuple[Dict[str, float], List[float], Dict[str, float]]]:
    """Non-dominated sorting over (params, objective_vector, metrics) archive entries."""
    front: List[Tuple[Dict[str, float], List[float], Dict[str, float]]] = []
    for cand in points:
        cv = cand[1]
        if any(_dominates(other[1], cv) for other in points if other is not cand):
            continue
        # de-duplicate near-identical objective vectors
        if any(all(abs(x - y) < 1e-9 for x, y in zip(cv, f[1])) for f in front):
            continue
        front.append(cand)
    front.sort(key=lambda c: sum(c[1]), reverse=True)
    return front


# --------------------------------------------------------------------------------------------------
# The result of a design run
# --------------------------------------------------------------------------------------------------
@dataclass
class DeviceDesign:
    """A finished (or honestly-refused) device design — the auditable artefact of the foundry."""
    name: str
    archetype: str
    status: str                                       # DESIGNED | INFEASIBLE | SPECULATIVE | FAILED
    feasibility: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    objectives: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    constraints_ok: bool = True
    pareto_front: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    lineage: List[str] = field(default_factory=list)
    created: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "archetype": self.archetype,
            "status": self.status,
            "feasibility": dict(self.feasibility),
            "parameters": {k: round(float(v), 6) for k, v in self.parameters.items()},
            "metrics": {k: round(float(v), 6) for k, v in self.metrics.items()},
            "objectives": {k: round(float(v), 6) for k, v in self.objectives.items()},
            "score": round(float(self.score), 6),
            "constraints_ok": bool(self.constraints_ok),
            "pareto_points": len(self.pareto_front),
            "pareto_front": self.pareto_front,
            "provenance": dict(self.provenance),
            "version": int(self.version),
            "lineage": list(self.lineage),
        }


# --------------------------------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------------------------------
class EngineeringFoundry:
    """Design, validate, and iteratively upgrade physics-grounded devices — her own compute, no LLM.

    All collaborators are optional and composed defensively so a missing one degrades to a
    weaker-but-honest path rather than an error:

    * ``law_discovery`` — a :class:`~nyxara.growth.law_discovery.LawDiscoveryEngine`; the source of
      invented formulas and of new experiments requested mid-design (the formula↔device feedback).
    * ``first_principles`` — a :class:`~nyxara.mind.first_principles.FirstPrinciplesEngine` consulted
      by the feasibility gate.
    * ``path`` — a JSON device tower persisted across sessions; a sibling ``*_ledger.json`` records
      every impossible "magic" target and the law it breaks.
    """

    def __init__(self, *, law_discovery: Any = None, first_principles: Any = None,
                 path: Optional[str] = None, seed: Optional[int] = None,
                 default_budget: int = 160) -> None:
        self.law_discovery = law_discovery
        self.first_principles = first_principles
        self.gate = FeasibilityGate(first_principles=first_principles)
        self.path = path
        self.ledger_path = (os.path.join(os.path.dirname(os.path.abspath(path)),
                                         "engineering_ledger.json") if path else None)
        self._rng = random.Random(seed if seed is not None else (int(time.time() * 1000) & 0x7FFFFFFF))
        self.default_budget = int(default_budget)
        self._devices: Dict[str, DeviceDesign] = {}
        self._ledger: List[Dict[str, Any]] = []
        self._opt_stats: Dict[str, Dict[str, float]] = {}
        self.total_devices = 0
        self.total_upgrades = 0
        self.infeasible_logged = 0
        self._load()

    # ---- feasibility -------------------------------------------------------------------------- #
    def feasibility(self, target: str, *, claims: Optional[Dict[str, Any]] = None
                    ) -> FeasibilityVerdict:
        """Judge whether ``target`` can be engineered at all, from first principles."""
        return self.gate.classify(target, claims=claims)

    # ---- the core design routine -------------------------------------------------------------- #
    def design(self, spec: DeviceSpec, *, budget: Optional[int] = None,
               feasibility: Optional[FeasibilityVerdict] = None) -> DeviceDesign:
        """Run the full pipeline for a fully-specified device: gate → optimise → validate → persist."""
        verdict = feasibility or self.feasibility(spec.target or spec.name)
        if not verdict.feasible:
            design = DeviceDesign(name=spec.name, archetype=spec.archetype,
                                  status=verdict.status, feasibility=verdict.to_dict(),
                                  provenance=dict(spec.provenance))
            if verdict.status == "INFEASIBLE":
                self._log_infeasible(spec.target or spec.name, verdict)
            self._devices[spec.name] = design
            self._save()
            return design

        budget = int(budget or self.default_budget)
        n_obj = len(spec.objectives)
        # archive entry: (params, raw signed objective vector, metrics, penalty)
        archive: List[Tuple[Dict[str, float], List[float], Dict[str, float], float]] = []
        obj_range: List[List[float]] = [[math.inf, -math.inf] for _ in range(n_obj)]
        portfolio = OptimizerPortfolio(self._opt_stats)

        def evaluate(point: Dict[str, float]) -> Tuple[List[float], Dict[str, float], float]:
            metrics = spec.evaluator(point)
            objvec = [obj.signed(metrics) for obj in spec.objectives]
            for i, v in enumerate(objvec):
                obj_range[i][0] = min(obj_range[i][0], v)
                obj_range[i][1] = max(obj_range[i][1], v)
            penalty = sum(c.violation(metrics) for c in spec.constraints)
            archive.append((dict(point), list(objvec), dict(metrics), penalty))
            return objvec, metrics, penalty

        def normalize(objvec: Sequence[float]) -> List[float]:
            """Scale each objective to [0, 1] by its observed range so no single objective — merely
            by having a larger numeric magnitude — dominates the weighted sum."""
            out: List[float] = []
            for i, v in enumerate(objvec):
                lo, hi = obj_range[i]
                out.append((v - lo) / (hi - lo) if hi > lo else 0.5)
            return out

        # Seed the objective ranges with a small random sample so normalisation is sane from step 1.
        for _ in range(max(8, 2 * n_obj)):
            evaluate(spec.space.sample(self._rng))

        # Weighted-sum scalarisations over *normalised* objectives trace the Pareto front.
        weight_sets = self._weight_sets(n_obj, spec.objectives)
        used_optimizers: List[str] = []
        for weights in weight_sets:
            def scalar(vec: List[float]) -> float:
                point = spec.space.from_vector(vec)
                objvec, _metrics, penalty = evaluate(point)
                nrm = normalize(objvec)
                return sum(w * n for w, n in zip(weights, nrm)) - 1e3 * penalty

            _vec, _val, name, _per = portfolio.optimize(
                spec.space, scalar, budget=max(24, budget // len(weight_sets)), rng=self._rng)
            used_optimizers.append(name)

        # Headline design = the archive point that best satisfies the PRIMARY objective (with
        # constraints), tie-broken by the other objectives — so the returned device actually hits
        # its target regardless of the secondary objectives' numeric scale.
        def headline_key(e: Tuple[Dict[str, float], List[float], Dict[str, float], float]) -> Tuple[float, float]:
            _p, ov, _m, pen = e
            secondary = sum(normalize(ov)[1:]) if n_obj > 1 else 0.0
            return (ov[0] - 1e3 * pen, secondary)

        head = max(archive, key=headline_key)
        point, objvec, metrics, penalty = head
        score = objvec[0] - penalty
        best_name = max(set(used_optimizers), key=used_optimizers.count) if used_optimizers else "random"
        front = _pareto_front([(p, ov, m) for p, ov, m, _pen in archive])
        pareto_serialised = [{"parameters": {k: round(v, 6) for k, v in p.items()},
                              "objectives": [round(o, 6) for o in ov],
                              "metrics": {k: round(v, 6) for k, v in m.items()}}
                             for p, ov, m in front[:24]]
        constraints_ok = all(c.violation(metrics) <= c.tol for c in spec.constraints)
        provenance = dict(spec.provenance)
        provenance.update({"optimizer": best_name, "evaluators": [], "budget": budget})
        design = DeviceDesign(
            name=spec.name, archetype=spec.archetype, status="DESIGNED",
            feasibility=verdict.to_dict(), parameters=point, metrics=metrics,
            objectives={spec.objectives[i].name: objvec[i] for i in range(len(spec.objectives))},
            score=float(score), constraints_ok=constraints_ok, pareto_front=pareto_serialised,
            provenance=provenance, version=1)
        self._devices[spec.name] = design
        self.total_devices += 1
        self._opt_stats = portfolio.stats
        self._save()
        return design

    # ---- public entry points ------------------------------------------------------------------ #
    def engineer_device(self, target: Optional[Any] = None, *, spec: Optional[DeviceSpec] = None,
                        archetype: Optional[str] = None, budget: Optional[int] = None,
                        law: Any = None) -> DeviceDesign:
        """Design a device. With an explicit ``spec`` she designs it; otherwise she assembles one —
        from a supplied/invented :class:`Law` (formula→device) or a named/default archetype — and
        designs that, so the capability is always demonstrable end-to-end."""
        # ``target`` is a text intent (gated for feasibility) OR a numeric archetype/law target.
        text_target = target if isinstance(target, str) and target.strip() else None
        num_target = float(target) if isinstance(target, (int, float)) else None
        if text_target:
            verdict = self.feasibility(text_target)
            if not verdict.feasible:
                name = spec.name if spec else (archetype or "target")
                design = DeviceDesign(name=name, archetype=(spec.archetype if spec else
                                      (archetype or "custom")), status=verdict.status,
                                      feasibility=verdict.to_dict(),
                                      provenance={"target": text_target})
                if verdict.status == "INFEASIBLE":
                    self._log_infeasible(text_target, verdict)
                self._devices[name] = design
                self._save()
                return design
        if spec is not None:
            return self.design(spec, budget=budget)
        if law is None and archetype is None and self.law_discovery is not None:
            law = self._latest_or_invent_law()
        if law is not None and archetype is None:
            return self.design_from_law(law, target=num_target, budget=budget)
        built = build_archetype(archetype or "rc_filter", rng=self._rng, target=num_target)
        return self.design(built, budget=budget)

    def design_from_law(self, law: Any, *, target: Optional[str] = None,
                        budget: Optional[int] = None) -> DeviceDesign:
        """Build a device whose behaviour model IS an invented formula, and optimise it.

        Design space = the law's variables (bounded around a working range); objective = drive the
        law's predicted output to a target (or maximise it). This is literally *using the formula she
        invented to design the machine*."""
        spec = self._law_spec(law, target=target)
        return self.design(spec, budget=budget)

    def upgrade_device(self, name: Optional[str] = None, *, budget: Optional[int] = None
                       ) -> DeviceDesign:
        """Re-open a prior design, widen its space, re-optimise, and keep it *only if measurably
        better* — the mechanism by which she keeps upgrading her machines and compounds power."""
        if name is None:
            designed = [d for d in self._devices.values() if d.status == "DESIGNED"]
            if not designed:
                return self.engineer_device(budget=budget)
            name = max(designed, key=lambda d: d.created).name
        incumbent = self._devices.get(name)
        if incumbent is None or incumbent.status != "DESIGNED":
            return self.engineer_device(target=(incumbent.provenance.get("target") if incumbent
                                                else None), budget=budget)
        spec = self._respec(incumbent)
        if spec is None:
            return incumbent
        spec.space = spec.space.widened(0.5)
        challenger = self.design(spec, budget=int((budget or self.default_budget) * 2))
        self.total_upgrades += 1
        if challenger.status == "DESIGNED" and challenger.score > incumbent.score + 1e-9:
            challenger.version = incumbent.version + 1
            challenger.lineage = list(incumbent.lineage) + [f"v{incumbent.version}:{incumbent.score:.4f}"]
            self._devices[name] = challenger
            self._save()
            return challenger
        # keep the incumbent; record that an upgrade was attempted and rejected (honest)
        incumbent.lineage = list(incumbent.lineage) + [
            f"upgrade-rejected@v{incumbent.version}:challenger={challenger.score:.4f}"]
        self._save()
        return incumbent

    # ---- archetype / law spec builders -------------------------------------------------------- #
    def _law_spec(self, law: Any, *, target: Optional[float | str] = None) -> DeviceSpec:
        names = list(getattr(law, "var_names", []) or [])
        if not names:
            names = ["x0"]
        space = DesignSpace([Parameter(n, 0.1, 10.0) for n in names])
        evaluator = MultiPhysicsEvaluator().add("law", MultiPhysicsEvaluator.from_law(law))
        target_val: Optional[float] = None
        if isinstance(target, (int, float)):
            target_val = float(target)
        if target_val is not None:
            objectives = [Objective("hit_target",
                                    lambda m, t=target_val: -abs(m.get("law_output", 0.0) - t),
                                    maximize=True)]
        else:
            objectives = [Objective("maximize_output",
                                    lambda m: m.get("law_output", 0.0), maximize=True)]
        sig = getattr(law, "signature", lambda: "law")()
        expr = getattr(law, "expression", "")
        return DeviceSpec(
            name=f"law-device:{sig}", space=space, objectives=objectives, evaluator=evaluator,
            archetype="law_device", target=(target if isinstance(target, str) else expr),
            provenance={"law": expr, "law_signature": sig, "variables": names,
                        "target_output": target_val})

    def _respec(self, design: DeviceDesign) -> Optional[DeviceSpec]:
        """Rebuild a :class:`DeviceSpec` from a persisted design so it can be re-optimised."""
        if design.archetype == "law_device":
            law = None
            if self.law_discovery is not None:
                sig = design.provenance.get("law_signature")
                for k in self._known_laws():
                    if getattr(k, "signature", lambda: "")() == sig:
                        law = k
                        break
            if law is None:
                return None
            return self._law_spec(law, target=design.provenance.get("target_output"))
        try:
            spec = build_archetype(design.archetype, rng=self._rng,
                                   target=design.provenance.get("target"))
            spec.name = design.name
            return spec
        except Exception:  # noqa: BLE001
            return None

    # ---- formula↔device feedback -------------------------------------------------------------- #
    def _latest_or_invent_law(self) -> Any:
        laws = self._known_laws()
        if laws:
            return laws[-1]
        try:
            self.law_discovery.discover_laws(rounds=1)   # invent one on demand, then design from it
        except Exception:  # noqa: BLE001
            return None
        laws = self._known_laws()
        return laws[-1] if laws else None

    def _known_laws(self) -> List[Any]:
        if self.law_discovery is None:
            return []
        try:
            return list(self.law_discovery.known_laws())
        except Exception:  # noqa: BLE001
            return []

    # ---- multi-objective weighting ------------------------------------------------------------ #
    @staticmethod
    def _weight_sets(k: int, objectives: Sequence[Objective]) -> List[List[float]]:
        base = [max(1e-6, o.weight) for o in objectives]
        if k <= 1:
            return [[1.0]]
        if k == 2:
            grid = [(1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0)]
            return [[g[0] * base[0], g[1] * base[1]] for g in grid]
        # k>=3: the k unit axes plus the balanced centroid
        sets = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
        sets.append([1.0 / k] * k)
        return [[w * base[i] for i, w in enumerate(s)] for s in sets]

    # ---- persistence -------------------------------------------------------------------------- #
    def _log_infeasible(self, target: str, verdict: FeasibilityVerdict) -> None:
        entry = {"target": target, "verdict": verdict.to_dict(), "at": time.time()}
        self._ledger.append(entry)
        self.infeasible_logged += 1
        if self.ledger_path:
            try:
                os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
                with open(self.ledger_path, "w", encoding="utf-8") as fh:
                    json.dump({"impossible_targets": self._ledger}, fh, indent=2)
            except Exception:  # noqa: BLE001 — the ledger is a convenience, never load-bearing
                pass

    def _save(self) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            payload = {
                "total_devices": self.total_devices,
                "total_upgrades": self.total_upgrades,
                "infeasible_logged": self.infeasible_logged,
                "optimizer_stats": self._opt_stats,
                "devices": [d.to_dict() for d in self._devices.values()],
            }
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception:  # noqa: BLE001
            pass

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.total_devices = int(data.get("total_devices", 0))
            self.total_upgrades = int(data.get("total_upgrades", 0))
            self.infeasible_logged = int(data.get("infeasible_logged", 0))
            self._opt_stats = dict(data.get("optimizer_stats", {}) or {})
            for d in data.get("devices", []):
                design = DeviceDesign(
                    name=d.get("name", "device"), archetype=d.get("archetype", "custom"),
                    status=d.get("status", "DESIGNED"), feasibility=d.get("feasibility", {}),
                    parameters=d.get("parameters", {}), metrics=d.get("metrics", {}),
                    objectives=d.get("objectives", {}), score=float(d.get("score", 0.0)),
                    constraints_ok=bool(d.get("constraints_ok", True)),
                    pareto_front=d.get("pareto_front", []), provenance=d.get("provenance", {}),
                    version=int(d.get("version", 1)), lineage=list(d.get("lineage", [])))
                self._devices[design.name] = design
        except Exception:  # noqa: BLE001
            pass
        if self.ledger_path and os.path.exists(self.ledger_path):
            try:
                with open(self.ledger_path, "r", encoding="utf-8") as fh:
                    self._ledger = list(json.load(fh).get("impossible_targets", []))
            except Exception:  # noqa: BLE001
                self._ledger = []

    # ---- inspection --------------------------------------------------------------------------- #
    def known_devices(self) -> List[DeviceDesign]:
        return list(self._devices.values())

    def impossibility_ledger(self) -> List[Dict[str, Any]]:
        return list(self._ledger)

    def report(self) -> Dict[str, Any]:
        designed = [d for d in self._devices.values() if d.status == "DESIGNED"]
        latest = max(designed, key=lambda d: d.created).to_dict() if designed else None
        return {
            "devices_designed": self.total_devices,
            "upgrades_applied": self.total_upgrades,
            "infeasible_targets_logged": self.infeasible_logged,
            "devices_held": len(self._devices),
            "archetypes": sorted(_ARCHETYPES.keys()),
            "optimizers": sorted(_OPTIMIZERS.keys()),
            "latest_device": latest,
            "has_numpy": _HAS_NUMPY,
            "has_scipy": _HAS_SCIPY,
        }


# --------------------------------------------------------------------------------------------------
# Device archetype library — parametric, real-physics templates backed by nyxara.sim
# --------------------------------------------------------------------------------------------------
def _rc_filter_spec(rng: random.Random, target: Any = None) -> DeviceSpec:
    """An RC low-pass filter: choose R, C to hit a target time-constant τ = R·C, measured by the
    real :class:`~nyxara.sim.circuit_world.RCCircuit` integrator (never the formula)."""
    from nyxara.sim.circuit_world import RCCircuit
    circuit = RCCircuit()
    tau_target = float(target) if isinstance(target, (int, float)) else 0.5

    def evaluate(p: Dict[str, float]) -> Dict[str, float]:
        tau = circuit.measure_time_constant(p["R"], p["C"], resolution=900)
        tau = float(tau) if tau is not None else 0.0
        return {"tau": tau, "component_cost": p["R"] * 0.1 + p["C"] * 10.0}

    space = DesignSpace([Parameter("R", 0.1, 10.0), Parameter("C", 0.01, 2.0)])
    objectives = [
        Objective("tau_match", lambda m: -abs(m["tau"] - tau_target), maximize=True),
        Objective("low_cost", lambda m: -m["component_cost"], maximize=True, weight=0.4),
    ]
    return DeviceSpec(name="rc-filter", space=space, objectives=objectives, evaluator=evaluate,
                      archetype="rc_filter", target=f"RC filter with time-constant {tau_target}",
                      provenance={"target": tau_target, "sim": "circuit_world.RCCircuit"})


def _rl_current_spec(rng: random.Random, target: Any = None) -> DeviceSpec:
    """An R-L branch driven to a target steady-state current V/R, measured by the real integrator."""
    from nyxara.sim.circuit_world import RCCircuit
    circuit = RCCircuit()
    i_target = float(target) if isinstance(target, (int, float)) else 1.0

    def evaluate(p: Dict[str, float]) -> Dict[str, float]:
        i = circuit.measure_current(p["V"], p["R"], inductance=p["L"], steps=2500)
        return {"current": float(i or 0.0), "voltage": p["V"]}

    space = DesignSpace([Parameter("V", 0.5, 12.0), Parameter("R", 0.5, 12.0),
                         Parameter("L", 0.2, 4.0)])
    objectives = [
        Objective("current_match", lambda m: -abs(m["current"] - i_target), maximize=True),
        Objective("low_voltage", lambda m: -m["voltage"], maximize=True, weight=0.3),
    ]
    return DeviceSpec(name="rl-source", space=space, objectives=objectives, evaluator=evaluate,
                      archetype="rl_current", target=f"R-L current source at {i_target} A",
                      provenance={"target": i_target, "sim": "circuit_world.RCCircuit"})


def _resonator_spec(rng: random.Random, target: Any = None) -> DeviceSpec:
    """A vibrating-string resonator tuned to a target wave speed v = √(T/μ), measured by the real
    finite-difference :class:`~nyxara.sim.optics_world.WaveString`."""
    from nyxara.sim.optics_world import WaveString
    string = WaveString(n_points=161)
    v_target = float(target) if isinstance(target, (int, float)) else 2.0

    def evaluate(p: Dict[str, float]) -> Dict[str, float]:
        v = string.measure_speed(p["tension"], p["mu"])
        return {"wave_speed": float(v or 0.0), "mass": p["mu"]}

    space = DesignSpace([Parameter("tension", 0.5, 20.0), Parameter("mu", 0.2, 5.0)])
    objectives = [
        Objective("speed_match", lambda m: -abs(m["wave_speed"] - v_target), maximize=True),
        Objective("light_string", lambda m: -m["mass"], maximize=True, weight=0.3),
    ]
    return DeviceSpec(name="string-resonator", space=space, objectives=objectives, evaluator=evaluate,
                      archetype="resonator", target=f"string resonator at wave speed {v_target}",
                      provenance={"target": v_target, "sim": "optics_world.WaveString"})


def _electrostatic_trap_spec(rng: random.Random, target: Any = None) -> DeviceSpec:
    """An electrostatic trap producing a target Coulomb force at a chosen geometry, measured by the
    real :class:`~nyxara.sim.em_world.ElectrostaticWorld` integrator."""
    from nyxara.sim.em_world import ElectrostaticWorld
    world = ElectrostaticWorld()
    f_target = float(target) if isinstance(target, (int, float)) else 5.0

    def evaluate(p: Dict[str, float]) -> Dict[str, float]:
        f = world.measure_force(p["q_source"], p["q_test"], p["r"], steps=6)
        return {"force": abs(float(f)), "separation": p["r"]}

    space = DesignSpace([Parameter("q_source", 0.5, 10.0), Parameter("q_test", 0.5, 10.0),
                         Parameter("r", 0.3, 3.0)])
    objectives = [
        Objective("force_match", lambda m: -abs(m["force"] - f_target), maximize=True),
        Objective("wide_gap", lambda m: m["separation"], maximize=True, weight=0.3),
    ]
    return DeviceSpec(name="electrostatic-trap", space=space, objectives=objectives,
                      evaluator=evaluate, archetype="electrostatic_trap",
                      target=f"electrostatic trap at force {f_target}",
                      provenance={"target": f_target, "sim": "em_world.ElectrostaticWorld"})


def _pressure_vessel_spec(rng: random.Random, target: Any = None) -> DeviceSpec:
    """A kinetic-gas pressure vessel: choose particle count and temperature to hit a target wall
    force (pressure), measured by the real molecular-dynamics :class:`~nyxara.sim.thermo_world.GasBox`."""
    from nyxara.sim.thermo_world import GasBox
    box = GasBox(seed=7)
    p_target = float(target) if isinstance(target, (int, float)) else 40.0

    def evaluate(p: Dict[str, float]) -> Dict[str, float]:
        n = max(1, int(round(p["n_particles"])))
        force = box.measure_wall_force(n, p["temperature"], steps=700, warmup=100)
        return {"wall_force": float(force), "n": float(n), "temperature": p["temperature"]}

    space = DesignSpace([Parameter("n_particles", 5, 120), Parameter("temperature", 0.2, 8.0)])
    objectives = [
        Objective("pressure_match", lambda m: -abs(m["wall_force"] - p_target), maximize=True),
        Objective("fewer_particles", lambda m: -m["n"], maximize=True, weight=0.25),
    ]
    return DeviceSpec(name="pressure-vessel", space=space, objectives=objectives, evaluator=evaluate,
                      archetype="pressure_vessel", target=f"kinetic pressure vessel at {p_target}",
                      provenance={"target": p_target, "sim": "thermo_world.GasBox"})


# name → builder(rng, target) -> DeviceSpec
_ARCHETYPES: Dict[str, Callable[..., DeviceSpec]] = {
    "rc_filter": _rc_filter_spec,
    "rl_current": _rl_current_spec,
    "resonator": _resonator_spec,
    "electrostatic_trap": _electrostatic_trap_spec,
    "pressure_vessel": _pressure_vessel_spec,
}


def build_archetype(name: str, *, rng: Optional[random.Random] = None, target: Any = None
                    ) -> DeviceSpec:
    """Construct a :class:`DeviceSpec` for a named device archetype (raises on unknown name)."""
    builder = _ARCHETYPES.get(name)
    if builder is None:
        raise KeyError(f"unknown archetype {name!r}; known: {sorted(_ARCHETYPES)}")
    return builder(rng or random.Random(0), target)


def archetype_names() -> List[str]:
    return sorted(_ARCHETYPES.keys())


if __name__ == "__main__":  # pragma: no cover — a real, assertion-backed self-demo
    foundry = EngineeringFoundry(seed=1, default_budget=120)

    # 1) A real design run against real physics: hit a target RC time-constant.
    d = foundry.engineer_device(archetype="rc_filter", target=None)
    assert d.status == "DESIGNED", d.status
    print("RC filter →", d.to_dict()["parameters"], "tau≈", round(d.metrics["tau"], 4))
    # the optimiser must genuinely find τ ≈ 0.5 (R·C), not a random guess
    assert abs(d.metrics["tau"] - 0.5) < 0.15, d.metrics

    # 2) The honest gate: a magic target is proven infeasible, never faked.
    inf = foundry.engineer_device(target="a zero-point energy over-unity reactor")
    assert inf.status == "INFEASIBLE", inf.status
    print("magic target →", inf.feasibility["status"], "|", inf.feasibility["law"])
    assert "energy" in inf.feasibility["law"].lower()

    ag = foundry.feasibility("an anti-gravity engine")
    assert ag.status == "INFEASIBLE" and "momentum" in ag.law.lower(), ag.to_dict()

    # 3) A resonator, multi-objective → a real Pareto front.
    r = foundry.engineer_device(archetype="resonator", target=2.0)
    assert r.status == "DESIGNED" and r.pareto_front, r.to_dict()
    print("resonator →", r.to_dict()["parameters"], "v≈", round(r.metrics["wave_speed"], 4),
          "| pareto pts:", len(r.pareto_front))

    print("report:", json.dumps(foundry.report(), indent=2)[:400])
    print("OK — engineering_foundry self-demo passed")
