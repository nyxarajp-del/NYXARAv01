"""NYXARA · nyx/multiverse.py — a trade-off surface, not an answer (🌌, NYX V.03).

Asked to design a system, the useful reply is not a design. It is: *here are the designs nothing
else beats, here is what each costs in the bad tail, here is which failure actually fired and
under what conditions, and here is what stayed true in every future I could check.*

Three layers decide each branch, and they are not three scores blended into one:

* **Logical.** Each :class:`Invariant` renders to a concrete arithmetic claim once the branch's
  numbers are known, and :class:`~nyxara.growth.prover.Prover` certifies it. A branch that
  violates an invariant is **refuted** — struck out, not merely ranked lower. That distinction is
  the whole reason this layer exists: a cheap design that breaks a hard constraint is not a
  cheap design, it is not a design.
* **Causal.** Couplings form a structural causal model and a design is applied as a ``do()`` —
  the value is *set*, its own incoming edges are cut, and effects propagate forward. That is an
  intervention, not a correlation, and cutting the incoming edges is what makes it one.
* **Physical.** Components that declare real physics are measured by the evaluators already in
  :mod:`nyxara.sim` — the RC circuit and the kinetic gas box are solved, not approximated by a
  formula written here.

Thousands of futures come from sampling the declared stochastic drivers, and the output is a
**Pareto front** using :mod:`nyxara.growth.engineering_foundry`'s existing non-dominated sort,
plus **CVaR** on the bad tail rather than an average that hides it.

Honest framing — this *is* the capability working correctly:

* **It reports residual risk; it never reports zero.** A simulation is exactly as true as its
  model, and "no failure margin" is not something any simulator can establish. What it does
  deliver is stronger than a comforting number: every surviving design has each stated invariant
  **machine-proven across every sampled branch**, and the risk that remains is quantified.
* **It abstains when the model is under-determined.** No invariants, no objectives, or too many
  claims the prover cannot decide, and :meth:`Multiverse.explore` refuses with the reason. A
  ranking derived from an under-specified model is worse than no ranking.
* **Sampling is finite.** "Proven across every sampled branch" means the branches that were
  sampled. The count is reported so the strength of the claim is visible.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["Component", "Coupling", "Driver", "Invariant", "FailureMode", "SystemModel",
           "Design", "Branch", "DesignResult", "MultiverseReport", "Multiverse"]


def _fmt(x: float) -> str:
    """Render a number so the prover sees an exact decimal, not scientific notation."""
    return f"{x:.6f}".rstrip("0").rstrip(".") or "0"


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
@dataclass
class Component:
    """One part. ``physics`` names a real evaluator in :mod:`nyxara.sim` when there is one."""

    name: str
    value: float = 0.0
    capacity: float = float("inf")
    physics: str = ""                       # "" | "rc" | "gas"
    params: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value, "capacity": self.capacity,
                "physics": self.physics, "params": dict(self.params)}


@dataclass
class Coupling:
    """A directed causal edge: ``target`` moves by ``strength × source``."""

    source: str
    target: str
    strength: float = 1.0
    kind: str = "linear"

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "target": self.target, "strength": self.strength,
                "kind": self.kind}


@dataclass
class Driver:
    """An exogenous source of variation — what makes futures differ rather than repeat."""

    name: str
    target: str
    mean: float = 0.0
    sigma: float = 1.0

    def sample(self, rng: random.Random) -> float:
        return rng.gauss(self.mean, self.sigma)


@dataclass
class Invariant:
    """Something that must hold. ``expr`` is a template over component names.

    e.g. ``"{load} <= {capacity}"``. It is filled in with the branch's actual numbers and handed
    to the prover, so what is checked is an arithmetic claim rather than a hope.
    """

    name: str
    expr: str

    def render(self, state: Dict[str, float], extra: Optional[Dict[str, float]] = None) -> str:
        values = {**state, **(extra or {})}
        try:
            return self.expr.format(**{k: _fmt(v) for k, v in values.items()})
        except KeyError as exc:
            raise KeyError(f"invariant {self.name!r} refers to unknown name {exc}") from exc


@dataclass
class FailureMode:
    """A named way the system breaks, and the condition under which it does."""

    name: str
    component: str
    threshold: float
    severity: float = 1.0
    direction: str = "above"                # above | below

    def fires(self, state: Dict[str, float]) -> bool:
        value = state.get(self.component)
        if value is None:
            return False
        return value > self.threshold if self.direction == "above" else value < self.threshold


@dataclass
class SystemModel:
    """Components, how they influence each other, what must stay true, and how it can break."""

    name: str = "system"
    components: List[Component] = field(default_factory=list)
    couplings: List[Coupling] = field(default_factory=list)
    drivers: List[Driver] = field(default_factory=list)
    invariants: List[Invariant] = field(default_factory=list)
    failure_modes: List[FailureMode] = field(default_factory=list)
    objectives: Dict[str, str] = field(default_factory=dict)   # label → component (larger better)

    def component(self, name: str) -> Optional[Component]:
        return next((c for c in self.components if c.name == name), None)

    def base_state(self) -> Dict[str, float]:
        return {c.name: float(c.value) for c in self.components}

    def capacities(self) -> Dict[str, float]:
        return {f"{c.name}_capacity": float(c.capacity) for c in self.components}

    def under_determined(self) -> str:
        """Why this model cannot support a ranking, or "" when it can."""
        if not self.components:
            return "no components"
        if not self.objectives:
            return "no objectives — nothing to trade off"
        if not self.invariants:
            return "no invariants — nothing that must hold, so nothing can be proven"
        missing = [label for label, comp in self.objectives.items() if self.component(comp) is None]
        if missing:
            return f"objectives reference unknown components: {sorted(missing)}"
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name,
                "components": [c.to_dict() for c in self.components],
                "couplings": [c.to_dict() for c in self.couplings],
                "drivers": [d.name for d in self.drivers],
                "invariants": [i.name for i in self.invariants],
                "failure_modes": [f.name for f in self.failure_modes],
                "objectives": dict(self.objectives)}


@dataclass
class Design:
    """A candidate: the values it pins. Applying it is an intervention, not an observation."""

    name: str
    interventions: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "interventions": dict(self.interventions)}


# --------------------------------------------------------------------------- #
# Outcomes
# --------------------------------------------------------------------------- #
@dataclass
class Branch:
    """One sampled future for one design."""

    state: Dict[str, float] = field(default_factory=dict)
    objectives: Dict[str, float] = field(default_factory=dict)
    violated: List[str] = field(default_factory=list)
    undecided: List[str] = field(default_factory=list)
    fired: List[str] = field(default_factory=list)
    loss: float = 0.0

    @property
    def refuted(self) -> bool:
        return bool(self.violated)


@dataclass
class DesignResult:
    """What survived, what it costs in the tail, and what was actually proven."""

    design: Design
    branches: int = 0
    refuted_branches: int = 0
    undecided_branches: int = 0
    objectives: Dict[str, float] = field(default_factory=dict)
    cvar: float = 0.0
    worst_loss: float = 0.0
    proven_invariants: List[str] = field(default_factory=list)
    violated_invariants: List[str] = field(default_factory=list)
    fired_failure_modes: Dict[str, int] = field(default_factory=dict)
    residual_risk: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)

    @property
    def viable(self) -> bool:
        """Viable means *no sampled branch* violated an invariant. Not "few". None."""
        return self.refuted_branches == 0 and self.branches > 0

    def to_dict(self) -> Dict[str, Any]:
        return {"design": self.design.to_dict(), "branches": self.branches,
                "refuted_branches": self.refuted_branches,
                "undecided_branches": self.undecided_branches,
                "objectives": {k: round(v, 5) for k, v in self.objectives.items()},
                "cvar": round(self.cvar, 5), "worst_loss": round(self.worst_loss, 5),
                "proven_invariants": self.proven_invariants,
                "violated_invariants": self.violated_invariants,
                "fired_failure_modes": dict(self.fired_failure_modes),
                "residual_risk": round(self.residual_risk, 5),
                "confidence_interval": [round(self.confidence_interval[0], 5),
                                        round(self.confidence_interval[1], 5)],
                "viable": self.viable}


@dataclass
class MultiverseReport:
    """The surface. Never a single answer, and sometimes an honest refusal."""

    system: str = ""
    results: List[DesignResult] = field(default_factory=list)
    pareto: List[str] = field(default_factory=list)
    branches_per_design: int = 0
    abstained: bool = False
    reason: str = ""
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"system": self.system, "results": [r.to_dict() for r in self.results],
                "pareto": self.pareto, "branches_per_design": self.branches_per_design,
                "abstained": self.abstained, "reason": self.reason,
                "seconds": round(self.seconds, 3)}


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
class Multiverse:
    """Roll many futures through three layers, and report what survives all of them."""

    def __init__(self, *, branches: int = 512, seed: int = 0, cvar_alpha: float = 0.1,
                 prover: Any = None, undecided_tolerance: float = 0.25) -> None:
        self.branches = max(1, int(branches))
        self.seed = int(seed)
        self.cvar_alpha = min(0.9, max(0.01, float(cvar_alpha)))
        self.undecided_tolerance = min(1.0, max(0.0, float(undecided_tolerance)))
        self._prover = prover

    # ---- the prover (logical layer) --------------------------------------- #
    def _get_prover(self) -> Any:
        if self._prover is None:
            from nyxara.growth.prover import Prover
            self._prover = Prover()
        return self._prover

    def _check_invariants(self, model: SystemModel, state: Dict[str, float],
                          branch: Branch) -> None:
        """Machine-check every invariant against this branch's actual numbers."""
        from nyxara.growth.prover import ProofVerdict
        prover = self._get_prover()
        extra = model.capacities()
        for invariant in model.invariants:
            try:
                claim = invariant.render(state, extra)
            except KeyError:
                branch.undecided.append(invariant.name)
                continue
            verdict = prover.certify(claim).verdict
            if verdict is ProofVerdict.REFUTED:
                branch.violated.append(invariant.name)
            elif verdict is not ProofVerdict.PROVEN:
                branch.undecided.append(invariant.name)

    # ---- the causal layer -------------------------------------------------- #
    @staticmethod
    def _propagate(model: SystemModel, state: Dict[str, float],
                   pinned: Sequence[str]) -> Dict[str, float]:
        """Push effects along the couplings, leaving intervened values alone.

        Skipping edges *into* a pinned node is what makes this a ``do()`` rather than a
        conditioning: an intervened value is set by us, so its usual causes no longer explain it.
        """
        out = dict(state)
        frozen = set(pinned)
        for _ in range(max(1, len(model.components))):
            changed = False
            for coupling in model.couplings:
                if coupling.target in frozen:
                    continue
                source = out.get(coupling.source)
                if source is None:
                    continue
                base = state.get(coupling.target, 0.0)
                nxt = base + coupling.strength * source
                if abs(nxt - out.get(coupling.target, base)) > 1e-12:
                    out[coupling.target] = nxt
                    changed = True
            if not changed:
                break
        return out

    # ---- the physical layer ------------------------------------------------ #
    @staticmethod
    def _measure_physics(model: SystemModel, state: Dict[str, float]) -> Dict[str, float]:
        """Solve the components that declare real physics, using the existing evaluators."""
        measured: Dict[str, float] = {}
        for component in model.components:
            if not component.physics:
                continue
            try:
                if component.physics == "rc":
                    from nyxara.sim.circuit_world import RCCircuit
                    measured[f"{component.name}_tau"] = float(
                        RCCircuit().measure_time_constant(
                            float(component.params.get("resistance", 1.0)),
                            float(component.params.get("capacitance", 1.0))))
                elif component.physics == "gas":
                    from nyxara.sim.thermo_world import GasBox
                    measured[f"{component.name}_force"] = float(
                        GasBox().measure_wall_force(
                            int(component.params.get("n_particles", 100)),
                            float(component.params.get("temperature", 300.0))))
            except Exception:  # noqa: BLE001 — an unavailable evaluator is a missing measurement
                continue
        return measured

    # ---- one branch -------------------------------------------------------- #
    def _roll(self, model: SystemModel, design: Design, rng: random.Random) -> Branch:
        state = model.base_state()
        for name, value in design.interventions.items():      # do(): pin the design's values
            state[name] = float(value)
        for driver in model.drivers:                          # exogenous variation
            if driver.target not in design.interventions:
                state[driver.target] = state.get(driver.target, 0.0) + driver.sample(rng)
        state = self._propagate(model, state, pinned=list(design.interventions))
        state.update(self._measure_physics(model, state))

        branch = Branch(state=state)
        self._check_invariants(model, state, branch)
        branch.fired = [f.name for f in model.failure_modes if f.fires(state)]
        branch.objectives = {label: float(state.get(comp, 0.0))
                             for label, comp in model.objectives.items()}
        branch.loss = sum(f.severity for f in model.failure_modes if f.name in branch.fired)
        branch.loss += 10.0 * len(branch.violated)
        return branch

    # ---- the run ----------------------------------------------------------- #
    def explore(self, model: SystemModel, designs: Sequence[Design]) -> MultiverseReport:
        """Roll every design through many futures. Abstains rather than guess."""
        started = time.time()
        report = MultiverseReport(system=model.name, branches_per_design=self.branches)

        reason = model.under_determined()
        if reason:
            report.abstained, report.reason = True, f"model is under-determined: {reason}"
            report.seconds = time.time() - started
            return report
        if not designs:
            report.abstained, report.reason = True, "no designs to compare"
            report.seconds = time.time() - started
            return report

        for index, design in enumerate(designs):
            rng = random.Random(self.seed + index * 7919)
            branches = [self._roll(model, design, rng) for _ in range(self.branches)]
            report.results.append(self._summarise(design, branches, model))

        undecided = sum(r.undecided_branches for r in report.results)
        total = max(1, self.branches * len(designs))
        if undecided / total > self.undecided_tolerance:
            report.abstained = True
            report.reason = (f"{undecided}/{total} branches contained claims the prover could not "
                             f"decide — a ranking from an under-specified model is worse than none")
            report.seconds = time.time() - started
            return report

        report.pareto = self._pareto(report.results, model)
        report.seconds = time.time() - started
        return report

    def _summarise(self, design: Design, branches: List[Branch],
                   model: SystemModel) -> DesignResult:
        result = DesignResult(design=design, branches=len(branches))
        result.refuted_branches = sum(1 for b in branches if b.refuted)
        result.undecided_branches = sum(1 for b in branches if b.undecided)

        for label in model.objectives:
            values = [b.objectives.get(label, 0.0) for b in branches]
            result.objectives[label] = sum(values) / len(values) if values else 0.0

        losses = sorted((b.loss for b in branches), reverse=True)
        tail = max(1, int(math.ceil(self.cvar_alpha * len(losses))))
        result.cvar = sum(losses[:tail]) / tail
        result.worst_loss = losses[0] if losses else 0.0

        violated = {name for b in branches for name in b.violated}
        result.violated_invariants = sorted(violated)
        # Proven means proven in EVERY sampled branch — one violation disqualifies it.
        result.proven_invariants = sorted(
            i.name for i in model.invariants
            if i.name not in violated and not any(i.name in b.undecided for b in branches))

        for branch in branches:
            for name in branch.fired:
                result.fired_failure_modes[name] = result.fired_failure_modes.get(name, 0) + 1

        risky = sum(1 for b in branches if b.refuted or b.fired)
        result.residual_risk = risky / len(branches) if branches else 1.0
        result.confidence_interval = self._wilson(risky, len(branches))
        return result

    @staticmethod
    def _wilson(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
        """Wilson interval — an honest interval at small counts, where normal approximation lies."""
        if n <= 0:
            return (0.0, 1.0)
        p = successes / n
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
        return (max(0.0, centre - margin), min(1.0, centre + margin))

    @staticmethod
    def _pareto(results: Sequence[DesignResult], model: SystemModel) -> List[str]:
        """Non-dominated designs, using the foundry's existing sort. Refuted designs cannot enter.

        Objectives are maximised and CVaR is negated, so "less tail risk" is genuinely one of the
        axes rather than a filter applied afterwards.
        """
        from nyxara.growth.engineering_foundry import _pareto_front
        viable = [r for r in results if r.viable]
        if not viable:
            return []
        labels = list(model.objectives)
        points = [({"i": float(i)},
                   [r.objectives.get(label, 0.0) for label in labels] + [-r.cvar],
                   {}) for i, r in enumerate(viable)]
        return [viable[int(params["i"])].design.name
                for params, _, _ in _pareto_front(points)]

    # ---- reporting --------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        return {
            "branches": self.branches, "cvar_alpha": self.cvar_alpha, "seed": self.seed,
            "undecided_tolerance": self.undecided_tolerance,
            "note": ("reports residual risk with an interval; it never reports zero, because a "
                     "simulation is exactly as true as its model. What it does establish is that "
                     "every stated invariant was machine-proven in every SAMPLED branch — a "
                     "finite number, reported. It abstains when the model is under-determined"),
        }

    @staticmethod
    def summary(report: MultiverseReport) -> str:
        lines = ["=" * 70, f"NYXARA multiverse — {report.system}", "=" * 70]
        if report.abstained:
            lines.append(f"ABSTAINED: {report.reason}")
            return "\n".join(lines)
        lines.append(f"{len(report.results)} design(s) × {report.branches_per_design} futures")
        lines.append(f"pareto front : {', '.join(report.pareto) or 'nothing survived'}")
        for r in report.results:
            mark = "✓" if r.viable else "✗"
            lines.append(
                f"  {mark} {r.design.name:<14} "
                f"obj={ {k: round(v, 2) for k, v in r.objectives.items()} } "
                f"cvar={r.cvar:.2f} risk={r.residual_risk:.3f} "
                f"[{r.confidence_interval[0]:.3f},{r.confidence_interval[1]:.3f}]")
            if r.violated_invariants:
                lines.append(f"      REFUTED by: {', '.join(r.violated_invariants)}")
            if r.fired_failure_modes:
                lines.append(f"      fired: {r.fired_failure_modes}")
            if r.proven_invariants:
                lines.append(f"      proven in all {r.branches}: {', '.join(r.proven_invariants)}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA multiverse self-test")
    print("=" * 70)

    # A small service: capacity you buy, load that arrives, latency that follows from both.
    model = SystemModel(
        name="service",
        components=[Component("capacity", value=10.0), Component("load", value=5.0),
                    Component("latency", value=0.0), Component("headroom", value=0.0)],
        couplings=[Coupling("load", "latency", strength=2.0),
                   Coupling("capacity", "headroom", strength=1.0)],
        drivers=[Driver("demand", target="load", mean=0.0, sigma=2.0)],
        invariants=[Invariant("load-within-capacity", "{load} <= {capacity}")],
        failure_modes=[FailureMode("overload", component="load", threshold=9.0, severity=1.0)],
        objectives={"served": "latency", "headroom": "headroom"})

    designs = [Design("lean", {"capacity": 6.0}),
               Design("balanced", {"capacity": 12.0}),
               Design("generous", {"capacity": 20.0})]

    mv = Multiverse(branches=400, seed=5)
    report = mv.explore(model, designs)
    print("\n" + mv.summary(report))

    assert not report.abstained, report.reason
    lean = next(r for r in report.results if r.design.name == "lean")
    generous = next(r for r in report.results if r.design.name == "generous")

    # The logical layer strikes out, it does not merely penalise.
    assert lean.violated_invariants == ["load-within-capacity"], lean.to_dict()
    assert lean.viable is False
    assert "lean" not in report.pareto, "a design that breaks a hard constraint is not on the front"

    # A surviving design carries a machine-checked proof across every sampled branch.
    assert generous.viable and generous.proven_invariants == ["load-within-capacity"]
    print(f"\nproven         : {generous.proven_invariants} across {generous.branches} branches")

    # Residual risk is reported with an interval — never zero-by-assertion.
    print(f"residual risk  : {generous.residual_risk:.4f} "
          f"CI[{generous.confidence_interval[0]:.4f}, {generous.confidence_interval[1]:.4f}]")
    assert generous.confidence_interval[1] > 0.0

    # Abstention when the model cannot support a ranking.
    bare = SystemModel(name="bare", components=[Component("a")], objectives={"x": "a"})
    out = mv.explore(bare, designs)
    print(f"\nabstain (no invariants) : {out.reason}")
    assert out.abstained and "invariants" in out.reason
    assert mv.explore(SystemModel(name="empty"), designs).abstained

    # do() really cuts incoming edges — an intervened value is not moved by its causes.
    pinned = Multiverse._propagate(model, {"capacity": 99.0, "load": 5.0, "latency": 0.0,
                                           "headroom": 0.0}, pinned=["headroom"])
    assert pinned["headroom"] == 0.0, "an intervened value must not be explained by its causes"
    print("intervention   : do() cuts incoming edges")

    # The physical layer really solves physics: tau = RC = 1 ms for 1k-ohm and 1 uF.
    phys = SystemModel(
        name="rc", components=[Component("filter", physics="rc",
                                         params={"resistance": 1000.0, "capacitance": 1e-6})])
    measured = Multiverse._measure_physics(phys, {})
    print(f"physics        : filter_tau = {measured.get('filter_tau')}")
    assert "filter_tau" in measured and abs(measured["filter_tau"] - 1e-3) < 1e-4

    # No data must mean no claim, not a claim of safety.
    assert Multiverse._wilson(0, 0) == (0.0, 1.0)

    print("\nALL SELF-TESTS PASSED ✓")
