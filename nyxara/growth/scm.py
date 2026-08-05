"""NYXARA · growth/scm.py — structural causal models, with the graph criteria that decide them.

The repo has a real causal engine (``mind/causal_world_model.py``: a working ``do()``, genuine
abduction-action-prediction counterfactuals, real NOTEARS). What it does **not** have, confirmed
by grep, is the graph-theoretic half — **no d-separation, no collider awareness, no adjustment-set
enumeration, no identifiability**. ``_find_confounder`` is a permutation-tested statistical
stand-in that would not notice it was conditioning on a collider, and the front-door
implementation argues its exclusion restriction in a comment rather than checking it.

Generating a causal training corpus needs exactly the missing half, and it needs it to be
*correct*, because the whole point of a synthetic causal domain is that the answer is ground
truth rather than an estimate. So this module implements the criteria directly:

* :func:`d_separated` — the standard path-blocking rules, including the collider rule that
  conditioning on a collider *opens* a path rather than closing it. That is the single most
  common causal mistake, and a corpus that got it wrong would teach it.
* :func:`backdoor_sets` — Pearl's back-door criterion: no member may be a descendant of the
  treatment, and the set must block every path with an arrow into the treatment.
* :class:`SCM` — a model whose mechanisms are known, so ``do()`` and counterfactuals are
  computed from the truth rather than inferred from samples.

Deliberately small and exact: integers and :class:`~fractions.Fraction` throughout, so a
generated document contains no floating-point noise that a reader could not reproduce by hand.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "SCM",
    "Mechanism",
    "d_separated",
    "backdoor_sets",
    "ancestors",
    "descendants",
]


@dataclass(frozen=True)
class Mechanism:
    """How one variable is produced from its parents, plus its exogenous noise term.

    ``kind`` is ``"linear"`` or ``"threshold"``. A corpus containing only linear mechanisms
    teaches that effects always add up, which is precisely the intuition that fails on real
    systems — so the generator mixes them.
    """

    node: str
    parents: Tuple[str, ...]
    coeffs: Tuple[Fraction, ...]
    intercept: Fraction = Fraction(0)
    kind: str = "linear"
    threshold: Fraction = Fraction(0)

    def evaluate(self, values: Dict[str, Fraction], noise: Fraction) -> Fraction:
        total = self.intercept + sum(
            (c * values[p] for c, p in zip(self.coeffs, self.parents)), Fraction(0))
        if self.kind == "threshold":
            # A saturating mechanism: above the threshold the parents stop mattering. Nonlinear,
            # still exact, still explainable in one line of prose.
            return (self.threshold + noise) if total > self.threshold else total + noise
        return total + noise

    def formula(self) -> str:
        terms = " + ".join(f"{_num(c)}*{p}" for c, p in zip(self.coeffs, self.parents))
        body = terms or "0"
        if self.intercept:
            body = f"{body} + {_num(self.intercept)}" if terms else _num(self.intercept)
        if self.kind == "threshold":
            return f"{self.node} := min({body}, {_num(self.threshold)}) + noise_{self.node}"
        return f"{self.node} := {body} + noise_{self.node}"


def _num(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


@dataclass
class SCM:
    """A structural causal model with known mechanisms — so every answer is ground truth."""

    nodes: Tuple[str, ...]
    mechanisms: Dict[str, Mechanism]
    #: Variables the observer cannot measure. An effect is unidentifiable when every valid
    #: adjustment set needs one of these.
    latent: FrozenSet[str] = field(default_factory=frozenset)

    # ---- graph ---- #
    def parents(self, node: str) -> Tuple[str, ...]:
        return self.mechanisms[node].parents

    def children(self, node: str) -> Tuple[str, ...]:
        return tuple(n for n in self.nodes if node in self.parents(n))

    def edges(self) -> List[Tuple[str, str]]:
        return [(p, n) for n in self.nodes for p in self.parents(n)]

    def observed(self) -> Tuple[str, ...]:
        return tuple(n for n in self.nodes if n not in self.latent)

    def topological(self) -> List[str]:
        order: List[str] = []
        seen: Set[str] = set()

        def visit(node: str) -> None:
            if node in seen:
                return
            seen.add(node)
            for parent in self.parents(node):
                visit(parent)
            order.append(node)

        for node in self.nodes:
            visit(node)
        return order

    # ---- evaluation ---- #
    def simulate(self, noise: Dict[str, Fraction],
                 do: Optional[Dict[str, Fraction]] = None) -> Dict[str, Fraction]:
        """Run the model forward. ``do`` replaces a variable's mechanism outright.

        That replacement *is* the intervention: the node keeps its children and loses its
        parents, which is why ``P(Y | do(X))`` and ``P(Y | X)`` differ whenever X has a parent
        that also influences Y.
        """
        do = do or {}
        values: Dict[str, Fraction] = {}
        for node in self.topological():
            if node in do:
                values[node] = do[node]
            else:
                values[node] = self.mechanisms[node].evaluate(values, noise.get(node, Fraction(0)))
        return values

    def abduct(self, observed: Dict[str, Fraction]) -> Dict[str, Fraction]:
        """Recover the exogenous noise that must have produced ``observed`` — step 1 of a
        counterfactual. Exact, because the mechanisms are known."""
        noise: Dict[str, Fraction] = {}
        for node in self.topological():
            mech = self.mechanisms[node]
            base = mech.intercept + sum(
                (c * observed[p] for c, p in zip(mech.coeffs, mech.parents)), Fraction(0))
            if mech.kind == "threshold" and base > mech.threshold:
                noise[node] = observed[node] - mech.threshold
            else:
                noise[node] = observed[node] - base
        return noise

    def counterfactual(self, observed: Dict[str, Fraction], node: str,
                       value: Fraction) -> Dict[str, Fraction]:
        """Abduction → action → prediction: what *would* have happened, holding this world fixed."""
        return self.simulate(self.abduct(observed), do={node: value})

    def describe(self) -> str:
        return "\n".join(f"  {self.mechanisms[n].formula()}" for n in self.topological())


# --------------------------------------------------------------------------- #
# Graph criteria
# --------------------------------------------------------------------------- #
def ancestors(scm: SCM, nodes: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    stack = list(nodes)
    while stack:
        node = stack.pop()
        for parent in scm.parents(node):
            if parent not in out:
                out.add(parent)
                stack.append(parent)
    return out


def descendants(scm: SCM, node: str) -> Set[str]:
    out: Set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        for child in scm.children(current):
            if child not in out:
                out.add(child)
                stack.append(child)
    return out


def _undirected_paths(scm: SCM, start: str, end: str) -> List[List[str]]:
    """Every simple path ignoring edge direction — the paths d-separation must block."""
    adjacency: Dict[str, Set[str]] = {n: set() for n in scm.nodes}
    for parent, child in scm.edges():
        adjacency[parent].add(child)
        adjacency[child].add(parent)

    paths: List[List[str]] = []

    def walk(node: str, trail: List[str]) -> None:
        if node == end:
            paths.append(list(trail))
            return
        for nxt in sorted(adjacency[node]):
            if nxt not in trail:
                trail.append(nxt)
                walk(nxt, trail)
                trail.pop()

    walk(start, [start])
    return paths


def _is_collider(scm: SCM, before: str, middle: str, after: str) -> bool:
    return before in scm.parents(middle) and after in scm.parents(middle)


def d_separated(scm: SCM, x: str, y: str, given: Sequence[str] = ()) -> bool:
    """Are ``x`` and ``y`` d-separated given ``given``?

    A path is blocked when any node on it blocks it:

    * a **chain** or **fork** whose middle node is conditioned on — blocked;
    * a **collider** whose node *and all its descendants* are unconditioned — blocked.

    The collider rule runs the other way from the intuition, and it is the mistake that matters:
    conditioning on a collider **opens** a path that was closed, manufacturing a dependence
    between causes that have nothing to do with each other. `_find_confounder` in the existing
    engine has no notion of this at all.
    """
    conditioned = set(given)
    activated = _activated_colliders(scm, conditioned)

    for path in _undirected_paths(scm, x, y):
        blocked = False
        for i in range(1, len(path) - 1):
            before, middle, after = path[i - 1], path[i], path[i + 1]
            if _is_collider(scm, before, middle, after):
                if middle not in activated:
                    blocked = True          # a closed collider blocks
                    break
            elif middle in conditioned:
                blocked = True              # a conditioned chain or fork blocks
                break
        if not blocked:
            return False                    # one open path is enough to connect them
    return True


def _activated_colliders(scm: SCM, conditioned: Set[str]) -> Set[str]:
    """Nodes whose collider is *opened* — the node itself or ANY DESCENDANT is conditioned on.

    The direction matters and is easy to invert. A collider is opened by conditioning on it or
    on something downstream of it, so the test is "does this node's descendant set intersect the
    conditioning set" — not "is this node in the conditioning set's descendants", which is the
    reverse relation and silently leaves `d_sep(A, C | D)` reporting separation for
    ``A -> B <- C, B -> D``. That is the exact error a causal corpus must not contain.
    """
    return {node for node in scm.nodes
            if ({node} | descendants(scm, node)) & conditioned}


def backdoor_sets(scm: SCM, treatment: str, outcome: str, *,
                  allowed: Optional[Sequence[str]] = None,
                  max_size: int = 3) -> List[Tuple[str, ...]]:
    """Every admissible back-door adjustment set, smallest first.

    Pearl's criterion: no member may be a descendant of ``treatment``, and the set must block
    every path from treatment to outcome that begins with an arrow *into* the treatment.
    """
    candidates = [n for n in (allowed if allowed is not None else scm.observed())
                  if n not in (treatment, outcome)]
    forbidden = descendants(scm, treatment) | {treatment, outcome}
    candidates = [n for n in candidates if n not in forbidden]

    backdoor_paths = [p for p in _undirected_paths(scm, treatment, outcome)
                      if len(p) > 1 and p[1] in scm.parents(treatment)]

    admissible: List[Tuple[str, ...]] = []
    for size in range(0, max_size + 1):
        for combo in itertools.combinations(sorted(candidates), size):
            if _blocks_all(scm, backdoor_paths, combo):
                admissible.append(combo)
    return admissible


def _blocks_all(scm: SCM, paths: Sequence[Sequence[str]], given: Sequence[str]) -> bool:
    conditioned = set(given)
    activated = _activated_colliders(scm, conditioned)
    for path in paths:
        blocked = False
        for i in range(1, len(path) - 1):
            before, middle, after = path[i - 1], path[i], path[i + 1]
            if _is_collider(scm, before, middle, after):
                if middle not in activated:
                    blocked = True
                    break
            elif middle in conditioned:
                blocked = True
                break
        if not blocked:
            return False
    return True


# --------------------------------------------------------------------------- #
# Random models with named, plausible variables
# --------------------------------------------------------------------------- #
_SCENARIOS = (
    ("Rainfall", "SoilMoisture", "Yield", "Fertiliser", "Temperature"),
    ("Advertising", "SiteVisits", "Revenue", "Discount", "Season"),
    ("StudyHours", "PracticeTests", "ExamScore", "Tutoring", "SleepQuality"),
    ("MaintenanceSpend", "Downtime", "OutputRate", "OperatorSkill", "MachineAge"),
    ("Dosage", "BloodLevel", "Recovery", "Comorbidity", "Age"),
    ("Insulation", "HeatLoss", "FuelCost", "WindExposure", "BuildingAge"),
    ("Irrigation", "RootDepth", "FruitMass", "SoilType", "Sunlight"),
    ("TrainingHours", "ErrorRate", "Throughput", "ShiftLength", "Tenure"),
)


#: ``(cause A, cause B, their common effect)`` — for the collider shape specifically.
#:
#: Reusing the general scenario tuple here produced graphs like "Rainfall -> Fertiliser <- Yield",
#: which is structurally a correct collider and semantically nonsense. The structure is what the
#: document teaches, but the variable names are read as world facts alongside it, and a corpus
#: asserting that rainfall causes fertiliser teaches that too. These triples are chosen so the
#: third variable is genuinely a common effect of the first two.
_COLLIDER_SCENARIOS = (
    ("Rainfall", "Irrigation", "SoilMoisture"),
    ("Advertising", "Discount", "SiteVisits"),
    ("StudyHours", "Tutoring", "ExamScore"),
    ("OperatorSkill", "MachineAge", "Downtime"),
    ("Dosage", "Comorbidity", "BloodLevel"),
    ("Insulation", "WindExposure", "FuelCost"),
    ("Sunlight", "SoilType", "FruitMass"),
    ("TrainingHours", "ShiftLength", "ErrorRate"),
)


def random_scm(rng: random.Random, *, shape: str = "confounded") -> SCM:
    """Build a small labelled SCM in one of a few structures worth teaching.

    ``confounded`` — C causes both T and Y, so the naive association is biased.
    ``mediated``   — T -> M -> Y, so conditioning on M wrongly removes the effect.
    ``collider``   — T and Y both cause K, so conditioning on K manufactures a dependence.
    ``latent``     — the confounder is unobserved, so the effect is not identifiable.
    """
    treatment, mediator, outcome, cofactor, extra = _SCENARIOS[rng.randrange(len(_SCENARIOS))]
    coefficient = lambda: Fraction(rng.choice([-4, -3, -2, 2, 3, 4, 5]))  # noqa: E731

    def mech(node, parents, kind="linear"):
        return Mechanism(node=node, parents=tuple(parents),
                         coeffs=tuple(coefficient() for _ in parents),
                         intercept=Fraction(rng.randint(-5, 5)), kind=kind,
                         threshold=Fraction(rng.randint(20, 60)))

    if shape == "mediated":
        mechanisms = {treatment: mech(treatment, ()), mediator: mech(mediator, (treatment,)),
                      outcome: mech(outcome, (mediator,))}
        nodes = (treatment, mediator, outcome)
        return SCM(nodes, mechanisms)

    if shape == "collider":
        cause_a, cause_b, effect = _COLLIDER_SCENARIOS[rng.randrange(len(_COLLIDER_SCENARIOS))]
        mechanisms = {cause_a: mech(cause_a, ()), cause_b: mech(cause_b, ()),
                      effect: mech(effect, (cause_a, cause_b))}
        return SCM((cause_a, cause_b, effect), mechanisms)

    kind = "threshold" if rng.random() < 0.3 else "linear"
    mechanisms = {cofactor: mech(cofactor, ()),
                  treatment: mech(treatment, (cofactor,)),
                  outcome: mech(outcome, (treatment, cofactor), kind)}
    nodes = (cofactor, treatment, outcome)
    latent = frozenset({cofactor}) if shape == "latent" else frozenset()
    return SCM(nodes, mechanisms, latent=latent)
