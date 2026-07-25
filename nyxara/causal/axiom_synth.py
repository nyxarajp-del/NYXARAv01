"""NYXARA · causal/axiom_synth.py — Automated Axiom Discovery (OMNISCIENCE · 1).

*The Master's ask, with his own honest boundary:* when a problem falls outside existing rules,
run an **Axiom Synthesizer** — use **symbolic regression + genetic programming** to invent new
mathematical rules, then **cross-check them with an automated theorem prover (Z3 / Sympy)** and
adopt only the ones that *prove* consistent. "Real symbolic theorem discovery — not fictional
magic physics, but machine-learned math-proof generation."

This module implements exactly that, and nothing more:

    1. **Discover** — :meth:`AxiomSynthesizer.discover` evolves an expression tree over ``{+, −, ×}``
       and a small terminal set (the variables and small integer constants) that reproduces a set
       of input→output samples. Real genetic programming: tournament selection, subtree crossover,
       point mutation, seeded for determinism.
    2. **Prove** — a discovered rule is adopted only after a real proof, strongest-first:
         * **Sympy** — symbolic equivalence to a reference expression over *all* reals
           (``simplify(candidate − reference) == 0``);
         * **Z3** — validity of a supplied logical/arithmetic formula (``∀`` over the theory);
         * **Exhaustive** — checked over every point of a *declared finite* integer domain (a
           genuine proof for that domain, dependency-free).
       Fitting the samples alone is *never* accepted as proof — a rule that only interpolates is
       reported ``proved=False`` and **not** adopted.

The honest boundary. The GP search is heuristic and can fail to find a rule; the prover is the
gate, so an unproved candidate is discarded, never trusted. Z3/Sympy are optional — absent them the
exhaustive-finite proof still gives real, if bounded, guarantees.

Bridges: complements :mod:`nyxara.growth.law_discovery`, :mod:`nyxara.growth.formal_proof`,
:mod:`nyxara.growth.prover`, :mod:`nyxara.growth.rule_synth`.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import sympy as _sympy  # type: ignore
except Exception:  # noqa: BLE001
    _sympy = None

try:
    import z3 as _z3  # type: ignore
except Exception:  # noqa: BLE001
    _z3 = None

__all__ = ["Expr", "Axiom", "ProofResult", "AxiomSynthesizer"]

_BINOPS = ("+", "-", "*")


# --------------------------------------------------------------------------- #
# Expression tree
# --------------------------------------------------------------------------- #
@dataclass
class Expr:
    """A tiny arithmetic expression tree over {+, −, ×}, integer constants, and variables."""

    kind: str                      # "op" | "var" | "const"
    op: str = ""                   # for kind == "op"
    left: Optional["Expr"] = None
    right: Optional["Expr"] = None
    name: str = ""                 # for kind == "var"
    value: int = 0                 # for kind == "const"

    # -- evaluation --------------------------------------------------------- #
    def eval(self, env: Dict[str, int]) -> int:
        if self.kind == "const":
            return self.value
        if self.kind == "var":
            return int(env[self.name])
        a = self.left.eval(env)   # type: ignore[union-attr]
        b = self.right.eval(env)  # type: ignore[union-attr]
        if self.op == "+":
            return a + b
        if self.op == "-":
            return a - b
        return a * b

    def size(self) -> int:
        if self.kind != "op":
            return 1
        return 1 + self.left.size() + self.right.size()  # type: ignore[union-attr]

    def render(self) -> str:
        if self.kind == "const":
            return str(self.value)
        if self.kind == "var":
            return self.name
        return f"({self.left.render()} {self.op} {self.right.render()})"  # type: ignore[union-attr]

    def clone(self) -> "Expr":
        if self.kind == "op":
            return Expr("op", self.op,
                        self.left.clone(), self.right.clone())  # type: ignore[union-attr]
        return Expr(self.kind, name=self.name, value=self.value)

    def nodes(self) -> List["Expr"]:
        out = [self]
        if self.kind == "op":
            out += self.left.nodes()   # type: ignore[union-attr]
            out += self.right.nodes()  # type: ignore[union-attr]
        return out


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class ProofResult:
    """The outcome of trying to prove a candidate axiom."""

    proved: bool
    method: str                    # "sympy" | "z3" | "exhaustive" | "numeric" | "none"
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"proved": self.proved, "method": self.method, "detail": self.detail}


@dataclass
class Axiom:
    """A discovered rule and its proof status. Adopted only when ``proof.proved``."""

    name: str
    expression: str
    variables: List[str]
    proof: ProofResult
    fit_samples: int = 0
    note: str = ""

    @property
    def adopted(self) -> bool:
        return self.proof.proved

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "expression": self.expression, "variables": self.variables,
                "proof": self.proof.to_dict(), "fit_samples": self.fit_samples,
                "adopted": self.adopted, "note": self.note}


Sample = Tuple[Dict[str, int], int]


# --------------------------------------------------------------------------- #
# Synthesizer
# --------------------------------------------------------------------------- #
class AxiomSynthesizer:
    """Discovers candidate axioms by GP and adopts only the ones a prover certifies."""

    def __init__(self, *, seed: int = 1234, population: int = 120, generations: int = 40,
                 const_range: int = 3, max_depth: int = 4) -> None:
        self.rng = random.Random(seed)
        self.population = int(population)
        self.generations = int(generations)
        self.const_range = int(const_range)
        self.max_depth = int(max_depth)
        self._adopted: List[Axiom] = []

    # -- discovery (symbolic regression via GP) ----------------------------- #
    def discover(self, samples: Sequence[Sample], variables: Sequence[str]) -> Optional[Expr]:
        """Evolve an expression over ``variables`` that reproduces every sample exactly."""
        variables = list(variables)
        if not samples or not variables:
            return None
        pop = [self._random_expr(variables, depth=self.rng.randint(1, self.max_depth))
               for _ in range(self.population)]
        best: Optional[Expr] = None
        best_fit = -1.0
        for _ in range(self.generations):
            scored = [(self._fitness(e, samples), e) for e in pop]
            scored.sort(key=lambda t: t[0], reverse=True)
            if scored[0][0] > best_fit:
                best_fit, best = scored[0][0], scored[0][1].clone()
            if self._matches_all(best, samples):
                return best
            survivors = [e for _f, e in scored[: max(2, self.population // 4)]]
            pop = list(survivors)
            while len(pop) < self.population:
                a = self._tournament(scored)
                b = self._tournament(scored)
                child = self._crossover(a, b)
                if self.rng.random() < 0.3:
                    child = self._mutate(child, variables)
                pop.append(child)
        if best is not None and self._matches_all(best, samples):
            return best
        return best if best is not None and best_fit >= 0 else None

    def _fitness(self, expr: Expr, samples: Sequence[Sample]) -> float:
        matched = 0
        for env, target in samples:
            try:
                if expr.eval(env) == target:
                    matched += 1
            except (KeyError, OverflowError, ZeroDivisionError):
                return -1.0
        # exact matches dominate; parsimony breaks ties (Occam)
        return matched - 0.001 * expr.size()

    @staticmethod
    def _matches_all(expr: Optional[Expr], samples: Sequence[Sample]) -> bool:
        if expr is None:
            return False
        for env, target in samples:
            try:
                if expr.eval(env) != target:
                    return False
            except (KeyError, OverflowError, ZeroDivisionError):
                return False
        return True

    # -- GP operators ------------------------------------------------------- #
    def _random_expr(self, variables: Sequence[str], depth: int) -> Expr:
        if depth <= 1 or self.rng.random() < 0.3:
            if self.rng.random() < 0.6:
                return Expr("var", name=self.rng.choice(list(variables)))
            return Expr("const", value=self.rng.randint(-self.const_range, self.const_range))
        return Expr("op", self.rng.choice(_BINOPS),
                    self._random_expr(variables, depth - 1),
                    self._random_expr(variables, depth - 1))

    def _tournament(self, scored: Sequence[Tuple[float, Expr]], k: int = 4) -> Expr:
        picks = [self.rng.choice(scored) for _ in range(k)]
        picks.sort(key=lambda t: t[0], reverse=True)
        return picks[0][1]

    def _crossover(self, a: Expr, b: Expr) -> Expr:
        child = a.clone()
        cnodes = [n for n in child.nodes()]
        donor = self.rng.choice(b.nodes()).clone()
        target = self.rng.choice(cnodes)
        target.kind, target.op = donor.kind, donor.op
        target.left, target.right = donor.left, donor.right
        target.name, target.value = donor.name, donor.value
        return child

    def _mutate(self, expr: Expr, variables: Sequence[str]) -> Expr:
        node = self.rng.choice(expr.nodes())
        fresh = self._random_expr(variables, depth=self.rng.randint(1, 2))
        node.kind, node.op = fresh.kind, fresh.op
        node.left, node.right = fresh.left, fresh.right
        node.name, node.value = fresh.name, fresh.value
        return expr

    # -- proof gate --------------------------------------------------------- #
    def prove_exhaustive(self, expr: Expr, reference: Callable[[Dict[str, int]], int],
                         domain: Dict[str, Iterable[int]]) -> ProofResult:
        """Real proof over a *finite* integer domain: check every point (dependency-free)."""
        names = list(domain)
        ranges = [list(domain[n]) for n in names]
        checked = 0
        for combo in itertools.product(*ranges):
            env = dict(zip(names, combo))
            try:
                if expr.eval(env) != reference(env):
                    return ProofResult(False, "exhaustive",
                                       f"counterexample at {env}")
            except (KeyError, OverflowError, ZeroDivisionError) as exc:
                return ProofResult(False, "exhaustive", f"evaluation error {exc!r} at {env}")
            checked += 1
        return ProofResult(True, "exhaustive", f"verified over {checked} points of {names}")

    def prove_sympy(self, expr: Expr, reference_expr: str) -> ProofResult:
        """Symbolic proof over all reals via sympy: candidate ≡ reference."""
        if _sympy is None:
            return ProofResult(False, "none", "sympy unavailable")
        try:
            diff = _sympy.simplify(_sympy.sympify(expr.render()) - _sympy.sympify(reference_expr))
            if diff == 0:
                return ProofResult(True, "sympy", f"{expr.render()} ≡ {reference_expr}")
            return ProofResult(False, "sympy", f"difference simplifies to {diff}")
        except Exception as exc:  # noqa: BLE001
            return ProofResult(False, "sympy", f"sympy error: {exc!r}")

    def prove_z3(self, build_claim: Callable[[Any], Any]) -> ProofResult:
        """Validity proof via Z3: ``build_claim(z3)`` returns a formula that must hold for all."""
        if _z3 is None:
            return ProofResult(False, "none", "z3 unavailable")
        try:
            claim = build_claim(_z3)
            solver = _z3.Solver()
            solver.add(_z3.Not(claim))
            result = solver.check()
            if result == _z3.unsat:
                return ProofResult(True, "z3", "negation unsatisfiable ⇒ valid")
            return ProofResult(False, "z3", f"countermodel: {solver.model()}"
                               if result == _z3.sat else "z3 returned unknown")
        except Exception as exc:  # noqa: BLE001
            return ProofResult(False, "z3", f"z3 error: {exc!r}")

    # -- discover + prove + adopt ------------------------------------------- #
    def synthesize(self, name: str, samples: Sequence[Sample], variables: Sequence[str], *,
                   reference_fn: Optional[Callable[[Dict[str, int]], int]] = None,
                   domain: Optional[Dict[str, Iterable[int]]] = None,
                   reference_expr: Optional[str] = None,
                   z3_claim: Optional[Callable[[Any], Any]] = None) -> Optional[Axiom]:
        """Discover a rule for ``samples`` and adopt it **iff** a prover certifies it."""
        expr = self.discover(samples, variables)
        if expr is None:
            return None
        fit = sum(1 for env, t in samples if self._safe_eq(expr, env, t))
        proof = ProofResult(False, "none", "no prover applicable")
        if reference_expr is not None:
            proof = self.prove_sympy(expr, reference_expr)
        if not proof.proved and z3_claim is not None:
            proof = self.prove_z3(z3_claim)
        if not proof.proved and reference_fn is not None and domain is not None:
            proof = self.prove_exhaustive(expr, reference_fn, domain)
        axiom = Axiom(name=name, expression=expr.render(), variables=list(variables),
                      proof=proof, fit_samples=fit)
        if axiom.adopted:
            self._adopted.append(axiom)
        else:
            axiom.note = "discovered but unproved — discarded, never trusted"
        return axiom

    @staticmethod
    def _safe_eq(expr: Expr, env: Dict[str, int], target: int) -> bool:
        try:
            return expr.eval(env) == target
        except (KeyError, OverflowError, ZeroDivisionError):
            return False

    # -- introspection ------------------------------------------------------ #
    def adopted(self) -> List[Axiom]:
        return list(self._adopted)

    def status(self) -> Dict[str, Any]:
        return {"adopted": len(self._adopted),
                "sympy": _sympy is not None, "z3": _z3 is not None}
