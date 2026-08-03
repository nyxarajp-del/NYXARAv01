"""NYXARA · growth/meta_epistemology.py — Autonomous Synthetic Mathematics (🜁∞, Rule 4).

The critique this module answers, in the Master's words: *"AI available data, rules aur patterns ke
andar hi adapt karti hai. Naya: ek meta-epistemology core — agar koi problem aaj ke mathematics ke daayre
mein solve nahi ho sakti, toh woh khud naye mathematical axioms invent kar sake — naya knowledge generate
kare jo pehle exist nahi karta tha."*

The rest of the growth stack discovers *within* a fixed system: :mod:`growth.law_discovery` fits laws
in a fixed operator library, :mod:`growth.eureka` invents+proves conjectures over a fixed algebra,
:mod:`growth.noesis` composes new primitives from existing ones. **None of them can extend the axiom
system itself.** This module can: when a goal is **unprovable in her current axioms**, she **invents a
new axiom**, and admits it only if it is:

* **consistent** — it must not let her prove a designated falsehood (two distinct constants equal),
* **non-trivial** — the goal must have been *unprovable before* and provable *after* (it genuinely adds
  power, it does not merely restate a theorem she already had), and
* **generative** — it must also make an independent **held-out** goal provable (it is a real law, not an
  overfit patch for one problem).

The engine is a genuine equational-logic decision procedure — **congruence closure** over ground terms
(union-find + the congruence rule ``x≈x' ∧ y≈y' ⇒ f(x,y)≈f(x',y')``) — so "provable" means *actually
derivable*, never asserted. New axioms are drawn from a template bank (commutativity, associativity,
identity, idempotence, distributivity) instantiated over the constants in play, and the admitted ones
**persist** so her mathematics compounds. This is honest and bounded: real automated theory extension
over a small algebra — **not** a claim to have solved open physics. **No LLM anywhere.**
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

__all__ = ["Term", "Axiom", "TheoremCertificate", "MetaEpistemologyCore"]

# A term is either an atom (str) or a compound (op, left, right) with op a str and left/right Terms.
Term = Any
Equation = Tuple[Term, Term]


def atom(name: str) -> str:
    return name


def op(name: str, a: Term, b: Term) -> Tuple[str, Term, Term]:
    return (name, a, b)


def _is_compound(t: Term) -> bool:
    return isinstance(t, tuple) and len(t) == 3


def _subterms(t: Term, into: set) -> None:
    into.add(t)
    if _is_compound(t):
        _subterms(t[1], into)
        _subterms(t[2], into)


# --------------------------------------------------------------------------- #
# Congruence closure — decide ground equality under a set of equations
# --------------------------------------------------------------------------- #
class _CongruenceClosure:
    """Union-find + the congruence rule: decides ``a ≈ b`` from ground equations, soundly."""

    def __init__(self, terms: Sequence[Term]) -> None:
        self.parent: Dict[Term, Term] = {}
        for t in terms:
            self._ensure(t)

    def _ensure(self, t: Term) -> None:
        if t in self.parent:
            return
        self.parent[t] = t
        if _is_compound(t):
            self._ensure(t[1])
            self._ensure(t[2])

    def find(self, t: Term) -> Term:
        self._ensure(t)
        root = t
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[t] != root:
            self.parent[t], t = root, self.parent[t]
        return root

    def union(self, a: Term, b: Term) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def close(self) -> None:
        """Merge classes to a fixpoint under the congruence rule."""
        changed = True
        while changed:
            changed = False
            compounds = [t for t in list(self.parent) if _is_compound(t)]
            for i in range(len(compounds)):
                for j in range(i + 1, len(compounds)):
                    a, b = compounds[i], compounds[j]
                    if self.find(a) == self.find(b):
                        continue
                    if a[0] == b[0] and self.find(a[1]) == self.find(b[1]) \
                            and self.find(a[2]) == self.find(b[2]):
                        self.union(a, b)
                        changed = True

    def equal(self, a: Term, b: Term) -> bool:
        return self.find(a) == self.find(b)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class Axiom:
    """A universally-quantified equational axiom (a named template over variables)."""

    name: str
    lhs: Term
    rhs: Term
    variables: FrozenSet[str]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "lhs": _render(self.lhs), "rhs": _render(self.rhs)}


@dataclass
class TheoremCertificate:
    """The auditable record of an admitted (or rejected) new axiom."""

    goal: str
    admitted: bool
    axiom: Optional[Dict[str, Any]] = None
    consistent: bool = False
    non_trivial: bool = False
    generative: bool = False
    heldout_goal: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"goal": self.goal, "admitted": self.admitted, "axiom": self.axiom,
                "consistent": self.consistent, "non_trivial": self.non_trivial,
                "generative": self.generative, "heldout_goal": self.heldout_goal,
                "reason": self.reason}


# --------------------------------------------------------------------------- #
# Axiom templates (the raw material of new mathematics)
# --------------------------------------------------------------------------- #
def _templates(ops: Sequence[str]) -> List[Axiom]:
    x, y, z = "?x", "?y", "?z"
    out: List[Axiom] = []
    for o in ops:
        out.append(Axiom(f"comm[{o}]", op(o, x, y), op(o, y, x), frozenset({x, y})))
        out.append(Axiom(f"assoc[{o}]", op(o, op(o, x, y), z), op(o, x, op(o, y, z)),
                         frozenset({x, y, z})))
        out.append(Axiom(f"idem[{o}]", op(o, x, x), x, frozenset({x})))
    if len(ops) >= 2:
        a, m = ops[0], ops[1]
        out.append(Axiom(f"distrib[{m},{a}]", op(m, x, op(a, y, z)),
                         op(a, op(m, x, y), op(m, x, z)), frozenset({x, y, z})))
    return out


def _ground_instances(ax: Axiom, constants: Sequence[str]) -> List[Equation]:
    """Instantiate an axiom template over every assignment of its variables to constants."""
    vars_ = sorted(ax.variables)
    out: List[Equation] = []
    for combo in itertools.product(constants, repeat=len(vars_)):
        sub = dict(zip(vars_, combo))
        out.append((_subst(ax.lhs, sub), _subst(ax.rhs, sub)))
    return out


def _subst(t: Term, sub: Dict[str, str]) -> Term:
    if _is_compound(t):
        return (t[0], _subst(t[1], sub), _subst(t[2], sub))
    return sub.get(t, t)


def _render(t: Term) -> str:
    if _is_compound(t):
        return f"({_render(t[1])}{t[0]}{_render(t[2])})"
    return str(t)


# --------------------------------------------------------------------------- #
# The core
# --------------------------------------------------------------------------- #
class MetaEpistemologyCore:
    """Extend her own axiom system when a goal is unprovable — consistently, non-trivially, generatively."""

    def __init__(self, *, ops: Sequence[str] = ("+", "*"), constants: Sequence[str] = ("a", "b", "c"),
                 base_axioms: Optional[Sequence[Axiom]] = None,
                 distinct: Optional[Tuple[str, str]] = None, persist_path: Optional[Path] = None) -> None:
        self.ops = tuple(ops)
        self.constants = tuple(constants)
        self.axioms: List[Axiom] = list(base_axioms) if base_axioms is not None else []
        # a designated falsehood: these two constants must never become provably equal
        self.distinct = distinct or (constants[0], constants[1])
        self.persist_path = Path(persist_path) if persist_path else None
        self.admitted: List[Axiom] = []

    # ---- the decision procedure ----------------------------------------- #
    def _all_equations(self, extra: Sequence[Axiom] = ()) -> List[Equation]:
        eqs: List[Equation] = []
        for ax in list(self.axioms) + list(extra):
            eqs.extend(_ground_instances(ax, self.constants))
        return eqs

    def provable(self, lhs: Term, rhs: Term, *, extra: Sequence[Axiom] = ()) -> bool:
        """Is ``lhs ≈ rhs`` derivable from the current axioms (+ any ``extra``)? Real congruence closure."""
        eqs = self._all_equations(extra)
        terms: set = set()
        _subterms(lhs, terms); _subterms(rhs, terms)
        for a, b in eqs:
            _subterms(a, terms); _subterms(b, terms)
        cc = _CongruenceClosure(list(terms))
        for a, b in eqs:
            cc.union(a, b)
        cc.close()
        return cc.equal(lhs, rhs)

    def _consistent(self, extra: Sequence[Axiom] = ()) -> bool:
        """The system must not prove the designated falsehood (two distinct constants equal)."""
        d0, d1 = self.distinct
        return not self.provable(d0, d1, extra=extra)

    # ---- the meta-move: invent an axiom when stuck ---------------------- #
    def prove_or_invent(self, lhs: Term, rhs: Term, *,
                        heldout: Optional[Equation] = None) -> TheoremCertificate:
        """Try to prove ``lhs ≈ rhs``. If she cannot, invent the axiom that makes it provable —
        admitted only if it is consistent, non-trivial (was unprovable before), and generative
        (also proves the held-out goal). Returns an auditable certificate."""
        goal = f"{_render(lhs)} = {_render(rhs)}"
        if self.provable(lhs, rhs):
            return TheoremCertificate(goal, admitted=False, non_trivial=False,
                                      reason="already provable in the current axioms — nothing to invent")
        # she is genuinely stuck: search the template bank for an axiom that unlocks the goal.
        heldout = heldout or self._auto_heldout(lhs, rhs)
        for cand in _templates(self.ops):
            if any(c.name == cand.name for c in self.axioms):
                continue
            extra = (cand,)
            if not self.provable(lhs, rhs, extra=extra):
                continue                                  # this axiom does not unlock the goal
            consistent = self._consistent(extra=extra)
            generative = bool(heldout) and self.provable(heldout[0], heldout[1], extra=extra)
            if consistent and generative:
                self.axioms.append(cand)
                self.admitted.append(cand)
                self._persist()
                return TheoremCertificate(
                    goal, admitted=True, axiom=cand.to_dict(), consistent=True,
                    non_trivial=True, generative=True,
                    heldout_goal=(f"{_render(heldout[0])} = {_render(heldout[1])}" if heldout else None),
                    reason=f"invented and admitted new axiom '{cand.name}' — consistent + generative")
            # a candidate that unlocks the goal but fails a gate is rejected (kept honest)
            last = TheoremCertificate(goal, admitted=False, axiom=cand.to_dict(),
                                      consistent=consistent, non_trivial=True, generative=generative,
                                      reason=("would prove a falsehood (inconsistent)" if not consistent
                                              else "unlocks the goal but is not generative (overfit)"))
            if not consistent:
                return last
        return TheoremCertificate(goal, admitted=False,
                                  reason="no consistent, generative axiom in the template bank unlocks it")

    def _auto_heldout(self, lhs: Term, rhs: Term) -> Optional[Equation]:
        """Manufacture an independent held-out goal of the same *shape* over different constants, so a
        candidate axiom must generalise rather than patch the single problem."""
        if len(self.constants) < 3 or not _is_compound(lhs) or not _is_compound(rhs):
            return None
        # remap the two leading constants to a fresh pair (rotate through the constant set)
        remap = {self.constants[0]: self.constants[1], self.constants[1]: self.constants[2]}
        return (_subst(lhs, remap), _subst(rhs, remap))

    # ---- persistence ---------------------------------------------------- #
    def _persist(self) -> None:
        if self.persist_path is None:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(json.dumps(
                [a.to_dict() for a in self.admitted], indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001 — persistence is best-effort
            pass

    def report(self) -> Dict[str, Any]:
        return {"ops": list(self.ops), "constants": list(self.constants),
                "n_axioms": len(self.axioms), "n_admitted": len(self.admitted),
                "admitted": [a.name for a in self.admitted]}


# --------------------------------------------------------------------------- #
# Self-test / demo — invent a real axiom to solve an unsolvable problem (no LLM)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA meta-epistemology self-test — invent a new axiom when stuck")
    print("=" * 70)

    # Start with an algebra that knows ONLY associativity of '+': commutativity is NOT derivable.
    x, y, z = "?x", "?y", "?z"
    assoc_plus = Axiom("assoc[+]", op("+", op("+", x, y), z), op("+", x, op("+", y, z)),
                       frozenset({x, y, z}))
    core = MetaEpistemologyCore(ops=("+", "*"), constants=("a", "b", "c"), base_axioms=[assoc_plus])

    # 1) the goal a+b = b+a is UNPROVABLE from associativity alone --------------------------------
    goal = (op("+", "a", "b"), op("+", "b", "a"))
    assert not core.provable(*goal), "commutativity must not follow from associativity"
    print("stuck               : a+b = b+a is NOT provable from associativity alone ✓")

    # 2) she invents the missing axiom (commutativity), verified consistent + generative ----------
    cert = core.prove_or_invent(*goal)
    print("\n" + json.dumps(cert.to_dict(), indent=2))
    assert cert.admitted and cert.axiom["name"] == "comm[+]", cert.to_dict()
    assert cert.consistent and cert.non_trivial and cert.generative
    print("\ninvented axiom      : commutativity — consistent + generative, ADMITTED ✓")

    # 3) with the new axiom the goal (and a held-out sibling) is now provable ---------------------
    assert core.provable(*goal), "the goal must be provable after admitting the axiom"
    assert core.provable(op("+", "b", "c"), op("+", "c", "b")), "the held-out sibling too"
    print("compounding         : the new mathematics proves the goal AND its held-out sibling ✓")

    # 4) inventing again for an already-provable goal is an honest no-op --------------------------
    cert2 = core.prove_or_invent(*goal)
    assert not cert2.admitted and "already provable" in cert2.reason
    print("no-op               : nothing invented for an already-provable goal ✓")

    # 5) consistency is enforced: a bogus 'axiom' that would equate distinct constants is refused.
    #    (idempotence over '+' with a=b would not fire here; we assert the guard directly.)
    bad = Axiom("bogus", "a", "b", frozenset())     # a = b — the designated falsehood
    assert not core._consistent(extra=(bad,)), "an inconsistent axiom must be detectable"
    print("consistency guard   : an axiom equating distinct constants is rejected ✓")

    print(f"\nreport              : {core.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
