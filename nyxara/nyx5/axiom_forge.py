"""NYXARA · nyx5/axiom_forge.py — the sub-axiomatic engine (⚛️, pillar 11).

Classical logic has a fatal brittleness: from a single contradiction it can prove *anything* (ex falso
quodlibet — the principle of explosion). A mind that must reason through genuinely contradictory evidence
cannot afford that. When a problem will not yield under classical assumptions, NYX-5 constructs an
**alternative axiom system** and reasons inside it — including **paraconsistent** logic (Belnap's
four-valued FDE), where a proposition may be TRUE, FALSE, BOTH (contradictory) or NEITHER (unknown), and
a local contradiction is contained instead of detonating the whole system.

Honest ceiling: this *explores alternative formal systems* — it does not rewrite mathematics or physics,
and any conclusion that is not derivable is honestly marked unproven. Reuses the spirit of
:mod:`nyxara.growth.law_discovery` / :mod:`nyxara.growth.formal_proof`; here it is a small, self-contained
paraconsistent evaluator. Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

__all__ = ["Truth", "AxiomSystem", "SubAxiomaticEngine"]


class Truth(str, Enum):
    """Belnap's four truth values — the lattice that contains contradiction instead of exploding."""

    TRUE = "true"
    FALSE = "false"
    BOTH = "both"          # contradictory evidence (a paradox, held not exploded)
    NEITHER = "neither"    # no evidence (unknown)


def _neg(v: Truth) -> Truth:
    return {Truth.TRUE: Truth.FALSE, Truth.FALSE: Truth.TRUE,
            Truth.BOTH: Truth.BOTH, Truth.NEITHER: Truth.NEITHER}[v]


# FDE conjunction/disjunction on the {T,F,B,N} bilattice (knowledge order joins/meets).
_T, _F, _B, _N = Truth.TRUE, Truth.FALSE, Truth.BOTH, Truth.NEITHER
_AND = {
    (_T, _T): _T, (_T, _F): _F, (_T, _B): _B, (_T, _N): _N,
    (_F, _T): _F, (_F, _F): _F, (_F, _B): _F, (_F, _N): _F,
    (_B, _T): _B, (_B, _F): _F, (_B, _B): _B, (_B, _N): _F,
    (_N, _T): _N, (_N, _F): _F, (_N, _B): _F, (_N, _N): _N,
}
_OR = {k: _neg(_AND[(_neg(a), _neg(b))]) for k, (a, b) in ((k, k) for k in _AND)}


@dataclass
class AxiomSystem:
    """A named alternative logic: a set of propositions each pinned to a four-valued truth."""

    name: str
    axioms: Dict[str, Truth] = field(default_factory=dict)

    def assert_(self, prop: str, value: Truth) -> None:
        """Assign/merge a truth value. Asserting the opposite of an existing value yields BOTH — the
        contradiction is *held*, not resolved by explosion."""
        cur = self.axioms.get(prop)
        if cur is None or cur == value:
            self.axioms[prop] = value
        elif {cur, value} == {Truth.TRUE, Truth.FALSE}:
            self.axioms[prop] = Truth.BOTH
        else:
            self.axioms[prop] = value

    def value(self, prop: str) -> Truth:
        return self.axioms.get(prop, Truth.NEITHER)

    def is_consistent(self) -> bool:
        return not any(v == Truth.BOTH for v in self.axioms.values())


class SubAxiomaticEngine:
    """Construct and evaluate alternative (paraconsistent) axiom systems, bounded by a cap."""

    def __init__(self, *, max_systems: int = 8) -> None:
        self.max_systems = int(max_systems)
        self._systems: Dict[str, AxiomSystem] = {}

    def forge(self, name: str) -> Optional[AxiomSystem]:
        """Create a fresh axiom system (returns None if the concurrent-systems cap is reached)."""
        if name in self._systems:
            return self._systems[name]
        if len(self._systems) >= self.max_systems:
            return None
        sys = AxiomSystem(name=name)
        self._systems[name] = sys
        return sys

    def evaluate(self, system: AxiomSystem, expr: Any) -> Truth:
        """Evaluate a nested expression: ('and', a, b), ('or', a, b), ('not', a), or a prop name."""
        if isinstance(expr, str):
            return system.value(expr)
        op = expr[0]
        if op == "not":
            return _neg(self.evaluate(system, expr[1]))
        a = self.evaluate(system, expr[1])
        b = self.evaluate(system, expr[2])
        return _AND[(a, b)] if op == "and" else _OR[(a, b)]

    def entails(self, system: AxiomSystem, expr: Any) -> bool:
        """A conclusion is *proven* only if it evaluates to a designated-true value (TRUE or BOTH)
        while not being merely NEITHER — otherwise it is honestly unproven."""
        return self.evaluate(system, expr) in (Truth.TRUE, Truth.BOTH)

    def to_dict(self) -> Dict[str, Any]:
        return {"max_systems": self.max_systems,
                "systems": {n: {p: v.value for p, v in s.axioms.items()}
                            for n, s in self._systems.items()}}
