"""NYXARA · njp/boundary.py — what cannot work, and exactly why (⛔, NJP V.46).

Every reasoning organ in this package answers *what is the case* or *what would follow*. None of
them answers the other half of the Master's point::

    Intelligence is not only "what solution works?"
    It is also "what solutions cannot possibly work?"

Those are not the same question asked from opposite ends. A search for what works explores; a
search for what cannot **eliminates**, and eliminating is what makes a large space tractable::

    100,000 possibilities
        ↓  constraint reasoning
      5,000
        ↓  causal reasoning
        200
        ↓  counterexample search
          7
        ↓  verification
          1

The engine that does the first arrow is this file, and the number it prints at each stage is the
whole product: a system that says *"no solution"* has told you nothing you can act on, and one
that says *"no solution, because D follows from A and B, and D and C were stated to exclude each
other"* has told you which of A, B or C to give up.

The three things it does
------------------------

**Closure.** From what is asserted and what implies what, derive everything that must also hold.
The derived conditions are :class:`Necessary` and each carries the chain that produced it — the
same provenance rule :mod:`nyxara.njp.provenance` applies to a claim, applied to a constraint.

**Contradiction, with a core.** When a derived condition collides with a stated one, the answer is
not ``False``. It is the **unsatisfiable core**: the smallest set of constraints that still
conflicts, found by removing them one at a time and seeing whether the conflict survives. A core
of three out of forty is a fixable problem; "unsatisfiable" is not.

**Pruning, with a funnel.** :meth:`Boundary.prune` eliminates candidates that violate the closure
and reports how many fell at each constraint. That report is the useful output: it names the
constraint doing the work, and a constraint that eliminates nothing is a constraint that was never
binding.

What makes a necessary condition a finding
-------------------------------------------

Only one thing: **nobody stated it.** :attr:`Necessary.derived` is False for anything that was
asserted outright, and :meth:`Boundary.necessary` marks them so, because a closure that proudly
returns its own inputs looks like it did work and did none. The interesting output is the
condition that had to be worked out — and it is exactly that condition which usually turns out to
be the one that cannot hold.

What it may not do
------------------

**It may not conclude from silence.** An unstated literal is *unknown*, not false. This is a
monotone closure, not a database: ``not stated`` and ``stated false`` are different, and treating
the first as the second is how a planner starts proving things impossible because nobody mentioned
them.

**It may not report a contradiction without a core.** Every :class:`Impossible` names the
constraints responsible and the derivation that reached them.

**It may not silently drop a constraint it could not read.** Anything unparsed is returned in
:attr:`Boundary.ignored`.

Pure standard library, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Constraint", "Necessary", "Impossible", "Funnel", "Boundary",
    "MAX_CORE", "MAX_DEPTH",
]

#: A core larger than this is reported as "at most these" rather than minimised further. Finding a
#: truly minimal core is exponential; a small one is what a person can act on.
MAX_CORE = 8

#: How far the closure chains implications. Deep enough to be useful, bounded so a cyclic rule set
#: terminates.
MAX_DEPTH = 12


@dataclass(frozen=True)
class Constraint:
    """One thing that must hold. Three shapes, and they are all the file needs.

    ``requires``   ``a`` being true forces ``b``      (``a → b``)
    ``excludes``   ``a`` and ``b`` cannot both hold
    ``asserts``    ``a`` holds
    """

    kind: str
    left: str
    right: str = ""
    label: str = ""

    def render(self) -> str:
        name = f"[{self.label}] " if self.label else ""
        if self.kind == "asserts":
            return f"{name}{self.left}"
        if self.kind == "requires":
            return f"{name}{self.left} → {self.right}"
        return f"{name}{self.left} ⊗ {self.right}"

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "left": self.left, "right": self.right,
                "label": self.label, "render": self.render()}


@dataclass
class Necessary:
    """A condition that must hold, with the chain that produced it.

    ``derived`` is the only thing that makes one interesting. A closure that returns its own
    inputs looks like it did work and did none.
    """

    condition: str
    derived: bool = True
    because: Tuple[Constraint, ...] = ()

    def render(self) -> str:
        if not self.derived:
            return f"{self.condition}  (stated)"
        chain = " ; ".join(c.render() for c in self.because)
        return f"{self.condition}  (derived: {chain})"

    def to_dict(self) -> Dict[str, Any]:
        return {"condition": self.condition, "derived": self.derived,
                "because": [c.to_dict() for c in self.because]}


@dataclass
class Impossible:
    """No solution — and which constraints are responsible."""

    conflict: Tuple[str, str]
    core: List[Constraint] = field(default_factory=list)
    derivation: List[Necessary] = field(default_factory=list)

    def __bool__(self) -> bool:
        return True

    def render(self) -> str:
        lines = [f"NO SOLUTION: {self.conflict[0]} and {self.conflict[1]} cannot both hold",
                 f"  because of {len(self.core)} constraint(s):"]
        lines += [f"    {c.render()}" for c in self.core]
        if self.derivation:
            lines.append("  reached by:")
            lines += [f"    {n.render()}" for n in self.derivation]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"conflict": list(self.conflict),
                "core": [c.to_dict() for c in self.core],
                "derivation": [n.to_dict() for n in self.derivation]}


@dataclass
class Funnel:
    """How a space of candidates narrowed, and which constraint did the narrowing."""

    start: int = 0
    stages: List[Tuple[str, int]] = field(default_factory=list)
    survivors: List[Any] = field(default_factory=list)

    @property
    def end(self) -> int:
        return len(self.survivors)

    @property
    def idle(self) -> List[str]:
        """Constraints that eliminated nothing. A constraint that never binds is not one."""
        return [name for name, removed in self.stages if removed == 0]

    def render(self) -> str:
        lines = [f"{self.start} candidates"]
        left = self.start
        for name, removed in self.stages:
            left -= removed
            lines.append(f"  ↓ {name}  (−{removed})")
            lines.append(f"{left}")
        if self.idle:
            lines.append(f"never binding: {', '.join(self.idle)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"start": self.start, "end": self.end, "idle": self.idle,
                "stages": [list(s) for s in self.stages]}


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
class Boundary:
    """Derives what must hold, finds what cannot, and narrows what is left."""

    def __init__(self, constraints: Sequence[Constraint] = ()) -> None:
        self.constraints: List[Constraint] = []
        self.ignored: List[Any] = []
        for constraint in constraints:
            self.add(constraint)

    def add(self, constraint: Any) -> "Boundary":
        if isinstance(constraint, Constraint) and constraint.kind in (
                "asserts", "requires", "excludes"):
            self.constraints.append(constraint)
        else:
            self.ignored.append(constraint)
        return self

    # ---- closure ---------------------------------------------------------- #
    def necessary(self) -> List[Necessary]:
        """Everything that must hold, stated and derived, with the chain for each.

        Monotone: a condition is added when something forces it and never removed. Nothing is
        concluded from an absence — an unstated literal is *unknown*, and treating it as false is
        how a planner starts proving things impossible because nobody mentioned them.
        """
        stated = {c.left: c for c in self.constraints if c.kind == "asserts"}
        rules = [c for c in self.constraints if c.kind == "requires"]
        out: Dict[str, Necessary] = {
            name: Necessary(condition=name, derived=False, because=(rule,))
            for name, rule in stated.items()}
        for _ in range(MAX_DEPTH):
            grew = False
            for rule in rules:
                if rule.left in out and rule.right not in out:
                    out[rule.right] = Necessary(
                        condition=rule.right, derived=True,
                        because=out[rule.left].because + (rule,))
                    grew = True
            if not grew:
                break
        return sorted(out.values(), key=lambda n: (not n.derived, n.condition))

    # ---- impossibility ----------------------------------------------------- #
    def impossible(self) -> Optional[Impossible]:
        """Is this set unsatisfiable, and which constraints are responsible?

        The core is minimised by removing constraints one at a time and asking whether the conflict
        survives without them. It is the difference between an answer and a diagnosis: a core of
        three out of forty is a fixable problem; "unsatisfiable" is not.
        """
        conflict = self._conflict(self.constraints)
        if conflict is None:
            return None
        core = list(self.constraints)
        for candidate in list(core):
            trimmed = [c for c in core if c is not candidate]
            if self._conflict(trimmed) is not None:
                core = trimmed
            if len(core) <= 2:
                break
        derivation = [n for n in Boundary(core).necessary() if n.derived]
        return Impossible(conflict=conflict, core=core[:MAX_CORE], derivation=derivation)

    def _conflict(self, constraints: Sequence[Constraint]) -> Optional[Tuple[str, str]]:
        holds = {n.condition for n in Boundary(constraints).necessary()} \
            if constraints else set()
        for constraint in constraints:
            if constraint.kind != "excludes":
                continue
            if constraint.left in holds and constraint.right in holds:
                return (constraint.left, constraint.right)
        return None

    # ---- pruning ------------------------------------------------------------ #
    def prune(self, candidates: Sequence[Any],
              tests: Sequence[Tuple[str, Callable[[Any], bool]]]) -> Funnel:
        """Narrow a space, stage by stage, and say which stage did the work.

        The report is the product. A funnel that names the constraint eliminating 95% of the space
        tells you where the problem actually lives; one that says only *"1 survivor"* tells you
        nothing about why, and cannot tell you that thirty of your forty constraints never bound at
        all.
        """
        out = Funnel(start=len(candidates), survivors=list(candidates))
        for name, test in tests:
            before = len(out.survivors)
            out.survivors = [c for c in out.survivors if test(c)]
            out.stages.append((name, before - len(out.survivors)))
        return out

    # ---- reading ------------------------------------------------------------ #
    def render(self) -> str:
        blocked = self.impossible()
        if blocked is not None:
            return blocked.render()
        lines = ["SATISFIABLE so far. What must hold:"]
        for got in self.necessary():
            lines.append(f"  {got.render()}")
        if self.ignored:
            lines.append(f"  (ignored {len(self.ignored)} unreadable constraint(s))")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        blocked = self.impossible()
        return {"constraints": [c.to_dict() for c in self.constraints],
                "necessary": [n.to_dict() for n in self.necessary()],
                "impossible": blocked.to_dict() if blocked else None,
                "ignored": len(self.ignored)}
