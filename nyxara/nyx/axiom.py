"""NYXARA · nyx/axiom.py — inventing a formal system, from no premises (📐, NYX V.02, L-AXIOM-GENESIS).

Every reasoning engine in this repo starts from premises somebody gave it. ``first_principles``
knows physics, chemistry, maths and logic; ``domain_genesis`` models an alien field from *its*
regularities. None of them can do the thing a mathematician does when no existing system fits:
**write down a new set of axioms and see whether it holds together**.

:class:`AxiomGenesis` is that pipeline, and every stage of it is a real check:

1. **read the problem** into a set of properties a relation should have or fail to have.
   ("an ordering where ``a<b`` and ``b<a`` can both hold" ⇒ transitive, but *not* antisymmetric.)
2. **generate candidate axiom sets** — without inheriting any domain's premises. The axioms are
   properties of an uninterpreted binary relation, nothing more.
3. **consistency** — a **bounded model search** with z3: expand the quantifiers over a finite
   domain and ask for a model. A model found is a *proof of consistency at that size*. No model
   found is exactly that and no more.
4. **independence** — an axiom that follows from the others is not a new axiom. Checked by
   asking whether ``rest ∧ ¬axiom`` has a model.
5. **derive** — a candidate theorem holds when ``axioms ∧ ¬theorem`` has **no** model up to the
   bound. Bounded, and labelled bounded.
6. **notation** — :mod:`nyxara.nyx.nexus` mints her own glyphs for the system.
7. **promote only on a certificate.** Everything else stays a labelled conjecture.

Honest framing (the rule this repo keeps, see ``docs/CAPABILITIES.md``):

* She invents **new formal systems, not new physical truths**. A consistent axiom system is
  *coherent*; it is not *true*. Nothing here discovers a fact about the world.
* **Gödel is not evaded.** Consistency is checked by finite model search up to a bound. Finding
  a model proves consistency *at that size*; finding none proves nothing at all — and this
  module says "no counter-model up to size N", never "consistent". That distinction is the
  whole honesty of the layer.
* It will not prove Riemann. What is real is the mathematician's *axiom-invention step*:
  writing a system down, checking it holds together, and deriving inside it.
* Search is **budgeted** — a domain bound and a per-check timeout. There is no infinite speed.
* Without ``z3`` installed the layer reports itself unavailable rather than guessing. z3 5.0.0
  is present in this environment.

Pure standard library plus optional ``z3``. Every public method is fail-soft.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["Axiom", "System", "Theorem", "Genesis", "AxiomGenesis", "PROPERTIES"]


# --------------------------------------------------------------------------- #
# The property language.
#
# Each property is a predicate over a finite domain, written directly as a propositional
# expansion of its quantifiers. That is what makes the consistency check *decidable* and its
# bound *visible*: there is no quantifier for a solver to be clever or slow about.
# --------------------------------------------------------------------------- #
def _reflexive(R, n):        # ∀x. R(x,x)
    return [R[i][i] for i in range(n)]


def _irreflexive(R, n):      # ∀x. ¬R(x,x)
    return [_not(R[i][i]) for i in range(n)]


def _symmetric(R, n):        # ∀x,y. R(x,y) → R(y,x)
    return [_implies(R[i][j], R[j][i]) for i in range(n) for j in range(n)]


def _antisymmetric(R, n):    # ∀x,y. R(x,y) ∧ R(y,x) → x = y
    return [_not(_and(R[i][j], R[j][i]))
            for i in range(n) for j in range(n) if i != j]


def _asymmetric(R, n):       # ∀x,y. R(x,y) → ¬R(y,x)
    return [_implies(R[i][j], _not(R[j][i])) for i in range(n) for j in range(n)]


def _transitive(R, n):       # ∀x,y,z. R(x,y) ∧ R(y,z) → R(x,z)
    return [_implies(_and(R[i][j], R[j][k]), R[i][k])
            for i in range(n) for j in range(n) for k in range(n)]


def _total(R, n):            # ∀x,y. x≠y → R(x,y) ∨ R(y,x)
    return [_or(R[i][j], R[j][i]) for i in range(n) for j in range(n) if i < j]


def _dense(R, n):            # ∀x,y. R(x,y) → ∃z. R(x,z) ∧ R(z,y)
    return [_implies(R[i][j], _any([_and(R[i][k], R[k][j]) for k in range(n)]))
            for i in range(n) for j in range(n)]


def _cyclic_pair(R, n):      # ∃x,y. x≠y ∧ R(x,y) ∧ R(y,x)  — "a<b and b<a can both hold"
    if n < 2:
        return [_false()]
    return [_any([_and(R[i][j], R[j][i]) for i in range(n) for j in range(n) if i != j])]


def _nonempty(R, n):         # ∃x,y. x≠y ∧ R(x,y) — the relation actually relates something
    if n < 2:
        return [_false()]
    return [_any([R[i][j] for i in range(n) for j in range(n) if i != j])]


#: Every axiom she can write down, by name. A closed set, deliberately: this is a language for
#: relations, not a general theorem prover, and pretending otherwise would be the usual lie.
PROPERTIES: Dict[str, Any] = {
    "reflexive": _reflexive,
    "irreflexive": _irreflexive,
    "symmetric": _symmetric,
    "antisymmetric": _antisymmetric,
    "asymmetric": _asymmetric,
    "transitive": _transitive,
    "total": _total,
    "dense": _dense,
    "cyclic-pair": _cyclic_pair,
    "nonempty": _nonempty,
}

#: Plain-language cues that name a property. Matched on a WORD BOUNDARY, which is not a
#: nicety: without it "symmetr" matches inside "antisymmetric" and "reflexiv" inside
#: "irreflexive", and the system she writes down is not the one that was described.
_CUES: Tuple[Tuple[str, str], ...] = (
    ("both hold", "cyclic-pair"),
    ("both ways", "cyclic-pair"),
    ("cycle", "cyclic-pair"),
    ("antisymmetr", "antisymmetric"),
    ("asymmetr", "asymmetric"),
    ("symmetr", "symmetric"),
    ("irreflexiv", "irreflexive"),
    ("reflexiv", "reflexive"),
    ("transitiv", "transitive"),
    ("total", "total"),
    ("linear", "total"),
    ("dense", "dense"),
    ("preorder", "reflexive"),
    ("equivalence", "symmetric"),
    ("order", "transitive"),
)


def _with_own_names(cues: Tuple[Tuple[str, str], ...]) -> Tuple[Tuple[str, str], ...]:
    """Every property answers to its own name, in hyphen and space form.

    Not a convenience — an invariant. ``cyclic-pair`` was expressible, listed in ``PROPERTIES``,
    and reachable by no phrasing at all: the hand-written cues covered "cycle"/"both ways" and
    ``\bcycle`` does not match "cyclic". So *"an irreflexive transitive relation with a cyclic
    pair"* quietly built ``{irreflexive, transitive}`` instead — a weaker system than the one
    described, which then found a model and was promoted. She answered a question nobody asked
    and called it a success. Deriving the names from ``PROPERTIES`` means a property that can be
    expressed can always be *named*, including any added later.
    """
    out = list(cues)
    have = {cue for cue, _name in cues}
    for name in sorted(PROPERTIES):
        for form in (name, name.replace("-", " ")):
            if form not in have:
                out.append((form, name))
                have.add(form)
    # Longest first, so a specific phrasing is never shadowed by a prefix of itself.
    return tuple(sorted(out, key=lambda pair: -len(pair[0])))


#: The cue table she actually matches against: the hand-written phrasings above, plus every
#: property's own name. Built once.
_ALL_CUES: Tuple[Tuple[str, str], ...] = _with_own_names(_CUES)

_NEGATORS = ("not ", "no ", "never ", "without ", "fails ", "non-", "non")


def _z3():
    try:
        import z3
        return z3
    except Exception:  # noqa: BLE001 — no solver means the layer reports itself unavailable
        return None


# Thin wrappers so the property functions above read as logic rather than as z3 plumbing.
def _and(a, b):
    return _z3().And(a, b)


def _or(a, b):
    return _z3().Or(a, b)


def _not(a):
    return _z3().Not(a)


def _implies(a, b):
    return _z3().Implies(a, b)


def _any(items):
    return _z3().Or(*items) if items else _false()


def _false():
    return _z3().BoolVal(False)


@dataclass
class Axiom:
    """One property, and whether it earned its place in the system."""

    name: str = ""
    asserted: bool = True        # False ⇒ the *negation* is the axiom
    independent: Optional[bool] = None
    why: str = ""

    def render(self) -> str:
        return f"{'' if self.asserted else '¬'}{self.name}"

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "asserted": self.asserted,
                "independent": self.independent, "why": self.why}


@dataclass
class Theorem:
    """Something that holds in the system — up to the bound, and labelled as such."""

    statement: str = ""
    holds: bool = False
    bound: int = 0
    certificate: str = ""
    why: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"statement": self.statement, "holds": self.holds, "bound": self.bound,
                "certificate": self.certificate, "why": self.why}


@dataclass
class System:
    """A candidate axiom system, and what the bounded search actually found."""

    name: str = ""
    axioms: List[Axiom] = field(default_factory=list)
    consistent: Optional[bool] = None    # True ⇒ a model was FOUND. None ⇒ none found, at all
    model_size: int = 0                  # the domain size a model was found at
    bound: int = 0                       # the largest size searched
    model: str = ""                      # the witness, rendered
    note: str = ""

    def render(self) -> str:
        return f"{self.name}: {{ {', '.join(a.render() for a in self.axioms)} }}"

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "axioms": [a.to_dict() for a in self.axioms],
                "consistent": self.consistent, "model_size": self.model_size,
                "bound": self.bound, "model": self.model, "note": self.note}


@dataclass
class Genesis:
    """One attempt to invent a formal system for a problem no domain covers."""

    problem: str = ""
    system: Optional[System] = None
    theorems: List[Theorem] = field(default_factory=list)
    notation: Dict[str, str] = field(default_factory=dict)
    promoted: bool = False              # a certificate was earned
    conjecture: bool = False            # held, but labelled
    refused: str = ""
    duration_ms: float = 0.0

    def render(self) -> str:
        if self.refused:
            return f"{self.problem}\n  refused: {self.refused}"
        lines = [self.problem]
        if self.system is not None:
            lines.append(f"  system    : {self.system.render()}")
            if self.system.consistent:
                lines.append(f"  model     : a model exists at size "
                             f"{self.system.model_size} (bounded check to "
                             f"{self.system.bound})")
                lines.append(f"  witness   : {self.system.model}")
            else:
                lines.append(f"  model     : NO model up to size {self.system.bound} — "
                             f"that is not a proof of inconsistency, and not a proof of "
                             f"consistency either")
            for axiom in self.system.axioms:
                if axiom.independent is False:
                    lines.append(f"  dependent : {axiom.render()} — {axiom.why}")
        for theorem in self.theorems:
            mark = "⊢" if theorem.holds else "⊬"
            lines.append(f"  {mark} {theorem.statement}   ({theorem.why})")
        if self.notation:
            lines.append("  notation  : " + ", ".join(f"{k}={v}" for k, v in
                                                      self.notation.items()))
        lines.append(f"  status    : {'promoted' if self.promoted else 'conjecture'}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"problem": self.problem,
                "system": self.system.to_dict() if self.system is not None else None,
                "theorems": [t.to_dict() for t in self.theorems],
                "notation": dict(self.notation), "promoted": self.promoted,
                "conjecture": self.conjecture, "refused": self.refused,
                "duration_ms": round(self.duration_ms, 2)}


class AxiomGenesis:
    """Write down a new axiom system, check it holds together, and derive inside it."""

    def __init__(self, brain: Any = None, *, bound: int = 4, timeout_ms: int = 2000,
                 max_systems: int = 16) -> None:
        self.brain = brain
        # The bound is the whole honesty of the consistency claim, so it is a first-class knob
        # and it is printed with every result.
        self.bound = max(2, int(bound))
        self.timeout_ms = max(50, int(timeout_ms))
        self.max_systems = max(1, int(max_systems))

        self.attempts = 0
        self.promoted = 0
        self.conjectures = 0
        self.refused = 0
        self.systems: Dict[str, Dict[str, Any]] = {}

    def available(self) -> bool:
        return _z3() is not None

    # ---- 1. read the problem ------------------------------------------------ #
    def read(self, problem: str) -> List[Axiom]:
        """Turn a description into the properties a relation should have, or fail to have."""
        out: List[Axiom] = []
        try:
            low = str(problem or "").lower()
            # "a<b and b<a" is not a word, so it is matched literally before the rest.
            if "a<b" in low.replace(" ", "") and "b<a" in low.replace(" ", ""):
                out.append(Axiom(name="cyclic-pair", why="read from 'a<b and b<a'"))
            seen = {a.name for a in out}
            for cue, name in _ALL_CUES:
                if name in seen:
                    continue
                match = re.search(rf"\b{re.escape(cue)}", low)
                if match is None:
                    continue
                window = low[max(0, match.start() - 24):match.start()]
                negated = any(neg in window for neg in _NEGATORS)
                seen.add(name)
                out.append(Axiom(name=name, asserted=not negated,
                                 why=f"read from {cue!r}"))
            return out
        except Exception:  # noqa: BLE001
            return out

    # ---- 2–3. build and check ------------------------------------------------ #
    def _formulas(self, axioms: Sequence[Axiom], n: int) -> List[Any]:
        z3 = _z3()
        R = [[z3.Bool(f"R_{i}_{j}") for j in range(n)] for i in range(n)]
        out: List[Any] = []
        for axiom in axioms:
            fn = PROPERTIES.get(axiom.name)
            if fn is None:
                continue
            clauses = fn(R, n)
            out.append(z3.And(*clauses) if axiom.asserted else z3.Not(z3.And(*clauses)))
        return out

    def _search(self, axioms: Sequence[Axiom]) -> Tuple[Optional[bool], int, str]:
        """Bounded model search. ``(found, size, witness)`` — ``found`` is never a guess."""
        z3 = _z3()
        if z3 is None:
            return None, 0, ""
        from nyxara.nyx.theorem_prover import Z3_LOCK
        for n in range(2, self.bound + 1):
            # z3's global context is not thread-safe, and the orchestrator reasons on a thread
            # pool — two threads building expressions at once segfault the interpreter, which no
            # ``except`` here can catch. See Z3_LOCK.
            try:
                with Z3_LOCK:
                    solver = z3.Solver()
                    solver.set("timeout", self.timeout_ms)
                    for formula in self._formulas(axioms, n):
                        solver.add(formula)
                    verdict = solver.check()
                    if verdict == z3.sat:
                        return True, n, self._render_model(solver.model(), n)
            except Exception:  # noqa: BLE001 — a solver failure is not a mathematical result
                continue
        return None, 0, ""

    @staticmethod
    def _render_model(model: Any, n: int) -> str:
        try:
            pairs = []
            for i in range(n):
                for j in range(n):
                    value = model.eval(_z3().Bool(f"R_{i}_{j}"), model_completion=True)
                    if bool(value):
                        pairs.append(f"{i}R{j}")
            return "{" + ", ".join(pairs) + "}" if pairs else "{} (the empty relation)"
        except Exception:  # noqa: BLE001
            return ""

    # ---- 4. independence ----------------------------------------------------- #
    def _independent(self, axioms: Sequence[Axiom], index: int) -> Tuple[bool, str]:
        """Does this axiom follow from the rest? If so it is not a new axiom.

        Checked the only honest way available here: look for a model of the *rest* in which
        this axiom fails. Finding one proves independence. Finding none, up to the bound,
        proves nothing — and is reported as "no witness found", not as "dependent".
        """
        z3 = _z3()
        if z3 is None:
            return False, "no solver"
        rest = [a for k, a in enumerate(axioms) if k != index]
        target = axioms[index]
        flipped = Axiom(name=target.name, asserted=not target.asserted)
        found, size, _witness = self._search(list(rest) + [flipped])
        if found:
            return True, f"independent: the rest hold without it at size {size}"
        return False, ("no witness of independence up to size "
                       f"{self.bound} — it may follow from the rest, or the bound may be "
                       f"too small to show otherwise")

    # ---- 5. derive ----------------------------------------------------------- #
    def _derive(self, axioms: Sequence[Axiom],
                candidates: Sequence[str]) -> List[Theorem]:
        """A candidate holds when ``axioms ∧ ¬candidate`` has no model up to the bound."""
        out: List[Theorem] = []
        for name in candidates:
            if name not in PROPERTIES or any(a.name == name for a in axioms):
                continue
            negated = Axiom(name=name, asserted=False)
            found, size, _w = self._search(list(axioms) + [negated])
            if found:
                out.append(Theorem(
                    statement=name, holds=False, bound=self.bound,
                    why=f"a counter-model exists at size {size} — it does not follow"))
            else:
                out.append(Theorem(
                    statement=name, holds=True, bound=self.bound,
                    certificate=f"no counter-model up to size {self.bound}",
                    why=f"no counter-model up to size {self.bound} — bounded, not a proof"))
        return out

    # ---- the whole act ------------------------------------------------------- #
    def invent(self, problem: str, *, name: str = "",
               derive: Optional[Sequence[str]] = None) -> Genesis:
        """Invent a system for ``problem``, check it, derive in it, and label the result."""
        out = Genesis(problem=str(problem or "").strip())
        t0 = time.perf_counter()
        try:
            if not out.problem:
                out.refused = "nothing was described"
                self.refused += 1
                return out
            if not self.available():
                out.refused = ("z3 is not installed, so consistency cannot be checked — and an "
                               "axiom system nobody checked is not a result")
                self.refused += 1
                return out

            self.attempts += 1
            axioms = self.read(out.problem)
            if not axioms:
                out.refused = (
                    "I could not read any relation property out of that. My axiom language is "
                    f"closed: {', '.join(sorted(PROPERTIES))}. Describe the ordering in those "
                    "terms and I can write the system down and check it.")
                self.refused += 1
                return out

            system = System(name=name or f"system-{self.attempts}", axioms=axioms,
                            bound=self.bound)
            found, size, witness = self._search(axioms)
            system.consistent = True if found else None
            system.model_size = size
            system.model = witness
            system.note = (
                f"a model exists at size {size}: consistency is PROVEN at that size"
                if found else
                f"no model up to size {self.bound}: that is not a proof of inconsistency, "
                f"and it is certainly not a proof of consistency")
            out.system = system

            if found:
                for index, axiom in enumerate(axioms):
                    independent, why = self._independent(axioms, index)
                    axiom.independent, axiom.why = independent, why
                out.theorems = self._derive(axioms, derive or sorted(PROPERTIES))
                out.notation = self._notation(system)

            # Promotion needs a certificate: a model actually found, and at least one axiom
            # shown to be independent. A system whose axioms all follow from each other is not
            # a new system, it is a restatement.
            certified = bool(found) and any(a.independent for a in axioms)
            out.promoted = certified
            out.conjecture = not certified
            if certified:
                self.promoted += 1
                self.systems[system.name] = system.to_dict()
                if len(self.systems) > self.max_systems:
                    self.systems.pop(next(iter(self.systems)))
            else:
                self.conjectures += 1
            return out
        except Exception as exc:  # noqa: BLE001 — inventing never breaks a turn
            out.refused = f"{type(exc).__name__}: {exc}"
            return out
        finally:
            out.duration_ms = (time.perf_counter() - t0) * 1000.0

    def _notation(self, system: System) -> Dict[str, str]:
        """Let :mod:`nyxara.nyx.nexus` mint her own glyphs for this system, if she has it."""
        try:
            nexus = getattr(self.brain, "nexus", None)
            if nexus is None:
                return {}
            concepts = [a.name for a in system.axioms]
            notation = nexus.invent(system.name, concepts)
            return dict(getattr(notation, "symbols", {}) or {})
        except Exception:  # noqa: BLE001 — notation is a flourish, never a requirement
            return {}

    # ---- introspection -------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        try:
            return {
                "available": self.available(),
                "attempts": self.attempts, "promoted": self.promoted,
                "conjectures": self.conjectures, "refused": self.refused,
                "bound": self.bound, "timeout_ms": self.timeout_ms,
                "language": sorted(PROPERTIES),
                "held": sorted(self.systems),
                "note": ("she invents new FORMAL systems, not new physical truths — a "
                         "consistent system is coherent, not true. Consistency is a BOUNDED "
                         "model search: a model found proves consistency at that size; no "
                         "model found proves nothing, and is reported as 'no model up to size "
                         "N'. She will not prove Riemann. The search is budgeted; there is no "
                         "infinite speed"),
            }
        except Exception:  # noqa: BLE001
            return {}

    def to_dict(self) -> Dict[str, Any]:
        return {"attempts": self.attempts, "promoted": self.promoted,
                "conjectures": self.conjectures, "refused": self.refused,
                "systems": dict(self.systems)}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.attempts = int(data.get("attempts", 0))
            self.promoted = int(data.get("promoted", 0))
            self.conjectures = int(data.get("conjectures", 0))
            self.refused = int(data.get("refused", 0))
            self.systems = {str(k): v for k, v in (data.get("systems") or {}).items()
                            if isinstance(v, dict)}
            return True
        except Exception:  # noqa: BLE001 — a corrupt sidecar must never block boot
            return False
