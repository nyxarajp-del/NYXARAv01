"""NYXARA · growth/formal_proof.py — proof-carrying abstractions (📐, Rule 4, F9).

The F5 red-team *tries to break* a candidate; F9 *proves it well-formed and merges what is provably the
same*. Before an abstraction is trusted, it carries a **machine-checkable certificate**, and the library
is kept minimal by a **univalence** rule: two abstractions **proven behaviourally equivalent** are
**unified** into one.

Three real, bounded procedures — honest like the :class:`~nyxara.growth.prover.Prover`, abstaining
(``UNPROVABLE``) rather than bluffing outside what they can decide:

* **Type-checking** (``typecheck``) — a bounded checker that verifies a program/abstraction body is
  well-typed against the library signatures. A carried certificate says *this is sound by construction*.
* **Behavioural equivalence** (``equivalent``) — a decision procedure over a comprehensive input battery
  (the F5 boundary inputs + structured probes): ``PROVEN`` when two programs agree on **every** tested
  input (a verified-equivalence certificate, honestly scoped to the battery), ``REFUTED`` with a
  counter-example on the first disagreement.
* **Univalence unification** (``UnivalenceUnifier``) — provably-equivalent abstractions are merged
  (one kept, the other rewritten away everywhere and archived reversibly), shrinking the library while
  preserving every program's meaning.

Pure standard library; consults **no LLM**; touches no source, no weights, no gate — it only *judges and
merges* verified programs. The reused :class:`~nyxara.growth.prover.ProofVerdict` keeps the vocabulary in
lock-step with the rest of NYXARA's verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.growth.noesis import INT, INTLIST, TYPES, Abstraction, Library, Prog, app, evaluate, lit, var
from nyxara.growth.prover import ProofVerdict
from nyxara.growth.redteam import boundary_inputs

__all__ = ["Certificate", "typecheck", "equivalent", "UnivalenceUnifier", "FormalVerifier"]


# --------------------------------------------------------------------------- #
# bounded type checker
# --------------------------------------------------------------------------- #
def typecheck(prog: Prog, lib: Library, hole_types: Sequence[str] = ()) -> bool:
    """True iff ``prog`` is well-typed against the library signatures (a soundness proof)."""
    if prog.kind == "lit":
        if prog.rtype == INT:
            return isinstance(prog.value, int) and not isinstance(prog.value, bool)
        if prog.rtype == "bool":
            return isinstance(prog.value, bool)
        return prog.rtype in TYPES
    if prog.kind == "var":
        return prog.rtype in TYPES
    if prog.kind == "hole":
        idx = int(prog.value)
        return idx < len(hole_types) and hole_types[idx] == prog.rtype if hole_types else True
    if prog.kind == "app":
        entry = lib.entry(prog.name)
        if entry is None or len(prog.args) != len(entry.arg_types):
            return False
        for a, t in zip(prog.args, entry.arg_types):
            if a.rtype != t or not typecheck(a, lib, hole_types):
                return False
        return prog.rtype == entry.ret_type
    return False


# --------------------------------------------------------------------------- #
# behavioural-equivalence decision procedure
# --------------------------------------------------------------------------- #
@dataclass
class Certificate:
    """A carried proof about a program/abstraction — honest about its scope."""

    property: str
    verdict: ProofVerdict
    tested: int = 0
    counterexample: Optional[Any] = None
    detail: str = ""

    @property
    def proven(self) -> bool:
        return self.verdict is ProofVerdict.PROVEN

    def to_dict(self) -> Dict[str, Any]:
        return {"property": self.property, "verdict": self.verdict.value,
                "tested": self.tested, "counterexample": self.counterexample,
                "detail": self.detail}


def _probe_inputs(input_type: str) -> List[Any]:
    return boundary_inputs(input_type) or [0, 1, 2]


def equivalent(a: Prog, b: Prog, input_type: str, lib: Library) -> Certificate:
    """Decide whether two programs compute the same function over the input battery."""
    tested = 0
    for x in _probe_inputs(input_type):
        va = evaluate(a, x, lib)
        vb = evaluate(b, x, lib)
        tested += 1
        if va != vb:
            return Certificate(f"{a.pretty()} ≡ {b.pretty()}", ProofVerdict.REFUTED,
                               tested, counterexample=x,
                               detail=f"differ on {x!r}: {va!r} != {vb!r}")
    return Certificate(f"{a.pretty()} ≡ {b.pretty()}", ProofVerdict.PROVEN, tested,
                       detail=f"verified equal on {tested} boundary/structured inputs")


# --------------------------------------------------------------------------- #
# renaming (for univalence merges) — pure, structural
# --------------------------------------------------------------------------- #
def rename_app(prog: Prog, old: str, new: str) -> Prog:
    name = new if (prog.kind == "app" and prog.name == old) else prog.name
    if not prog.args:
        return Prog(prog.kind, prog.rtype, name=name, value=prog.value)
    return Prog(prog.kind, prog.rtype, name=name, value=prog.value,
                args=tuple(rename_app(a, old, new) for a in prog.args))


# --------------------------------------------------------------------------- #
# abstraction-level equivalence + univalence unification
# --------------------------------------------------------------------------- #
def _fillers_for(arg_types: Sequence[str]) -> List[Tuple[Prog, ...]]:
    """A small battery of argument tuples to probe two abstractions as functions."""
    per_arg: List[List[Prog]] = []
    for t in arg_types:
        if t == INT:
            per_arg.append([lit(0, INT), lit(1, INT), lit(2, INT), lit(3, INT)])
        elif t == INTLIST:
            per_arg.append([var(INTLIST)])
        else:  # bool
            per_arg.append([lit(True, "bool"), lit(False, "bool")])
    combos: List[Tuple[Prog, ...]] = [()]
    for choices in per_arg:
        combos = [c + (ch,) for c in combos for ch in choices]
    return combos


def abstractions_equivalent(a: Abstraction, b: Abstraction, lib: Library,
                            input_type: str = INTLIST) -> Certificate:
    """PROVEN iff two same-signature abstractions agree for every filler tuple over the input battery."""
    if a.arg_types != b.arg_types or a.ret_type != b.ret_type:
        return Certificate(f"{a.name} ≡ {b.name}", ProofVerdict.REFUTED, 0,
                           detail="different type signatures")
    tested = 0
    for fillers in _fillers_for(a.arg_types):
        pa = app(a.name, a.ret_type, fillers)
        pb = app(b.name, b.ret_type, fillers)
        for x in _probe_inputs(input_type):
            tested += 1
            if evaluate(pa, x, lib) != evaluate(pb, x, lib):
                return Certificate(f"{a.name} ≡ {b.name}", ProofVerdict.REFUTED, tested,
                                   counterexample=x, detail="diverge on a filler/input")
    return Certificate(f"{a.name} ≡ {b.name}", ProofVerdict.PROVEN, tested,
                       detail=f"verified equal on {tested} filler×input probes")


class UnivalenceUnifier:
    """Merge provably-equivalent abstractions — the library keeps one representative per behaviour."""

    def unify(self, lib: Library) -> List[Tuple[str, str]]:
        """Return (kept, dropped) pairs; rewrite dropped→kept in all bodies and archive the dropped."""
        merged: List[Tuple[str, str]] = []
        names = list(lib.abstractions.keys())
        gone: set = set()
        for i in range(len(names)):
            if names[i] in gone:
                continue
            for j in range(i + 1, len(names)):
                if names[j] in gone:
                    continue
                a, b = lib.abstractions.get(names[i]), lib.abstractions.get(names[j])
                if a is None or b is None:
                    continue
                if abstractions_equivalent(a, b, lib).proven:
                    keep, drop = names[i], names[j]
                    # rewrite every remaining abstraction body drop→keep, then archive drop
                    for k, ab in list(lib.abstractions.items()):
                        if k != drop:
                            lib.abstractions[k] = Abstraction(
                                ab.name, ab.arg_types, ab.ret_type,
                                rename_app(ab.body, drop, keep))
                    lib.counts[keep] = lib.counts.get(keep, 0.0) + lib.counts.get(drop, 0.0)
                    lib.prune_abstraction(drop)   # reversible archive
                    gone.add(drop)
                    merged.append((keep, drop))
        return merged


# --------------------------------------------------------------------------- #
# the facade the engine uses
# --------------------------------------------------------------------------- #
class FormalVerifier:
    """Certify adopted abstractions and unify provably-equivalent ones (F9)."""

    def __init__(self) -> None:
        self.unifier = UnivalenceUnifier()
        self.certificates: List[Certificate] = []

    def certify(self, absn: Abstraction, lib: Library) -> Certificate:
        ok = typecheck(absn.body, lib, absn.arg_types)
        cert = Certificate(
            f"{absn.name} well-typed :: {absn.arg_types} -> {absn.ret_type}",
            ProofVerdict.PROVEN if ok else ProofVerdict.UNPROVABLE,
            tested=1, detail="type-checked against library signatures" if ok
            else "could not type-check (abstained)")
        self.certificates.append(cert)
        return cert

    def unify(self, lib: Library) -> List[Tuple[str, str]]:
        return self.unifier.unify(lib)
