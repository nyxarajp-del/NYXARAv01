"""NYXARA · njp/theory.py — what never changes, and running what survives (📐, NJP V.45).

Two of the Master's items, and they are one pipeline::

    #9   examples → what survives every change → invariant → abstract law
    #11  law → formal representation → executable predictor → compare → revise

Separating them would have produced a module that *names* a regularity and a module that *runs*
one, with nothing to run. A law you cannot execute is a sentence; a predictor with no law behind
it is a lookup table. The point is the join.

The demand that makes it hard is one line of the Master's spec::

    system state changes
    parameters change
    surface vocabulary changes
    but:  relationship X → Y survives

**Surface vocabulary changes.** So an invariant cannot be found by asking which variable names
recur — in the interesting case, none of them do. What recurs is *shape*, which is the same claim
:mod:`nyxara.njp.fusion` makes across domains, made here across **situations**. Fusion asks what
biology and engineering have in common; this asks what Monday and Thursday have in common when
Monday's variables are called nothing that Thursday's are.

So the hunt runs on structure, and the alignment is a bijection.

What an invariant is, and what it is not
-----------------------------------------

An :class:`Invariant` is a relation that holds in **every** situation shown, under an alignment of
each situation's variables onto shared **roles**. Roles are positions, not names — ``role0``,
``role1`` — for the reason ``fusion.Abstraction`` leaves its shapes unnamed: inventing the word
would be the module supplying the finding it claims to have made.

It is **not** a correlation, and the difference is enforced rather than asserted:

**One situation is not an invariant.** Anything at all holds in one situation.
:data:`MIN_SITUATIONS` is three, because two can be a coincidence with a name.

**A relation that could not have failed is not a finding.** If every situation has the edge because
every situation has *every* edge, the invariant is a description of the encoding, not of the world.
:attr:`Invariant.discriminating` says whether it ever could have come out otherwise, and
:meth:`Hunter.hunt` will not promote one that could not.

**An exception is recorded, never hidden.** A relation holding in four of five situations is a
:class:`Law` with a **scope** and a listed counterexample, not an invariant. Rounding that to
"usually" is how a theory stops being checkable.

The compiler, and what makes it a compiler
-------------------------------------------

:meth:`Law.compile` returns a :class:`Theory` — an object with :meth:`Theory.predict`, which takes
a situation it has never seen, aligns it to the roles, and **fills in the relation the law says
must be there**. That is the whole of #11: the learned regularity becomes something that runs.

And because it runs, it can be **wrong in public**. :meth:`Theory.check` takes a situation with its
真 relations included, predicts without looking, and reports every miss — which is what closes the
loop the Master drew: theory → simulate → predict → compare → *invalid or incomplete*.

What it may not do
------------------

**It may not predict from an alignment it does not have.** A situation whose variables cannot be
mapped onto the roles gets :attr:`Prediction.aligned` ``False`` and no prediction — not a guess
against a partial match.

**It may not quietly widen a law's scope.** A law induced from situations of one shape does not
claim situations of another; :attr:`Law.scope` is the shape, and a situation outside it is
declined.

**It may not hide the counterexample that made it a law rather than an invariant.**

Pure standard library, deterministic.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "Situation", "Invariant", "Law", "Theory", "Prediction", "Hunter",
    "MIN_SITUATIONS", "MAX_ROLES",
]

#: Below this an invariant is a coincidence with a name. Two situations can agree by accident;
#: three agreeing under a *structural* alignment is a claim.
MIN_SITUATIONS = 3

#: How many variables a situation may have before alignment stops being affordable. Alignment is a
#: bijection search, and it is factorial.
MAX_ROLES = 6


# --------------------------------------------------------------------------- #
# What is observed
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Situation:
    """One occasion: some named things, and the relations that held between them.

    The names are **local to the situation** and are expected to differ between them. That is the
    whole difficulty and it is why :class:`Hunter` never compares them.
    """

    name: str
    variables: Tuple[str, ...]
    relations: FrozenSet[Tuple[str, str, str]] = frozenset()   # (subject, relation, object)

    @property
    def shape(self) -> Tuple[Tuple[str, int], ...]:
        """A vocabulary-free fingerprint: how many of each relation, and the degree profile."""
        counts: Dict[str, int] = defaultdict(int)
        degree: Dict[str, int] = defaultdict(int)
        for subject, relation, obj in self.relations:
            counts[relation] += 1
            degree[subject] += 1
            degree[obj] += 1
        profile = tuple(sorted(degree.get(v, 0) for v in self.variables))
        return tuple(sorted(counts.items())) + (("·degree", hash(profile) % 9973),)

    def holds(self, subject: str, relation: str, obj: str) -> bool:
        return (subject, relation, obj) in self.relations

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "variables": list(self.variables),
                "relations": sorted(list(r) for r in self.relations)}


# --------------------------------------------------------------------------- #
# What survives
# --------------------------------------------------------------------------- #
@dataclass
class Invariant:
    """A relation that held in every situation, stated over roles rather than over names."""

    edges: Tuple[Tuple[str, str, str], ...]      # over role names
    roles: Tuple[str, ...]
    #: situation name -> {role: that situation's variable}
    alignment: Dict[str, Dict[str, str]] = field(default_factory=dict)
    #: How often each edge could have been absent and was not. See :attr:`discriminating`.
    could_have_failed: int = 0

    @property
    def situations(self) -> int:
        return len(self.alignment)

    @property
    def discriminating(self) -> bool:
        """Could this ever have come out otherwise?

        A relation present in every situation because every situation contains every relation is a
        description of the encoding rather than of the world, and promoting one is how a theory
        acquires content it never earned.
        """
        return self.could_have_failed > 0

    def render(self) -> str:
        body = "; ".join(f"{s} —{r}→ {o}" for s, r, o in self.edges)
        lines = [body]
        for role in self.roles:
            fills = ", ".join(f"{name}: {mapping.get(role, '?')}"
                              for name, mapping in sorted(self.alignment.items()))
            lines.append(f"  {role} = {fills}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"edges": [list(e) for e in self.edges], "roles": list(self.roles),
                "situations": self.situations, "discriminating": self.discriminating,
                "alignment": {k: dict(v) for k, v in self.alignment.items()}}


@dataclass
class Law:
    """An invariant promoted to something that claims about situations it has not seen.

    ``scope`` is the shape the law was induced from. It is not decoration: a law induced from
    four-variable situations with two relation kinds does not claim a six-variable one, and
    :meth:`Theory.predict` declines rather than guessing outside it.
    """

    invariant: Invariant
    scope: Tuple[Tuple[str, int], ...]
    exceptions: List[str] = field(default_factory=list)
    support: int = 0

    @property
    def exact(self) -> bool:
        """No counterexample. An invariant; otherwise a law with a scope."""
        return not self.exceptions

    def render(self) -> str:
        head = "INVARIANT" if self.exact else f"LAW (with {len(self.exceptions)} exception(s))"
        lines = [f"{head} over {self.support} situation(s)", self.invariant.render()]
        for name in self.exceptions:
            lines.append(f"  EXCEPT in {name}")
        return "\n".join(lines)

    def compile(self) -> "Theory":
        """Turn the law into something that runs. The whole of #11."""
        return Theory(law=self)

    def to_dict(self) -> Dict[str, Any]:
        return {"exact": self.exact, "support": self.support,
                "exceptions": list(self.exceptions), "scope": [list(s) for s in self.scope],
                "invariant": self.invariant.to_dict()}


@dataclass
class Prediction:
    """What a theory says about a situation, and whether it was in a position to say anything."""

    situation: str
    aligned: bool = False
    expected: List[Tuple[str, str, str]] = field(default_factory=list)
    missing: List[Tuple[str, str, str]] = field(default_factory=list)
    why: str = ""

    @property
    def ok(self) -> bool:
        return self.aligned and not self.missing

    def to_dict(self) -> Dict[str, Any]:
        return {"situation": self.situation, "aligned": self.aligned, "ok": self.ok,
                "expected": [list(e) for e in self.expected],
                "missing": [list(e) for e in self.missing], "why": self.why}


class Theory:
    """A compiled law: give it a situation it has never seen and it says what must be there."""

    def __init__(self, law: Law) -> None:
        self.law = law
        self.checked = 0
        self.failed = 0

    # ---- alignment -------------------------------------------------------- #
    def align(self, situation: Situation,
              *, ignore: Sequence[Tuple[str, str, str]] = ()) -> Optional[Dict[str, str]]:
        """Map this situation's variables onto the law's roles, or ``None``.

        ``ignore`` is what to align *without* — the relations being predicted. Aligning on the
        answer and then predicting it is the mistake every one of this package's exams has had to
        be rescued from at least once, so the caller says which edges are hidden and the search
        never sees them.
        """
        roles = self.law.invariant.roles
        if len(situation.variables) < len(roles) or len(roles) > MAX_ROLES:
            return None
        hidden = {tuple(e) for e in ignore}
        visible = {r for r in situation.relations if tuple(r) not in hidden}
        wanted = [e for e in self.law.invariant.edges if tuple(e) not in hidden]
        for candidate in itertools.permutations(situation.variables, len(roles)):
            mapping = dict(zip(roles, candidate))
            if all((mapping[s], r, mapping[o]) in visible for s, r, o in wanted):
                return mapping
        return None

    # ---- running it -------------------------------------------------------- #
    def predict(self, situation: Situation,
                *, hidden: Sequence[Tuple[str, str, str]] = ()) -> Prediction:
        """What the law says must hold here. Declines outside its scope, and outside an alignment."""
        out = Prediction(situation=situation.name)
        if situation.shape != self.law.scope and not hidden:
            out.why = "outside the shape this law was induced from"
            return out
        mapping = self.align(situation, ignore=hidden)
        if mapping is None:
            out.why = "this situation's variables do not map onto the roles"
            return out
        out.aligned = True
        for subject, relation, obj in self.law.invariant.edges:
            edge = (mapping[subject], relation, mapping[obj])
            out.expected.append(edge)
            if not situation.holds(*edge):
                out.missing.append(edge)
        out.why = "every edge the law requires is present" if not out.missing else (
            f"{len(out.missing)} edge(s) the law requires are absent")
        return out

    def check(self, situations: Sequence[Situation]) -> Dict[str, Any]:
        """Predict on each, compare, and report. Theory → simulate → predict → compare."""
        results = [self.predict(s) for s in situations]
        self.checked += len(results)
        self.failed += sum(1 for r in results if r.aligned and not r.ok)
        aligned = [r for r in results if r.aligned]
        return {"asked": len(results), "aligned": len(aligned),
                "held": sum(1 for r in aligned if r.ok),
                "declined": len(results) - len(aligned),
                "verdict": ("holds" if aligned and all(r.ok for r in aligned)
                            else "incomplete" if not aligned else "invalid"),
                "misses": [r.to_dict() for r in aligned if not r.ok][:5]}

    def to_dict(self) -> Dict[str, Any]:
        return {"law": self.law.to_dict(), "checked": self.checked, "failed": self.failed}


# --------------------------------------------------------------------------- #
# The hunt
# --------------------------------------------------------------------------- #
class Hunter:
    """Finds what survives across situations whose vocabularies share nothing."""

    def __init__(self, *, min_situations: int = MIN_SITUATIONS,
                 max_roles: int = MAX_ROLES) -> None:
        self.min_situations = max(2, int(min_situations))
        self.max_roles = max(2, int(max_roles))

    def hunt(self, situations: Sequence[Situation]) -> List[Law]:
        """Every law these situations support, strongest first.

        The alignment is anchored on the first situation and every other is searched for a
        bijection onto it — the same structure-preserving match :mod:`nyxara.njp.fusion` makes
        across domains, made here across occasions. What survives every alignment is the invariant;
        what survives all but a few is a law with those few listed.
        """
        usable = [s for s in situations if len(s.variables) <= self.max_roles]
        if len(usable) < self.min_situations:
            return []
        anchor = usable[0]
        roles = tuple(f"role{i}" for i in range(len(anchor.variables)))
        base = dict(zip(anchor.variables, roles))
        candidate_edges = {(base[s], r, base[o]) for s, r, o in anchor.relations
                           if s in base and o in base}
        if not candidate_edges:
            return []

        alignments: Dict[str, Dict[str, str]] = {anchor.name: {v: k for k, v in
                                                               zip(anchor.variables, roles)}}
        alignments[anchor.name] = dict(zip(roles, anchor.variables))
        held: Dict[Tuple[str, str, str], List[str]] = {e: [anchor.name] for e in candidate_edges}
        missed: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        could_fail = 0

        for other in usable[1:]:
            mapping = self._match(roles, candidate_edges, other)
            if mapping is None:
                # No alignment at all: this situation is not about the same thing, and saying so
                # is better than forcing a partial map and calling the leftovers exceptions.
                for edge in candidate_edges:
                    missed[edge].append(other.name)
                continue
            alignments[other.name] = mapping
            present = {(mapping[s], r, mapping[o]) for s, r, o in candidate_edges
                       if other.holds(mapping[s], r, mapping[o])}
            # How many edges *could* have been absent here — the discrimination count.
            possible = len(other.variables) * (len(other.variables) - 1) * \
                len({r for _s, r, _o in other.relations} or {"x"})
            could_fail += max(0, possible - len(other.relations))
            for edge in candidate_edges:
                held.setdefault(edge, [])
                if (mapping[edge[0]], edge[1], mapping[edge[2]]) in \
                        {(mapping[s], r, mapping[o]) for s, r, o in candidate_edges} and \
                        (mapping[edge[0]], edge[1], mapping[edge[2]]) in present:
                    held[edge].append(other.name)
                else:
                    missed[edge].append(other.name)

        out: List[Law] = []
        survivors = [e for e in candidate_edges if len(held.get(e, [])) >= self.min_situations]
        if not survivors:
            return out
        # **Both groups, always.** Preferring the perfect subset dropped the edge that had a
        # counterexample and reported an INVARIANT over what was left — which is the module's own
        # "an exception is recorded, never hidden" rule, broken by the code that states it. An
        # edge holding in four situations of five is a finding *with a listed exception*, and
        # silently removing it turns a checkable law into a smaller true one that says less.
        perfect = tuple(sorted(e for e in survivors if not missed.get(e)))
        flawed = tuple(sorted(e for e in survivors if missed.get(e)))
        for group, exceptions in ((perfect, ()),
                                  (flawed, tuple(sorted({name for e in flawed
                                                         for name in missed.get(e, [])})))):
            if not group:
                continue
            invariant = Invariant(edges=group, roles=roles, alignment=alignments,
                                  could_have_failed=could_fail)
            law = Law(invariant=invariant, scope=anchor.shape, exceptions=list(exceptions),
                      support=len(alignments) - len(exceptions))
            if invariant.discriminating and law.support >= self.min_situations:
                out.append(law)
        return sorted(out, key=lambda l: (not l.exact, -l.support))

    def _match(self, roles: Sequence[str], edges: Set[Tuple[str, str, str]],
               situation: Situation) -> Optional[Dict[str, str]]:
        """A bijection from roles onto this situation's variables preserving as much as possible.

        Best-effort rather than exact, because an *exception* is a situation that aligns and then
        fails an edge — and an exact-only matcher could never find one, so a law with exceptions
        could never be discovered at all.
        """
        if len(situation.variables) < len(roles):
            return None
        best: Optional[Tuple[int, Dict[str, str]]] = None
        for candidate in itertools.permutations(situation.variables, len(roles)):
            mapping = dict(zip(roles, candidate))
            score = sum(1 for s, r, o in edges
                        if situation.holds(mapping[s], r, mapping[o]))
            if best is None or score > best[0]:
                best = (score, mapping)
            if best[0] == len(edges):
                break
        return best[1] if best and best[0] > 0 else None
