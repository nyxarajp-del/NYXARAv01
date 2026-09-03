"""NYXARA · njp/surgery.py — rival structures, not one graph walked (🔪🕸, NJP V.40).

Everything in :mod:`nyxara.njp.explain` walks *the* graph. The graph is whatever somebody wrote
down, and the walk's only question is which path through it answers the question. That is a
reader, and it has one blind spot that no amount of better walking closes: **if the graph is wrong,
every path through it is wrong, and nothing in the walk can tell.**

The Master's version of the point::

    Model 1:  A → B → C
    Model 2:  A → C   and   B → C
    Model 3:  A ← B → C

Three structures over the same three names. A walk reads whichever one it was handed. This module
is the thing that asks **which of the three the evidence actually supports**, and it is allowed to
change the graph to find out: add an edge, remove one, reverse one, merge two nodes, split one.

What counts as evidence
-----------------------

Not the edges. Asking the graph whether the graph is right is how a model confirms itself, and
this package has made that mistake twice already — V.26's figurative guard suppressing the evidence
that would correct it, V.36's prediction graded on its own output. So the evidence here is
**observations, which are a different kind of statement from a structure**:

``depends_on``
    These two vary together. Written by whoever observed them, and it says nothing about which
    causes which — that is the whole point, and it is what a structure has to *explain*.
``independent_of``
    These two do not vary together. The most informative observation there is, and the reason is
    :class:`Structure`'s v-rule below.

From those two, a structure is **recovered** rather than read. The procedure is the one causal
discovery has used since the PC algorithm and it is three steps:

1. **Skeleton.** Two nodes are adjacent exactly when they are dependent. Nothing about direction
   yet — an undirected graph is what dependence alone can support.
2. **V-structures.** For a path ``A — C — B`` where A and B are **independent of each other**, the
   only orientation consistent with that is ``A → C ← B``. A collider. This is the one place
   direction falls out of the data rather than being assumed, and it is why ``independent_of`` is
   worth more than ``depends_on``.
3. **Propagation.** Any orientation forced by avoiding a new collider or a cycle is applied, and
   then no more. What is left unoriented is **genuinely unoriented**.

Which brings the rule this repository has applied to two parses of a sentence, two equally
supported triples, and two admissible orders of a procedure, now applied to a causal structure:

    **Where several structures fit the evidence equally, the answer is that there are several.**

``A → B → C``, ``A ← B → C`` and ``A ← B ← C`` imply *exactly the same* dependencies and
independencies. No observation of this kind can separate them — they are one **Markov equivalence
class** — and a system that names one of them has invented a direction. :attr:`Verdict.equivalent`
holds all of them and :attr:`Verdict.determined` says whether there was one.

Scoring, when the evidence does not decide
------------------------------------------

Where several structures survive, they are scored, and the score is the Master's #14 written as
something computable::

    explains  −  unsupported  −  λ · complexity

``explains``
    Observations the structure accounts for: a dependence it connects, an independence it leaves
    disconnected or blocks with a collider.
``unsupported``
    Edges the structure asserts that no observation calls for. This is the term that stops "add
    every edge" from winning, and without it the complete graph explains everything.
``complexity``
    Edge count, priced by :data:`COMPLEXITY_COST`. Simplicity is a tie-break and never a truth:
    a simpler model that explains less loses, which is why it is subtracted last and weighted low.

**A score never overrules the evidence.** Members of one equivalence class are indistinguishable
*by construction*, so scoring them against each other would be scoring noise; the score only ever
ranks structures that differ in what they explain. :meth:`Surgeon.discover` returns the class and
the score separately for that reason.

What surgery may not do
-----------------------

**It may not invent an observation.** Every operation is checked against what was observed; a
structure is proposed, never asserted.

**It may not silently pick.** A tie is returned as a tie.

**It never edits the store.** :meth:`Surgeon.discover` returns structures. Writing one back is a
decision with an owner, and in this package that owner is a person.

Pure standard library, deterministic.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "Observation", "Structure", "Score", "Verdict", "Surgeon",
    "DEPENDS", "INDEPENDENT", "COMPLEXITY_COST", "MAX_NODES", "MAX_EQUIVALENT",
]

#: The relation that says *these two vary together*. Direction-free on purpose.
DEPENDS: Tuple[str, ...] = ("depends_on", "co_occurs_with", "varies_with")

#: The relation that says *these two do not*. Worth more than a dependence, because it is what
#: orients a collider.
INDEPENDENT: Tuple[str, ...] = ("independent_of", "unrelated_to")

#: What one edge costs in the score. Low, because simplicity is a tie-break and not a truth.
COMPLEXITY_COST = 0.1

#: Enumerating orientations is exponential in the number of edges. Above this the skeleton and the
#: forced orientations are still returned and the equivalence class is reported as *not
#: enumerated* rather than as empty — a difference a caller must be able to see.
MAX_NODES = 7

#: How many members of an equivalence class are listed. The **count** is exact; the list is capped,
#: for the reason `explain.Plan` caps its orders and keeps its count.
MAX_EQUIVALENT = 12


# --------------------------------------------------------------------------- #
# What is observed, and what is proposed
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Observation:
    """One statement about how two things vary. Never about which causes which.

    ``given`` is the set of things held fixed while looking, and leaving it out was a defect worth
    recording because it silently halves what can be recovered. With **marginal** observations
    only — nothing held fixed — a chain ``A → B → C`` and a triangle are indistinguishable: A and C
    vary together in both. Measured, ``discover`` returned an equivalence class of **six** for a
    chain, which is every orientation of a triangle, because a triangle is what the observations
    described.

    One conditional observation fixes it. *A and C do not vary together **once B is held fixed***
    is what says there is no A–C edge, and it is the statement causal discovery has been built on
    since the PC algorithm. The set that made a pair independent is its **separating set**, and it
    is also what decides whether the middle of a path is a collider — so this one field does both
    jobs.
    """

    left: str
    right: str
    dependent: bool
    given: FrozenSet[str] = frozenset()

    @property
    def pair(self) -> FrozenSet[str]:
        return frozenset({self.left.lower(), self.right.lower()})

    @property
    def sepset(self) -> FrozenSet[str]:
        return frozenset(g.lower() for g in self.given)

    def to_dict(self) -> Dict[str, Any]:
        return {"left": self.left, "right": self.right, "dependent": self.dependent,
                "given": sorted(self.given)}


@dataclass(frozen=True)
class Structure:
    """A directed graph over named nodes, and what it implies.

    Frozen and hashable so an equivalence class can be a set: two structures that are the same
    graph written twice must not be counted as two answers.
    """

    nodes: Tuple[str, ...]
    edges: FrozenSet[Tuple[str, str]] = frozenset()

    def parents(self, node: str) -> Set[str]:
        return {a for a, b in self.edges if b == node}

    def children(self, node: str) -> Set[str]:
        return {b for a, b in self.edges if a == node}

    def adjacent(self, node: str) -> Set[str]:
        return self.parents(node) | self.children(node)

    @property
    def skeleton(self) -> FrozenSet[FrozenSet[str]]:
        return frozenset(frozenset({a, b}) for a, b in self.edges)

    @property
    def colliders(self) -> FrozenSet[Tuple[str, str, str]]:
        """Every ``A → C ← B`` where A and B are not themselves adjacent.

        The *unshielded* qualifier matters: with an ``A — B`` edge present the pattern implies
        nothing about A and B being independent, so it is not a v-structure and must not be
        counted as one.
        """
        out: Set[Tuple[str, str, str]] = set()
        for node in self.nodes:
            parents = sorted(self.parents(node))
            for a, b in itertools.combinations(parents, 2):
                if frozenset({a, b}) not in self.skeleton:
                    out.add((a, node, b))
        return frozenset(out)

    @property
    def acyclic(self) -> bool:
        colour: Dict[str, int] = {}

        def visit(node: str) -> bool:
            state = colour.get(node, 0)
            if state == 1:
                return False
            if state == 2:
                return True
            colour[node] = 1
            for child in self.children(node):
                if not visit(child):
                    return False
            colour[node] = 2
            return True

        return all(visit(n) for n in self.nodes)

    def reachable(self, start: str) -> Set[str]:
        seen: Set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            for child in self.children(node):
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        return seen

    def connected(self, left: str, right: str) -> bool:
        """Is there an undirected path? What a dependence needs in order to be explained."""
        seen = {left}
        stack = [left]
        while stack:
            node = stack.pop()
            if node == right:
                return True
            for other in self.adjacent(node):
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        return right in seen

    def render(self) -> str:
        return ", ".join(f"{a} → {b}" for a, b in sorted(self.edges)) or "(no edges)"

    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": list(self.nodes), "edges": sorted(list(e) for e in self.edges),
                "render": self.render()}


@dataclass
class Score:
    """What a structure is worth against a set of observations."""

    explains: int = 0
    unsupported: int = 0
    complexity: int = 0

    @property
    def total(self) -> float:
        return round(self.explains - self.unsupported - COMPLEXITY_COST * self.complexity, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {"explains": self.explains, "unsupported": self.unsupported,
                "complexity": self.complexity, "total": self.total}


@dataclass
class Verdict:
    """What the evidence supports, including how much it fails to settle."""

    nodes: Tuple[str, ...] = ()
    skeleton: FrozenSet[FrozenSet[str]] = frozenset()
    #: Orientations the data forces. An edge here is a claim about direction the observations make.
    forced: FrozenSet[Tuple[str, str]] = frozenset()
    #: Every DAG consistent with the evidence, capped at :data:`MAX_EQUIVALENT`.
    equivalent: List[Structure] = field(default_factory=list)
    #: The exact size of the equivalence class, uncapped.
    equivalent_count: int = 0
    scores: Dict[str, Score] = field(default_factory=dict)
    enumerated: bool = True
    why: str = ""

    @property
    def determined(self) -> bool:
        """Did the evidence pin exactly one structure?"""
        return self.equivalent_count == 1

    @property
    def best(self) -> Optional[Structure]:
        return self.equivalent[0] if self.equivalent else None

    def holds(self, cause: str, effect: str) -> bool:
        """Is this edge in **every** structure the evidence allows?

        The question a discovered graph is actually for, and it is deliberately not *"is it in the
        first one"*. An arrow that appears in one of three admissible structures is not a finding
        about the world.
        """
        if not self.equivalent:
            return False
        return all((cause, effect) in s.edges for s in self.equivalent)

    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": list(self.nodes),
                "skeleton": sorted(sorted(p) for p in self.skeleton),
                "forced": sorted(list(e) for e in self.forced),
                "determined": self.determined, "count": self.equivalent_count,
                "enumerated": self.enumerated, "why": self.why,
                "equivalent": [s.to_dict() for s in self.equivalent],
                "scores": {k: v.to_dict() for k, v in self.scores.items()}}


# --------------------------------------------------------------------------- #
# The surgeon
# --------------------------------------------------------------------------- #
class Surgeon:
    """Recovers a structure from observations, and scores rivals where they survive together."""

    def __init__(self, grounder: Any = None, *, max_nodes: int = MAX_NODES) -> None:
        self.grounder = grounder
        self.max_nodes = max(2, int(max_nodes))

    # ---- reading observations off a store -------------------------------- #
    def observations(self, nodes: Sequence[str]) -> List[Observation]:
        """Every dependence and independence the store states among these nodes."""
        out: List[Observation] = []
        seen: Set[FrozenSet[str]] = set()
        wanted = {self._key(n) for n in nodes}
        for node in nodes:
            for relation in DEPENDS + INDEPENDENT:
                for other in self._out(node, relation):
                    if self._key(other) not in wanted:
                        continue
                    got = Observation(left=node, right=other,
                                      dependent=relation in DEPENDS)
                    if got.pair in seen or len(got.pair) < 2:
                        continue
                    seen.add(got.pair)
                    out.append(got)
        return out

    def _out(self, subject: str, relation: str) -> List[str]:
        g = self.grounder
        if g is None:
            return []
        try:
            facts = getattr(g, "facts", {}) or {}
            got = facts.get((self._key(subject), relation), ())
            return [str(t.object) for t in got if not getattr(t, "superseded", False)]
        except Exception:  # noqa: BLE001
            return []

    def _key(self, text: str) -> str:
        g = self.grounder
        if g is not None:
            try:
                return g._key(text)
            except Exception:  # noqa: BLE001
                pass
        return " ".join(str(text or "").split()).lower()

    # ---- the three steps -------------------------------------------------- #
    def discover(self, nodes: Sequence[str],
                 observations: Optional[Sequence[Observation]] = None) -> Verdict:
        """Skeleton, v-structures, propagation — then enumerate what is left."""
        names = tuple(dict.fromkeys(str(n) for n in nodes))
        got = list(observations) if observations is not None else self.observations(names)
        out = Verdict(nodes=names)
        if len(names) < 2:
            out.why = "fewer than two things to relate"
            return out

        # 1. Skeleton. Two nodes are adjacent unless **some** observation separates them — with
        #    nothing held fixed, or with something held fixed. A single conditional independence is
        #    enough to delete an edge, which is exactly what marginal observations cannot do.
        separated: Dict[FrozenSet[str], FrozenSet[str]] = {}
        dependent: Set[FrozenSet[str]] = set()
        for observed in got:
            if len(observed.pair) < 2:
                continue
            if observed.dependent:
                dependent.add(observed.pair)
            else:
                separated[observed.pair] = observed.sepset
        pairs = {frozenset({a.lower(), b.lower()})
                 for a, b in itertools.combinations(names, 2)}
        out.skeleton = frozenset(p for p in pairs
                                 if p in dependent and p not in separated)

        # 2. V-structures. For an unshielded ``A — C — B``, C is a **collider** exactly when it is
        #    not in the set that separated A from B. That is the whole rule, and it is why
        #    `Observation.given` does two jobs: the set that deleted the edge also decides the
        #    direction. A marginal independence has an empty separating set, so C is never in it
        #    and the triple is always a collider — which is the classic case and falls out rather
        #    than being special-cased.
        adjacency: Dict[str, Set[str]] = {n.lower(): set() for n in names}
        by_low = {n.lower(): n for n in names}
        for pair in out.skeleton:
            a, b = sorted(pair)
            adjacency[a].add(b)
            adjacency[b].add(a)
        forced: Set[Tuple[str, str]] = set()
        for middle in names:
            low = middle.lower()
            for a_low, b_low in itertools.combinations(sorted(adjacency[low]), 2):
                pair = frozenset({a_low, b_low})
                if pair in out.skeleton:
                    continue            # shielded: implies nothing about direction
                if pair not in separated:
                    continue            # never observed apart: nothing to conclude
                if low in separated[pair]:
                    continue            # the middle is *in* the separating set: not a collider
                forced.add((by_low[a_low], middle))
                forced.add((by_low[b_low], middle))
        out.forced = frozenset(forced)
        independent = set(separated)

        # 3. Every acyclic orientation of the skeleton that keeps the forced edges and invents no
        #    collider the data does not have. What survives is the equivalence class.
        if len(names) > self.max_nodes:
            out.enumerated = False
            out.why = (f"{len(names)} nodes: the skeleton and {len(forced)} forced orientations "
                       f"are here, the class was not enumerated")
            return out
        members, count = self._orientations(names, out.skeleton, forced, separated)
        out.equivalent, out.equivalent_count = members, count
        if not members:
            out.why = "no acyclic structure fits these observations"
        elif count == 1:
            out.why = "the observations determine one structure"
        else:
            out.why = (f"{count} structures fit these observations equally — they imply the same "
                       f"dependencies and no observation of this kind can separate them")
        out.scores = {s.render(): self.score(s, got) for s in members}
        return out

    def _orientations(self, names: Sequence[str], skeleton: FrozenSet[FrozenSet[str]],
                      forced: Set[Tuple[str, str]],
                      separated: Dict[FrozenSet[str], FrozenSet[str]]
                      ) -> Tuple[List[Structure], int]:
        pairs = [tuple(sorted(p)) for p in skeleton]
        found: List[Structure] = []
        count = 0
        forced_low = {(a.lower(), b.lower()) for a, b in forced}
        for bits in itertools.product((0, 1), repeat=len(pairs)):
            edges = frozenset((a, b) if bit == 0 else (b, a)
                              for (a, b), bit in zip(pairs, bits))
            structure = Structure(nodes=tuple(names), edges=edges)
            if not structure.acyclic:
                continue
            low = {(a.lower(), b.lower()) for a, b in edges}
            if not forced_low <= low:
                continue
            # It must not invent a collider the data denies, and must keep every one it has.
            # A collider at C requires that C is **not** in what separated A from B. Testing only
            # whether the pair was separated at all accepted `a → b ← c` for a chain, where b is
            # precisely what makes a and c independent — and the equivalence class came back as 4
            # where it is 3. The middle is the whole rule; the pair is only half of it.
            invented = False
            for a, c, b in structure.colliders:
                pair = frozenset({a.lower(), b.lower()})
                if pair not in separated or c.lower() in separated[pair]:
                    invented = True
                    break
            if invented:
                continue
            count += 1
            if len(found) < MAX_EQUIVALENT:
                found.append(structure)
        return found, count

    # ---- scoring ---------------------------------------------------------- #
    def score(self, structure: Structure, observations: Sequence[Observation]) -> Score:
        """``explains − unsupported − λ·complexity``, and never used to break a Markov tie."""
        out = Score(complexity=len(structure.edges))
        # An edge is called for by a dependence and **cancelled by any separation**, including a
        # conditional one. Counting only the marginal dependence made the term dead: in a chain,
        # `a` and `c` do vary together, so `a → c` looked called for and the complete graph scored
        # an `unsupported` of zero. A conditional independence is precisely what deletes an edge —
        # it is the whole reason `Observation.given` exists — and the score has to read it the same
        # way the skeleton does or the two disagree about the same graph.
        separated = {o.pair for o in observations if not o.dependent}
        called_for = {o.pair for o in observations if o.dependent} - separated
        for observed in observations:
            left, right = observed.left, observed.right
            if observed.dependent:
                if structure.connected(left, right):
                    out.explains += 1
            else:
                # An independence is explained by *not* being connected, or by being connected
                # only through a collider — which blocks, rather than transmits.
                if not structure.connected(left, right):
                    out.explains += 1
                elif any(a in (left, right) and b in (left, right)
                         for a, _c, b in structure.colliders):
                    out.explains += 1
        for a, b in structure.edges:
            if frozenset({a.lower(), b.lower()}) not in {
                    frozenset({x.lower() for x in p}) for p in called_for}:
                out.unsupported += 1
        return out

    # ---- the operations, named ------------------------------------------- #
    @staticmethod
    def add_edge(structure: Structure, cause: str, effect: str) -> Structure:
        return Structure(structure.nodes, structure.edges | {(cause, effect)})

    @staticmethod
    def remove_edge(structure: Structure, cause: str, effect: str) -> Structure:
        return Structure(structure.nodes, structure.edges - {(cause, effect)})

    @staticmethod
    def reverse_edge(structure: Structure, cause: str, effect: str) -> Structure:
        return Structure(structure.nodes,
                         (structure.edges - {(cause, effect)}) | {(effect, cause)})

    @staticmethod
    def merge_nodes(structure: Structure, first: str, second: str, *, into: str = "") -> Structure:
        """Two names for one thing. The inverse of the split below, and the repair for a store
        that filed one entity under two spellings."""
        name = into or f"{first}/{second}"
        rename = {first: name, second: name}
        nodes = tuple(dict.fromkeys(rename.get(n, n) for n in structure.nodes))
        edges = frozenset((rename.get(a, a), rename.get(b, b)) for a, b in structure.edges)
        return Structure(nodes, frozenset(e for e in edges if e[0] != e[1]))

    @staticmethod
    def split_node(structure: Structure, node: str, *, incoming: str, outgoing: str) -> Structure:
        """One name for two things — the identity defect V.38 fixed in the corpus, expressed as a
        graph operation so a discovered graph can propose it rather than a person having to."""
        edges: Set[Tuple[str, str]] = set()
        for a, b in structure.edges:
            if b == node:
                edges.add((a, incoming))
            elif a == node:
                edges.add((outgoing, b))
            else:
                edges.add((a, b))
        nodes = tuple(dict.fromkeys(
            [n for n in structure.nodes if n != node] + [incoming, outgoing]))
        return Structure(nodes, frozenset(edges))
