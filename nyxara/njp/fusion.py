"""NYXARA · njp/fusion.py — the same shape in two subjects that never met (🧩, NJP V.40).

Forty-two domains sit in ``scripts/knowledge`` and nothing has ever put two of them in the same
room. ``biology.kb`` knows a body holds its temperature steady; ``engineering.kb`` knows a control
system holds a quantity at a set value; ``economics.kb`` knows price moves until supply meets
demand. Three files, three vocabularies, **one shape** — and a store that keeps them apart has
three facts where it could have an idea.

That is the whole of this module: find subgraphs in different domains that are the **same graph
under a renaming**, and say what they have in common.

What an analogy is here, and what it is not
-------------------------------------------

It is a **structure-preserving bijection**. Two subgraphs are analogous when their nodes can be
paired up so that every relation on one side lands on the matching relation on the other. Not
"these two words are similar", not "these two things co-occur" — the words are deliberately
unavailable, because the point is to relate things whose vocabularies share nothing.

So the match is computed on the **shape alone**::

    sensor    → controller → actuator → quantity → sensor       (engineering)
    receptor  → hypothalamus → effector → temperature → receptor (biology)

Same five edges, same relations, no shared word. The bijection is the finding, and the abstraction
is the shape with its roles named by position rather than by either side's vocabulary.

Three rules, and each is a way this could have been fake
--------------------------------------------------------

**Different domains, or it is not a fusion.** Two subgraphs from ``biology.kb`` matching each other
is a duplicate, not an analogy, and a module that counted those would report a large number and
mean nothing by it. :meth:`Fusion.analogies` takes the domains explicitly and refuses a pair from
one.

**A shape has to be big enough to be a shape.** Two nodes joined by one edge are isomorphic to
every other two nodes joined by one edge, so at :data:`MIN_EDGES` = 3 a match is a claim and below
it, it is arithmetic. The number is a floor with a reason, not a taste.

**A near-miss is a miss.** The bijection must preserve **every** edge, both directions. A pattern
matching on four of five edges is exactly the false analogy that makes this kind of system
untrustworthy — *"the atom is a solar system"* is right about four edges and wrong about the one
that matters. :attr:`Analogy.exact` is the only thing scored, and the gauntlet's ``fusion`` paper
mints a distractor domain that differs by a single edge, for that reason alone.

What it may not do
------------------

**It may not name the abstraction.** The roles come out as ``role0``, ``role1`` — positions in a
shape — because inventing the word *"feedback regulation"* would be the module supplying the
insight it claims to have found. What it returns is the shape, the members, and the mapping; a
name is a human's to give, and :meth:`Abstraction.gloss` says so by naming roles after the members
that fill them.

**It may not merge the store.** An analogy is a proposal. Writing it back as fact would assert that
a controller *is* a hypothalamus.

Pure standard library, deterministic.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

__all__ = ["Pattern", "Analogy", "Abstraction", "Fusion", "MIN_EDGES", "MAX_RADIUS",
           "STRUCTURAL"]

#: Below this an isomorphism is arithmetic rather than a finding: every one-edge subgraph matches
#: every other.
MIN_EDGES = 3

#: How far a pattern reaches from its seed.
#:
#: **Three**, and the number was measured rather than chosen. At two, the four-edge feedback loop
#: this module exists to find came back as a two-edge line: from ``sensor`` you reach
#: ``controller`` and ``actuator`` and stop, and the edge back to the seed — the one that *makes*
#: it feedback — is a hop further than the walk went. A shape needs a radius at least as large as
#: its longest cycle, and the cycles worth finding here are three and four long.
MAX_RADIUS = 3

#: The relations a shape is built from. Deliberately the *functional* ones — what acts on what —
#: rather than ``is_a``, which would match every taxonomy to every other taxonomy and report the
#: fact that both files have hierarchies as an insight.
STRUCTURAL: Tuple[str, ...] = ("causes", "requires", "purpose", "produces", "has_part",
                               "consists_of", "increases", "decreases", "occurs_when")


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Pattern:
    """A small subgraph around a seed, as edges over its own nodes."""

    seed: str
    nodes: Tuple[str, ...]
    edges: FrozenSet[Tuple[str, str, str]] = frozenset()   # (subject, relation, object)
    domain: str = ""

    @property
    def size(self) -> int:
        return len(self.edges)

    @property
    def shape(self) -> Tuple[Tuple[str, int], ...]:
        """A vocabulary-free fingerprint: how many edges of each relation, and the degree profile.

        Cheap, and it is a **filter rather than a decision**: two patterns with the same
        fingerprint still have to survive the bijection below. Its only job is to keep the
        expensive check off pairs that cannot possibly match, which matters because the expensive
        check is a backtracking search.
        """
        by_relation: Dict[str, int] = defaultdict(int)
        degree: Dict[str, int] = defaultdict(int)
        for subject, relation, obj in self.edges:
            by_relation[relation] += 1
            degree[subject] += 1
            degree[obj] += 1
        profile = tuple(sorted(degree.values()))
        return tuple(sorted(by_relation.items())) + (("·degree", hash(profile) % 9973),)

    def render(self) -> str:
        return "; ".join(f"{s} —{r}→ {o}" for s, r, o in sorted(self.edges))

    def to_dict(self) -> Dict[str, Any]:
        return {"seed": self.seed, "domain": self.domain, "size": self.size,
                "nodes": list(self.nodes), "edges": sorted(list(e) for e in self.edges)}


@dataclass
class Analogy:
    """Two patterns from different domains, and the renaming that turns one into the other."""

    left: Pattern
    right: Pattern
    mapping: Dict[str, str] = field(default_factory=dict)

    @property
    def exact(self) -> bool:
        """Does the mapping carry **every** edge, both ways? A near-miss is a miss."""
        if not self.mapping or len(self.left.edges) != len(self.right.edges):
            return False
        carried = {(self.mapping.get(s, s), r, self.mapping.get(o, o))
                   for s, r, o in self.left.edges}
        return carried == set(self.right.edges)

    @property
    def size(self) -> int:
        return len(self.left.edges)

    def to_dict(self) -> Dict[str, Any]:
        return {"left": self.left.to_dict(), "right": self.right.to_dict(),
                "mapping": dict(self.mapping), "exact": self.exact, "size": self.size}


@dataclass
class Abstraction:
    """One shape and everything found to have it. The output of a fusion, and it is unnamed."""

    shape: Tuple[Tuple[str, str, str], ...]      # edges over role names
    members: List[Pattern] = field(default_factory=list)
    roles: Dict[str, Dict[str, str]] = field(default_factory=dict)   # role -> {domain: node}

    @property
    def domains(self) -> List[str]:
        return sorted({m.domain for m in self.members if m.domain})

    @property
    def reach(self) -> int:
        return len(self.domains)

    def gloss(self) -> str:
        """The shape, with each role read out as what fills it in each domain.

        Roles are positions, not names. Calling this one *"feedback regulation"* would be the
        module supplying the insight it claims to have found — so it says instead: *role0 is the
        sensor here and the receptor there*, and leaves the word to a person.
        """
        lines = ["; ".join(f"{s} —{r}→ {o}" for s, r, o in self.shape)]
        for role in sorted(self.roles):
            fills = ", ".join(f"{d}: {n}" for d, n in sorted(self.roles[role].items()))
            lines.append(f"  {role} = {fills}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"shape": [list(e) for e in self.shape], "reach": self.reach,
                "domains": self.domains, "roles": {k: dict(v) for k, v in self.roles.items()},
                "members": [m.to_dict() for m in self.members]}


# --------------------------------------------------------------------------- #
# The fuser
# --------------------------------------------------------------------------- #
class Fusion:
    """Pulls patterns out of a store and matches them across domains."""

    def __init__(self, explainer: Any = None, *, min_edges: int = MIN_EDGES,
                 radius: int = MAX_RADIUS) -> None:
        self.explainer = explainer
        self.min_edges = max(1, int(min_edges))
        self.radius = max(1, int(radius))

    # ---- pulling a shape out of the store -------------------------------- #
    def _out(self, subject: str, relation: str) -> List[str]:
        ex = self.explainer
        if ex is None:
            return []
        try:
            return [obj for obj, _c in ex._out(subject, relation)]
        except Exception:  # noqa: BLE001
            return []

    def pattern(self, seed: str, *, domain: str = "") -> Pattern:
        """Everything structurally reachable from *seed* within :attr:`radius` hops."""
        nodes: List[str] = [seed]
        edges: Set[Tuple[str, str, str]] = set()
        frontier = [seed]
        for _ in range(self.radius):
            nxt: List[str] = []
            for node in frontier:
                for relation in STRUCTURAL:
                    for obj in self._out(node, relation):
                        edges.add((node, relation, obj))
                        if obj not in nodes:
                            nodes.append(obj)
                            nxt.append(obj)
            frontier = nxt
        # And the edges *among* the nodes already collected, so a loop closes. Without this a
        # feedback shape is read as a line — the edge that makes it feedback is the one back to
        # the seed, and it is only found by looking again once the far end is known.
        for node in list(nodes):
            for relation in STRUCTURAL:
                for obj in self._out(node, relation):
                    if obj in nodes:
                        edges.add((node, relation, obj))
        return Pattern(seed=seed, nodes=tuple(nodes), edges=frozenset(edges), domain=domain)

    # ---- matching --------------------------------------------------------- #
    def match(self, left: Pattern, right: Pattern) -> Optional[Analogy]:
        """The bijection, or ``None``. Backtracking over node pairings, edges as the constraint."""
        if left.size < self.min_edges or left.size != right.size:
            return None
        if len(left.nodes) != len(right.nodes):
            return None
        target = set(right.edges)
        left_nodes, right_nodes = list(left.nodes), list(right.nodes)

        def consistent(mapping: Dict[str, str]) -> bool:
            for s, r, o in left.edges:
                if s in mapping and o in mapping:
                    if (mapping[s], r, mapping[o]) not in target:
                        return False
            return True

        def walk(index: int, mapping: Dict[str, str], used: Set[str]) -> Optional[Dict[str, str]]:
            if index == len(left_nodes):
                return dict(mapping) if consistent(mapping) else None
            node = left_nodes[index]
            for candidate in right_nodes:
                if candidate in used:
                    continue
                mapping[node] = candidate
                if consistent(mapping):
                    got = walk(index + 1, mapping, used | {candidate})
                    if got is not None:
                        return got
                del mapping[node]
            return None

        mapping = walk(0, {}, set())
        if mapping is None:
            return None
        got = Analogy(left=left, right=right, mapping=mapping)
        return got if got.exact else None

    def analogies(self, seeds: Dict[str, Sequence[str]]) -> List[Analogy]:
        """Every exact cross-domain analogy among these seeds. ``{domain: [seed]}``."""
        patterns: List[Pattern] = []
        for domain, names in seeds.items():
            for name in names:
                got = self.pattern(name, domain=domain)
                if got.size >= self.min_edges:
                    patterns.append(got)
        by_shape: Dict[Any, List[Pattern]] = defaultdict(list)
        for got in patterns:
            by_shape[got.shape].append(got)
        found: List[Analogy] = []
        for group in by_shape.values():
            for left, right in itertools.combinations(group, 2):
                if left.domain and left.domain == right.domain:
                    continue        # a duplicate, not an analogy
                got = self.match(left, right)
                if got is not None:
                    found.append(got)
        return sorted(found, key=lambda a: (-a.size, a.left.seed, a.right.seed))

    def abstract(self, seeds: Dict[str, Sequence[str]]) -> List[Abstraction]:
        """Group the analogies into shapes, one :class:`Abstraction` per shape.

        Transitive by construction: three domains with one shape between them produce three
        pairwise analogies and **one** abstraction with three members, which is the object worth
        having — an idea that reaches three places, not three separate resemblances.
        """
        found = self.analogies(seeds)
        groups: Dict[Any, List[Pattern]] = defaultdict(list)
        for got in found:
            for side in (got.left, got.right):
                if side not in groups[got.left.shape]:
                    groups[got.left.shape].append(side)
        out: List[Abstraction] = []
        for _shape_key, members in groups.items():
            anchor = members[0]
            roles = {node: f"role{i}" for i, node in enumerate(anchor.nodes)}
            shape = tuple(sorted((roles[s], r, roles[o]) for s, r, o in anchor.edges
                                 if s in roles and o in roles))
            filled: Dict[str, Dict[str, str]] = {role: {} for role in roles.values()}
            kept: List[Pattern] = []
            for member in members:
                # Every member is aligned **to the anchor**, not to whichever member happened to
                # be paired with it first. The first version took one analogy's two sides as the
                # membership, so three domains sharing one shape produced an abstraction reaching
                # two of them — the third was in the analogies and not in the idea. An abstraction
                # is the transitive closure over a shape, and that is the whole reason it is a
                # different object from a list of resemblances.
                mapping = ({node: node for node in anchor.nodes} if member is anchor
                           else (self.match(anchor, member) or Analogy(anchor, member)).mapping)
                if not mapping:
                    continue
                kept.append(member)
                for node, role in roles.items():
                    other = mapping.get(node)
                    if other and member.domain:
                        filled[role][member.domain] = other
            if len(kept) < 2:
                continue
            out.append(Abstraction(shape=shape, members=kept, roles=filled))
        return sorted(out, key=lambda a: (-a.reach, -len(a.shape)))
