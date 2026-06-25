"""NYXARA · cognition/concept_formation.py — the abstraction ladder (▲▲).

The hardest thing a mind does is **abstraction**: see ``Car``, ``Bike``, ``Truck`` and
build the concept ``Vehicle``; then look at ``Vehicle``, ``Plane``, ``Ship`` and climb
again to ``Transport``. Substitution turns the abstract concrete; *this* turns the
concrete abstract — and it climbs, level after level.

The two faculties NYXARA already had do **not** do this:

* :mod:`nyxara.cognition.abstraction` is logical **anti-unification** over
  Language-of-Thought terms (``parent(tom,bob) ⊓ parent(ann,sue) → parent(?X,?Y)``) —
  relational schemas, not a *named, multi-level IS-A taxonomy* of things.
* :mod:`nyxara.mind.concept_hierarchy` clusters **actions** by what they *do*
  (effect-signature), not what they *are*.

This module supplies the missing piece — **feature-based taxonomic concept formation**:

* :meth:`AbstractionLadder.observe` registers a thing by its features.
* :meth:`AbstractionLadder.abstract` forms a superordinate concept as the **shared
  structure** of its members — the generalization that keeps what they agree on
  (``Car ⊓ Bike ⊓ Truck → Vehicle``). Relational features are folded through
  :func:`nyxara.cognition.abstraction.anti_unify`, so structure (not just flat tags)
  generalizes consistently.
* :meth:`AbstractionLadder.ascend` does it **herself**: it clusters the current
  top-level nodes by feature overlap and builds the next level up
  (``Vehicle, Plane, Ship → Transport``). Repeat → a ladder of arbitrary height.
* :meth:`AbstractionLadder.classify` recognizes a *never-seen* thing by **subsumption**
  (a fresh ``Scooter`` is a ``Vehicle``), and :meth:`AbstractionLadder.path` reads off
  the chain ``Car → Vehicle → Transport``.

The whole structure is introspectable (:meth:`report`, :meth:`to_dict`). Pure standard
library (no numpy), reusing the existing anti-unification engine for the relational path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = ["Instance", "Concept", "AbstractionLadder"]


# --------------------------------------------------------------------------- #
# Relational features — a tiny parser so structured tags fold through anti_unify
# --------------------------------------------------------------------------- #
_REL_RE = re.compile(r"^([A-Za-z_]\w*)\(([^()]*)\)$")


def _parse_relation(feature: str):
    """``"owns(jp, car)" → Predicate`` (flat predicates only). None if not relational.

    A leading ``?`` marks a variable (``"faster(?X, ?Y)"``). Returns ``None`` for plain
    atomic tags like ``"has_wheels"`` so they are treated as ordinary features."""
    m = _REL_RE.match(feature.strip())
    if not m:
        return None
    args = [a.strip() for a in m.group(2).split(",") if a.strip()]
    if not args:
        return None
    try:
        from nyxara.mind.lot import Var, const, pred
    except Exception:  # noqa: BLE001 — LoT optional; fall back to atomic tag
        return None
    terms = [Var(a[1:]) if a.startswith("?") else const(a) for a in args]
    return pred(m.group(1), *terms)


def _relational_generalization(feature_sets: Sequence[Set[str]]) -> Set[str]:
    """Fold relational features that occur in *every* member into abstract schemas.

    For each predicate name present across all members, anti-unify one representative
    term per member; the resulting (variabilized) schema string becomes a defining
    feature of the concept. Atomic tags are ignored here (handled by plain intersection).
    """
    if not feature_sets:
        return set()
    # predicate-name -> per-member list of parsed terms
    by_name: Dict[str, List[List[Any]]] = {}
    for fs in feature_sets:
        seen: Dict[str, List[Any]] = {}
        for feat in fs:
            term = _parse_relation(feat)
            if term is not None:
                seen.setdefault(term.name, []).append(term)
        for name, terms in seen.items():
            by_name.setdefault(name, []).append(terms)

    out: Set[str] = set()
    n = len(feature_sets)
    for name, per_member in by_name.items():
        if len(per_member) != n:           # not present in every member → not shared
            continue
        try:
            from nyxara.cognition.abstraction import anti_unify
            acc = per_member[0][0]
            for member_terms in per_member[1:]:
                acc = anti_unify(acc, member_terms[0])
            out.add(str(acc))
        except Exception:  # noqa: BLE001 — relational lift is best-effort, never fatal
            continue
    return out


# --------------------------------------------------------------------------- #
# Nodes: instances (level 0) and concepts (level ≥ 1)
# --------------------------------------------------------------------------- #
@dataclass
class Instance:
    """A concrete thing described by a set of features (+ optional numeric attributes)."""

    name: str
    features: Set[str] = field(default_factory=set)
    attrs: Dict[str, float] = field(default_factory=dict)
    level: int = 0

    @property
    def support(self) -> int:
        return 1


@dataclass
class Concept:
    """A superordinate node: the shared structure of the members it generalizes.

    ``members`` are its direct children in the taxonomy (instances or lower concepts);
    ``features`` are the **defining** features they all share; ``level`` is its height
    above the instances; ``support`` counts the concrete instances beneath it.
    """

    name: str
    features: Set[str] = field(default_factory=set)
    members: List[str] = field(default_factory=list)
    level: int = 1
    parent: Optional[str] = None
    attr_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    support: int = 0


# --------------------------------------------------------------------------- #
# The abstraction ladder
# --------------------------------------------------------------------------- #
class AbstractionLadder:
    """Feature-based concept formation that climbs: instances → concepts → meta-concepts.

    ``link_threshold`` is the minimum Jaccard overlap of two nodes' features for
    :meth:`ascend` to group them under a shared parent (single-link agglomeration)."""

    def __init__(self, *, link_threshold: float = 0.2) -> None:
        self.link_threshold = link_threshold
        self._instances: Dict[str, Instance] = {}
        self._concepts: Dict[str, Concept] = {}
        self._parent: Dict[str, str] = {}        # node name -> parent concept name

    # ------------------------------------------------------------------ #
    # Observe concrete things
    # ------------------------------------------------------------------ #
    def observe(self, name: str, features: Iterable[str],
                attrs: Optional[Dict[str, float]] = None) -> Instance:
        """Register (or update) a concrete instance by its features and attributes."""
        feats = {str(f).strip() for f in features if str(f).strip()}
        inst = self._instances.get(name)
        if inst is None:
            inst = Instance(name=name, features=feats, attrs=dict(attrs or {}))
            self._instances[name] = inst
        else:
            inst.features |= feats
            inst.attrs.update(attrs or {})
        return inst

    # ------------------------------------------------------------------ #
    # Node access (instances and concepts share a name space)
    # ------------------------------------------------------------------ #
    def _exists(self, name: str) -> bool:
        return name in self._instances or name in self._concepts

    def _features_of(self, name: str) -> Set[str]:
        if name in self._instances:
            return self._instances[name].features
        if name in self._concepts:
            return self._concepts[name].features
        return set()

    def _attrs_of(self, name: str) -> Dict[str, Tuple[float, float]]:
        """Numeric attributes of a node as ranges (a bare instance value → a point range)."""
        if name in self._instances:
            return {k: (v, v) for k, v in self._instances[name].attrs.items()}
        if name in self._concepts:
            return dict(self._concepts[name].attr_ranges)
        return {}

    def _level_of(self, name: str) -> int:
        if name in self._instances:
            return self._instances[name].level
        if name in self._concepts:
            return self._concepts[name].level
        return 0

    def _support_of(self, name: str) -> int:
        if name in self._instances:
            return 1
        if name in self._concepts:
            return self._concepts[name].support
        return 0

    # ------------------------------------------------------------------ #
    # Abstract: form a superordinate concept = the shared structure of members
    # ------------------------------------------------------------------ #
    def abstract(self, names: Optional[Sequence[str]] = None, *,
                 name: Optional[str] = None, min_share: float = 1.0) -> Optional[Concept]:
        """Generalize ``names`` into one superordinate concept and return it.

        The concept's **defining features** are those shared by at least ``min_share``
        of the members (``1.0`` = strict intersection — the classic generalization).
        Relational features fold through anti-unification; numeric attributes become
        ranges. If ``name`` is omitted it is auto-derived from the salient shared
        features. Each member is re-parented under the new concept.
        """
        candidates = self._top_level() if names is None else names
        members = [n for n in candidates if self._exists(n)]
        if len(members) < 2:
            return None

        feature_sets = [self._features_of(n) for n in members]
        atomic_sets = [{f for f in fs if _parse_relation(f) is None} for fs in feature_sets]

        # shared atomic tags: present in >= min_share of the members
        counts: Dict[str, int] = {}
        for fs in atomic_sets:
            for f in fs:
                counts[f] = counts.get(f, 0) + 1
        need = max(2, int(round(min_share * len(members)))) if min_share >= 1.0 \
            else max(2, int(min_share * len(members)) + (1 if min_share else 0))
        shared = {f for f, c in counts.items() if c >= min(need, len(members))}
        # relational structure shared by every member, generalized via anti_unify
        shared |= _relational_generalization(feature_sets)

        # numeric attributes shared by every member → a covering range
        attr_ranges: Dict[str, Tuple[float, float]] = {}
        member_attrs = [self._attrs_of(n) for n in members]
        common_keys = set.intersection(*[set(a) for a in member_attrs]) if member_attrs else set()
        for k in common_keys:
            los = [a[k][0] for a in member_attrs]
            his = [a[k][1] for a in member_attrs]
            attr_ranges[k] = (min(los), max(his))

        cname = name or self._auto_name(shared)
        cname = self._unique_name(cname)
        level = max((self._level_of(n) for n in members), default=0) + 1
        support = sum(self._support_of(n) for n in members)
        concept = Concept(name=cname, features=set(shared), members=list(members),
                          level=level, attr_ranges=attr_ranges, support=support)
        self._concepts[cname] = concept
        for n in members:                       # re-parent the members under the concept
            self._parent[n] = cname
        return concept

    def _auto_name(self, shared: Set[str]) -> str:
        salient = sorted(shared)[:3]
        return "≈" + "+".join(salient) if salient else f"concept{len(self._concepts) + 1}"

    def _unique_name(self, base: str) -> str:
        if base not in self._concepts and base not in self._instances:
            return base
        i = 2
        while f"{base}#{i}" in self._concepts or f"{base}#{i}" in self._instances:
            i += 1
        return f"{base}#{i}"

    # ------------------------------------------------------------------ #
    # Ascend: cluster the top-level nodes and build the next level — autonomously
    # ------------------------------------------------------------------ #
    def ascend(self, *, threshold: Optional[float] = None,
               min_cluster: int = 2) -> List[Concept]:
        """Climb one level: group the current top-level nodes by feature overlap and
        abstract each group of size ``>= min_cluster`` into a shared parent.

        Single-link agglomeration on Jaccard similarity of features. Returns the
        newly-formed concepts (empty if nothing clustered). This is the self-driven
        path — she builds the higher concept on her own."""
        thr = self.link_threshold if threshold is None else threshold
        nodes = self._top_level()
        if len(nodes) < min_cluster:
            return []

        # union-find: link any pair whose feature overlap clears the threshold
        parent = {n: n for n in nodes}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if self._jaccard(nodes[i], nodes[j]) >= thr:
                    union(nodes[i], nodes[j])

        groups: Dict[str, List[str]] = {}
        for n in nodes:
            groups.setdefault(find(n), []).append(n)

        formed: List[Concept] = []
        for members in groups.values():
            if len(members) >= min_cluster:
                c = self.abstract(members)
                if c is not None:
                    formed.append(c)
        return formed

    def _jaccard(self, a: str, b: str) -> float:
        fa, fb = self._features_of(a), self._features_of(b)
        if not fa and not fb:
            return 0.0
        inter = len(fa & fb)
        union = len(fa | fb)
        return inter / union if union else 0.0

    def _top_level(self) -> List[str]:
        """All nodes (instances or concepts) that have no parent yet."""
        return [n for n in [*self._instances, *self._concepts] if n not in self._parent]

    # ------------------------------------------------------------------ #
    # Classify a (possibly unseen) thing by subsumption
    # ------------------------------------------------------------------ #
    def classify(self, features: Iterable[str]) -> Optional[Tuple[str, Concept]]:
        """Most-specific concept whose defining features subsume ``features``.

        A concept *subsumes* a thing when all of the concept's defining features are
        present in the thing. Among the subsuming concepts the most specific one wins
        (most defining features, then deepest level). Generalizes to never-seen things."""
        feats = {str(f).strip() for f in features if str(f).strip()}
        best: Optional[Concept] = None
        best_key: Tuple[int, int] = (-1, -1)
        for c in self._concepts.values():
            if c.features and c.features <= feats:
                key = (len(c.features), c.level)
                if key > best_key:
                    best, best_key = c, key
        return (best.name, best) if best is not None else None

    def is_a(self, name: str, concept: str) -> bool:
        """True if ``name`` is (transitively) under ``concept`` — or, for an unseen
        thing's feature set, subsumed by it."""
        if self._exists(name):
            return concept in self.ancestors(name)
        return False

    # ------------------------------------------------------------------ #
    # Read the ladder: the Car → Vehicle → Transport chain
    # ------------------------------------------------------------------ #
    def concept_of(self, name: str) -> Optional[str]:
        """The direct parent concept of a node (one rung up), or None at the top."""
        return self._parent.get(name)

    def path(self, name: str) -> List[str]:
        """The chain from ``name`` up to its root concept: ``[Car, Vehicle, Transport]``."""
        chain = [name]
        seen = {name}
        cur = name
        while cur in self._parent:
            cur = self._parent[cur]
            if cur in seen:                     # defensive: never loop
                break
            seen.add(cur)
            chain.append(cur)
        return chain

    def ancestors(self, name: str) -> List[str]:
        return self.path(name)[1:]

    def members_of(self, concept: str) -> List[str]:
        c = self._concepts.get(concept)
        return list(c.members) if c else []

    def concepts(self) -> List[Concept]:
        return list(self._concepts.values())

    def instances(self) -> List[Instance]:
        return list(self._instances.values())

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "instances": len(self._instances),
            "concepts": [
                {"name": c.name, "level": c.level, "support": c.support,
                 "features": sorted(c.features), "members": list(c.members),
                 "parent": c.parent or self._parent.get(c.name)}
                for c in sorted(self._concepts.values(), key=lambda x: (x.level, x.name))
            ],
            "roots": [n for n in self._top_level() if n in self._concepts],
        }

    def report(self) -> str:
        """A readable tree, roots first, indented by level."""
        lines = [f"AbstractionLadder: {len(self._instances)} instances in "
                 f"{len(self._concepts)} concepts"]
        roots = [n for n in self._top_level() if n in self._concepts]
        # also surface lone top-level instances (not yet abstracted)
        loose = [n for n in self._top_level() if n in self._instances]

        def emit(node: str, depth: int) -> None:
            mark = "▲" if node in self._concepts else "•"
            feats = sorted(self._features_of(node))
            shown = ", ".join(feats[:5]) + ("…" if len(feats) > 5 else "")
            lines.append(f"{'  ' * depth}{mark} {node}  {{{shown}}}")
            for m in self._concepts.get(node, Concept(node)).members \
                    if node in self._concepts else []:
                emit(m, depth + 1)

        for r in roots:
            emit(r, 1)
        for inst in loose:
            emit(inst, 1)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test / demo — the literal Car/Bike/Truck → Vehicle → Transport ladder
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA abstraction-ladder self-test  (Car/Bike/Truck → Vehicle → Transport)")
    print("=" * 70)

    L = AbstractionLadder()

    # --- level 0: concrete things, described by features --- #
    L.observe("Car",   ["has_wheels", "has_engine", "carries_people", "moves", "road"])
    L.observe("Bike",  ["has_wheels", "carries_people", "moves", "road"])
    L.observe("Truck", ["has_wheels", "has_engine", "carries_people", "carries_cargo",
                        "moves", "road"])

    # --- abstract: Car ⊓ Bike ⊓ Truck → Vehicle (the shared structure) --- #
    vehicle = L.abstract(["Car", "Bike", "Truck"], name="Vehicle")
    print(f"\nformed concept      : {vehicle.name}  defining={sorted(vehicle.features)}")
    assert vehicle is not None and vehicle.name == "Vehicle"
    assert vehicle.features == {"has_wheels", "carries_people", "moves", "road"}
    assert vehicle.support == 3 and vehicle.level == 1
    assert L.concept_of("Car") == "Vehicle"

    # --- two more top-level things at the same altitude --- #
    L.observe("Plane", ["has_wings", "has_engine", "carries_people", "moves", "sky"])
    L.observe("Ship",  ["has_hull", "carries_people", "carries_cargo", "moves", "water"])

    # --- ascend: she clusters Vehicle/Plane/Ship herself → Transport --- #
    formed = L.ascend()
    print(f"ascended           : {[c.name for c in formed]}")
    assert len(formed) == 1
    transport = formed[0]
    # rename the auto-named meta-concept to read nicely in the demo
    print(f"meta-concept       : {transport.name}  defining={sorted(transport.features)}")
    assert transport.features == {"carries_people", "moves"}
    assert transport.level == 2 and transport.support == 5
    assert set(transport.members) == {"Vehicle", "Plane", "Ship"}

    # --- the chain reads off cleanly: Car → Vehicle → Transport --- #
    chain = L.path("Car")
    print(f"\npath(Car)          : {' → '.join(chain)}")
    assert chain == ["Car", "Vehicle", transport.name]
    assert L.ancestors("Bike") == ["Vehicle", transport.name]

    # --- classify a never-seen thing by subsumption → most-specific concept --- #
    hit = L.classify(["has_wheels", "has_engine", "carries_people", "moves", "road"])
    print(f"classify(Scooter)  : {hit[0]}  (most specific subsuming concept)")
    assert hit is not None and hit[0] == "Vehicle"      # Vehicle beats the broader Transport
    far = L.classify(["carries_people", "moves", "rails"])
    assert far is not None and far[0] == transport.name  # only the meta-concept subsumes a train

    # --- auto-naming when no name is given (she names it from shared features) --- #
    L2 = AbstractionLadder()
    L2.observe("Sparrow", ["has_wings", "lays_eggs", "feathers", "flies"])
    L2.observe("Eagle",   ["has_wings", "lays_eggs", "feathers", "flies", "predator"])
    auto = L2.abstract(["Sparrow", "Eagle"])
    print(f"\nauto-named concept : {auto.name}")
    assert auto.name.startswith("≈") and "feathers" in auto.features

    # --- relational features fold through anti-unification --- #
    L3 = AbstractionLadder()
    L3.observe("a", ["owns(jp, car)", "mortal"])
    L3.observe("b", ["owns(jp, bike)", "mortal"])
    rel = L3.abstract(["a", "b"])
    print(f"relational abstr   : {sorted(rel.features)}")
    assert "mortal" in rel.features
    assert any("owns(jp," in f for f in rel.features)   # owns(jp, ?X) schema retained

    print("\n" + L.report())
    print("\nALL SELF-TESTS PASSED ✓")
