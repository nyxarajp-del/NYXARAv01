"""NYXARA · mind/analogy.py — structure-mapping engine (Gentner) (✦).

Analogy is not decoration; for NYXARA it is *core cognition* — the way one domain is
understood in terms of another. This is a faithful (if compact) **Structure-Mapping
Engine** built on Gentner's Structure-Mapping Theory:

* **Relations matter, attributes don't.** A good analogy aligns the *relational*
  structure (``attracts``, ``revolves-around``, ``causes``), not surface features
  (both are yellow).
* **Systematicity.** Mappings that cohere into a deep, interconnected *system* —
  bound together by higher-order relations like ``cause`` — beat piles of isolated
  matches. The engine weights higher-order structure more.
* **One-to-one & parallel connectivity.** Each base element maps to exactly one
  target element, and if a relation maps, its arguments must map correspondingly.
* **Candidate inferences.** Base relations whose arguments are mapped but which have
  no target counterpart are *projected* into the target — the predictive payoff of
  analogy ("so the nucleus must *cause* the electron to revolve").

Expressions are built from :mod:`mind.lot` (``Const`` = entity, ``Predicate`` =
relation/attribute; nested predicates = higher-order relations).

Depends on :mod:`mind.lot`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from nyxara.mind.lot import Const, Predicate, Term

__all__ = [
    "entity",
    "relation",
    "Correspondence",
    "AnalogyResult",
    "StructureMapper",
    "analogy",
]


# --------------------------------------------------------------------------- #
# Representation helpers
# --------------------------------------------------------------------------- #
def entity(name: str) -> Const:
    return Const(name)


def relation(name: str, *args: Term) -> Predicate:
    return Predicate(name, tuple(args))


def _is_entity(t: Term) -> bool:
    return isinstance(t, Const)


def _is_relation(t: Term) -> bool:
    return isinstance(t, Predicate)


def _order(expr: Term) -> int:
    """Relational order: entity=0, first-order rel=1, higher-order adds one per nesting."""
    if _is_entity(expr):
        return 0
    assert isinstance(expr, Predicate)
    if not expr.args:
        return 1
    return 1 + max(_order(a) for a in expr.args)


def _weight(expr: Term) -> float:
    """Systematicity weight: higher-order relations carry their whole subsystem."""
    if _is_entity(expr):
        return 0.0
    assert isinstance(expr, Predicate)
    return 1.0 + sum(_weight(a) for a in expr.args if _is_relation(a))


# --------------------------------------------------------------------------- #
# Match hypotheses
# --------------------------------------------------------------------------- #
def _match(b: Term, t: Term, analogical: bool = False
           ) -> Optional[Tuple[Dict[str, str], List[Tuple[str, str]], bool]]:
    """Try to align base expr ``b`` with target expr ``t``.

    Returns ``(entity_map, relation_pairs, renamed)`` or ``None``. ``renamed`` is True when the
    alignment relied on relations whose *names differ* (only possible when ``analogical`` is set).

    Strict (``analogical=False``, the default): relations must share an **identical** predicate
    name *and* arity — Gentner's relational-identicality constraint — and entities map freely.

    Analogical (``analogical=True``): relations of **equal arity** may align even when their
    names differ. This is what lets a genuinely-new domain (which uses its *own* verbs) map onto
    a domain NYXARA already understands: the correspondence is licensed not by matching labels but
    by the entity mapping staying globally one-to-one consistent across every aligned relation —
    corroboration is the evidence. Arity must still match, and an entity never aligns to a relation.
    """
    if _is_entity(b) and _is_entity(t):
        return ({b.name: t.name}, [], False)
    if _is_relation(b) and _is_relation(t):
        if len(b.args) != len(t.args):
            return None
        renamed = b.name != t.name
        if renamed and not analogical:
            return None  # strict identicality: differing predicate names never align
        emap: Dict[str, str] = {}
        rels: List[Tuple[str, str]] = [(str(b), str(t))]
        for ba, ta in zip(b.args, t.args):
            sub = _match(ba, ta, analogical)
            if sub is None:
                return None
            se, sr, sub_renamed = sub
            for k, v in se.items():
                if k in emap and emap[k] != v:
                    return None
                emap[k] = v
            rels.extend(sr)
            renamed = renamed or sub_renamed
        return (emap, rels, renamed)
    return None


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class Correspondence:
    base: str
    target: str


@dataclass
class AnalogyResult:
    entity_mapping: Dict[str, str]
    relation_matches: List[Correspondence]
    candidate_inferences: List[Predicate]
    structural_score: float          # systematicity-weighted support (SES)
    systematicity: float             # fraction of matched support that is higher-order
    used_renamed: bool = False       # True if any alignment relied on non-identical relations

    def maps(self, base_entity: str) -> Optional[str]:
        return self.entity_mapping.get(base_entity)

    def to_dict(self) -> Dict[str, object]:
        return {
            "entity_mapping": self.entity_mapping,
            "relation_matches": [(c.base, c.target) for c in self.relation_matches],
            "candidate_inferences": [str(p) for p in self.candidate_inferences],
            "structural_score": round(self.structural_score, 3),
            "systematicity": round(self.systematicity, 3),
            "used_renamed": self.used_renamed,
        }


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
@dataclass
class _Partial:
    entity_map: Dict[str, str]
    rel_pairs: List[Tuple[str, str]]
    base_expr: Predicate
    target_expr: Predicate
    weight: float
    renamed: bool = False


class StructureMapper:
    """Gentner-style structure-mapping engine.

    ``analogical`` (default ``False``) selects the alignment discipline:

    * ``False`` — **strict relational identicality**: only relations with the *same* predicate
      name align. This is the classic, high-precision behaviour every existing caller relies on.
    * ``True`` — **analogical**: relations of equal arity may also align when their names differ,
      as long as the entity mapping stays globally one-to-one consistent. This is what lets a
      *genuinely novel* domain — described in its own fresh vocabulary — map onto a domain NYXARA
      already understands, instead of bouncing to the base LLM. Renamed alignments are charged a
      ``rename_penalty`` so an identical (strict) correspondence always outranks a loose one, and
      the whole mapping only survives when several relations *corroborate* the same entity map.
    """

    def __init__(self, *, attribute_weight: float = 0.2, analogical: bool = False,
                 rename_penalty: float = 0.5) -> None:
        # attributes (1-place predicates) are down-weighted vs relations
        self.attribute_weight = attribute_weight
        self.analogical = bool(analogical)
        self.rename_penalty = float(rename_penalty)

    def _expr_weight(self, expr: Predicate) -> float:
        base = _weight(expr)
        if len(expr.args) == 1 and _is_entity(expr.args[0]):
            return base * self.attribute_weight
        return base

    def _compatible(self, current: Dict[str, str], add: Dict[str, str]) -> bool:
        reverse = {v: k for k, v in current.items()}
        for k, v in add.items():
            if k in current and current[k] != v:
                return False
            if v in reverse and reverse[v] != k:
                return False  # violates one-to-one (two base entities -> one target)
        return True

    def map(self, base: Sequence[Predicate], target: Sequence[Predicate]) -> AnalogyResult:
        # 1) generate local match hypotheses between every base/target relation pair
        partials: List[_Partial] = []
        for b in base:
            for t in target:
                m = _match(b, t, self.analogical)
                if m is not None:
                    emap, rels, renamed = m
                    weight = self._expr_weight(b)
                    if renamed:
                        weight *= self.rename_penalty  # loose (renamed) matches rank below strict
                    partials.append(_Partial(emap, rels, b, t, weight, renamed))

        # 2) greedily merge compatible matches, biggest systematic structure first. Because
        #    renamed matches carry the penalised weight, an identical correspondence is always
        #    merged before a loose one — strict structure wins whenever it exists (tiered SME).
        partials.sort(key=lambda p: p.weight, reverse=True)
        entity_map: Dict[str, str] = {}
        matched: List[_Partial] = []
        matched_base_keys: Set[str] = set()
        matched_target_keys: Set[str] = set()
        for p in partials:
            # one-to-one on relations too (parallel connectivity): a base relation and a target
            # relation each take part in at most one correspondence. This matters in analogical
            # mode, where without it several base relations could pile onto one target relation
            # and wrongly suppress the very inferences that should be projected.
            if str(p.base_expr) in matched_base_keys or str(p.target_expr) in matched_target_keys:
                continue
            if self._compatible(entity_map, p.entity_map):
                entity_map.update(p.entity_map)
                matched.append(p)
                matched_base_keys.add(str(p.base_expr))
                matched_target_keys.add(str(p.target_expr))

        relation_matches = [Correspondence(str(p.base_expr), str(p.target_expr))
                            for p in matched]

        # 3) structural evaluation score + systematicity ratio
        ses = sum(p.weight for p in matched)
        higher = sum(p.weight for p in matched if _order(p.base_expr) >= 2)
        systematicity = (higher / ses) if ses > 0 else 0.0

        # 4) candidate inferences: project un-matched base relations whose args map
        target_strs = {str(t) for t in target}
        candidate_inferences: List[Predicate] = []
        for b in base:
            if str(b) in matched_base_keys:
                continue
            projected = self._project(b, entity_map)
            if projected is not None and str(projected) not in target_strs:
                candidate_inferences.append(projected)

        return AnalogyResult(entity_mapping=entity_map,
                             relation_matches=relation_matches,
                             candidate_inferences=candidate_inferences,
                             structural_score=ses, systematicity=systematicity,
                             used_renamed=any(p.renamed for p in matched))

    def _project(self, expr: Term, emap: Dict[str, str]) -> Optional[Predicate]:
        """Project a base expression into the target via the entity mapping."""
        result = self._project_term(expr, emap)
        return result if isinstance(result, Predicate) else None

    def _project_term(self, expr: Term, emap: Dict[str, str]) -> Optional[Term]:
        if _is_entity(expr):
            return Const(emap[expr.name]) if expr.name in emap else None
        assert isinstance(expr, Predicate)
        new_args: List[Term] = []
        for a in expr.args:
            pa = self._project_term(a, emap)
            if pa is None:
                return None
            new_args.append(pa)
        return Predicate(expr.name, tuple(new_args))


def analogy(base: Sequence[Predicate], target: Sequence[Predicate],
            **kw) -> AnalogyResult:
    """Convenience: map ``base`` onto ``target`` and return the analogy."""
    return StructureMapper(**kw).map(base, target)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA analogy (structure-mapping) self-test")
    print("=" * 70)

    e, r = entity, relation

    # The classic: the atom is like the solar system (Rutherford).
    sun, planet = e("sun"), e("planet")
    nucleus, electron = e("nucleus"), e("electron")

    base = [
        r("attracts", sun, planet),
        r("revolves_around", planet, sun),
        r("hotter", sun, planet),                       # a surface relation, no target match
        r("causes", r("attracts", sun, planet),         # higher-order: the system's glue
          r("revolves_around", planet, sun)),
    ]
    target = [
        r("attracts", nucleus, electron),
        r("revolves_around", electron, nucleus),
    ]

    result = analogy(base, target)
    print(f"\nentity mapping      : {result.entity_mapping}")
    assert result.entity_mapping == {"sun": "nucleus", "planet": "electron"}

    print(f"relation matches    : {[(c.base, c.target) for c in result.relation_matches]}")
    matched_base = {c.base for c in result.relation_matches}
    assert any("attracts" in m for m in matched_base)
    assert any("revolves_around" in m for m in matched_base)

    print(f"candidate inferences: {[str(p) for p in result.candidate_inferences]}")
    # the higher-order CAUSE is projected into the atom domain — the predictive payoff
    inferred = {str(p) for p in result.candidate_inferences}
    assert any("causes" in s and "nucleus" in s and "electron" in s for s in inferred)

    print(f"structural score    : {result.structural_score:.2f}")
    print(f"systematicity       : {result.systematicity:.2f}")
    assert result.structural_score > 0

    # one-to-one is enforced: a base entity can't map to two targets
    bad_target = [r("attracts", nucleus, electron), r("attracts", nucleus, e("proton"))]
    res2 = analogy([r("attracts", sun, planet)], bad_target)
    # only one consistent match is kept
    assert len(res2.entity_mapping) == 2
    print(f"\none-to-one enforced : sun->{res2.maps('sun')}, planet->{res2.maps('planet')}")

    # systematicity preference: a deep system scores higher than scattered matches
    deep = analogy(base, target)
    flat = analogy([r("hotter", sun, planet)], [r("hotter", nucleus, electron)])
    print(f"\ndeep vs flat score  : {deep.structural_score:.2f} vs {flat.structural_score:.2f}")
    assert deep.structural_score > flat.structural_score

    # an analogy with no relational overlap yields nothing (STRICT default)
    empty = analogy([r("foo", e("a"))], [r("bar", e("x"))])
    assert not empty.relation_matches
    print("no-overlap analogy   : empty (correct)")

    # ANALOGICAL mode: a novel domain that uses its OWN verbs still maps by structure.
    # The magnet/filing domain never says "attracts" or "revolves_around" — it says "tugs" and
    # "circles" — yet the higher-order CAUSE is still projected in NYXARA's known vocabulary.
    magnet, filing = e("magnet"), e("filing")
    novel = [r("tugs", magnet, filing), r("circles", filing, magnet)]
    strict = StructureMapper().map(base, novel)
    assert not strict.relation_matches, "strict identicality cannot map fresh verbs"
    loose = StructureMapper(analogical=True).map(base, novel)
    assert loose.used_renamed and loose.entity_mapping == {"sun": "magnet", "planet": "filing"}
    inferred = {str(p) for p in loose.candidate_inferences}
    assert any("causes" in s and "magnet" in s and "filing" in s for s in inferred)
    print(f"analogical (renamed) : sun->magnet, planet->filing; projects {sorted(inferred)}")

    print("\nALL SELF-TESTS PASSED ✓")
