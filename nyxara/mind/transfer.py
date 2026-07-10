"""NYXARA · mind/transfer.py — cross-domain generalization by her OWN faculties (⇄).

The critique this module answers: *"NYXARA can only generalize as far as her 9B base model;
the scaffolding polishes output but never breaks the base model's ceiling."* That was true for
one honest reason — NYXARA already owns a genuine Structure-Mapping Engine
(:mod:`nyxara.mind.analogy`, Gentner), but nothing on the *inference* path ever ran it. A new-
domain query flowed straight to the LLM.

:class:`RelationalTransferEngine` closes that gap. When NYXARA meets a novel domain it does what
a scientist does before reaching for memorized facts: it **maps the new domain's relational
structure onto a domain it already understands** and *projects* the known structure across the
mapping. The transferred content — the candidate inferences — is derived by NYXARA's own
symbolic engine, not sampled from the base model. The LLM, if present, only verbalizes what the
mapping already established; with no LLM at all the transfer *is* the answer.

This is deliberately honest about its reach:

* It only fires when it can recover **≥2 relations and a real mapping** from the query. On
  free-form chat with no extractable structure it returns ``None`` and the normal LLM path runs
  — no faking a transfer that isn't there.
* A projected inference is a *hypothesis by analogy* ("so X probably Y"), surfaced as such, not
  asserted as ground truth. The base model's parametric ceiling is not magically lifted; what
  changes is that on structurally-transferable questions the *reasoning content is hers*.

Pure standard library. Reuses :class:`nyxara.mind.analogy.StructureMapper` and the Language of
Thought (:mod:`nyxara.mind.lot`). Pairs with :class:`nyxara.mind.self_model_router` (which routes
a query here before the teacher) and :mod:`nyxara.mind.general_intelligence` (novel-domain solver).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from nyxara.mind.analogy import AnalogyResult, StructureMapper, entity, relation
from nyxara.mind.lot import Const, Predicate, Term

__all__ = [
    "DomainSchema",
    "DomainSchemaStore",
    "TransferResult",
    "RelationalTransferEngine",
    "seed_library",
]

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")

# Surface words that never carry domain structure — stripped before entity detection.
_STOP = frozenset("""
a an the of to in on at by for with from into onto over under and or but if then than as is are
was were be been being do does did will would can could should may might must this that these
those it its it's their there here what which who whom whose why how when where does do can what's
some any each every both all no not more most much many few least how's like about between within
""".split())


# --------------------------------------------------------------------------- #
# A known domain: its relational skeleton + the surface words that signal it
# --------------------------------------------------------------------------- #
@dataclass
class DomainSchema:
    """One domain NYXARA already understands, as a set of :mod:`lot` relations plus the
    surface vocabulary (relation verbs + entity nouns) that flags the domain in free text."""

    name: str
    relations: List[Predicate]
    keywords: frozenset = field(default_factory=frozenset)

    def entities(self) -> List[str]:
        seen: List[str] = []
        def walk(t: Term) -> None:
            if isinstance(t, Const):
                if t.name not in seen:
                    seen.append(t.name)
            elif isinstance(t, Predicate):
                for a in t.args:
                    walk(a)
        for r in self.relations:
            walk(r)
        return seen

    def relation_names(self) -> set:
        names: set = set()
        def walk(t: Term) -> None:
            if isinstance(t, Predicate):
                names.add(t.name)
                for a in t.args:
                    walk(a)
        for r in self.relations:
            walk(r)
        return names

    def vocabulary(self) -> frozenset:
        return frozenset(set(self.keywords)
                         | {n.lower() for n in self.relation_names()}
                         | {e.lower() for e in self.entities()})


# --------------------------------------------------------------------------- #
# The store of known domains — seeded, and grown from lived structure
# --------------------------------------------------------------------------- #
class DomainSchemaStore:
    """Known domains NYXARA can transfer *from*, ranked for a query by surface + structural fit."""

    def __init__(self, schemas: Optional[Sequence[DomainSchema]] = None) -> None:
        self._schemas: Dict[str, DomainSchema] = {}
        for s in (schemas if schemas is not None else seed_library()):
            self._schemas[s.name] = s

    def __len__(self) -> int:
        return len(self._schemas)

    def names(self) -> List[str]:
        return list(self._schemas)

    def get(self, name: str) -> Optional[DomainSchema]:
        return self._schemas.get(name)

    def add(self, schema: DomainSchema) -> None:
        self._schemas[schema.name] = schema

    def learn(self, name: str, relations: Sequence[Predicate],
              keywords: Sequence[str] = ()) -> DomainSchema:
        """Grow the store from lived structure so a domain met once transfers next time.

        Merges into an existing domain of the same name (union of relations/keywords) rather
        than clobbering it, so recognition and structure *accumulate*."""
        rels = list(relations)
        kw = frozenset(k.lower() for k in keywords if k)
        prior = self._schemas.get(name)
        if prior is not None:
            seen = {str(r) for r in prior.relations}
            merged = list(prior.relations) + [r for r in rels if str(r) not in seen]
            schema = DomainSchema(name, merged, prior.keywords | kw)
        else:
            schema = DomainSchema(name, rels, kw)
        self._schemas[name] = schema
        return schema

    def rank(self, tokens: Sequence[str], target_rel_names: Sequence[str] = ()) -> List[
            Tuple[float, DomainSchema]]:
        """Rank known domains for a query by surface-word overlap and relation-name overlap.

        A shared *relation name* (the query says "attracts", a domain has ``attracts``) counts
        double a shared surface word — relational structure is what actually transfers."""
        toks = {t.lower() for t in tokens}
        trn = {n.lower() for n in target_rel_names}
        scored: List[Tuple[float, DomainSchema]] = []
        for s in self._schemas.values():
            vocab = s.vocabulary()
            surface = len(toks & vocab)
            rel_hit = len(trn & {n.lower() for n in s.relation_names()})
            score = surface + 2.0 * rel_hit
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda p: p[0], reverse=True)
        return scored


# --------------------------------------------------------------------------- #
# The transfer result — a mapping plus the projected (candidate) inferences
# --------------------------------------------------------------------------- #
@dataclass
class TransferResult:
    base_domain: str
    entity_mapping: Dict[str, str]
    candidate_inferences: List[Predicate]
    structural_score: float
    systematicity: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "base_domain": self.base_domain,
            "entity_mapping": dict(self.entity_mapping),
            "candidate_inferences": [str(p) for p in self.candidate_inferences],
            "structural_score": round(self.structural_score, 3),
            "systematicity": round(self.systematicity, 3),
        }

    def render(self) -> str:
        """A human-readable, honestly-hedged statement of the transfer."""
        if not self.candidate_inferences:
            pairs = ", ".join(f"{b}→{t}" for b, t in self.entity_mapping.items())
            return (f"By analogy to {self.base_domain} ({pairs}), the same relational structure "
                    f"holds, but nothing new is projected.")
        lines = [f"By analogy to {self.base_domain}, NYXARA's structure-mapper projects "
                 f"(hypotheses to verify, not asserted facts):"]
        for p in self.candidate_inferences:
            lines.append(f"  • so it is likely that {_humanize(p)}")
        return "\n".join(lines)


def _humanize(p: Predicate) -> str:
    """Render a relation as a readable clause: ``causes(a, b)`` → ``a causes b``."""
    verb = p.name.replace("_", " ")
    args = [a.name if isinstance(a, Const) else _humanize(a)  # type: ignore[arg-type]
            if isinstance(a, Predicate) else str(a) for a in p.args]
    if len(args) == 2:
        return f"{args[0]} {verb} {args[1]}"
    if len(args) == 1:
        return f"{args[0]} is {verb}"
    return f"{verb}({', '.join(args)})"


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
class RelationalTransferEngine:
    """Generalize a novel-domain query by structure-mapping from a known domain.

    ``min_score`` is the minimum systematicity-weighted structural score to accept a mapping;
    below it the engine declines (returns ``None``) and the caller falls through to the LLM.
    """

    def __init__(self, *, store: Optional[DomainSchemaStore] = None,
                 mapper: Optional[StructureMapper] = None, min_score: float = 1.0,
                 max_bases: int = 4) -> None:
        self.store = store if store is not None else DomainSchemaStore()
        self.mapper = mapper if mapper is not None else StructureMapper()
        self.min_score = float(min_score)
        self.max_bases = int(max_bases)

    def learn_domain(self, name: str, relations: Sequence[Predicate],
                     keywords: Sequence[str] = ()) -> DomainSchema:
        return self.store.learn(name, relations, keywords)

    # ---- the public entry point ---- #
    def generalize(self, query: str, *,
                   target_relations: Optional[Sequence[Predicate]] = None
                   ) -> Optional[TransferResult]:
        """Transfer known structure onto ``query``'s domain, or ``None`` if none maps.

        ``target_relations`` lets a structured caller (the domain solver, a test) supply the
        new domain's relational skeleton directly; otherwise a conservative text extractor
        recovers it from the query. Either way the result's inferences are projected by
        NYXARA's own structure-mapper — never sampled from a language model."""
        target = list(target_relations) if target_relations else self._extract(query)
        if len(target) < 2:
            return None  # too little structure to transfer honestly
        tokens = _WORD.findall(query or "")
        target_rel_names = {p.name for p in target}
        ranked = self.store.rank(tokens, target_rel_names)
        best: Optional[TransferResult] = None
        for _score, schema in ranked[:self.max_bases]:
            result = self.mapper.map(schema.relations, target)
            cand = self._novel_inferences(result, target)
            if result.structural_score >= self.min_score and cand:
                tr = TransferResult(schema.name, dict(result.entity_mapping), cand,
                                    result.structural_score, result.systematicity)
                if best is None or tr.structural_score > best.structural_score:
                    best = tr
        return best

    @staticmethod
    def _novel_inferences(result: AnalogyResult,
                          target: Sequence[Predicate]) -> List[Predicate]:
        """Keep only projected inferences that add something the target didn't already state."""
        have = {str(t) for t in target}
        return [p for p in result.candidate_inferences if str(p) not in have]

    # ---- best-effort text → relational structure (honest, conservative) ---- #
    def _extract(self, query: str) -> List[Predicate]:
        """Recover relation(entity, entity) facts from free text using the store's known
        relation vocabulary. Conservative by design: emits a relation only when a known
        relation word sits between two entity-like tokens. Yields [] when it can't — the
        engine then declines rather than inventing structure."""
        text = (query or "").lower()
        toks = _WORD.findall(text)
        if len(toks) < 3:
            return []
        # the union of relation names across known domains is our relation lexicon
        rel_vocab: set = set()
        for name in self.store.names():
            s = self.store.get(name)
            if s is not None:
                rel_vocab |= {n.lower() for n in s.relation_names()}
        rels: List[Predicate] = []
        for i, tok in enumerate(toks):
            if tok in rel_vocab:
                left = _prev_entity(toks, i)
                right = _next_entity(toks, i)
                if left and right and left != right:
                    rels.append(relation(tok, entity(left), entity(right)))
        # dedupe, preserve order
        seen: set = set()
        out: List[Predicate] = []
        for r in rels:
            if str(r) not in seen:
                seen.add(str(r))
                out.append(r)
        return out


def _prev_entity(toks: Sequence[str], i: int) -> Optional[str]:
    for j in range(i - 1, -1, -1):
        if toks[j] not in _STOP and len(toks[j]) > 1:
            return toks[j]
    return None


def _next_entity(toks: Sequence[str], i: int) -> Optional[str]:
    for j in range(i + 1, len(toks)):
        if toks[j] not in _STOP and len(toks[j]) > 1:
            return toks[j]
    return None


# --------------------------------------------------------------------------- #
# Seed library — a handful of well-understood domains with real relational glue
# --------------------------------------------------------------------------- #
def seed_library() -> List[DomainSchema]:
    e, r = entity, relation

    # Solar system — the classic base for the Rutherford atom analogy. The higher-order
    # ``causes`` relation is the systematic glue that gets *projected* into a target.
    sun, planet = e("sun"), e("planet")
    solar = DomainSchema(
        "solar_system",
        [
            r("attracts", sun, planet),
            r("more_massive", sun, planet),
            r("revolves_around", planet, sun),
            r("causes", r("attracts", sun, planet), r("revolves_around", planet, sun)),
        ],
        frozenset({"gravity", "orbit", "orbits", "star", "mass", "space", "celestial"}),
    )

    # Fluid flow through a pipe — the base for electrical-circuit reasoning (V↔pressure,
    # I↔flow, R↔narrowness). Ohm's-law structure as higher-order cause.
    pressure, flow, narrowness = e("pressure"), e("flow"), e("narrowness")
    fluid = DomainSchema(
        "fluid_flow",
        [
            r("drives", pressure, flow),
            r("impedes", narrowness, flow),
            r("causes", r("drives", pressure, flow), r("increases", pressure, flow)),
        ],
        frozenset({"water", "pipe", "hydraulic", "liquid", "pump", "tank"}),
    )

    # Predator–prey population dynamics — a base for any regulated feedback system.
    predator, prey = e("predator"), e("prey")
    ecosystem = DomainSchema(
        "predator_prey",
        [
            r("consumes", predator, prey),
            r("reduces", predator, prey),
            r("sustains", prey, predator),
            r("causes", r("consumes", predator, prey), r("reduces", predator, prey)),
        ],
        frozenset({"population", "species", "ecology", "food", "hunt", "wolves", "rabbits"}),
    )

    # Supply & demand — a base for market / resource-allocation reasoning.
    supply, demand, price = e("supply"), e("demand"), e("price")
    market = DomainSchema(
        "supply_demand",
        [
            r("raises", demand, price),
            r("lowers", supply, price),
            r("causes", r("raises", demand, price), r("clears", price, supply)),
        ],
        frozenset({"market", "economy", "cost", "buyers", "sellers", "goods", "trade"}),
    )

    return [solar, fluid, ecosystem, market]


# --------------------------------------------------------------------------- #
# Self-test / demo (offline, no LLM anywhere)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA relational-transfer self-test (her OWN faculties, no LLM)")
    print("=" * 70)

    e, r = entity, relation
    eng = RelationalTransferEngine()

    # The atom (a "new domain"): NYXARA is given only that the nucleus attracts the electron
    # and the electron revolves around it — NOT that the attraction *causes* the revolution.
    nucleus, electron = e("nucleus"), e("electron")
    atom = [
        r("attracts", nucleus, electron),
        r("revolves_around", electron, nucleus),
    ]
    tr = eng.generalize("model of the atom with a nucleus and an electron",
                        target_relations=atom)
    assert tr is not None, "should transfer from a known domain"
    print(f"\nbase domain         : {tr.base_domain}")
    assert tr.base_domain == "solar_system"
    print(f"entity mapping      : {tr.entity_mapping}")
    assert tr.entity_mapping.get("sun") == "nucleus"
    assert tr.entity_mapping.get("planet") == "electron"
    inferred = {str(p) for p in tr.candidate_inferences}
    print(f"candidate inferences: {sorted(inferred)}")
    # the higher-order CAUSE is projected — the predictive payoff NYXARA derived herself
    assert any("causes" in s and "nucleus" in s and "electron" in s for s in inferred)
    print("\n" + tr.render())

    # Honest decline: a query with no extractable relational structure yields None.
    assert eng.generalize("hi how are you today") is None
    print("\nno structure        : declines (None) — LLM path would run  ✓")

    # Learning a domain grows the store so it transfers next time.
    spring, mass = e("spring"), e("mass")
    eng.learn_domain("harmonic_oscillator",
                     [r("pulls", spring, mass),
                      r("oscillates", mass, spring),
                      r("causes", r("pulls", spring, mass), r("oscillates", mass, spring))],
                     keywords=["pendulum", "vibration"])
    assert eng.store.get("harmonic_oscillator") is not None
    lc, ind = e("inductor"), e("charge")
    tr2 = eng.generalize("an LC circuit",
                         target_relations=[r("pulls", lc, ind), r("oscillates", ind, lc)])
    assert tr2 is not None and tr2.base_domain == "harmonic_oscillator"
    print(f"learned-domain xfer : LC circuit ← {tr2.base_domain}  ✓")

    print("\nALL SELF-TESTS PASSED ✓")
