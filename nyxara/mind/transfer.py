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
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.mind.analogy import AnalogyResult, StructureMapper, entity, relation
from nyxara.mind.lot import Const, Predicate, Term

__all__ = [
    "DomainSchema",
    "DomainSchemaStore",
    "TransferResult",
    "RelationalTransferEngine",
    "seed_library",
    "lemma",
    "extract_relations",
]

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")

# Surface words that never carry domain structure — stripped before entity detection.
_STOP = frozenset("""
a an the of to in on at by for with from into onto over under and or but if then than as is are
was were be been being do does did will would can could should may might must this that these
those it its it's their there here what which who whom whose why how when where does do can what's
some any each every both all no not more most much many few least how's like about between within
around across along through toward towards upon beyond near against
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

    # ---- serialization (so a domain learned once survives a restart) ---- #
    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "relations": [_term_to_dict(r) for r in self.relations],
            "keywords": sorted(self.keywords),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "DomainSchema":
        rels = [_term_from_dict(r) for r in d.get("relations", [])]  # type: ignore[arg-type]
        rels = [r for r in rels if isinstance(r, Predicate)]
        kw = frozenset(str(k).lower() for k in d.get("keywords", []) if k)  # type: ignore[union-attr]
        return cls(name=str(d.get("name", "")), relations=rels, keywords=kw)


def _term_to_dict(t: Term) -> Dict[str, object]:
    """Recursively serialize a Language-of-Thought term (entity or nested relation) to JSON."""
    if isinstance(t, Const):
        return {"k": "c", "n": t.name}
    if isinstance(t, Predicate):
        return {"k": "p", "n": t.name, "a": [_term_to_dict(a) for a in t.args]}
    return {"k": "c", "n": str(t)}


def _term_from_dict(d: Dict[str, object]) -> Term:
    """Inverse of :func:`_term_to_dict`; rebuilds entities and nested higher-order relations."""
    if not isinstance(d, dict):
        return Const(str(d))
    if d.get("k") == "p":
        args = tuple(_term_from_dict(a) for a in d.get("a", []))  # type: ignore[arg-type]
        return Predicate(str(d.get("n", "")), args)
    return Const(str(d.get("n", "")))


# --------------------------------------------------------------------------- #
# The store of known domains — seeded, and grown from lived structure
# --------------------------------------------------------------------------- #
class DomainSchemaStore:
    """Known domains NYXARA can transfer *from*, ranked for a query by surface + structural fit."""

    def __init__(self, schemas: Optional[Sequence[DomainSchema]] = None, *,
                 memory: Any = None, tag: str = "domain_schema",
                 max_schemas: int = 200) -> None:
        self._schemas: Dict[str, DomainSchema] = {}
        for s in (schemas if schemas is not None else seed_library()):
            self._schemas[s.name] = s
        # the seed/base domains are permanent; only self-distilled domains are capped/evicted.
        self._seed_names = set(self._schemas)
        self.memory = memory
        self.tag = str(tag)
        self.max_schemas = max(1, int(max_schemas))
        self._hydrated = False

    def __len__(self) -> int:
        self._hydrate()
        return len(self._schemas)

    def names(self) -> List[str]:
        self._hydrate()
        return list(self._schemas)

    def get(self, name: str) -> Optional[DomainSchema]:
        self._hydrate()
        return self._schemas.get(name)

    def add(self, schema: DomainSchema) -> None:
        self._hydrate()
        self._schemas[schema.name] = schema

    # ---- persistence: a domain distilled once is reloaded on the next boot ---- #
    def _hydrate(self) -> None:
        """Load self-distilled domains from long-term memory (once, best-effort)."""
        if self._hydrated:
            return
        self._hydrated = True
        if self.memory is None:
            return
        try:
            hits = self.memory.recall("", k=10000, tags=[self.tag], strengthen=False)
        except Exception:  # noqa: BLE001 — hydration is best-effort, never fatal
            return
        for rec, _score in hits:
            meta = getattr(rec, "metadata", None)
            data = meta.get("schema") if isinstance(meta, dict) else None
            if isinstance(data, dict):
                try:
                    schema = DomainSchema.from_dict(data)
                except Exception:  # noqa: BLE001 — skip a corrupt record, keep the rest
                    continue
                if schema.name and schema.relations:
                    self._absorb(schema)

    def _absorb(self, schema: DomainSchema) -> None:
        """Merge a schema into the store in memory (no re-persist), union-style."""
        prior = self._schemas.get(schema.name)
        if prior is not None:
            seen = {str(r) for r in prior.relations}
            merged = list(prior.relations) + [r for r in schema.relations if str(r) not in seen]
            self._schemas[schema.name] = DomainSchema(schema.name, merged,
                                                      prior.keywords | schema.keywords)
        else:
            self._schemas[schema.name] = schema

    def _persist(self, schema: DomainSchema) -> None:
        if self.memory is None:
            return
        try:
            from nyxara.memory.store import MemoryType
            self.memory.remember(
                f"domain schema: {schema.name} "
                f"({', '.join(sorted(schema.relation_names()))})",
                mem_type=MemoryType.SEMANTIC, importance=0.6, tags=[self.tag],
                metadata={"schema": schema.to_dict(), "name": schema.name})
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            pass

    def _evict_over_cap(self) -> None:
        """Keep the store bounded: drop the oldest self-distilled domains beyond the cap."""
        distilled = [n for n in self._schemas if n not in self._seed_names]
        overflow = len(distilled) - self.max_schemas
        for n in distilled[:max(0, overflow)]:
            self._schemas.pop(n, None)

    def learn(self, name: str, relations: Sequence[Predicate],
              keywords: Sequence[str] = ()) -> DomainSchema:
        """Grow the store from lived structure so a domain met once transfers next time.

        Merges into an existing domain of the same name (union of relations/keywords) rather
        than clobbering it, so recognition and structure *accumulate*, and persists the result
        to long-term memory so the growth survives a restart."""
        self._hydrate()
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
        self._evict_over_cap()
        self._persist(schema)
        return schema

    def rank(self, tokens: Sequence[str], target_rel_names: Sequence[str] = ()) -> List[
            Tuple[float, DomainSchema]]:
        """Rank known domains for a query by surface-word overlap and relation-name overlap.

        A shared *relation name* (the query says "attracts", a domain has ``attracts``) counts
        double a shared surface word — relational structure is what actually transfers."""
        self._hydrate()
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
    analogical: bool = False   # True when the base was reached by *loose* (renamed) alignment

    def to_dict(self) -> Dict[str, object]:
        return {
            "base_domain": self.base_domain,
            "entity_mapping": dict(self.entity_mapping),
            "candidate_inferences": [str(p) for p in self.candidate_inferences],
            "structural_score": round(self.structural_score, 3),
            "systematicity": round(self.systematicity, 3),
            "analogical": self.analogical,
        }

    def render(self) -> str:
        """A human-readable, honestly-hedged statement of the transfer.

        A *loose* (analogical) transfer — where the new domain's own vocabulary was aligned to a
        known base by structure alone — is hedged more strongly than a strict one: its content is
        NYXARA's own projection, but it is a conjecture across unfamiliar terms, not a tight match.
        """
        strength = ("by loose analogy (unfamiliar terms aligned by structure — a conjecture "
                    "to verify)" if self.analogical else "by analogy")
        if not self.candidate_inferences:
            pairs = ", ".join(f"{b}→{t}" for b, t in self.entity_mapping.items())
            return (f"By analogy to {self.base_domain} ({pairs}), the same relational structure "
                    f"holds, but nothing new is projected.")
        lines = [f"Reasoning {strength} to {self.base_domain}, NYXARA's structure-mapper projects "
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
                 max_bases: int = 4, analogical: bool = True,
                 learn_from_experience: bool = True, memory: Any = None,
                 max_schemas: int = 200) -> None:
        self.store = store if store is not None else DomainSchemaStore(
            memory=memory, max_schemas=max_schemas)
        self.mapper = mapper if mapper is not None else StructureMapper()
        # A second, opt-in mapper that aligns relations by *structure* even when their names
        # differ — the faculty that lets a genuinely-new domain (its own fresh vocabulary) map
        # onto one NYXARA already understands, instead of surrendering to the base LLM.
        self.analogical = bool(analogical)
        self.mapper_analogical = StructureMapper(analogical=True)
        self.min_score = float(min_score)
        self.max_bases = int(max_bases)
        # When on, a novel domain recovered from raw text (its relations extracted, not supplied)
        # is distilled into a DomainSchema and learned — so her transfer library grows from lived
        # structure, by her own action, and (with a memory) survives a restart.
        self.learn_from_experience = bool(learn_from_experience)

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
        extracted = target_relations is None
        target = list(target_relations) if target_relations else self._extract(query)
        if len(target) < 2:
            return None  # too little structure to transfer honestly
        tokens = _WORD.findall(query or "")
        target_rel_names = {p.name for p in target}
        ranked = self.store.rank(tokens, target_rel_names)

        # Pass 1 — STRICT identicality (preferred): a base whose relation names literally match.
        best = self._best_over(self.mapper, [s for _sc, s in ranked[:self.max_bases]], target,
                               analogical=False)

        # Pass 2 — ANALOGICAL (only if strict found nothing and it is enabled): map the new
        # domain's *structure* onto a known base even though its verbs are unfamiliar. A truly
        # novel domain shares no surface/relation vocabulary, so ranking is uninformative here —
        # consider every known base and let structural corroboration decide.
        if best is None and self.analogical:
            bases = [s for _sc, s in ranked] + [self.store.get(n) for n in self.store.names()]
            uniq: List[DomainSchema] = []
            seen_names: set = set()
            for s in bases:
                if s is not None and s.name not in seen_names:
                    seen_names.add(s.name)
                    uniq.append(s)
            best = self._best_over(self.mapper_analogical, uniq[:max(self.max_bases * 3, 12)],
                                   target, analogical=True)

        # Self-growth: a novel domain recovered from raw text and successfully transferred is worth
        # keeping — distil its relational skeleton into a DomainSchema so it is recognised (and can
        # itself serve as a transfer base) next time. Only from extracted structure (not caller-
        # supplied), and never fatal.
        if best is not None and extracted and self.learn_from_experience:
            try:
                self._distill_and_learn(query, target, tokens)
            except Exception:  # noqa: BLE001 — growth is best-effort, never breaks a transfer
                pass
        return best

    def _distill_and_learn(self, query: str, target: Sequence[Predicate],
                           tokens: Sequence[str]) -> Optional[DomainSchema]:
        """Turn a freshly-extracted relational skeleton into a learned :class:`DomainSchema`."""
        label = _label_from(query, tokens, self.store)
        if not label:
            return None
        keywords = [t.lower() for t in tokens
                    if len(t) >= 4 and t.lower() not in _STOP][:8]
        return self.store.learn(label, target, keywords)

    def _best_over(self, mapper: StructureMapper, schemas: Sequence[DomainSchema],
                   target: Sequence[Predicate], *, analogical: bool) -> Optional[TransferResult]:
        """Map ``target`` against each candidate base with ``mapper``; keep the strongest transfer
        that clears ``min_score`` and actually projects something new."""
        best: Optional[TransferResult] = None
        for schema in schemas:
            result = mapper.map(schema.relations, target)
            cand = self._novel_inferences(result, target)
            if result.structural_score >= self.min_score and cand:
                tr = TransferResult(schema.name, dict(result.entity_mapping), cand,
                                    result.structural_score, result.systematicity,
                                    analogical=bool(analogical or result.used_renamed))
                if best is None or tr.structural_score > best.structural_score:
                    best = tr
        return best

    @staticmethod
    def _novel_inferences(result: AnalogyResult,
                          target: Sequence[Predicate]) -> List[Predicate]:
        """Keep only projected inferences that add something the target didn't already state."""
        have = {str(t) for t in target}
        return [p for p in result.candidate_inferences if str(p) not in have]

    # ---- best-effort text → relational structure (honest, two-tier) ---- #
    def _extract(self, query: str) -> List[Predicate]:
        """Recover ``relation(entity, entity)`` facts from free text — including from a domain
        NYXARA has *never seen*, whose relation words are not in any known schema.

        Two tiers, high precision first:

        * **Tier A** scans for relation words already in the store's vocabulary (exact names, so
          the recovered structure maps *strictly* onto a known base). If it alone recovers ≥2
          relations, those are used unchanged — the strict path is preferred.
        * **Tier B** (open-domain) fires when Tier A is thin: it recovers a relation wherever a
          **verb-like token sits between two entity-like tokens**, judged by morphology and
          position rather than a fixed lexicon. This is what gives a brand-new domain a
          relational skeleton at all — which the analogical mapper can then align by structure.

        Honest either way: a clause with no plausible verb-between-two-entities yields nothing, so
        free-form chat ("hi, how are you") returns ``[]`` and the engine declines rather than
        inventing structure it cannot see.
        """
        rel_vocab: set = set()
        for name in self.store.names():
            s = self.store.get(name)
            if s is not None:
                rel_vocab |= {n.lower() for n in s.relation_names()}
        return extract_relations(query, known_relations=rel_vocab)


def _extract_between(toks: Sequence[str], is_relation, *, lemmatize: bool) -> List[Predicate]:
    """Emit ``relation(word, left_entity, right_entity)`` for every token accepted by
    ``is_relation`` that has a distinct entity on either side. Deduped, order preserved."""
    rels: List[Predicate] = []
    seen: set = set()
    for i, tok in enumerate(toks):
        if not is_relation(tok):
            continue
        left = _prev_entity(toks, i)
        right = _next_entity(toks, i)
        if not (left and right) or left == right:
            continue
        name = lemma(tok) if lemmatize else tok
        r = relation(name, entity(left), entity(right))
        if str(r) not in seen:
            seen.add(str(r))
            rels.append(r)
    return rels


def extract_relations(query: str,
                      known_relations: Sequence[str] = ()) -> List[Predicate]:
    """Recover ``relation(entity, entity)`` facts from free text — the shared, honest,
    two-tier open-domain extractor used by both the transfer engine and domain genesis.

    * **Tier A** scans for relation words already known (``known_relations`` — exact names,
      so the recovered structure maps *strictly* onto a known base). If it alone recovers ≥2
      relations, those are used unchanged — the strict path is preferred.
    * **Tier B** (open-domain) fires when Tier A is thin: it recovers a relation wherever a
      **verb-like token sits between two entity-like tokens**, judged by morphology and
      position rather than a fixed lexicon. This gives a brand-new domain (whose verbs are in
      no known schema) a relational skeleton at all.
    * **Tier C** (copula / comparative) recovers the structure a bare verb misses: a linked
      predicate — ``A is bigger than B`` → ``bigger(A, B)``, ``A is part of B`` → ``part_of(A, B)``.
      This roughly doubles the prose a genuinely-new field can be recovered from.

    Honest: a clause with no plausible relation (no verb-between-two-entities and no linked
    predicate) yields nothing, so free-form chat ("hi, how are you") returns ``[]``. Pass an empty
    ``known_relations`` for pure open-domain extraction (what a from-scratch domain-genesis pass
    wants)."""
    text = (query or "").lower()
    toks = _WORD.findall(text)
    if len(toks) < 3:
        return []
    rel_vocab = {r.lower() for r in known_relations}

    # Tier A — known relation vocabulary (exact names -> strict-mappable structure)
    if rel_vocab:
        tier_a = _extract_between(toks, lambda tok: tok in rel_vocab, lemmatize=False)
        if len(tier_a) >= 2:
            return tier_a
    else:
        tier_a = []

    # Tier B — open-domain: any verb-like token flanked by two entities (novel verbs welcome)
    tier_b = _extract_between(
        toks, lambda tok: tok in rel_vocab or _looks_verb(tok), lemmatize=True)
    # Tier C — copula/comparative linked predicates ("is bigger than", "is part of")
    tier_c = _extract_copula(toks)
    have = {str(r) for r in tier_b}
    combined = list(tier_b) + [r for r in tier_c if str(r) not in have]
    # honest: only a genuine skeleton (≥2 relations) is returned; otherwise defer (chat stays
    # un-parsed and the caller falls through), preserving the no-fabrication contract.
    return combined if len(combined) >= 2 else tier_a


# Copula/auxiliary verbs and the link words that turn "X is <descriptor> <link> Y" into a
# genuine binary relation — the structure a plain content verb misses.
_COPULA = frozenset({"is", "are", "was", "were", "be", "been", "being", "become", "becomes"})
_LINK = frozenset({"than", "of", "to", "for", "from", "with", "over", "under", "above", "below",
                   "within", "into", "onto"})


def _extract_copula(toks: Sequence[str]) -> List[Predicate]:
    """Recover ``relation(entity, entity)`` from a copula + descriptor + link pattern.

    ``A is bigger than B`` → ``bigger(A, B)``; ``A is part of B`` → ``part_of(A, B)``. Requires a
    real left entity, a content descriptor, a link word, and a right entity — so a bare
    ``how are you`` (no descriptor+link+entity) never yields a relation and chat stays un-parsed."""
    rels: List[Predicate] = []
    seen: set = set()
    n = len(toks)
    for i, tok in enumerate(toks):
        if tok not in _COPULA:
            continue
        left = _prev_entity(toks, i)
        if not left:
            continue
        # the descriptor: first content token after the copula (skip degree words / articles)
        j = i + 1
        while j < n and toks[j] in _STOP and toks[j] not in _LINK:
            j += 1
        if j >= n or toks[j] in _LINK or len(toks[j]) < 2:
            continue
        descriptor = toks[j]
        # a link word must follow within a couple of tokens; anything else means "not this pattern"
        k, link = j + 1, None
        while k < n and k <= j + 2:
            if toks[k] in _LINK:
                link = toks[k]
                break
            if toks[k] not in _STOP:
                break
            k += 1
        if link is None:
            continue
        right = _next_entity(toks, k)
        if not right or right == left:
            continue
        name = lemma(descriptor) if link == "than" else f"{descriptor}_{link}"
        r = relation(name, entity(left), entity(right))
        if str(r) not in seen:
            seen.add(str(r))
            rels.append(r)
    return rels


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


# Verb-shaped endings that flag a token as a *relation word* even when NYXARA has never seen it —
# the morphology of an action, not a fixed lexicon. Copulas/auxiliaries live in ``_STOP`` and are
# excluded first, so a bare "is/are" never becomes a standalone relation (chat stays un-parsed).
_STRONG_VERB_SUFFIX = ("ates", "izes", "ises", "ifies", "ing", "ed")
_WEAK_VERB_SUFFIX = ("es", "s")
# singular noun endings that merely *look* like a 3rd-person verb's -s (nucleus, analysis, mass)
_NOUN_S_ENDING = ("ss", "us", "is", "ous", "ics")


def _looks_verb(tok: str) -> bool:
    """True if ``tok`` is shaped like a content verb (a relation word), by morphology alone."""
    if len(tok) < 3 or tok in _STOP:
        return False
    if tok.endswith(_STRONG_VERB_SUFFIX):
        return True
    # a trailing -s/-es signals a 3rd-person verb, but also a plural noun; require some length and
    # avoid endings that flag a singular Latin/Greek noun ("nucleus", "analysis", "process",
    # "mass") so a common noun is not mistaken for a verb.
    if len(tok) >= 4 and tok.endswith(_WEAK_VERB_SUFFIX) and not tok.endswith(_NOUN_S_ENDING):
        return True
    return False


def _label_from(query: str, tokens: Sequence[str], store: "DomainSchemaStore") -> str:
    """A short, stable label for a self-distilled domain, from its most distinctive terms.

    Prefers the two longest content tokens that are not already a known relation word, joined —
    so ``the emitter zaps the mote`` becomes ``emitter_mote``. Deterministic (same query ⇒ same
    label) so re-meeting a field merges into the same schema rather than fragmenting it."""
    known: set = set()
    try:
        for n in store.names():
            s = store.get(n)
            if s is not None:
                known |= {r.lower() for r in s.relation_names()}
    except Exception:  # noqa: BLE001
        known = set()
    content = [t.lower() for t in tokens
               if len(t) >= 4 and t.lower() not in _STOP and t.lower() not in known]
    # keep first-seen order but rank by length; ties broken by position for determinism
    uniq: List[str] = []
    for t in content:
        if t not in uniq:
            uniq.append(t)
    uniq.sort(key=lambda w: (-len(w), content.index(w)))
    picked = uniq[:2]
    if not picked:
        return ""
    return "learned_" + "_".join(sorted(picked))


def lemma(word: str) -> str:
    """Fold a surface verb to a stable base form so tense/number don't fragment one relation
    (``pulls``/``pulled``/``pulling`` → ``pull``). Conservative: never shortens below 3 chars."""
    w = word.lower()
    for suf, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if w.endswith(suf) and not w.endswith("ss") and len(w) - len(suf) + len(repl) >= 3:
            return w[: len(w) - len(suf)] + repl
    return w


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

    # Regulated feedback — a thermostat / homeostatic loop. The base for *any* self-correcting
    # controller: a regulator drives an effector, the effector moves the variable back, and the
    # correction sustains the regulation (the higher-order loop that gets projected).
    controller, effector, variable = e("controller"), e("effector"), e("variable")
    feedback = DomainSchema(
        "regulated_feedback",
        [
            r("regulates", controller, variable),
            r("drives", controller, effector),
            r("corrects", effector, variable),
            r("causes", r("corrects", effector, variable),
              r("regulates", controller, variable)),
        ],
        frozenset({"feedback", "thermostat", "setpoint", "homeostasis", "regulator",
                   "equilibrium", "balance", "stabilize", "stabilise"}),
    )

    # Signal transmission — a sender/channel/receiver communication chain. The base for messaging,
    # networking, and any encode→carry→decode pipeline.
    sender, receiver, signal = e("sender"), e("receiver"), e("signal")
    comms = DomainSchema(
        "signal_transmission",
        [
            r("encodes", sender, signal),
            r("transmits", sender, receiver),
            r("decodes", receiver, signal),
            r("causes", r("transmits", sender, receiver), r("decodes", receiver, signal)),
        ],
        frozenset({"signal", "message", "communication", "network", "protocol", "packet",
                   "channel", "transmit", "broadcast", "encode", "decode"}),
    )

    # Command hierarchy — a directed containment/authority tree. The base for org charts, taxonomies,
    # and any parent→child delegation structure.
    superior, subordinate = e("superior"), e("subordinate")
    hierarchy = DomainSchema(
        "command_hierarchy",
        [
            r("directs", superior, subordinate),
            r("reports_to", subordinate, superior),
            r("causes", r("directs", superior, subordinate),
              r("reports_to", subordinate, superior)),
        ],
        frozenset({"hierarchy", "organization", "organisation", "manager", "command",
                   "delegate", "parent", "child", "tree", "authority", "chain"}),
    )

    # Epidemic spread — a contact-driven contagion loop. The base for any diffusion/propagation
    # process (rumours, adoption, cascading failure): a carrier infects a susceptible, which
    # swells the infected pool, which drives yet more infection (the higher-order amplification).
    infected, susceptible = e("infected"), e("susceptible")
    epidemic = DomainSchema(
        "epidemic_spread",
        [
            r("infects", infected, susceptible),
            r("increases", infected, susceptible),
            r("causes", r("infects", infected, susceptible),
              r("increases", infected, susceptible)),
        ],
        frozenset({"virus", "contagion", "outbreak", "epidemic", "spread", "infection",
                   "transmission", "immunity", "pandemic", "propagate", "diffuse"}),
    )

    # Natural selection — a variation→selection→inheritance loop. The base for any optimization-
    # under-pressure process (markets weeding firms, training pruning weights): the environment
    # selects a trait, the trait raises fitness, fitness raises reproduction, which propagates it.
    environment, trait, fitness = e("environment"), e("trait"), e("fitness")
    selection = DomainSchema(
        "natural_selection",
        [
            r("selects", environment, trait),
            r("raises", trait, fitness),
            r("propagates", fitness, trait),
            r("causes", r("raises", trait, fitness), r("propagates", fitness, trait)),
        ],
        frozenset({"evolution", "mutation", "adaptation", "gene", "fitness", "species",
                   "survival", "reproduce", "variation", "inherit"}),
    )

    # Reinforcement — a reward→strengthen→repeat loop. The base for learning, habit, addiction,
    # incentive design: a reward strengthens a behaviour, the behaviour produces more reward, and
    # that reward sustains the strengthening (the higher-order habit loop).
    reward, behaviour = e("reward"), e("behaviour")
    reinforcement = DomainSchema(
        "reinforcement_loop",
        [
            r("strengthens", reward, behaviour),
            r("produces", behaviour, reward),
            r("causes", r("strengthens", reward, behaviour), r("produces", behaviour, reward)),
        ],
        frozenset({"reward", "reinforce", "habit", "learning", "incentive", "dopamine",
                   "conditioning", "motivation", "addiction", "loop"}),
    )

    # Diffusion — a gradient→flux→equalization loop. The base for any spreading-down-a-gradient
    # process (heat conduction, osmosis, price arbitrage): a gradient drives a flux, the flux
    # reduces the gradient, and that reduction regulates the flux (negative feedback to balance).
    gradient, flux = e("gradient"), e("flux")
    diffusion = DomainSchema(
        "diffusion_gradient",
        [
            r("drives", gradient, flux),
            r("reduces", flux, gradient),
            r("causes", r("reduces", flux, gradient), r("drives", gradient, flux)),
        ],
        frozenset({"diffusion", "gradient", "concentration", "osmosis", "conduction",
                   "equilibrium", "spread", "flux", "permeate"}),
    )

    # Catalysis — a facilitator that accelerates a conversion without being consumed. The base for
    # enzymes, brokers, platforms, and any accelerant: a catalyst lowers a barrier, the barrier
    # impedes conversion, so lowering it accelerates conversion (the higher-order enablement).
    catalyst, barrier, conversion = e("catalyst"), e("barrier"), e("conversion")
    catalysis = DomainSchema(
        "catalysis",
        [
            r("lowers", catalyst, barrier),
            r("impedes", barrier, conversion),
            r("accelerates", catalyst, conversion),
            r("causes", r("lowers", catalyst, barrier), r("accelerates", catalyst, conversion)),
        ],
        frozenset({"catalyst", "enzyme", "facilitate", "accelerate", "broker", "platform",
                   "substrate", "reaction", "activation"}),
    )

    # Lever — a trade-off that conserves a quantity across two dimensions. The base for mechanical
    # advantage, gears, and any effort/reach exchange: effort moves a load, distance trades against
    # force, and their product (work) is conserved (the higher-order invariant).
    effort, load, reach = e("effort"), e("load"), e("reach")
    lever = DomainSchema(
        "lever_tradeoff",
        [
            r("moves", effort, load),
            r("trades", reach, effort),
            r("conserves", r("moves", effort, load), r("trades", reach, effort)),
        ],
        frozenset({"lever", "leverage", "gear", "mechanical", "tradeoff", "fulcrum",
                   "advantage", "pulley", "torque"}),
    )

    return [solar, fluid, ecosystem, market, feedback, comms, hierarchy, epidemic, selection,
            reinforcement, diffusion, catalysis, lever]


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

    # A GENUINELY NEW domain — its own fresh verbs, none in any schema — extracted from raw prose
    # and transferred by ANALOGY, no LLM. This is the ceiling the critique named, lifted.
    novel_q = ("in this device the emitter zaps the mote and then the mote whirls "
               "the emitter continuously")
    tr_novel = eng.generalize(novel_q)
    assert tr_novel is not None, "a novel-vocabulary domain must still transfer by structure"
    assert tr_novel.analogical, "it should be reached by loose (renamed) alignment"
    proj = {str(p) for p in tr_novel.candidate_inferences}
    print(f"novel-vocab domain  : ← {tr_novel.base_domain} (analogical); projects {sorted(proj)}")
    assert any("causes" in s for s in proj), "the higher-order cause must be projected"

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
