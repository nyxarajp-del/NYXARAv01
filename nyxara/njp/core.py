"""NYXARA · njp/core.py — the Cognitive Learning Core (🧩, NJP V.05).

Every other organ in NJP is a *component* of intelligence. This one is the loop that turns those
components into learning, and it exists because measuring the components honestly showed that
having all of them is not the same as having the loop.

**What was actually missing.** Three findings, each reproduced before it was believed:

* **Nothing composed.** ``grounder.answer`` is a single dict lookup on ``(subject, predicate)``.
  A brain told ``aag causes garmi`` and ``garmi causes pasina`` could answer neither *"what does
  aag cause besides garmi"* nor *"why pasina"* — the two facts sat in the same store and were
  never put next to each other. A grep for transitive closure across ``nyxara/njp/`` returned
  nothing. The one multi-hop mechanism in the package, :meth:`nyxara.njp.universe.InternalUniverse.intervene`,
  needs both arrows numerically *fitted*, so it cannot touch a symbolic chain at all.
* **Nothing variabilised.** :mod:`nyxara.njp.discover` mines association rules over **literal
  strings**: ``{water} → growth`` and ``{light} → growth`` are two unrelated tuples, and nothing
  in that module can notice they share a consequent, because an ``Abstraction`` has no slot, no
  role and no variable. Meanwhile :mod:`nyxara.njp.concepts` *does* cluster subjects into kinds
  with real invariants — and :meth:`~nyxara.njp.concepts.ConceptGenesis.generalise`, the only
  defeasible-transfer method in the package, had **zero production callers**. The machinery for
  abstraction and the machinery for rules were both present and had never been introduced.
* **Nothing retried.** Both diagnosis layers — :meth:`nyxara.njp.predict.PredictionEngine.diagnose`
  and :meth:`nyxara.njp.field.RecursiveCognitiveField._diagnose` — end by *updating an organ*:
  widen the fabric, re-parse, replay a transition, move a clustering knob. Not one of them
  re-answers the question. A repair whose effect on the answer is never measured is a hope.

**The loop this closes**, and every arrow is a method below::

    EXPERIENCE → REPRESENT → PREDICT → EXPLAIN → SIMULATE
              → TEST → REVISE → GENERALIZE → new situation

**Schemas, and why they are the unit.** A :class:`Schema` is a relation with its subject slot
replaced by a *role* — ``⟨kind-3⟩ --increases--> growth`` rather than ``water --increases-->
growth``. It is induced only where two or more subjects that :mod:`nyxara.njp.concepts` has
independently clustered together share a relation, so the role is never a category anyone typed
into this file; it is whatever the clustering found. That is the difference between abstraction
and a synonym list, and it is why there is no ontology in this module.

**Held-out is enforced, not offered.** A schema is induced from the *fit* half of the fact store
and scored on the *held-out* half, split by a stable hash of the triple so the two folds are the
same across restarts. A schema that only predicts the facts that suggested it scores zero here
and dies. This is the whole reason the module can claim to measure generalisation rather than
recall: **she is never tested on what taught her.**

**Composition is conjecture, and says so.** ``A causes B`` and ``B causes C`` licence ``A causes
C`` — usually. Transitivity is a property of a relation and this module does not pretend to know
which relations have it: :class:`Transitivity` holds a per-predicate posterior that starts from a
weak prior and moves on evidence, every composed answer is capped below the KNOWN threshold, and
a chain's confidence is the product of its links so a long derivation is worth less than a short
one. A composed answer that later turns out wrong lowers that predicate's transitivity, which is
the only way this could be learning rather than a rule someone wrote down.

**Sovereignty is unchanged.** The Core only ever *proposes* — it returns a :class:`Derivation`
that the brain may use as an answer, and every such answer still goes through the Truth-Seeking
Gauntlet and the kernel's gate like any other. Nothing here writes to the fact store, edits
source, or touches the safety core.

Pure standard library. Every public entry point is fail-soft: on error it degrades to a null
result rather than breaking a turn.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Collection, Dict, List, Optional, Sequence, Set, Tuple

from nyxara.njp.canon import canonical_entity

__all__ = [
    "Derivation", "Schema", "Transitivity",
    "RepresentReport", "TestReport", "ReviseReport", "CoreReport",
    "CognitiveLearningCore",
]


# --------------------------------------------------------------------------- #
# Priors
# --------------------------------------------------------------------------- #
#
# A weak prior on which relations chain, not a claim about which ones do. Every entry is a
# starting belief that :class:`Transitivity` moves on evidence, and the values are deliberately
# below the KNOWN floor so that no composed answer is ever stated as fact on the strength of
# this table alone. `causes` is the highest because a causal chain is the case this module was
# written for; `likes` is absent entirely because liking is not transitive and starting it at
# even a weak positive would licence "Jay likes tea, tea likes ..." shaped nonsense.
_TRANSITIVE_PRIOR: Dict[str, float] = {
    "causes": 0.62,
    "part_of": 0.70,
    "is_a": 0.66,
    "requires": 0.55,
    "produces": 0.50,
    "located_in": 0.48,
}
#: What an unlisted relation starts at. Low, and deliberately not zero: a relation nobody
#: anticipated should be *allowed* to prove itself transitive, just not assumed to be.
_TRANSITIVE_DEFAULT = 0.25

#: A composed answer can never be stated as knowledge, however good the chain. It is an inference
#: from facts rather than one of them, and the distinction is the point.
_COMPOSED_CEILING = 0.70

#: How much a link costs. Each additional hop multiplies confidence by the relation's own
#: transitivity, so depth is priced rather than merely capped.
_MIN_LINK_CONFIDENCE = 0.05


def _norm(text: Any) -> str:
    """The one spelling rule for a name, so two paths to it reach the same key.

    It must be the *same* rule :class:`~nyxara.njp.grounding.Grounder` files facts under, and for
    a while it was not: this was ``.strip().lower()`` while the store keyed on the surface form
    too, so both halves agreed and both halves were wrong together. The moment the store learned
    that ``birds`` and ``bird`` are one entity, this had to learn it as well — :meth:`_inherit`
    reads ``self._facts()`` directly, so a kind reached through a stated ``is_a`` edge would look
    up a key the store no longer writes, and multi-level inheritance would break in exactly the
    way canonicalisation was introduced to fix.
    """
    return canonical_entity(text)


#: How a question says "not that one". Everything after the marker, up to the end or a comma, is
#: an answer she has already been given and is not being asked for again.
#:
#: This is a small grammar rather than a general one, and deliberately: the words below are the
#: ones that exclude a *named value*. "instead of" is absent because it usually replaces the
#: subject rather than the answer, and reading it as an exclusion would silently drop the wrong
#: half of the question.
_EXCLUSION = re.compile(
    r"\b(?:besides|apart\s+from|other\s+than|except(?:\s+for)?|aside\s+from)\s+"
    r"(?P<v>[^,?.;]+)", re.IGNORECASE)


def _exclusions(question: str) -> Set[str]:
    """Answers the question has ruled out, normalised the way a stored object is."""
    try:
        return {_norm(m.group("v")) for m in _EXCLUSION.finditer(str(question or ""))
                if _norm(m.group("v"))}
    except Exception:  # noqa: BLE001
        return set()


def _strip_exclusions(text: str) -> str:
    """The question with its exclusion clause removed, so the subject parse is not polluted."""
    try:
        return _norm(_EXCLUSION.sub("", str(text or "")))
    except Exception:  # noqa: BLE001
        return _norm(text)


def _fold(key: str, *, share: float) -> bool:
    """Is this key in the held-out fold? Stable across restarts and machines.

    Hashed rather than sampled, for the reason the rest of the repo hashes its splits: a random
    split reshuffles on every process start, so a schema refuted yesterday is scored against
    different evidence today and its history stops meaning anything. Same fact, same fold,
    forever.
    """
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=4).hexdigest()
    return (int(digest, 16) % 1000) < int(max(0.0, min(1.0, share)) * 1000)


# --------------------------------------------------------------------------- #
# What an answer is, once it has been derived rather than looked up
# --------------------------------------------------------------------------- #

@dataclass
class Derivation:
    """An answer with its working attached — the steps, not a story about them.

    ``kind`` names *how* it was got, and the three are genuinely different epistemic objects:
    ``direct`` is a fact she was told, ``composed`` is an inference from two or more such facts,
    and ``schema`` is a defeasible transfer from a kind to a member that was never observed to
    have the property. Only the first is ever eligible to be called knowledge.
    """

    answer: str = ""
    kind: str = ""                      # direct | composed | schema | simulated
    confidence: float = 0.0
    steps: List[str] = field(default_factory=list)
    support: List[Tuple[str, str, str]] = field(default_factory=list)
    role: str = ""                      # the concept a schema answer came through
    why: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.answer) and self.confidence > 0.0

    @property
    def defeasible(self) -> bool:
        """Could this be wrong even though every fact behind it is right?"""
        return self.kind in ("composed", "schema")

    def to_dict(self) -> Dict[str, Any]:
        return {"answer": self.answer, "kind": self.kind,
                "confidence": round(self.confidence, 3), "steps": self.steps[:8],
                "support": [list(s) for s in self.support[:8]], "role": self.role,
                "why": self.why, "defeasible": self.defeasible}


# --------------------------------------------------------------------------- #
# Learned structure
# --------------------------------------------------------------------------- #

@dataclass
class Schema:
    """A relation with its subject replaced by a role: ``⟨role⟩ --predicate--> object``.

    The unit of generalisation here, and the thing :mod:`nyxara.njp.discover` structurally cannot
    represent. ``members`` are the subjects observed to satisfy it; ``role`` is a concept id from
    :mod:`nyxara.njp.concepts`, so the grouping is one the clustering found rather than one this
    file declared.
    """

    role: str = ""
    predicate: str = ""
    object: str = ""
    members: Set[str] = field(default_factory=set)
    hits: int = 0                       # held-out facts it predicted correctly
    misses: int = 0                     # held-out facts it got wrong
    born: float = field(default_factory=time.time)
    tested: int = 0
    # The members whose held-out fact actually contradicted this schema, and what they had
    # instead. Kept because "this generalisation is wrong" is not a question anyone can answer,
    # and "you said every X is Y, but this X is Z — which is it?" is.
    refuted_by: Dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.role, self.predicate, self.object)

    @property
    def trials(self) -> int:
        return self.hits + self.misses

    @property
    def precision(self) -> float:
        """Laplace-smoothed, so one lucky held-out hit is not a precision of 1.0."""
        return (self.hits + 1.0) / (self.trials + 2.0)

    @property
    def status(self) -> str:
        if self.trials < 2:
            return "conjectured"
        if self.precision >= 0.70:
            return "confirmed"
        if self.precision < 0.40:
            return "refuted"
        return "conjectured"

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "predicate": self.predicate, "object": self.object,
                "members": sorted(self.members)[:12], "hits": self.hits,
                "misses": self.misses, "precision": round(self.precision, 3),
                "status": self.status}


@dataclass
class Transitivity:
    """Per-predicate posterior on "does this relation chain?", moved by outcomes.

    A Beta posterior in all but name: ``alpha``/``beta`` start from the prior above scaled by a
    weak pseudo-count, so early evidence moves the estimate a lot and later evidence moves it
    little. The point of learning this rather than tabling it is that a wrong composed answer has
    somewhere to go — without it, a bad chain is a one-off mistake she is free to repeat forever.
    """

    predicate: str = ""
    alpha: float = 1.0
    beta: float = 1.0

    @property
    def value(self) -> float:
        total = self.alpha + self.beta
        return self.alpha / total if total > 0 else _TRANSITIVE_DEFAULT

    def confirm(self, weight: float = 1.0) -> None:
        self.alpha += max(0.0, weight)

    def refute(self, weight: float = 1.0) -> None:
        self.beta += max(0.0, weight)

    def to_dict(self) -> Dict[str, Any]:
        return {"predicate": self.predicate, "value": round(self.value, 3),
                "observations": round(self.alpha + self.beta - 2.0, 1)}


@dataclass
class RepresentReport:
    """What one REPRESENT pass built."""

    proposed: int = 0
    kept: int = 0
    roles: int = 0
    members_covered: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"proposed": self.proposed, "kept": self.kept, "roles": self.roles,
                "members_covered": self.members_covered}


@dataclass
class TestReport:
    """What one TEST pass measured, on facts no schema here was built from."""

    scored: int = 0
    hits: int = 0
    misses: int = 0
    holdout_size: int = 0

    @property
    def accuracy(self) -> Optional[float]:
        return (self.hits / self.scored) if self.scored else None

    def to_dict(self) -> Dict[str, Any]:
        return {"scored": self.scored, "hits": self.hits, "misses": self.misses,
                "holdout": self.holdout_size,
                "accuracy": None if self.accuracy is None else round(self.accuracy, 3)}


@dataclass
class ReviseReport:
    """What one REVISE pass changed. A pass that changes nothing reports nothing."""

    refuted: int = 0
    dropped: int = 0
    transitivity_moved: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"refuted": self.refuted, "dropped": self.dropped,
                "transitivity_moved": self.transitivity_moved}


@dataclass
class CoreReport:
    """One turn through the Core."""

    turn: int = 0
    represented: Optional[RepresentReport] = None
    tested: Optional[TestReport] = None
    revised: Optional[ReviseReport] = None
    derived: Optional[Derivation] = None
    question: str = ""
    ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"turn": self.turn,
                "represent": self.represented.to_dict() if self.represented else None,
                "test": self.tested.to_dict() if self.tested else None,
                "revise": self.revised.to_dict() if self.revised else None,
                "derived": self.derived.to_dict() if self.derived else None,
                "question": self.question, "ms": round(self.ms, 3)}


# --------------------------------------------------------------------------- #
# The Core
# --------------------------------------------------------------------------- #

class CognitiveLearningCore:
    """REPRESENT · PREDICT · EXPLAIN · SIMULATE · TEST · REVISE · GENERALIZE.

    Holds no knowledge of its own. The fact store belongs to :mod:`nyxara.njp.grounding`, the
    kinds to :mod:`nyxara.njp.concepts` and the fitted quantities to :mod:`nyxara.njp.universe`;
    what lives here is the schemas induced over them, the transitivity posteriors, and the loop.
    That is deliberate — a Core with its own private copy of the facts would be a second brain to
    keep in step, which is the arrangement this package exists to not have.
    """

    def __init__(self, brain: Any = None, *, concepts: Any = None, grounder: Any = None,
                 universe: Any = None, world: Any = None, curiosity: Any = None,
                 max_depth: int = 4, min_members: int = 2, holdout_share: float = 0.3,
                 capacity: int = 512, represent_every: int = 4, test_every: int = 6) -> None:
        self.brain = brain
        self.concepts = concepts
        self.grounder = grounder
        self.universe = universe
        self.world = world
        self.curiosity = curiosity

        self.max_depth = max(1, int(max_depth))
        self.min_members = max(2, int(min_members))
        self.holdout_share = max(0.0, min(0.9, float(holdout_share)))
        self.capacity = max(16, int(capacity))
        self.represent_every = max(1, int(represent_every))
        self.test_every = max(1, int(test_every))

        self.schemas: Dict[Tuple[str, str, str], Schema] = {}
        self.transitivity: Dict[str, Transitivity] = {}
        # (subject, predicate, expected, found) harvested from schemas that failed held-out
        # scoring, waiting to become questions. Held between REVISE and the curiosity pass
        # because the schema that motivated the question is deleted in between.
        self._pending_questions: List[Tuple[str, str, str, str]] = []

        self.turns = 0
        self.cycles = 0
        self.represent_passes = 0
        self.test_passes = 0
        self.revise_passes = 0
        self.derived_direct = 0
        self.derived_composed = 0
        self.derived_schema = 0
        self.answers_given = 0
        self.questions_raised = 0
        self.schemas_formed = 0
        # Schemas refused because the kind was defined by the relation they would predict. Counted
        # rather than dropped silently: a high number says her concepts are being formed on the
        # same relation she is trying to generalise, which is a finding about the intake.
        self.schemas_tautological = 0
        self.schemas_dropped = 0
        self.last: Dict[str, Any] = {}

    # ---- the relation graph, read from wherever the facts actually live ---- #
    def _facts(self) -> Dict[Tuple[str, str], List[Any]]:
        """The live **positive** fact store. Read, never written — this module owns no knowledge.

        The polarity filter is here, at the one place this module reads the store, rather than in
        each of the six methods that walk it. That placement is the point: ``Grounder._lookup``
        already excluded denials and the leak persisted, because everything in this file goes to
        ``grounder.facts`` directly and never through the accessor that was fixed. Measured, a
        brain told ``"X does not need Y"`` and asked what X needs answered ``Y`` — derived,
        sourced, and stating the exact opposite of what it was told, with
        ``Derivation.why == "stated"`` attached to it.

        A denied relation is knowledge and stays in the store. What it may not do is enter a
        derivation as a premise, because every rule in this module — inheritance, composition,
        schema transfer — is written for relations that *hold*, and a premise that does not hold
        makes each of them produce a confident falsehood rather than nothing.
        """
        try:
            store = getattr(self.grounder, "facts", None) or {}
            out: Dict[Tuple[str, str], List[Any]] = {}
            for key, triples in store.items():
                live = [t for t in triples if not getattr(t, "negated", False)]
                if live:
                    out[key] = live
            return out
        except Exception:  # noqa: BLE001
            return {}

    def _edges(self, *, fit_only: bool = False) -> List[Tuple[str, str, str, float]]:
        """Every live ``(subject, predicate, object, confidence)`` she currently believes.

        ``fit_only`` drops the held-out fold, and is what REPRESENT uses. TEST uses its
        complement. Nothing else in this module may look at both.
        """
        out: List[Tuple[str, str, str, float]] = []
        for (subject, predicate), triples in self._facts().items():
            for triple in triples:
                try:
                    if getattr(triple, "superseded", False):
                        continue
                    obj = _norm(getattr(triple, "object", ""))
                    if not obj:
                        continue
                    if fit_only and self._held_out(subject, predicate, obj):
                        continue
                    out.append((_norm(subject), str(predicate), obj,
                                float(getattr(triple, "confidence", 0.5) or 0.5)))
                except Exception:  # noqa: BLE001
                    continue
        return out

    def _held_out(self, subject: str, predicate: str, obj: str) -> bool:
        return _fold(f"{_norm(subject)}|{predicate}|{_norm(obj)}", share=self.holdout_share)

    def _predicate(self, raw: str) -> str:
        """Fold a relation name the way the grounder folds it, so both reach the same edge.

        Borrowed rather than reimplemented. "needs" is stored as ``requires`` and "name" as
        ``has_name``; a Core with its own naming rule would look up keys the store does not use
        and report an honest-looking miss on a fact it holds. There is one naming authority in
        this package and it is :mod:`nyxara.njp.grounding`.
        """
        try:
            folder = getattr(self.grounder, "_predicate", None)
            if folder is not None:
                return str(folder(raw) or "")
        except Exception:  # noqa: BLE001
            pass
        return str(raw or "").strip().lower().replace(" ", "_")

    def _transitivity(self, predicate: str) -> Transitivity:
        """The posterior for this relation, seeded from the prior on first sight."""
        got = self.transitivity.get(predicate)
        if got is None:
            prior = _TRANSITIVE_PRIOR.get(predicate, _TRANSITIVE_DEFAULT)
            # Two pseudo-observations' worth of prior: enough to express a leaning, weak enough
            # that four real outcomes outweigh it.
            got = Transitivity(predicate=predicate, alpha=1.0 + prior, beta=1.0 + (1.0 - prior))
            self.transitivity[predicate] = got
        return got

    # ======================================================================= #
    # 1. REPRESENT — observations become roles, roles become schemas
    # ======================================================================= #
    def represent(self) -> RepresentReport:
        """Induce variabilised schemas from what several clustered subjects share.

        The one rule: a schema is proposed only where **two or more subjects that the concept
        layer independently grouped together** carry the same ``(predicate, object)``. Neither
        half is sufficient on its own — subjects sharing a relation might merely be two facts,
        and subjects sharing a concept might share nothing worth saying — and requiring both is
        what makes the role earned rather than assumed.

        Built from the fit fold only. A schema induced from a held-out fact would be scored
        against its own evidence in :meth:`test`, and would pass every time.
        """
        report = RepresentReport()
        try:
            if self.concepts is None:
                return report
            self.represent_passes += 1

            # (predicate, object) -> the subjects observed to have it, fit fold only.
            grouped: Dict[Tuple[str, str], Set[str]] = {}
            for subject, predicate, obj, _conf in self._edges(fit_only=True):
                grouped.setdefault((predicate, obj), set()).add(subject)

            roles: Set[str] = set()
            for (predicate, obj), subjects in grouped.items():
                if len(subjects) < self.min_members:
                    continue
                # Which concept, if any, actually claims these subjects together.
                for role, members in self._roles_over(subjects).items():
                    if len(members) < self.min_members:
                        continue
                    if self._tautological(role, predicate):
                        # The kind is defined by the very relation this schema would predict, so
                        # the schema says "things that are species are species". `generalize`
                        # already refuses to transfer such a role — `_roles_for_transfer` drops a
                        # concept whose invariants are empty once the target relation is removed —
                        # so inducing it produces a schema that can never answer anything and a
                        # count that looks like productivity.
                        #
                        # Measured after kind-heads gave many subjects a shared `is_a:<kind>`:
                        # every schema induced on the corpus was of this form. 43 schemas, 3 of
                        # them tested, and `derived_schema` structurally 0.
                        self.schemas_tautological += 1
                        continue
                    report.proposed += 1
                    key = (role, predicate, obj)
                    schema = self.schemas.get(key)
                    if schema is None:
                        if len(self.schemas) >= self.capacity:
                            self._evict()
                        if len(self.schemas) >= self.capacity:
                            continue
                        schema = Schema(role=role, predicate=predicate, object=obj)
                        self.schemas[key] = schema
                        self.schemas_formed += 1
                        report.kept += 1
                    schema.members |= members
                    roles.add(role)
                    report.members_covered += len(members)
            report.roles = len(roles)
            return report
        except Exception:  # noqa: BLE001
            return report

    def _roles_over(self, subjects: Set[str]) -> Dict[str, Set[str]]:
        """Group these subjects by the concept that claims each, dropping unclaimed ones.

        The concept id is used as the role name rather than any label, because a cluster found by
        similarity has no name — inventing one ("resource") would be this module quietly adding
        the ontology the whole design refuses to have.
        """
        out: Dict[str, Set[str]] = {}
        try:
            by_subject = getattr(self.concepts, "_by_subject", None) or {}
            for subject in subjects:
                obs = by_subject.get(_norm(subject))
                cid = str(getattr(obs, "concept", "") or "")
                if not cid:
                    continue
                out.setdefault(cid, set()).add(subject)
        except Exception:  # noqa: BLE001
            return {}
        return out

    def _evict(self) -> None:
        """Drop the worst-performing schema to make room. Refuted first, then least-tested."""
        try:
            if not self.schemas:
                return
            worst = min(self.schemas.values(), key=lambda s: (s.precision, s.trials))
            self.schemas.pop(worst.key, None)
            self.schemas_dropped += 1
        except Exception:  # noqa: BLE001
            pass

    # ======================================================================= #
    # 2. PREDICT — direct, then composed, then generalised
    # ======================================================================= #
    def predict(self, subject: str, predicate: str, *, allow_schema: bool = True,
                exclude: Collection[str] = ()) -> Derivation:
        """What she expects ``subject``'s ``predicate`` to be, and how sure by what route.

        Ordered strictly by epistemic strength: a fact she was told beats a chain she inferred,
        which beats a property transferred from a kind. Each rung is only tried where the one
        above it came back empty, so a derivation never overrides a fact — the failure mode this
        ordering exists to prevent is a confident generalisation burying the observation that
        contradicts it.

        ``exclude`` names answers the question has already ruled out, and it is the one thing that
        may push past a direct hit. Measured, with ``aag causes garmi`` and ``garmi causes pasina``
        stored::

            answer("what does aag cause besides garmi?")  ->  'garmi'   direct, 0.9
            reach("aag", "causes")                        ->  'pasina'  composed

        The ladder is right and the reading of the question was wrong: a stated fact *should* beat
        a chain, but "besides garmi" says that fact is not what was asked for. Without the
        exclusion the strongest rung answers with the very value the question excluded, and every
        rung below it — the one that had the answer — never runs.
        """
        out = Derivation()
        try:
            subject, predicate = _norm(subject), self._predicate(predicate)
            if not subject or not predicate:
                return out
            barred = {_norm(x) for x in exclude if _norm(x)}

            direct = self._direct(subject, predicate, exclude=barred)
            if direct.ok:
                self.derived_direct += 1
                return direct

            # Cross-predicate composition through the kind hierarchy, and it is deliberately
            # tried before same-predicate chaining. Same-predicate composition is close to dead
            # code on this path and it is worth saying why: `_compose` can only start where
            # ``facts[(subject, predicate)]`` is non-empty, which is exactly the condition under
            # which `_direct` has already answered. So a chain over one relation can never be
            # what rescues a *lookup miss* — it answers a different question ("what does this
            # eventually reach"), which is what :meth:`reach` and :meth:`simulate` are for.
            # What genuinely rescues a miss is inheritance: she was never told water increases
            # growth, but she was told water is a resource and that resources do.
            inherited = self._inherit(subject, predicate)
            if inherited.ok:
                self.derived_composed += 1
                return inherited

            composed = self._compose(subject, predicate)
            if composed.ok:
                self.derived_composed += 1
                return composed

            if allow_schema:
                transferred = self.generalize(subject, predicate)
                if transferred.ok:
                    self.derived_schema += 1
                    return transferred
            return out
        except Exception:  # noqa: BLE001
            return out

    def _direct(self, subject: str, predicate: str, *,
                exclude: Collection[str] = ()) -> Derivation:
        """The fact itself, if she has it and the question did not rule it out."""
        out = Derivation()
        try:
            triples = [t for t in self._facts().get((subject, predicate), [])
                       if not getattr(t, "superseded", False)
                       and _norm(str(getattr(t, "object", ""))) not in exclude]
            if not triples:
                return out
            best = max(triples, key=lambda t: float(getattr(t, "confidence", 0.0) or 0.0))
            out.answer = str(getattr(best, "object", ""))
            out.confidence = float(getattr(best, "confidence", 0.5) or 0.5)
            out.kind = "direct"
            out.steps = [f"{subject} —{predicate}→ {out.answer}"]
            out.support = [(subject, predicate, out.answer)]
            out.why = "stated"
            return out
        except Exception:  # noqa: BLE001
            return out

    def _inherit(self, subject: str, predicate: str) -> Derivation:
        """``X is_a K``, ``K —predicate→ O`` ⊢ ``X —predicate→ O``. Stated inheritance.

        The composition that actually answers a lookup miss, and it is a different mechanism from
        :meth:`generalize` even though both transfer from a kind to a member. This one walks
        ``is_a`` edges the Master **stated**, so the kind is a thing he named; that one walks
        clusters the concept layer **found**, so the kind is one she invented. The first is
        available immediately and needs no held-out evidence because the membership is not in
        doubt — only the inheritance is. The second has to earn its precision before it may speak.

        Multi-level, because ``is_a`` hierarchies are: sparrow is_a bird, bird is_a animal,
        animal needs water. Each level costs a factor of the relation's transitivity, so a
        property inherited from four levels up arrives visibly weaker than one from immediately
        above — which is right, since the further up it comes from the more exceptions it has.
        """
        out = Derivation()
        try:
            trans = self._transitivity("is_a").value
            seen: Set[str] = {subject}
            # (kind, confidence, path)
            queue: List[Tuple[str, float, List[Tuple[str, str, str]]]] = []
            for triple in self._facts().get((subject, "is_a"), []):
                if getattr(triple, "superseded", False):
                    continue
                kind = _norm(getattr(triple, "object", ""))
                if kind and kind not in seen:
                    seen.add(kind)
                    queue.append((kind, float(getattr(triple, "confidence", 0.5) or 0.5),
                                  [(subject, "is_a", kind)]))

            while queue:
                kind, conf, path = queue.pop(0)
                held = self._direct(kind, predicate)
                if held.ok:
                    chained = conf * held.confidence * trans
                    if chained >= _MIN_LINK_CONFIDENCE:
                        out.answer = held.answer
                        out.confidence = min(_COMPOSED_CEILING, chained)
                        out.kind = "composed"
                        out.role = kind
                        out.support = path + [(kind, predicate, held.answer)]
                        out.steps = [f"{a} —{p}→ {b}" for a, p, b in out.support]
                        out.steps.append(
                            f"so probably {subject} —{predicate}→ {held.answer} "
                            f"(inherited from {kind})")
                        out.why = f"inherited from the stated kind {kind}; defeasible"
                        return out
                if len(path) >= self.max_depth:
                    continue
                for triple in self._facts().get((kind, "is_a"), []):
                    if getattr(triple, "superseded", False):
                        continue
                    higher = _norm(getattr(triple, "object", ""))
                    if not higher or higher in seen:
                        continue
                    seen.add(higher)
                    queue.append((higher,
                                  conf * float(getattr(triple, "confidence", 0.5) or 0.5) * trans,
                                  path + [(kind, "is_a", higher)]))
            return out
        except Exception:  # noqa: BLE001
            return out

    def reach(self, subject: str, predicate: str = "causes") -> Derivation:
        """Everything ``subject`` eventually reaches along ``predicate``, beyond one hop.

        The question same-predicate composition actually answers. "What does fire cause" is a
        lookup; "what does fire *ultimately* cause" is this, and the two are different questions
        that the fact store cannot tell apart because it only ever stores one hop.
        """
        return self._compose(subject, predicate)

    def connects(self, subject: str, obj: str, predicate: str = "causes") -> Derivation:
        """Does ``subject`` reach ``obj`` along ``predicate``, and by what chain?

        A yes/no over the same walk, kept separate because the answer is the *path* rather than
        the endpoint — "does aag cause pyaas" wants the reason, not a repetition of "pyaas".
        """
        out = Derivation()
        try:
            subject, obj = _norm(subject), _norm(obj)
            trans = self._transitivity(predicate).value
            adjacency: Dict[str, List[Tuple[str, float]]] = {}
            for s, p, o, conf in self._edges():
                if p == predicate:
                    adjacency.setdefault(s, []).append((o, conf))
            queue: List[Tuple[str, float, List[Tuple[str, str, str]]]] = [(subject, 1.0, [])]
            seen: Set[str] = {subject}
            while queue:
                node, conf, path = queue.pop(0)
                if len(path) >= self.max_depth:
                    continue
                for neighbour, edge_conf in adjacency.get(node, []):
                    chained = conf * edge_conf * (trans if path else 1.0)
                    if chained < _MIN_LINK_CONFIDENCE:
                        continue
                    step = path + [(node, predicate, neighbour)]
                    if neighbour == obj:
                        out.answer = "yes"
                        out.confidence = min(_COMPOSED_CEILING, chained)
                        out.kind = "direct" if len(step) == 1 else "composed"
                        out.support = step
                        out.steps = [f"{a} —{p}→ {b}" for a, p, b in step]
                        out.why = f"reached in {len(step)} hop{'s' if len(step) > 1 else ''}"
                        return out
                    if neighbour in seen:
                        continue
                    seen.add(neighbour)
                    queue.append((neighbour, chained, step))
            return out
        except Exception:  # noqa: BLE001
            return out

    def walk_shape(self, subject: str, shape: Sequence[str]) -> Derivation:
        """Follow one named reasoning *form* from a subject: ``('is_a', 'needs')`` and so on.

        :meth:`_compose` chains a predicate with *itself* — ``A causes B``, ``B causes C`` —
        which is the right walk for transitivity and the wrong one for a shape. A shape as
        :mod:`nyxara.njp.genome` records it is a sequence of *different* predicates, one per hop,
        and it is the thing she keeps re-deriving: ``sparrow is_a bird``, ``bird needs water``
        ⊢ ``sparrow needs water``. There was no walk for it, which is part of why the genome's
        `candidates()` had no consumer — the promotion had nothing to promote *to*.

        Priced exactly as composition is, and for the same reason: the product of the edge
        confidences and the transitivity of each predicate, capped below the knowledge threshold.
        A four-hop shape is therefore worth visibly less than a two-hop one rather than merely
        being allowed. It is `composed`, never `direct`: a chain is defeasible by construction and
        an answer that reads like a stated fact is the failure this whole layer avoids.
        """
        out = Derivation()
        try:
            steps = [str(p).strip().lower() for p in (shape or []) if str(p).strip()]
            start = _norm(subject)
            if not steps or not start:
                return out

            by_key: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
            for s, p, o, conf in self._edges():
                by_key.setdefault((s, p), []).append((o, conf))

            node, confidence = start, 1.0
            support: List[Tuple[str, str, str]] = []
            visited = {start}
            for predicate in steps:
                options = by_key.get((node, predicate)) or []
                # Deterministic and best-first: the strongest edge this hop offers, and never one
                # that doubles back. `a causes b` and `b causes a` are both perfectly sayable, and
                # without the visited set a two-step shape would derive that a causes itself.
                options = sorted((o for o in options if o[0] not in visited),
                                 key=lambda pair: (-pair[1], pair[0]))
                if not options:
                    return Derivation()
                nxt, edge_conf = options[0]
                confidence *= edge_conf * self._transitivity(predicate).value
                if confidence < _MIN_LINK_CONFIDENCE:
                    return Derivation()
                support.append((node, predicate, nxt))
                visited.add(nxt)
                node = nxt

            out.answer = node
            out.kind = "composed"
            out.confidence = min(_COMPOSED_CEILING, confidence)
            out.support = support
            out.steps = [f"{a} —{p}→ {b}" for a, p, b in support]
            out.steps.append(f"therefore {start} —{'>'.join(steps)}→ {node} "
                             f"({len(support)} hops)")
            out.why = f"walked the shape {'>'.join(steps)} over {len(support)} stated facts"
            return out
        except Exception:  # noqa: BLE001 — a shape that does not walk simply answers nothing
            return Derivation()

    def _compose(self, subject: str, predicate: str) -> Derivation:
        """Chain the relation with itself: ``A→B``, ``B→C`` ⊢ ``A→C``. Novel recombination.

        A breadth-first walk so the *shortest* chain is found first — a two-hop explanation is
        both likelier and more useful than a five-hop one reaching the same place. Confidence is
        the product of the links and the relation's own transitivity per hop, capped below the
        knowledge threshold, so depth is priced rather than merely bounded and a long chain is
        worth visibly less than a short one.

        Cycles are cut by the visited set, which matters more than it looks: ``a causes b`` and
        ``b causes a`` are both perfectly sayable, and without this the walk would happily derive
        that a causes itself and hand it back as an insight.
        """
        out = Derivation()
        try:
            trans = self._transitivity(predicate).value
            if trans < _MIN_LINK_CONFIDENCE:
                return out

            adjacency: Dict[str, List[Tuple[str, float]]] = {}
            for s, p, o, conf in self._edges():
                if p == predicate:
                    adjacency.setdefault(s, []).append((o, conf))
            if subject not in adjacency:
                return out

            # A plain BFS queue of ``(node, confidence, path)``. Seeded one hop out: the direct
            # neighbours are what `_direct` already answered with, so re-offering one here as a
            # "derivation" would be dressing a lookup up as an inference. The first node reached
            # at depth two or more is the answer, and because the queue is breadth-first that is
            # the shortest chain to it.
            queue: List[Tuple[str, float, List[Tuple[str, str, str]]]] = []
            seen: Set[str] = {subject}
            for neighbour, conf in adjacency.get(subject, []):
                seen.add(neighbour)
                queue.append((neighbour, conf, [(subject, predicate, neighbour)]))

            while queue:
                node, conf, path = queue.pop(0)
                if len(path) >= self.max_depth:
                    continue
                for neighbour, edge_conf in adjacency.get(node, []):
                    if neighbour in seen:
                        continue        # cycle, or already reachable by a shorter route
                    seen.add(neighbour)
                    chained = conf * edge_conf * trans
                    if chained < _MIN_LINK_CONFIDENCE:
                        continue
                    step = path + [(node, predicate, neighbour)]
                    # Reached at depth two or more, by the shortest route the queue allows: this
                    # is an inference she could not have looked up, which is the whole product.
                    out.answer = neighbour
                    out.confidence = min(_COMPOSED_CEILING, chained)
                    out.kind = "composed"
                    out.support = step
                    out.steps = [f"{a} —{p}→ {b}" for a, p, b in step]
                    out.steps.append(
                        f"therefore {subject} —{predicate}→ {neighbour} "
                        f"({len(step)} hops, transitivity {trans:.2f})")
                    out.why = f"composed over {len(step)} stated facts"
                    return out
                # Nothing new off this node; keep walking outward from it. Enqueued after the
                # scan rather than before so a node whose every neighbour was already seen still
                # extends the frontier — without this the walk stops dead at depth two whenever
                # the graph doubles back, and chains that only open up further out are missed.
                for neighbour, edge_conf in adjacency.get(node, []):
                    chained = conf * edge_conf * trans
                    if chained >= _MIN_LINK_CONFIDENCE:
                        queue.append((neighbour, chained, path + [(node, predicate, neighbour)]))
            return out
        except Exception:  # noqa: BLE001
            return out

    # ======================================================================= #
    # 3. EXPLAIN — the derivation, not a narration of it
    # ======================================================================= #
    def explain(self, subject: str, predicate: str) -> Derivation:
        """Why she would say what she would say. Same computation, kept as the record.

        Not a separate mechanism on purpose. An explanation produced by different code from the
        answer is a rationalisation — it can be fluent and true and still not be the reason. The
        steps on a :class:`Derivation` are the ones the walk actually took.
        """
        return self.predict(subject, predicate)

    # ======================================================================= #
    # 4. SIMULATE — what moves if this moves
    # ======================================================================= #
    def simulate(self, variable: str, *, direction: float = 1.0) -> Derivation:
        """Qualitative propagation over the causal graph: what changes, and which way.

        Deliberately *directional* rather than numeric. :mod:`nyxara.njp.universe` already owns
        the numeric do-operator and does it properly — with a fitted coefficient, an observed
        range and a confidence that falls outside it. But it can only answer where a coefficient
        has been fitted, which needs paired measurements, which most conversations never provide.
        Between "I have regression-quality data" and "I know nothing" sits the case she is
        usually in: she knows fire causes heat and has never measured either. Saying "less fire,
        less heat" there is genuinely informative and genuinely honest, and refusing to say it
        because no number is attached was throwing away most of what she knows.
        """
        out = Derivation()
        try:
            variable = _norm(variable)
            if not variable:
                return out
            trans = self._transitivity("causes").value
            downstream: List[Tuple[str, float, int]] = []
            frontier = [(variable, 1.0, 0)]
            seen = {variable}
            while frontier:
                node, conf, hops = frontier.pop(0)
                if hops >= self.max_depth:
                    continue
                for triple in self._facts().get((node, "causes"), []):
                    if getattr(triple, "superseded", False):
                        continue
                    effect = _norm(getattr(triple, "object", ""))
                    if not effect or effect in seen:
                        continue
                    seen.add(effect)
                    chained = conf * float(getattr(triple, "confidence", 0.5) or 0.5)
                    if hops:
                        chained *= trans
                    if chained < _MIN_LINK_CONFIDENCE:
                        continue
                    downstream.append((effect, chained, hops + 1))
                    frontier.append((effect, chained, hops + 1))
            if not downstream:
                return out

            word = "increases" if direction > 0 else ("decreases" if direction < 0 else "stops")
            downstream.sort(key=lambda item: (-item[1], item[2]))
            out.answer = ", ".join(f"{name} {word}" for name, _c, _h in downstream[:4])
            out.confidence = min(_COMPOSED_CEILING, downstream[0][1])
            out.kind = "simulated"
            out.support = [(variable, "causes", name) for name, _c, _h in downstream[:4]]
            out.steps = [f"{variable} {word} → {name} {word} "
                         f"({hops} hop{'s' if hops > 1 else ''}, {conf:.2f})"
                         for name, conf, hops in downstream[:4]]
            out.why = "directional propagation over stated causal links (no fitted coefficient)"
            return out
        except Exception:  # noqa: BLE001
            return out

    # ======================================================================= #
    # 5. TEST — on held-out facts only
    # ======================================================================= #
    def test(self) -> TestReport:
        """Score every schema against facts it was never built from.

        This is the method that makes the module's claim checkable. A schema says *members of
        this role have this property*; the held-out fold contains facts about members that
        REPRESENT could not see. Where the fold says a member has the property, the schema was
        right; where it says the member has a **different** value for a relation that can only
        hold one, the schema was wrong. Silence is neither — an absent fact is not a refutation,
        and counting it as one would refute every true schema about a thing she simply has not
        been told everything about.
        """
        report = TestReport()
        try:
            self.test_passes += 1
            holdout = [(s, p, o) for s, p, o, _c in self._edges()
                       if self._held_out(s, p, o)]
            report.holdout_size = len(holdout)
            if not holdout:
                return report

            # (subject, predicate) -> the values the held-out fold actually records.
            observed: Dict[Tuple[str, str], Set[str]] = {}
            for subject, predicate, obj in holdout:
                observed.setdefault((subject, predicate), set()).add(obj)

            for schema in list(self.schemas.values()):
                for subject in self._members_of(schema.role):
                    values = observed.get((subject, schema.predicate))
                    if not values:
                        continue            # silence is not evidence either way
                    schema.tested += 1
                    report.scored += 1
                    if schema.object in values:
                        schema.hits += 1
                        report.hits += 1
                    else:
                        schema.misses += 1
                        report.misses += 1
                        schema.refuted_by[subject] = sorted(values)[0]
            return report
        except Exception:  # noqa: BLE001
            return report

    def _members_of(self, role: str) -> List[str]:
        """Every subject the concept layer currently places in this role.

        Read live from :mod:`nyxara.njp.concepts` rather than from the schema's own ``members``,
        and that is the whole point of testing: the schema's members are who *suggested* it, and
        scoring against those is scoring against the training set. The concept may since have
        claimed subjects the schema has never seen, and those are exactly the ones worth asking
        about.
        """
        try:
            concept = (getattr(self.concepts, "concepts", None) or {}).get(role)
            return [_norm(m) for m in (getattr(concept, "members", None) or [])]
        except Exception:  # noqa: BLE001
            return []

    # ======================================================================= #
    # 6. REVISE — what failed changes, and the change is recorded
    # ======================================================================= #
    def revise(self) -> ReviseReport:
        """Drop refuted schemas and move the transitivity posteriors that licensed them.

        Two different things are being revised and they are not the same claim. A schema that
        fails is a wrong generalisation about one role — it goes. A *predicate* whose schemas
        keep failing is weaker evidence that the relation chains at all, so its transitivity
        drops and every future composition over it is priced lower. Without the second half a
        refuted schema would be replaced next pass by an identical one, since the thing that
        proposed it has learned nothing.
        """
        report = ReviseReport()
        try:
            self.revise_passes += 1
            moved: Set[str] = set()
            for key, schema in list(self.schemas.items()):
                if schema.status != "refuted":
                    if schema.status == "confirmed" and schema.trials >= 2:
                        self._transitivity(schema.predicate).confirm(0.5)
                        moved.add(schema.predicate)
                    continue
                report.refuted += 1
                # Harvest the specific disagreements before the schema goes, so the question
                # outlives the generalisation that raised it.
                for member, found in list(schema.refuted_by.items())[:2]:
                    self._pending_questions.append(
                        (member, schema.predicate, schema.object, found))
                self.schemas.pop(key, None)
                self.schemas_dropped += 1
                report.dropped += 1
                self._transitivity(schema.predicate).refute(0.5)
                moved.add(schema.predicate)
            report.transitivity_moved = len(moved)
            return report
        except Exception:  # noqa: BLE001
            return report

    # ======================================================================= #
    # 7. GENERALIZE — a kind's property, applied to a member that never showed it
    # ======================================================================= #
    def generalize(self, subject: str, predicate: str) -> Derivation:
        """Transfer from role to member: the unseen-situation case.

        Defeasible by construction, and reported as such. The schema says members of this kind
        have this property; this subject is a member; so *probably* it does too. Confidence is
        the schema's held-out precision — not its support, which is how often the pattern was
        seen and says nothing about whether it holds up. A schema that has never been tested
        cannot answer at all: an untested generalisation is a guess with a role attached.
        """
        out = Derivation()
        try:
            subject = _norm(subject)
            roles = self._roles_for_transfer(subject, predicate)
            if not roles:
                return out
            role = ""
            best: Optional[Schema] = None
            for schema in self.schemas.values():
                if schema.role not in roles or schema.predicate != predicate:
                    continue
                if schema.status == "refuted" or schema.trials < 1:
                    continue
                if best is None or schema.precision > best.precision:
                    best = schema
                    role = schema.role
            if best is None:
                return out
            out.answer = best.object
            out.confidence = min(_COMPOSED_CEILING, best.precision)
            out.kind = "schema"
            out.role = role
            out.support = [(m, predicate, best.object) for m in sorted(best.members)[:4]]
            out.steps = [
                f"{subject} is a {role}",
                f"{role}s are {predicate} {best.object} "
                f"(held out {best.hits}/{best.trials}, precision {best.precision:.2f})",
                f"so probably {subject} —{predicate}→ {best.object}",
            ]
            out.why = f"transferred from {role}; defeasible"
            return out
        except Exception:  # noqa: BLE001
            return out

    def _role_of(self, subject: str) -> str:
        """The concept that currently claims this subject outright."""
        try:
            obs = (getattr(self.concepts, "_by_subject", None) or {}).get(_norm(subject))
            return str(getattr(obs, "concept", "") or "")
        except Exception:  # noqa: BLE001
            return ""

    def _tautological(self, role: str, predicate: str) -> bool:
        """Is this kind defined by nothing but the relation the schema would infer?

        The same test `_roles_for_transfer` applies at answer time, applied at induction time so
        the schema is never created rather than created and then left unusable.
        """
        try:
            concept = (getattr(self.concepts, "concepts", None) or {}).get(role)
            if concept is None:
                return False
            prefix = f"{predicate}:"
            residue = {f for f in (getattr(concept, "invariants", None) or set())
                       if not str(f).startswith(prefix)}
            return not residue
        except Exception:  # noqa: BLE001
            return False

    def _roles_for_transfer(self, subject: str, predicate: str) -> Set[str]:
        """The kinds this subject qualifies for **ignoring the relation being inferred**.

        This method is the difference between a system that can generalise and one that only
        looks like it can, and the bug it fixes is worth stating plainly because it is invisible
        until you test for transfer specifically.

        :mod:`nyxara.njp.concepts` forms a concept from features its members *all* share, so a
        cluster of things that increase growth has ``increases:growth`` among its **invariants**.
        Membership is then tested by requiring those invariants. Ask whether a newly met thing
        belongs — in order to predict whether it increases growth — and the test demands it
        already be known to increase growth. Measured: seven resources clustered, the schema
        ``⟨kind⟩ increases growth`` confirmed 3/3 on held-out facts, and ``potash``, stated to be
        a resource and nothing else, got no concept at all and no prediction. The kind was
        defined by the very property being inferred, so transfer was not merely failing, it was
        unreachable by construction — the circularity is in the definition, not the data.

        So membership here is tested against the invariants *minus* anything the target relation
        contributes. "You match this kind on everything except the thing I am predicting" is the
        only form of the question that can have an informative answer. The excluded relation is
        exactly the one the schema will supply, and the held-out precision on that schema is what
        says whether the inference is any good — the discipline is not weakened by the exclusion,
        it is relocated to where it can actually bite.
        """
        out: Set[str] = set()
        try:
            by_subject = getattr(self.concepts, "_by_subject", None) or {}
            obs = by_subject.get(_norm(subject))
            if obs is None:
                return out
            features: Set[str] = set(getattr(obs, "features", None) or set())
            if not features:
                return out
            prefix = f"{predicate}:"
            for cid, concept in (getattr(self.concepts, "concepts", None) or {}).items():
                invariants = {f for f in (getattr(concept, "invariants", None) or set())
                              if not str(f).startswith(prefix)}
                # An empty residue means the kind was *only* the target relation — every member
                # shared nothing else. Such a concept says "things that increase growth increase
                # growth" and admits anything at all once the relation is excluded, so it can
                # licence no inference and is skipped rather than matched.
                if not invariants or not invariants.issubset(features):
                    continue
                out.add(str(cid))
            return out
        except Exception:  # noqa: BLE001
            return out

    # ======================================================================= #
    # The loop, and the entry point the brain uses
    # ======================================================================= #
    def answer(self, question: str) -> Derivation:
        """Derive an answer to a question the fact store could not look up.

        Reads the question through the grounder's own question parser rather than a second one
        of its own — so anything the Master can *ask* reaches the Core, and a question form added
        there is answerable here for free.
        """
        out = Derivation()
        try:
            if self.grounder is None:
                return out
            reader = getattr(self.grounder, "_read_question", None)
            if reader is None:
                return out
            from nyxara.njp.grounding import _CAUSE_OF, _clean
            subject, predicate = reader(_clean(question).lower())
            barred = _exclusions(question)
            if barred:
                # The question named answers it has already been given, so the parse has to lose
                # them too — "what does aag cause besides garmi" parses its subject as
                # "aag ... besides garmi" or leaves the tail attached, and predicting on that
                # subject finds nothing at all.
                subject = _strip_exclusions(subject)
            if predicate and subject:
                if predicate == _CAUSE_OF:
                    # Asked from the effect end. Composition still applies, backwards.
                    derived = self._compose_inverse(_norm(subject))
                else:
                    derived = self.predict(subject, predicate, exclude=barred)
                if derived.ok:
                    self.answers_given += 1
                    return derived

            # The parse named nothing, or named something that answered nothing. Both are
            # failures of the same door, and until now both returned empty — so everything this
            # module can do was reachable only through the grounder's pattern table.
            #
            # Treating a *successful* parse as final was the worse half. `"aag se pasina hoti hai
            # kya"` parses to subject `"aag se pasina hoti hai"` under `is_a`, and
            # `"what is the root cause of pasina"` to `"root cause of pasina"` — both confident,
            # both meaningless, and both blocked the second route by not failing. A wrong parse
            # cost more than no parse at all.
            #
            # So the parse is a proposal: if it produced an answer, that answer wins, because it
            # read an actual relation. If it produced nothing, name the *entities* instead. That
            # asks a weaker question — "which things here do I hold causal edges for" — and it is
            # allowed to answer only by composition, never by lookup (the grounder owns lookup
            # and has already declined) and never with a relation it did not read.
            return self._compose_by_entity(question)
        except Exception:  # noqa: BLE001
            return out

    def _compose_by_entity(self, question: str) -> Derivation:
        """Compose from whichever entities the question names, when the parser named none.

        The relation is not read here — it is *assumed* to be ``causes``, and only ``causes``.
        That is the whole safety story of this method: it cannot invent a relation it did not
        parse, so the worst it can do is answer a causal question that was not asked, and the
        confidence cap on composed answers still applies on top.

        Direction is decided by evidence rather than by grammar. An entity that has causes
        upstream of it is read backwards ("why X"), one that has effects downstream is read
        forwards ("what does X lead to"), and an entity with both prefers the backward reading —
        because a question naming an effect is far more often asking for its cause than for what
        it goes on to produce. Where two entities are both known and one reaches the other, that
        connection wins over either single-entity reading: naming both ends is a more specific
        question than naming one.

        Only *composed* answers are returned. A one-hop edge is a stored fact and the grounder
        owns that lookup; if it declined, repeating it here would be a second retrieval path with
        different rules, which is the duplication this module exists to avoid.
        """
        out = Derivation()
        try:
            names = self._entities_in(question)
            if not names:
                return out

            forward: Dict[str, List[Tuple[str, float]]] = {}
            backward: Dict[str, List[Tuple[str, float]]] = {}
            for s, p, o, conf in self._edges():
                if p == "causes":
                    forward.setdefault(s, []).append((o, conf))
                    backward.setdefault(o, []).append((s, conf))

            known = [n for n in names if n in forward or n in backward]
            if not known:
                return out

            # Both ends named: "does aag cause pasina", "aag se pasina hoti hai kya".
            for head in known:
                for tail in known:
                    if head == tail:
                        continue
                    linked = self.connects(head, tail, "causes")
                    if linked.ok and len(linked.support) > 1:
                        self.answers_given += 1
                        return linked

            for name in known:
                if name in backward:
                    got = self._compose_inverse(name)
                    if got.ok:
                        self.answers_given += 1
                        return got
            for name in known:
                if name in forward:
                    got = self.reach(name, "causes")
                    if got.ok:
                        self.answers_given += 1
                        return got
            return out
        except Exception:  # noqa: BLE001
            return out

    def _entities_in(self, question: str) -> List[str]:
        """The content words of a question, in order, normalised to fact-store keys.

        Uses the package's own tokeniser (:mod:`nyxara.njp.tongue`), which is script-aware and
        already carries the function-word lists for English, romanised Hinglish and Devanagari —
        so "aag se kya kya hota hai" yields ``aag`` and not ``se``, ``kya`` or ``hota``. Writing a
        word filter here instead would be a second, worse stop-list beside the real one.
        """
        try:
            from nyxara.njp.tongue import concepts_in
            seen: Set[str] = set()
            out: List[str] = []
            for token in concepts_in(str(question or "")):
                name = _norm(token)
                if name and name not in seen:
                    seen.add(name)
                    out.append(name)
            return out
        except Exception:  # noqa: BLE001
            return []

    def _compose_inverse(self, effect: str) -> Derivation:
        """What upstream of ``effect`` could account for it, more than one hop back."""
        out = Derivation()
        try:
            trans = self._transitivity("causes").value
            back: Dict[str, List[Tuple[str, float]]] = {}
            for s, p, o, conf in self._edges():
                if p == "causes":
                    back.setdefault(o, []).append((s, conf))
            if effect not in back:
                return out
            seen = {effect}
            frontier = [(cause, conf, [(cause, "causes", effect)])
                        for cause, conf in back.get(effect, [])]
            for cause, _c, _p in frontier:
                seen.add(cause)
            depth = 1
            while frontier and depth < self.max_depth:
                nxt = []
                for node, conf, path in frontier:
                    for prior, edge_conf in back.get(node, []):
                        if prior in seen:
                            continue
                        seen.add(prior)
                        chained = conf * edge_conf * trans
                        if chained < _MIN_LINK_CONFIDENCE:
                            continue
                        step = [(prior, "causes", node)] + path
                        out.answer = prior
                        out.confidence = min(_COMPOSED_CEILING, chained)
                        out.kind = "composed"
                        out.support = step
                        out.steps = [f"{a} —{p}→ {b}" for a, p, b in step]
                        out.steps.append(f"therefore {prior} ultimately causes {effect}")
                        out.why = f"composed backwards over {len(step)} stated facts"
                        return out
                    nxt.append((node, conf, path))
                frontier = nxt
                depth += 1
            return out
        except Exception:  # noqa: BLE001
            return out

    def cycle(self, thought: Any = None) -> CoreReport:
        """One turn: represent on a schedule, test and revise on another, always be answerable.

        The two schedules are separate because they cost differently and pay differently.
        REPRESENT walks the fit fold and is cheap; TEST plus REVISE walks the held-out fold and
        the concept membership and is not, and running it every turn would grade schemas against
        a fold that has barely changed since the last time. Turn counts rather than a wall clock,
        for the reason :mod:`nyxara.njp.integrate` records: a clock does not tick in the paths a
        brain is actually used from.
        """
        report = CoreReport()
        started = time.perf_counter()
        try:
            self.turns += 1
            self.cycles += 1
            report.turn = self.turns

            if self.turns % self.represent_every == 0:
                report.represented = self.represent()
            if self.turns % self.test_every == 0:
                report.tested = self.test()
                report.revised = self.revise()
                self._wonder(report)

            self.last = report.to_dict()
            return report
        except Exception:  # noqa: BLE001
            return report
        finally:
            report.ms = (time.perf_counter() - started) * 1000.0

    def _wonder(self, report: CoreReport) -> None:
        """Turn a refuted schema into a question she actually asks.

        The gap this closes: :mod:`nyxara.njp.curiosity` self-generates questions from prediction
        failures and they are *terminal* — "why does my grounding keep failing?" enters a queue
        priced by expected information gain and is either voiced or aged out, and nothing ever
        investigates it. A refuted schema is a better question than that one because it is
        specific and it is answerable: she believed members of a kind shared a property, held-out
        evidence said otherwise, and the thing worth asking is what actually distinguishes them.
        """
        try:
            if self.curiosity is None or not self._pending_questions:
                return
            from nyxara.njp.curiosity import Gap, Question
            raise_one = getattr(self.curiosity, "_raise", None)
            if raise_one is None:
                return
            for subject, predicate, expected, found in self._pending_questions[:4]:
                question = Question(
                    text=(f"I generalised that {subject} —{predicate}→ {expected}, "
                          f"and it is {found} instead. "
                          f"What distinguishes {subject} from the others?"),
                    gap=Gap.REPEATED_FAILURE,
                    # Keyed on the pair the *answer* would arrive as, so the Master stating that
                    # very fact closes it. A question keyed on anything else can never be
                    # resolved and only ever ages out, which is what happens to every question
                    # curiosity currently raises about its own failures.
                    subject=subject, predicate=predicate,
                    uncertainty=0.8, stakes=0.6, cost=0.2)
                if raise_one(question) is question:
                    self.questions_raised += 1
            self._pending_questions.clear()
        except Exception:  # noqa: BLE001
            pass

    # ---- reporting --------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        confirmed = sum(1 for s in self.schemas.values() if s.status == "confirmed")
        tested = [s for s in self.schemas.values() if s.trials]
        precision = (sum(s.precision for s in tested) / len(tested)) if tested else None
        return {
            "turns": self.turns,
            "cycles": self.cycles,
            "schemas": len(self.schemas),
            "schemas_confirmed": confirmed,
            "schemas_formed": self.schemas_formed,
            "schemas_tautological": self.schemas_tautological,
            "schemas_dropped": self.schemas_dropped,
            "schemas_tested": len(tested),
            "mean_precision": None if precision is None else round(precision, 3),
            "represent_passes": self.represent_passes,
            "test_passes": self.test_passes,
            "revise_passes": self.revise_passes,
            "derived_direct": self.derived_direct,
            "derived_composed": self.derived_composed,
            "derived_schema": self.derived_schema,
            "answers_given": self.answers_given,
            "questions_raised": self.questions_raised,
            "holdout_share": self.holdout_share,
            "max_depth": self.max_depth,
            "transitivity": {p: round(t.value, 3) for p, t in sorted(self.transitivity.items())},
            "last": self.last,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turns": self.turns,
            "schemas": [s.to_dict() for s in self.schemas.values()],
            "transitivity": [t.to_dict() for t in self.transitivity.values()],
            "counters": {"formed": self.schemas_formed, "dropped": self.schemas_dropped,
                         "direct": self.derived_direct, "composed": self.derived_composed,
                         "schema": self.derived_schema},
        }

    def load_dict(self, d: Dict[str, Any]) -> None:
        """Restore. Fail-soft: a corrupt record leaves her with fewer schemas, never broken."""
        try:
            self.turns = int(d.get("turns", 0) or 0)
            for raw in d.get("schemas", []) or []:
                try:
                    schema = Schema(role=str(raw.get("role", "")),
                                    predicate=str(raw.get("predicate", "")),
                                    object=str(raw.get("object", "")),
                                    members=set(raw.get("members", []) or []),
                                    hits=int(raw.get("hits", 0) or 0),
                                    misses=int(raw.get("misses", 0) or 0))
                    if schema.role and schema.predicate:
                        self.schemas[schema.key] = schema
                except Exception:  # noqa: BLE001
                    continue
            for raw in d.get("transitivity", []) or []:
                try:
                    predicate = str(raw.get("predicate", ""))
                    if not predicate:
                        continue
                    got = self._transitivity(predicate)
                    value = float(raw.get("value", got.value))
                    observations = float(raw.get("observations", 0.0) or 0.0)
                    got.alpha = 1.0 + value * (1.0 + observations)
                    got.beta = 1.0 + (1.0 - value) * (1.0 + observations)
                except Exception:  # noqa: BLE001
                    continue
            counters = d.get("counters", {}) or {}
            self.schemas_formed = int(counters.get("formed", 0) or 0)
            self.schemas_dropped = int(counters.get("dropped", 0) or 0)
            self.derived_direct = int(counters.get("direct", 0) or 0)
            self.derived_composed = int(counters.get("composed", 0) or 0)
            self.derived_schema = int(counters.get("schema", 0) or 0)
        except Exception:  # noqa: BLE001
            pass
