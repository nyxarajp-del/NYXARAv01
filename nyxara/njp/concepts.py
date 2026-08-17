"""NYXARA · njp/concepts.py — Concept Genesis and Reality Compression (🧬, NJP V.04).

Nothing in here is given a concept. Concepts are what comes out.

**The problem this exists to solve.** :mod:`nyxara.njp.grounding` turns a sentence into relations
and :mod:`nyxara.njp.world` turns relations into causes, and between them there is a hole: both
work at the level of the *particular*. "dog is_a animal", "cat is_a animal", "horse is_a animal"
are three facts, stored three times, answering three questions. A mind that keeps them that way
has memorised its input. The thing that separates a memory from an understanding is that the
understanding is **smaller than what it explains** — and can then be applied to a wolf it has
never met.

**The pipeline, and every arrow in it is an algorithm rather than a table lookup.**::

    raw observations → similarity → invariants → prototype → concept → hierarchy → causal relations

* **Similarity** is a blend of Jaccard over an observation's symbolic features and cosine over its
  numeric ones, so "has four legs" and "boils at 100" are comparable evidence about kinship.
* **Invariants** are the features shared by at least :attr:`invariant_share` of a cluster's
  members. This is the part that makes a concept a claim rather than a bag: the invariants are
  what the concept *asserts*, and an observation that violates them is evidence against it.
* **A prototype** is the members' centroid — the thing the concept looks like, which is what
  makes a *near* match possible at all.
* **The hierarchy is derived, never declared.** Concept A is beneath concept B when A's
  invariants are a strict superset of B's: everything true of the general is true of the specific,
  plus more. That single subsumption test is where ``dog → animal → living thing`` comes from,
  and it is why nobody has to write the taxonomy down.

**Reality compression is measured, not asserted.** :meth:`ConceptGenesis.compression` is a real
minimum-description-length ratio — the cost in feature-symbols of storing every observation
verbatim, over the cost of storing the concepts plus each observation's *residual* against its
concept. A concept set that does not pay for itself scores at or below 1.0 and says so. This is
the honest form of "in 10,000 observations ke peeche ek simple rule kya hai": if the rule really
is simpler than the observations, the ratio proves it; if it is not, no amount of naming it a
concept makes it one.

**Insufficiency is the output that matters most.** :meth:`ConceptGenesis.explain` returns a
:class:`Coverage`, and when nothing covers an observation past threshold, :attr:`Coverage.gap`
carries *why* — unknown kind, or a violated invariant. :mod:`nyxara.njp.field` reads exactly that
signal to decide whether a prediction error should adjust weights or **restructure the concept
system**, which is the difference the Master named: a mind whose errors can only ever change
numbers has an architecture that no experience can reach.

Pure standard library, ``numpy`` optional (only to make the numeric half of the similarity fast).
No LLM anywhere in this file: a concept invented by asking a language model to name a cluster is
the language model's concept, not hers.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "Observation",
    "Concept",
    "Coverage",
    "GenesisReport",
    "ConceptGenesis",
]

# A cluster is a claim about a kind, and two examples is the smallest number that can be wrong in
# an interesting way. One is an anecdote — it has no invariants that are not simply its own
# features, so it would "explain" itself perfectly and compress nothing.
_MIN_MEMBERS = 2
_SIMILARITY = 0.34          # join an existing concept at or above this
_INVARIANT_SHARE = 0.6      # a feature is invariant if this fraction of members carry it
_COVER = 0.45               # an observation is covered by a concept at or above this


#: Marks an observation created for the *object* of a relation, so a superordinate has somewhere
#: to accumulate evidence. It records that something points at this node — not a property of it.
#: Every kind she has ever met carries it, so it discriminates nothing and must never be clustered
#: on. See :meth:`ConceptGenesis._similarity` for what happened when it was.
_ROLE_PREFIX = "is_target_of:"


def _content_features(features: Set[str]) -> Set[str]:
    """The features that actually say something about the thing."""
    return {f for f in features if not f.startswith(_ROLE_PREFIX)}


def _norm(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


# --------------------------------------------------------------------------- #
# What goes in
# --------------------------------------------------------------------------- #
@dataclass
class Observation:
    """One thing met once, described by what was true of it.

    ``features`` are symbolic (``"is_a:animal"``, ``"needs:water"``) and ``numbers`` are the
    quantities that came with it (``{"boils_at": 100.0}``). Both are optional and an observation
    with neither is dropped: there is nothing to be similar *about*.
    """

    subject: str = ""
    features: Set[str] = field(default_factory=set)
    numbers: Dict[str, float] = field(default_factory=dict)
    at: float = field(default_factory=time.time)
    seq: int = 0
    concept: str = ""            # id of the concept that claimed it, once one has

    def to_dict(self) -> Dict[str, Any]:
        return {"subject": self.subject, "features": sorted(self.features)[:16],
                "numbers": {k: round(v, 4) for k, v in list(self.numbers.items())[:8]},
                "seq": self.seq, "concept": self.concept}


# --------------------------------------------------------------------------- #
# What comes out
# --------------------------------------------------------------------------- #
@dataclass
class Concept:
    """An invented kind: what its members share, what they look like, and where it sits."""

    cid: str = ""
    members: List[str] = field(default_factory=list)     # subject names
    invariants: Set[str] = field(default_factory=set)
    # feature -> share of members carrying it, for features that are common but NOT universal.
    # The concept does not *claim* these — that is what invariants are — but they are what its
    # members mostly look like, and they are the only source of defeasible transfer there is.
    # Without them `generalise` is structurally empty: an ancestor's invariants are a subset of
    # its child's, which are satisfied by every member by construction, so inheritance down a
    # subsumption hierarchy can never tell a member anything it did not already have.
    typical: Dict[str, float] = field(default_factory=dict)
    prototype: Dict[str, float] = field(default_factory=dict)
    parent: str = ""
    children: List[str] = field(default_factory=list)
    born: float = field(default_factory=time.time)
    level: int = 0               # which cut of the tree first produced it; 0 = most specific
    # Times an observation this concept claimed turned out to violate its invariants. A concept
    # that keeps being wrong is a concept whose boundary is in the wrong place, and `restructure`
    # is what acts on that.
    violations: int = 0

    @property
    def support(self) -> int:
        return len(self.members)

    @property
    def name(self) -> str:
        """A label read off the invariants, because that is genuinely all she knows about it.

        Deliberately not a pretty English word. Calling a cluster "animal" requires knowing the
        word means what the cluster is, which is exactly the knowledge that would have had to be
        given to her. ``needs:water+is_a:animal`` is uglier and true.
        """
        if not self.invariants:
            return self.cid
        return "+".join(sorted(self.invariants)[:3])

    @property
    def order(self) -> int:
        """How general it is: fewer invariants ⇒ higher up the hierarchy."""
        return len(self.invariants)

    def covers(self, obs: "Observation") -> float:
        """How much of this concept's claim the observation actually satisfies, in ``[0, 1]``."""
        if not self.invariants:
            return 0.0
        held = len(self.invariants & obs.features)
        return held / len(self.invariants)

    def violated_by(self, obs: "Observation") -> Set[str]:
        """Invariants this observation is missing. Non-empty ⇒ the concept over-claims."""
        return set(self.invariants) - set(obs.features)

    def to_dict(self) -> Dict[str, Any]:
        return {"cid": self.cid, "name": self.name, "support": self.support,
                "invariants": sorted(self.invariants)[:12], "order": self.order,
                "typical": {k: round(v, 3) for k, v in
                            sorted(self.typical.items(), key=lambda kv: -kv[1])[:8]},
                "parent": self.parent, "children": list(self.children)[:12],
                "members": self.members[:12], "violations": self.violations,
                "prototype": {k: round(v, 3) for k, v in list(self.prototype.items())[:8]}}


@dataclass
class Coverage:
    """The answer to "what is this?" — and, when there is no answer, why not."""

    subject: str = ""
    concept: str = ""
    score: float = 0.0
    margin: float = 0.0          # how far clear of the runner-up
    gap: str = ""                # "" when covered; otherwise what is missing
    violated: List[str] = field(default_factory=list)

    @property
    def covered(self) -> bool:
        return bool(self.concept) and not self.gap

    def to_dict(self) -> Dict[str, Any]:
        return {"subject": self.subject, "concept": self.concept,
                "score": round(self.score, 4), "margin": round(self.margin, 4),
                "covered": self.covered, "gap": self.gap, "violated": self.violated[:8]}


@dataclass
class GenesisReport:
    """What one crystallisation pass changed."""

    observations: int = 0
    formed: List[str] = field(default_factory=list)
    grown: List[str] = field(default_factory=list)
    merged: List[Tuple[str, str]] = field(default_factory=list)
    split: List[str] = field(default_factory=list)
    links: int = 0
    compression_before: float = 1.0
    compression_after: float = 1.0
    ms: float = 0.0

    @property
    def changed(self) -> bool:
        return bool(self.formed or self.merged or self.split or self.grown)

    @property
    def gained(self) -> float:
        """Compression actually won by this pass. Negative means the pass made her worse."""
        return self.compression_after - self.compression_before

    def to_dict(self) -> Dict[str, Any]:
        return {"observations": self.observations, "formed": self.formed[:8],
                "grown": self.grown[:8], "merged": [list(m) for m in self.merged[:8]],
                "split": self.split[:8], "links": self.links, "changed": self.changed,
                "compression_before": round(self.compression_before, 4),
                "compression_after": round(self.compression_after, 4),
                "gained": round(self.gained, 4), "ms": round(self.ms, 3)}


# --------------------------------------------------------------------------- #
# The genesis
# --------------------------------------------------------------------------- #
class ConceptGenesis:
    """Invents concepts from observations, and reports how much reality they compress."""

    def __init__(self, *, min_members: int = _MIN_MEMBERS, similarity: float = _SIMILARITY,
                 invariant_share: float = _INVARIANT_SHARE, cover: float = _COVER,
                 capacity: int = 4000) -> None:
        self.min_members = max(2, int(min_members))
        self.similarity = float(similarity)
        self.invariant_share = min(1.0, max(0.1, float(invariant_share)))
        # Where the Master set the knobs. A restructure may move them, but only within a band
        # around this — otherwise a run of unclassifiable observations walks the threshold to its
        # floor one repair at a time, and at the floor everything resembles everything and every
        # concept formed is worthless. Measured: over a 28-turn session the threshold reached
        # 0.1 and the hierarchy collapsed to zero links. Local repairs must not be able to
        # accumulate into a global change nobody chose.
        self._base_similarity = float(similarity)
        self._base_share = float(self.invariant_share)
        self.cover = float(cover)
        self.capacity = max(64, int(capacity))

        self.observations: List[Observation] = []
        self.concepts: Dict[str, Concept] = {}
        self._by_subject: Dict[str, Observation] = {}
        self._seq = 0
        self._next_cid = 1
        self.passes = 0
        self.formed_total = 0
        self.merged_total = 0
        self.split_total = 0
        self.gaps_found = 0

    # ---- intake ----------------------------------------------------------- #
    def observe(self, subject: str, features: Iterable[str] = (),
                numbers: Optional[Dict[str, float]] = None) -> Optional[Observation]:
        """Record what was true of one thing. Repeat sightings *accumulate* onto that thing.

        Accumulating rather than appending is the whole reason a concept can form from a
        conversation: the Master says "paudhe ko pani chahiye" in one turn and "paudhe suraj se
        badhte hai" three turns later, and those are two facts about one plant, not two plants.
        """
        subject = _norm(subject)
        if not subject:
            return None
        feats = {_norm(f) for f in features if _norm(f)}
        nums = {_norm(k): float(v) for k, v in (numbers or {}).items()
                if isinstance(v, (int, float)) and math.isfinite(float(v))}
        if not feats and not nums:
            return None

        existing = self._by_subject.get(subject)
        if existing is not None:
            existing.features |= feats
            existing.numbers.update(nums)
            existing.at = time.time()
            return existing

        self._seq += 1
        obs = Observation(subject=subject, features=feats, numbers=nums, seq=self._seq)
        self.observations.append(obs)
        self._by_subject[subject] = obs
        if len(self.observations) > self.capacity:
            dropped = self.observations[: len(self.observations) - self.capacity]
            self.observations = self.observations[len(dropped):]
            for gone in dropped:
                self._by_subject.pop(gone.subject, None)
        return obs

    def observe_triples(self, triples: Iterable[Any]) -> int:
        """Feed grounded ``(subject, predicate, object)`` relations straight in.

        A relation *is* a feature of its subject — ``dog is_a animal`` says the feature
        ``is_a:animal`` holds of ``dog``. That equivalence is why no separate teaching channel is
        needed: everything the grounder already extracts is concept-forming evidence.
        """
        seen = 0
        for triple in (triples or []):
            try:
                subject = _norm(getattr(triple, "subject", ""))
                predicate = _norm(getattr(triple, "predicate", ""))
                obj = _norm(getattr(triple, "object", ""))
                if not subject or not predicate:
                    continue
                numbers: Dict[str, float] = {}
                feature = f"{predicate}:{obj}" if obj else predicate
                # A numeric object is a quantity, not a category: "boils:100" would make water
                # and mercury unrelated kinds, when what they share is *having* a boiling point.
                if obj:
                    try:
                        numbers[predicate] = float(obj)
                        feature = predicate
                    except (TypeError, ValueError):
                        pass
                if self.observe(subject, [feature], numbers) is not None:
                    seen += 1
                # The object of a relation is a thing too, and knowing it is pointed *at* is how
                # a superordinate ever accumulates evidence of its own.
                if obj and predicate not in ("has_name",) and not numbers:
                    self.observe(obj, [f"is_target_of:{predicate}"])
            except Exception:  # noqa: BLE001 — one malformed triple never stops the intake
                continue
        return seen

    # ---- similarity ------------------------------------------------------- #
    def _similarity(self, a: Observation, b: Observation) -> float:
        """Jaccard on the symbols, cosine on the numbers, blended by how much of each exists.

        Role markers are excluded, and that exclusion is load-bearing. ``observe_triples`` records
        the *object* of every relation as an observation carrying one feature —
        ``is_target_of:is_a`` — so that a superordinate has somewhere to accumulate evidence. But
        a role marker is not a property of the thing: it says something points at it, which is
        true of every kind she has ever met. Two such observations have identical feature sets and
        therefore Jaccard 1.0, so they all collapse into one concept.

        Measured: the largest concept after 1,200 corpus pairs was named ``is_target_of:is_a`` and
        had **230 members** — "things that something is a kind of", true of everything and
        informative about nothing. It dominated the concept set and dragged compression under
        break-even, which then vetoed every self-improvement trial.
        """
        af, bf = _content_features(a.features), _content_features(b.features)
        sym = 0.0
        sym_weight = 0.0
        if af or bf:
            union = af | bf
            if union:
                sym = len(af & bf) / len(union)
                sym_weight = 1.0

        num = 0.0
        num_weight = 0.0
        shared = set(a.numbers) & set(b.numbers)
        if shared:
            dot = sum(a.numbers[k] * b.numbers[k] for k in shared)
            na = math.sqrt(sum(a.numbers[k] ** 2 for k in shared))
            nb = math.sqrt(sum(b.numbers[k] ** 2 for k in shared))
            if na > 0 and nb > 0:
                num = max(0.0, dot / (na * nb))
                # Sharing a *dimension* is itself the evidence; the cosine only grades it. Two
                # things that both have a boiling point are more alike than two things where one
                # does, however far apart the values are.
                num_weight = 0.5
        total = sym_weight + num_weight
        if total <= 0:
            return 0.0
        return (sym * sym_weight + num * num_weight) / total

    def _prototype(self, members: Sequence[Observation]) -> Dict[str, float]:
        """The centroid of the members' numeric features."""
        acc: Dict[str, List[float]] = {}
        for obs in members:
            for key, val in obs.numbers.items():
                acc.setdefault(key, []).append(val)
        return {k: sum(v) / len(v) for k, v in acc.items() if v}

    def _invariants(self, members: Sequence[Observation]) -> Set[str]:
        """Features held by at least ``invariant_share`` of the members."""
        if not members:
            return set()
        counts: Dict[str, int] = {}
        for obs in members:
            # Content only. A concept whose *claim* is a role marker asserts that something points
            # at its members — true of every kind she has ever met, and the adversarial battery's
            # own rule that "no concept may assert nothing" in spirit if not in letter.
            for feat in _content_features(obs.features):
                counts[feat] = counts.get(feat, 0) + 1
        need = max(2, math.ceil(self.invariant_share * len(members)))
        return {f for f, n in counts.items() if n >= need}

    def _typical(self, members: Sequence[Observation],
                 invariants: Set[str]) -> Dict[str, float]:
        """Features most members carry but the concept does not claim, with their share.

        The floor is half: a feature a minority carries is not what the kind looks like, and
        transferring it would be guessing dressed as inference.
        """
        if not members:
            return {}
        counts: Dict[str, int] = {}
        for obs in members:
            for feat in obs.features:
                counts[feat] = counts.get(feat, 0) + 1
        total = len(members)
        return {f: n / total for f, n in counts.items()
                if f not in invariants and n / total >= 0.5}

    # ---- the pass --------------------------------------------------------- #
    def crystallise(self) -> GenesisReport:
        """One full pass: cluster, extract invariants, derive the hierarchy, measure the win.

        Recomputed from the observations each time rather than updated in place. That is more
        work and it is the honest choice: a concept system that only ever accretes cannot
        *unlearn* a boundary it drew badly, and unlearning is the whole point of the restructure
        signal this module exists to produce.
        """
        rep = GenesisReport(observations=len(self.observations))
        t0 = time.perf_counter()
        try:
            rep.compression_before = self.compression()
            before = set(self.concepts)
            self.concepts = {}

            # A hierarchy cannot come out of one clustering, however good. Cutting the tree at a
            # single height gives one row of sibling kinds and no ancestry — which is exactly
            # what a first attempt here produced: six animals and plants collapsed into one
            # concept "living+needs:energy", nothing above it and nothing below, `links = 0`.
            # ``dog → animal → living thing`` needs the same observations grouped *coarsely* and
            # *finely* at once, so the coarse grouping can turn out to subsume the fine one.
            # These are cuts of the same dendrogram at several heights.
            for level, members_group in enumerate(self._groupings()):
                for members in members_group:
                    if len(members) < self.min_members:
                        continue
                    invariants = self._invariants(members)
                    if not invariants:
                        # A cluster with nothing in common is a coincidence of the similarity
                        # threshold, not a kind. Dropping it is what stops the hierarchy filling
                        # with concepts that assert nothing.
                        continue
                    cid = self._cid_for(invariants)
                    existing = self.concepts.get(cid)
                    if existing is not None:
                        # The same kind found again at a different resolution. Keep the wider
                        # membership: a concept's extension is every observation that satisfies
                        # it, not just the ones the tightest cut happened to group.
                        for member in members:
                            if member.subject not in existing.members:
                                existing.members.append(member.subject)
                        continue
                    self.concepts[cid] = Concept(
                        cid=cid, members=[m.subject for m in members],
                        invariants=invariants, typical=self._typical(members, invariants),
                        prototype=self._prototype(members), level=level)

            # Membership is assigned once, globally, to the most *specific* concept that covers
            # each observation — so an observation lives at its leaf and inherits the rest.
            self._assign_members()

            for cid in self.concepts:
                (rep.grown if cid in before else rep.formed).append(cid)
            self.formed_total += len(rep.formed)

            rep.links = self._derive_hierarchy()
            rep.compression_after = self.compression()
            self.passes += 1
            return rep
        except Exception:  # noqa: BLE001 — a failed pass leaves the previous concepts standing
            return rep
        finally:
            rep.ms = (time.perf_counter() - t0) * 1000.0

    def _ladder(self) -> List[float]:
        """The heights this pass cuts the tree at — specific first, general last.

        Anchored on :attr:`similarity` so the Master's configured tightness stays the middle of
        the range rather than being ignored, and clipped into ``(0, 1)`` so a restructure that
        loosens the threshold a long way cannot produce a cut at zero, where everything is
        similar to everything and the "concept" formed is the whole world.
        """
        base = self.similarity
        raw = [base + 0.34, base + 0.17, base, max(0.1, base - 0.15)]
        out: List[float] = []
        for value in raw:
            value = round(min(0.92, max(0.08, value)), 4)
            if value not in out:
                out.append(value)
        return out

    def _groupings(self) -> List[List[List[Observation]]]:
        """Every grouping of the observations this pass will try to make a concept out of.

        Two sources, and the second is not an optimisation — it is the only way a superordinate
        ever forms. Similarity cuts find kinds whose members resemble each other *overall*, and
        by that measure a dog and an oak do not: they agree on one feature out of seven, which no
        honest similarity threshold calls kinship. Yet "living thing" is a real concept, and what
        makes it one is precisely that a single property is shared very widely. So a feature held
        by enough observations is itself a candidate grouping — the frequent-itemset reading of a
        kind — and it supplies exactly the general end of the taxonomy that similarity cannot see.

        Measured before this existed: six observations that all shared ``living`` produced a flat
        row of sibling concepts and ``hierarchy_links = 0``.
        """
        out: List[List[List[Observation]]] = []
        for threshold in self._ladder():
            out.append(self._cluster(threshold))

        by_feature: Dict[str, List[Observation]] = {}
        for obs in self.observations:
            # Content only, for the same reason the similarity cut excludes them: a role marker is
            # shared by every object of every relation, so bucketing on one produces a "kind"
            # holding a third of the store. Measured after the similarity fix alone, the largest
            # concept was still `is_target_of:known_for` with 122 members — the frequent-itemset
            # path had been reading raw features and rebuilding the same degenerate group.
            for feat in _content_features(obs.features):
                by_feature.setdefault(feat, []).append(obs)
        frequent = [members for members in by_feature.values()
                    if len(members) >= max(self.min_members, 3)]
        if frequent:
            # Widest first, so the most general candidate is offered before the narrower ones.
            frequent.sort(key=len, reverse=True)
            out.append(frequent[:32])
        return out

    def _cluster(self, threshold: float) -> List[List[Observation]]:
        """Single-pass agglomeration against running cluster centroids, at one cut height.

        Cheap on purpose, and it had stopped being cheap. Scoring every observation against every
        live centroid is ``O(n·k)`` and both grow, which profiled at **12.8 million**
        ``_similarity`` calls over 100 turns — 58% of the whole turn, and rising with the store.

        The comparison is skipped rather than approximated. Jaccard over disjoint feature sets is
        exactly zero, so a centroid sharing no content feature with the observation can never win
        and never needed scoring; an inverted index over the features finds the ones that can.
        The clustering produced is identical, which matters — a faster clustering that grouped
        differently would silently change what she believes a kind is.
        """
        clusters: List[List[Observation]] = []
        summaries: List[Observation] = []          # a synthetic centroid per cluster
        # feature -> indices of centroids carrying it, so only reachable centroids are scored.
        index: Dict[str, Set[int]] = {}

        def _reindex(idx: int, summary: Observation) -> None:
            for feature in _content_features(summary.features):
                index.setdefault(feature, set()).add(idx)

        for obs in self.observations:
            features = _content_features(obs.features)
            candidates: Set[int] = set()
            for feature in features:
                candidates |= index.get(feature, set())
            # A numeric observation can be similar without sharing a symbol, so those still see
            # every centroid. There are few of them and the fast path is for the symbolic bulk.
            if obs.numbers:
                candidates = set(range(len(summaries)))

            best = -1
            best_score = 0.0
            for idx in candidates:
                score = self._similarity(obs, summaries[idx])
                if score > best_score:
                    best, best_score = idx, score
            if best >= 0 and best_score >= threshold:
                clusters[best].append(obs)
                summaries[best] = self._summarise(clusters[best])
                _reindex(best, summaries[best])
            else:
                clusters.append([obs])
                summary = Observation(subject=obs.subject,
                                      features=set(obs.features),
                                      numbers=dict(obs.numbers))
                summaries.append(summary)
                _reindex(len(summaries) - 1, summary)
        return clusters

    def _assign_members(self) -> None:
        """Put every observation under the most specific concept whose claim it satisfies.

        "Most specific" is the one with the most invariants, because that is the concept that
        says the most about it. Ties go to the better-supported concept, so a well-evidenced kind
        beats a narrow accident with the same number of invariants.
        """
        for obs in self.observations:
            obs.concept = ""
            best: Optional[Concept] = None
            for concept in self.concepts.values():
                if concept.invariants - obs.features:
                    continue                       # it does not actually satisfy the claim
                if best is None or (concept.order, concept.support) > (best.order, best.support):
                    best = concept
            if best is not None:
                obs.concept = best.cid
        # Members are then exactly the observations that satisfy each concept — recomputed, so a
        # concept can never keep claiming a member the evidence stopped supporting.
        holders: Dict[str, List[str]] = {cid: [] for cid in self.concepts}
        for obs in self.observations:
            for concept in self.concepts.values():
                if not (concept.invariants - obs.features):
                    holders[concept.cid].append(obs.subject)
        for cid, members in holders.items():
            concept = self.concepts[cid]
            concept.members = members
            observations = [o for o in (self._by_subject.get(m) for m in members)
                            if o is not None]
            # Recomputed here, not left at whatever the forming cluster produced. A concept's
            # extension grows during assignment — the cluster that formed the animal concept was
            # three animals, all of which made a sound, so "makes:sound" counted as invariant and
            # never became typical; the fourth animal then inherited nothing. Typicality is a
            # property of the members a concept actually has, so it has to be measured after it
            # has them.
            concept.typical = self._typical(observations, concept.invariants)
            concept.prototype = self._prototype(observations) or concept.prototype
            concept.violations = sum(1 for o in observations if concept.violated_by(o))

    def _summarise(self, members: Sequence[Observation]) -> Observation:
        """A stand-in observation carrying the cluster's shared features and mean numbers."""
        return Observation(subject="", features=self._invariants(members) or
                           set().union(*[m.features for m in members]) if members else set(),
                           numbers=self._prototype(members))

    def _cid_for(self, invariants: Set[str]) -> str:
        """A stable id derived from the invariants, so the same kind keeps the same name.

        Deriving it from content rather than from a counter is what lets ``crystallise`` rebuild
        from scratch every pass without the hierarchy's ids churning underneath anything that
        stored one.
        """
        key = "|".join(sorted(invariants)[:6])
        return "c:" + str(abs(hash(key)) % (10 ** 9))

    def _derive_hierarchy(self) -> int:
        """Subsumption: A is under B when A's invariants strictly contain B's.

        The parent chosen is the *most specific* concept that still subsumes — the nearest
        ancestor — so ``dog`` hangs under ``animal`` rather than jumping straight to the root.
        """
        links = 0
        for concept in self.concepts.values():
            concept.parent = ""
            concept.children = []
        for child in self.concepts.values():
            best: Optional[Concept] = None
            for parent in self.concepts.values():
                if parent.cid == child.cid:
                    continue
                if not parent.invariants < child.invariants:
                    continue
                if best is None or parent.order > best.order:
                    best = parent
            if best is not None:
                child.parent = best.cid
                best.children.append(child.cid)
                links += 1
        return links

    # ---- reading it back --------------------------------------------------- #
    def explain(self, subject: str, features: Iterable[str] = (),
                numbers: Optional[Dict[str, float]] = None) -> Coverage:
        """What kind is this — and if nothing fits, say precisely what did not.

        The ``gap`` is the product. ``unknown`` means no concept comes close, which calls for a
        new one; ``violates`` means the best concept *nearly* fits but its claim is broken by
        this member, which calls for the boundary to move. Those are different repairs, and a
        system that reports only "wrong" cannot tell them apart.
        """
        subject = _norm(subject)
        feats = {_norm(f) for f in features if _norm(f)}
        if not feats:
            known = self._by_subject.get(subject)
            if known is not None:
                feats = set(known.features)
        probe = Observation(subject=subject, features=feats,
                            numbers={_norm(k): float(v) for k, v in (numbers or {}).items()})

        scored: List[Tuple[float, Concept]] = []
        for concept in self.concepts.values():
            scored.append((concept.covers(probe), concept))
        scored.sort(key=lambda p: p[0], reverse=True)

        out = Coverage(subject=subject)
        if not scored or scored[0][0] <= 0.0:
            out.gap = "unknown"
            self.gaps_found += 1
            return out

        score, best = scored[0]
        runner = scored[1][0] if len(scored) > 1 else 0.0
        out.concept, out.score, out.margin = best.cid, score, score - runner
        missing = best.violated_by(probe)
        if score >= self.cover and not missing:
            return out
        out.violated = sorted(missing)[:8]
        out.gap = "violates" if score >= self.cover else "unknown"
        self.gaps_found += 1
        return out

    def ancestors(self, cid: str) -> List[str]:
        """The chain upward — ``dog → animal → living thing``, as far as it goes."""
        out: List[str] = []
        seen: Set[str] = set()
        current = self.concepts.get(cid)
        while current is not None and current.parent and current.parent not in seen:
            seen.add(current.parent)
            out.append(current.parent)
            current = self.concepts.get(current.parent)
        return out

    def generalise(self, subject: str) -> List[str]:
        """What she can say about a thing from its kind that she never observed of the thing.

        This is transfer, and it is *defeasible* — "most animals have legs; this is an animal;
        so probably legs". It draws on the concept's :attr:`Concept.typical` features rather than
        its invariants, because the invariants are satisfied by every member by construction and
        so can never say anything new. Ordered by how typical, most-typical first, so a caller
        that takes only the head takes the safest inference.
        """
        obs = self._by_subject.get(_norm(subject))
        if obs is None or not obs.concept:
            return []
        scored: Dict[str, float] = {}
        for cid in [obs.concept] + self.ancestors(obs.concept):
            concept = self.concepts.get(cid)
            if concept is None:
                continue
            for feature, share in concept.typical.items():
                if feature in obs.features:
                    continue
                scored[feature] = max(scored.get(feature, 0.0), share)
        return [f for f, _ in sorted(scored.items(), key=lambda kv: -kv[1])][:16]

    def roots(self) -> List[Concept]:
        """The most general concepts — the ones nothing subsumes."""
        return sorted((c for c in self.concepts.values() if not c.parent),
                      key=lambda c: c.support, reverse=True)

    # ---- the measurement --------------------------------------------------- #
    def compression(self) -> float:
        """Minimum-description-length ratio: raw cost over modelled cost. ``> 1`` is a real win.

        Raw cost is one symbol per feature per observation — storing everything verbatim.
        Modelled cost is the concepts' invariants (paid once) plus every observation's residual
        against the concept that claimed it (the features the concept did *not* predict, plus the
        invariants it wrongly promised). Unclaimed observations pay full price, so a concept
        system that covers little cannot score well by covering it perfectly.

        The asymmetry matters: a concept that over-claims is charged for the invariants its
        member lacks, so inventing one enormous concept that swallows everything makes the number
        *worse*, not better. That is what stops this from being a metric she can game.
        """
        # Content only, throughout. A node recorded solely because a relation points at it has
        # nothing to compress: it costs one symbol raw and one symbol modelled, so every such
        # node drags the ratio toward exactly 1.0 regardless of how well the real concepts are
        # doing. With 445 observations of which most were relation targets, that dilution was
        # most of the number.
        raw = sum(len(_content_features(o.features)) for o in self.observations)
        if raw <= 0:
            return 1.0
        model = sum(len(c.invariants) for c in self.concepts.values())
        for obs in self.observations:
            features = _content_features(obs.features)
            if not features:
                continue                                       # nothing to describe either way
            concept = self.concepts.get(obs.concept)
            if concept is None:
                model += len(features)
                continue
            model += len(features - concept.invariants)        # not predicted
            model += len(concept.invariants - features)        # wrongly promised
        return raw / model if model > 0 else 1.0

    # ---- restructuring ------------------------------------------------------ #
    def restructure(self, coverage: Coverage) -> GenesisReport:
        """Act on an insufficiency: widen a broken boundary, or let a new kind form.

        This is the method :mod:`nyxara.njp.field` calls when a prediction error is diagnosed as
        conceptual rather than numeric. It does not invent an answer — it changes the conditions
        under which concepts form and then re-crystallises, so the new structure is still earned
        by the same clustering that earns every other concept. Adopted only if the compression
        ratio does not fall: a restructure that explains this one observation by making her
        model of everything else worse is a trade she should refuse, and here she does.
        """
        before_similarity = self.similarity
        before_share = self.invariant_share
        before_ratio = self.compression()
        if coverage.gap not in ("violates", "unknown"):
            return GenesisReport(observations=len(self.observations),
                                 compression_before=before_ratio,
                                 compression_after=before_ratio)
        try:
            # A single nudge is not a repair. The first version of this moved one knob by a fixed
            # step and returned whatever came out — and on the obvious test case ("horse" excluded
            # from `animal` because three of four animals happened to make a sound) the step was
            # too small to change the invariant set at all, so it reported a restructure that had
            # restructured nothing. What is wanted is the *smallest* change to the conditions
            # that actually closes this gap and does not cost compression elsewhere. That is a
            # bounded search, and it is short enough to run inside a turn.
            # The band a repair may move the knobs within, around what the Master configured.
            share_lo, share_hi = 0.6 * self._base_share, min(0.98, 1.6 * self._base_share)
            sim_lo, sim_hi = 0.5 * self._base_similarity, 1.5 * self._base_similarity

            candidates: List[Tuple[float, float]] = []
            loosening = coverage.gap == "unknown"
            if not loosening:
                # The concept promised something this member does not have. Demand more agreement
                # before a feature counts as invariant, so the boundary tightens onto what is
                # genuinely shared instead of over-claiming.
                step = before_share
                while step < share_hi:
                    step = round(min(share_hi, step + 0.07), 4)
                    candidates.append((before_similarity, step))
                    if step >= share_hi:
                        break
                self.split_total += 1
            else:
                # Nothing came close. Loosen what counts as kinship so a new kind can crystallise
                # around it rather than it sitting outside every concept forever.
                step = before_similarity
                while step > sim_lo:
                    step = round(max(sim_lo, step - 0.05), 4)
                    candidates.append((step, before_share))
                    if step <= sim_lo:
                        break
                self.merged_total += 1

            best: Optional[GenesisReport] = None
            best_setting = (before_similarity, before_share)
            for similarity, share in candidates[:12]:
                self.similarity, self.invariant_share = similarity, share
                rep = self.crystallise()
                # Never buy a local repair with a global loss. A restructure that explains this
                # one observation by making her model of everything else worse is a trade she
                # should refuse, and here she does.
                #
                # Loosening is held to the stricter test. Tightening a boundary that was
                # over-claiming is a correction and may reasonably break even; widening what
                # counts as kinship to swallow one outlier is a *generalisation*, and a
                # generalisation that buys no compression is not a discovery — it is the concept
                # system being talked into accepting anything.
                floor = before_ratio + (1e-9 if loosening else -1e-9)
                if rep.compression_after < floor:
                    continue
                closed = self.explain(coverage.subject).covered
                if closed:
                    rep.split = [coverage.concept] if coverage.gap == "violates" else []
                    rep.compression_before = before_ratio
                    return rep
                if best is None or rep.compression_after > best.compression_after:
                    best, best_setting = rep, (similarity, share)

            # Nothing closed the gap. Keep the best compression found — including the original
            # settings if no candidate beat them — rather than leaving the knobs wherever the
            # last trial left them.
            self.similarity, self.invariant_share = best_setting
            final = self.crystallise()
            final.compression_before = before_ratio
            return final
        except Exception:  # noqa: BLE001
            self.similarity = before_similarity
            self.invariant_share = before_share
            return GenesisReport(observations=len(self.observations))

    # ---- reporting ---------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        concepts = list(self.concepts.values())
        deepest = max((len(self.ancestors(c.cid)) for c in concepts), default=0)
        return {
            "observations": len(self.observations),
            "concepts": len(concepts),
            "roots": len(self.roots()),
            "depth": deepest,
            "hierarchy_links": sum(1 for c in concepts if c.parent),
            "passes": self.passes,
            "formed_total": self.formed_total,
            "merged_total": self.merged_total,
            "split_total": self.split_total,
            "gaps_found": self.gaps_found,
            "violations": sum(c.violations for c in concepts),
            "compression": round(self.compression(), 4),
            "similarity": round(self.similarity, 4),
            "invariant_share": round(self.invariant_share, 4),
            "largest": max(concepts, key=lambda c: c.support).to_dict() if concepts else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observations": [o.to_dict() | {"at": o.at} for o in self.observations[-self.capacity:]],
            "similarity": self.similarity, "invariant_share": self.invariant_share,
            "passes": self.passes, "formed_total": self.formed_total,
            "merged_total": self.merged_total, "split_total": self.split_total,
            "gaps_found": self.gaps_found, "seq": self._seq,
        }

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.observations = []
            self._by_subject = {}
            for row in (d.get("observations") or []):
                obs = Observation(
                    subject=str(row.get("subject", "")),
                    features={str(f) for f in (row.get("features") or [])},
                    numbers={str(k): float(v) for k, v in (row.get("numbers") or {}).items()},
                    at=float(row.get("at", 0.0)), seq=int(row.get("seq", 0)))
                if not obs.subject:
                    continue
                self.observations.append(obs)
                self._by_subject[obs.subject] = obs
            self.similarity = float(d.get("similarity", self.similarity))
            self.invariant_share = float(d.get("invariant_share", self.invariant_share))
            self.passes = int(d.get("passes", 0))
            self.formed_total = int(d.get("formed_total", 0))
            self.merged_total = int(d.get("merged_total", 0))
            self.split_total = int(d.get("split_total", 0))
            self.gaps_found = int(d.get("gaps_found", 0))
            self._seq = int(d.get("seq", len(self.observations)))
            # The concepts themselves are never persisted: they are a *function* of the
            # observations, and rebuilding them proves that on every load rather than trusting a
            # sidecar that may have been written by an older clustering.
            if self.observations:
                self.crystallise()
        except Exception:  # noqa: BLE001 — a corrupt sidecar leaves her with no concepts, not bad ones
            pass
