"""NYXARA · memory/retrieval.py — associative, context-cued, emotional recall (✦).

``store.py`` answers "what is similar to this text?". Real remembering is richer: it
is **cued by the whole situation** — your mood, your goals, where you are, what you
were just thinking — and it carries an **emotional colour**. This module layers that
on top of :class:`~nyxara.memory.store.MemoryStore`.

What it adds
------------
* **Multi-signal fusion** — a memory's retrieval score blends five cues:
  semantic similarity, contextual tag overlap, emotional congruence (mood-congruent
  recall), temporal proximity, and graph proximity (spreading activation from what
  you were just attending to).
* **Emotional indexing** — memories are tagged in valence/arousal space; a sad mood
  preferentially surfaces sad memories, an alert mood surfaces threats.
* **Déjà-vu = weighted signals** — familiarity (diffuse "I've seen something like
  this") is computed separately from recollection (a sharp, specific match). High
  familiarity with *low* recollection is exactly the déjà-vu sensation.

Depends on :mod:`memory.store` and :mod:`memory.provenance`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.memory.store import MemoryRecord, MemoryStore, MemoryType

__all__ = [
    "EmotionTag",
    "RetrievalContext",
    "FusionWeights",
    "LearnedReranker",
    "RetrievalResult",
    "DejaVuSignal",
    "AssociativeRetriever",
]

_RERANK_FEATURES: Tuple[str, ...] = ("semantic", "context", "emotion",
                                     "temporal", "graph", "goal")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Emotion
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EmotionTag:
    """A point in valence–arousal affect space.

    ``valence`` ∈ [-1, 1] (unpleasant…pleasant), ``arousal`` ∈ [0, 1] (calm…activated).
    """

    valence: float = 0.0
    arousal: float = 0.0

    def distance(self, other: "EmotionTag") -> float:
        # valence spans 2 units, arousal 1; normalise valence to comparable scale
        dv = (self.valence - other.valence) / 2.0
        da = self.arousal - other.arousal
        return math.sqrt(dv * dv + da * da)

    def congruence(self, other: "EmotionTag") -> float:
        """1.0 == identical affect, → 0 as they diverge (mood-congruence signal)."""
        # max distance in the normalised box is sqrt(1^2 + 1^2) = sqrt(2)
        return _clamp(1.0 - self.distance(other) / math.sqrt(2.0))

    @property
    def intensity(self) -> float:
        return _clamp(math.sqrt(self.valence * self.valence + self.arousal * self.arousal)
                      / math.sqrt(2.0))


NEUTRAL = EmotionTag(0.0, 0.0)


# --------------------------------------------------------------------------- #
# Context & weights
# --------------------------------------------------------------------------- #
@dataclass
class RetrievalContext:
    """The full situation a recall is cued by (encoding-specificity)."""

    query: str = ""
    mood: EmotionTag = NEUTRAL
    tags: frozenset = field(default_factory=frozenset)       # situational tags
    goals: Dict[str, float] = field(default_factory=dict)    # goal tag -> weight
    recent_ids: Sequence[str] = ()                           # what we just attended to
    target_time: Optional[float] = None                      # temporal cue (epoch)
    temporal_scale_days: float = 3.0
    now: Optional[float] = None

    def at(self) -> float:
        return self.now if self.now is not None else time.time()


@dataclass(frozen=True)
class FusionWeights:
    semantic: float = 1.0
    context: float = 0.6
    emotion: float = 0.5
    temporal: float = 0.3
    graph: float = 0.7
    goal: float = 0.8

    def total(self) -> float:
        return (self.semantic + self.context + self.emotion
                + self.temporal + self.graph + self.goal)


@dataclass
class LearnedReranker:
    """A small online-learned re-ranker over the retrieval signals (Pillar B5).

    Recall by fixed :class:`FusionWeights` answers "what is textually/contextually similar?".
    This learns "what actually *helps* here": it starts from the hand-tuned weights and, each
    time a surfaced memory is marked useful (reward→1) or not (reward→0), a logistic update
    nudges the per-signal weights toward what paid off. So if, for this mind, goal-overlap
    predicts usefulness better than raw cosine, the re-ranker discovers that from experience.
    Pure-stdlib, persistent, and off by default — the retriever keeps fixed weights until one
    is attached."""

    weights: Dict[str, float]
    bias: float = 0.0
    lr: float = 0.2

    @classmethod
    def from_fusion(cls, fusion: "FusionWeights", *, lr: float = 0.2) -> "LearnedReranker":
        return cls(weights={k: float(getattr(fusion, k)) for k in _RERANK_FEATURES}, lr=lr)

    def _raw(self, signals: Dict[str, float]) -> float:
        return self.bias + sum(self.weights.get(k, 0.0) * float(signals.get(k, 0.0))
                               for k in _RERANK_FEATURES)

    def score(self, signals: Dict[str, float]) -> float:
        """Predicted usefulness of a candidate, squashed to [0,1]."""
        return 1.0 / (1.0 + math.exp(-_clamp(self._raw(signals), -30.0, 30.0)))

    def learn(self, signals: Dict[str, float], reward: float) -> float:
        """One logistic step toward ``reward`` ∈ [0,1]; returns the prediction error."""
        err = _clamp(reward, 0.0, 1.0) - self.score(signals)
        for k in _RERANK_FEATURES:
            self.weights[k] += self.lr * err * float(signals.get(k, 0.0))
        self.bias += self.lr * err
        return err

    def to_dict(self) -> Dict[str, Any]:
        return {"weights": dict(self.weights), "bias": self.bias, "lr": self.lr}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LearnedReranker":
        return cls(weights={k: float(v) for k, v in d.get("weights", {}).items()},
                   bias=float(d.get("bias", 0.0)), lr=float(d.get("lr", 0.2)))


@dataclass
class RetrievalResult:
    record: MemoryRecord
    score: float
    signals: Dict[str, float]


@dataclass
class DejaVuSignal:
    is_deja_vu: bool
    familiarity: float
    recollection: float
    strength: float
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_deja_vu": self.is_deja_vu,
            "familiarity": round(self.familiarity, 4),
            "recollection": round(self.recollection, 4),
            "strength": round(self.strength, 4),
            "note": self.note,
        }


# --------------------------------------------------------------------------- #
# The retriever
# --------------------------------------------------------------------------- #
class AssociativeRetriever:
    """Context-cued, emotion-aware retrieval over a :class:`MemoryStore`."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        weights: Optional[FusionWeights] = None,
        reranker: Optional[LearnedReranker] = None,
        familiarity_threshold: float = 0.45,
        recollection_threshold: float = 0.6,
    ) -> None:
        self.store = store
        self.weights = weights or FusionWeights()
        # an optional learned re-ranker; when present it scores candidates from experience
        # instead of the fixed fusion weights (B5). None ⇒ original behaviour, unchanged.
        self.reranker = reranker
        self.familiarity_threshold = familiarity_threshold
        self.recollection_threshold = recollection_threshold
        self._emotion: Dict[str, EmotionTag] = {}

    # ---- emotional indexing ---- #
    def tag_emotion(self, mem_id: str, valence: float, arousal: float) -> EmotionTag:
        tag = EmotionTag(_clamp(valence, -1.0, 1.0), _clamp(arousal))
        self._emotion[mem_id] = tag
        # persist alongside the record for durability
        self.store.get(mem_id).metadata["emotion"] = {"valence": tag.valence, "arousal": tag.arousal}
        return tag

    def emotion_of(self, mem_id: str) -> EmotionTag:
        if mem_id in self._emotion:
            return self._emotion[mem_id]
        rec = self.store.get(mem_id)
        e = rec.metadata.get("emotion")
        if e:
            tag = EmotionTag(e["valence"], e["arousal"])
            self._emotion[mem_id] = tag
            return tag
        return NEUTRAL

    # ---- signal computations ---- #
    @staticmethod
    def _jaccard(a: frozenset, b: frozenset) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _goal_relevance(self, rec: MemoryRecord, goals: Dict[str, float]) -> float:
        if not goals:
            return 0.0
        hit = sum(goals.get(t, 0.0) for t in rec.tags)
        norm = sum(abs(v) for v in goals.values()) or 1.0
        return _clamp(hit / norm)

    def _temporal_proximity(self, rec: MemoryRecord, ctx: RetrievalContext) -> float:
        if ctx.target_time is not None:
            dt_days = abs(rec.created_at - ctx.target_time) / 86400.0
            return math.exp(-dt_days / max(0.01, ctx.temporal_scale_days))
        return rec.recency_factor(ctx.at())

    def _graph_proximity(self, ctx: RetrievalContext) -> Dict[str, float]:
        """Spreading activation seeded by everything we were just attending to."""
        if not ctx.recent_ids:
            return {}
        merged: Dict[str, float] = {}
        for seed in ctx.recent_ids:
            try:
                spread = self.store.spread(seed, hops=2, now=ctx.at())
            except Exception:
                continue
            for mid, act in spread.items():
                if act > merged.get(mid, 0.0):
                    merged[mid] = act
        return merged

    # ---- main retrieval ---- #
    def retrieve(
        self,
        ctx: RetrievalContext,
        *,
        k: int = 10,
        mem_type: Optional[MemoryType] = None,
        min_trust: float = 0.0,
        candidate_pool: int = 40,
    ) -> List[RetrievalResult]:
        now = ctx.at()
        graph_act = self._graph_proximity(ctx)

        # semantic candidate pool (plus anything reachable in the graph)
        sem_hits: Dict[str, float] = {}
        if ctx.query:
            for rec, _ in self.store.recall(ctx.query, k=candidate_pool, mem_type=mem_type,
                                            strengthen=False, now=now):
                sem_hits[rec.mem_id] = 0.0
            # recompute raw cosine similarity for the pool
            qvec = self.store.embedder.embed(ctx.query)
            for mid in list(sem_hits):
                emb = self.store.get(mid).embedding
                if emb:
                    from nyxara.memory.store import _cosine
                    sem_hits[mid] = max(0.0, _cosine(qvec, emb))

        candidate_ids = set(sem_hits) | set(graph_act)
        if not candidate_ids:
            candidate_ids = {r.mem_id for r in self.store.by_type(mem_type)} if mem_type \
                else set(self.store._kv.keys())

        w = self.weights
        results: List[RetrievalResult] = []
        for mid in candidate_ids:
            rec = self.store._kv.get(mid)
            if rec is None:
                continue
            if mem_type is not None and rec.mem_type is not mem_type:
                continue
            trust = rec.provenance.trust(now)
            if trust < min_trust:
                continue

            sem = sem_hits.get(mid, 0.0)
            cxt = self._jaccard(rec.tags, ctx.tags)
            emo = self.emotion_of(mid).congruence(ctx.mood) if (ctx.mood != NEUTRAL) else 0.0
            tmp = self._temporal_proximity(rec, ctx)
            grp = _clamp(graph_act.get(mid, 0.0))
            goal = self._goal_relevance(rec, ctx.goals)

            signals = {"semantic": sem, "context": cxt, "emotion": emo,
                       "temporal": tmp, "graph": grp, "goal": goal}
            if self.reranker is not None:
                base = self.reranker.score(signals)          # learned usefulness in [0,1]
            else:
                base = (w.semantic * sem + w.context * cxt + w.emotion * emo
                        + w.temporal * tmp + w.graph * grp + w.goal * goal) / w.total()
            # trust modulates the final pull (don't surface distrusted memories loudly)
            score = base * (0.5 + 0.5 * trust)

            signals["trust"] = trust
            results.append(RetrievalResult(record=rec, score=score, signals=signals))

        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:k]
        for r in results:
            r.record.touch(now)
        return results

    # ---- learning from feedback (B5) ---- #
    def record_feedback(self, results: Sequence[RetrievalResult],
                        useful_ids: Sequence[str], *, reward: float = 1.0) -> None:
        """Teach the re-ranker which surfaced memories actually helped this turn.

        Memories in ``useful_ids`` are reinforced toward ``reward``; the rest of the surfaced
        set are pushed toward 0 (they were retrieved but didn't help), so the model learns the
        signal mix that predicts usefulness for *this* mind. A no-op without a re-ranker."""
        if self.reranker is None:
            return
        useful = set(useful_ids)
        for res in results:
            target = reward if res.record.mem_id in useful else 0.0
            self.reranker.learn(res.signals, target)

    # ---- mood-congruent ---- #
    def mood_congruent(self, mood: EmotionTag, *, k: int = 5,
                       now: Optional[float] = None) -> List[RetrievalResult]:
        ctx = RetrievalContext(mood=mood, now=now)
        scored: List[RetrievalResult] = []
        for mid, rec in self.store._kv.items():
            cong = self.emotion_of(mid).congruence(mood)
            scored.append(RetrievalResult(record=rec, score=cong, signals={"emotion": cong}))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    # ---- familiarity, recollection, déjà-vu ---- #
    def recollection(self, ctx: RetrievalContext, now: Optional[float] = None) -> float:
        """Sharpness of the single best specific match (do we actually remember it?)."""
        if not ctx.query:
            return 0.0
        hits = self.store.recall(ctx.query, k=1, strengthen=False, now=now or ctx.at())
        if not hits:
            return 0.0
        qvec = self.store.embedder.embed(ctx.query)
        emb = hits[0][0].embedding
        if not emb:
            return 0.0
        from nyxara.memory.store import _cosine
        return _clamp(_cosine(qvec, emb))

    def familiarity(self, ctx: RetrievalContext, *, top: int = 8,
                    now: Optional[float] = None) -> float:
        """Diffuse sense of having-seen-something-like-this (averaged gist match)."""
        if not ctx.query:
            return 0.0
        now = now or ctx.at()
        hits = self.store.recall(ctx.query, k=top, strengthen=False, now=now)
        if not hits:
            return 0.0
        qvec = self.store.embedder.embed(ctx.query)
        from nyxara.memory.store import _cosine
        sims = []
        for rec, _ in hits:
            if rec.embedding:
                sims.append(max(0.0, _cosine(qvec, rec.embedding)))
        if not sims:
            return 0.0
        # gist = average of the diffuse field, gently boosted by contextual overlap
        gist = sum(sims) / len(sims)
        ctx_boost = max((self._jaccard(r.tags, ctx.tags) for r, _ in hits), default=0.0)
        return _clamp(0.8 * gist + 0.2 * ctx_boost)

    def deja_vu(self, ctx: RetrievalContext, now: Optional[float] = None) -> DejaVuSignal:
        """High familiarity + low recollection == the déjà-vu sensation."""
        now = now or ctx.at()
        fam = self.familiarity(ctx, now=now)
        rec = self.recollection(ctx, now=now)
        is_dv = fam >= self.familiarity_threshold and rec < self.recollection_threshold
        strength = _clamp(fam - rec) if is_dv else 0.0
        note = ("familiar gist with no specific source — déjà-vu" if is_dv
                else ("clear recollection" if rec >= self.recollection_threshold
                      else "novel"))
        return DejaVuSignal(is_deja_vu=is_dv, familiarity=fam, recollection=rec,
                            strength=strength, note=note)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.memory.provenance import Provenance, SourceType

    print("=" * 70)
    print("NYXARA associative-retrieval self-test")
    print("=" * 70)

    store = MemoryStore()
    r = AssociativeRetriever(store)

    # build an emotionally-coloured, linked memory landscape
    m_threat = store.remember("An intruder attacked the network at midnight",
                              mem_type=MemoryType.EPISODIC,
                              provenance=Provenance(SourceType.SENSOR, confidence=0.9),
                              tags=["security", "threat", "night"], importance=0.9)
    m_calm = store.remember("A quiet evening reading by the fire",
                            mem_type=MemoryType.EPISODIC, tags=["leisure", "night"],
                            importance=0.5)
    m_owner = store.remember("The owner praised my defense work",
                             mem_type=MemoryType.EPISODIC,
                             provenance=Provenance(SourceType.OWNER, confidence=1.0),
                             tags=["owner", "praise"], importance=0.8)
    m_chess = store.remember("Studied chess opening theory", mem_type=MemoryType.SEMANTIC,
                             tags=["chess", "strategy"], importance=0.6)
    m_war = store.remember("Read about defensive warfare strategy", mem_type=MemoryType.SEMANTIC,
                           tags=["war", "strategy", "security"], importance=0.6)

    r.tag_emotion(m_threat.mem_id, valence=-0.8, arousal=0.9)   # fearful
    r.tag_emotion(m_calm.mem_id, valence=0.6, arousal=0.1)      # serene
    r.tag_emotion(m_owner.mem_id, valence=0.9, arousal=0.5)     # proud/happy
    store.link(m_chess.mem_id, m_war.mem_id, "related")

    # 1) mood-congruent recall: an anxious mood surfaces the threat memory
    anxious = EmotionTag(valence=-0.7, arousal=0.85)
    mc = r.mood_congruent(anxious, k=2)
    print(f"\nmood-congruent (anxious): {[m.record.text()[:30] for m in mc]}")
    assert mc[0].record.mem_id == m_threat.mem_id

    happy = EmotionTag(valence=0.85, arousal=0.5)
    mc2 = r.mood_congruent(happy, k=1)
    assert mc2[0].record.mem_id == m_owner.mem_id
    print(f"mood-congruent (happy)  : {mc2[0].record.text()[:30]}")

    # 2) context-cued fusion: query + situational tags + graph seed + goal
    ctx = RetrievalContext(
        query="strategy for defense",
        tags=frozenset({"security"}),
        goals={"security": 1.0},
        recent_ids=[m_chess.mem_id],   # we were just thinking about chess
        mood=anxious,
    )
    fused = r.retrieve(ctx, k=3)
    print(f"\nfused retrieval         :")
    for res in fused:
        print(f"  {res.score:.3f}  {res.record.text()[:34]:36} "
              f"sem={res.signals['semantic']:.2f} ctx={res.signals['context']:.2f} "
              f"graph={res.signals['graph']:.2f} goal={res.signals['goal']:.2f}")
    assert fused

    # 3) déjà-vu: a gist-similar but never-stored phrasing
    store2 = MemoryStore()
    r2 = AssociativeRetriever(store2)
    for txt in ["walking through an old european town square at dusk",
                "strolling across a historic plaza in the evening light",
                "wandering an ancient cobblestone marketplace at twilight"]:
        store2.remember(txt, mem_type=MemoryType.EPISODIC, tags=["travel"])
    dv = r2.deja_vu(RetrievalContext(query="ambling through an antique town plaza come nightfall",
                                     tags=frozenset({"travel"})))
    print(f"\ndéjà-vu probe           : {dv.to_dict()}")

    # an exact-stored query should read as recollection, not déjà-vu
    dv2 = r2.deja_vu(RetrievalContext(query="walking through an old european town square at dusk"))
    print(f"recollection probe      : {dv2.to_dict()}")
    assert dv2.recollection >= dv.recollection

    print("\nALL SELF-TESTS PASSED ✓")
