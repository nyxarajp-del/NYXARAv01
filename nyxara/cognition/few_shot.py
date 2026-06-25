"""NYXARA · cognition/few_shot.py — sample-efficient / one-shot learning (🪶).

NYXARA's reasoner is fixed weights and her foundry retrains only on a slow cadence over
a whole corpus. Neither can pick up a *new concept from a single example* in the moment —
the cheap, fast learning a mind needs when the Master says "this is a frobnicator" once and
expects it to stick. This module is that channel: **prototype-based few-shot learning** in
the memory store's own embedding space (no gradient loop, no retrain).

Two complementary mechanisms:

* :class:`FewShotLearner` — a *prototypical-network* style classifier. Each concept owns a
  running **prototype** (centroid of its example embeddings). One example already forms a
  usable prototype (one-shot); more examples sharpen it (few-shot). Classification is
  nearest-prototype by cosine, with a calibrated margin → confidence. This is genuine
  abstraction-by-averaging: the centroid is the concept's *gist*, not any single example.

* :class:`OneShotEpisodicMemory` — remember a single ``(cue → response)`` pair and reproduce
  it on the next semantically-similar cue (the "you told me once" capability). One exposure,
  permanent recall — the extreme of sample efficiency.

Both reuse the active :class:`~nyxara.memory.store.MemoryStore` embedder (``.embed`` / ``.dim``)
so they live in the *same* vector space as the rest of recall, and both persist through the
store (tags ``concept`` / ``oneshot``) so what is learned once survives restarts. With no
store an in-process dict is used instead. Pure standard library.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Prototype", "FewShotLearner", "OneShotEpisodicMemory", "cosine"]


# --------------------------------------------------------------------------- #
# vector helpers (pure stdlib)
# --------------------------------------------------------------------------- #
def cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 on a zero vector)."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = na = nb = 0.0
    for i in range(n):
        x, y = a[i], b[i]
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _running_mean(centroid: List[float], support: int, vec: List[float]) -> List[float]:
    """Incrementally fold ``vec`` into a centroid of ``support`` prior vectors."""
    if support <= 0 or not centroid:
        return list(vec)
    n = min(len(centroid), len(vec))
    out = list(centroid[:n])
    inv = 1.0 / (support + 1)
    for i in range(n):
        out[i] = (centroid[i] * support + vec[i]) * inv
    return out


# --------------------------------------------------------------------------- #
# Prototype
# --------------------------------------------------------------------------- #
@dataclass
class Prototype:
    """The learned gist of a concept: a centroid in embedding space + a few exemplars."""
    label: str
    centroid: List[float]
    support: int = 0
    examples: List[str] = field(default_factory=list)
    proto_id: str = field(default_factory=lambda: "proto-" + uuid.uuid4().hex[:8])
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0

    def fold(self, vec: List[float], example: str = "") -> None:
        self.centroid = _running_mean(self.centroid, self.support, vec)
        self.support += 1
        if example and example not in self.examples:
            self.examples.append(example)
            if len(self.examples) > 5:           # keep a small, representative window
                self.examples = self.examples[-5:]

    def touch(self) -> None:
        self.last_used = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {"proto_id": self.proto_id, "label": self.label,
                "centroid": [round(x, 6) for x in self.centroid], "support": self.support,
                "examples": list(self.examples), "created_at": self.created_at,
                "last_used": self.last_used}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Prototype":
        return cls(label=str(d.get("label", "")),
                   centroid=[float(x) for x in d.get("centroid", [])],
                   support=int(d.get("support", 0)),
                   examples=list(d.get("examples", [])),
                   proto_id=str(d.get("proto_id", "proto-" + uuid.uuid4().hex[:8])),
                   created_at=float(d.get("created_at", time.time())),
                   last_used=float(d.get("last_used", 0.0)))


# --------------------------------------------------------------------------- #
# Few-shot prototype classifier
# --------------------------------------------------------------------------- #
class FewShotLearner:
    """Learn concepts from 1–few labelled examples; classify by nearest prototype."""

    TAG = "concept"

    def __init__(self, embedder: Any, *, store: Any = None) -> None:
        if embedder is None or not hasattr(embedder, "embed"):
            raise ValueError("FewShotLearner needs an embedder exposing .embed(text)->vector")
        self.embedder = embedder
        self.store = store
        self._protos: Dict[str, Prototype] = {}
        self._hydrated = False

    # ---- learn ---- #
    def learn(self, label: str, *examples: str) -> Prototype:
        """Teach ``label`` from one or more examples. One example is enough (one-shot)."""
        self._hydrate()
        label = str(label).strip()
        if not label:
            raise ValueError("a concept needs a non-empty label")
        proto = self._protos.get(label) or Prototype(label=label, centroid=[])
        learned = False
        for ex in examples or ("",):
            text = str(ex).strip()
            if not text:
                continue
            proto.fold(self.embedder.embed(text), example=text)
            learned = True
        if not learned and proto.support == 0:
            # label-only teaching: anchor the prototype on the label text itself
            proto.fold(self.embedder.embed(label), example=label)
        proto.touch()
        self._protos[label] = proto
        self._persist(proto)
        return proto

    def learn_one(self, label: str, example: str) -> Prototype:
        """Explicit one-shot: a single example forms (or refines) the prototype."""
        return self.learn(label, example)

    # ---- classify ---- #
    def _ranked(self, query: str) -> List[Tuple[str, float, float]]:
        """Rank concepts as (label, raw_cosine, softmax_confidence), most-similar first."""
        self._hydrate()
        if not self._protos:
            return []
        qvec = self.embedder.embed(str(query))
        sims = []
        for proto in self._protos.values():
            sims.append((proto.label, cosine(qvec, proto.centroid), proto))
        sims.sort(key=lambda t: t[1], reverse=True)
        # softmax over a temperature-scaled similarity gives a *relative* confidence that
        # falls when several prototypes are equally close (genuine ambiguity). The raw cosine
        # is kept too: it is the *absolute* recognition signal (one concept ⇒ softmax is always
        # 1.0, so the raw similarity is what tells us the query is about that concept at all).
        temp = 0.15
        exps = [math.exp(s / temp) for _, s, _ in sims]
        total = sum(exps) or 1.0
        out: List[Tuple[str, float, float]] = []
        for (label, raw, proto), e in zip(sims, exps):
            proto.touch()
            out.append((label, raw, e / total))
        return out

    def classify(self, query: str, *, k: int = 1) -> List[Tuple[str, float]]:
        """Rank concepts by similarity to ``query``; confidence is a softmax over similarities."""
        return [(label, conf) for label, _raw, conf in self._ranked(query)[:max(1, k)]]

    def best(self, query: str) -> Optional[Tuple[str, float]]:
        ranked = self.classify(query, k=1)
        return ranked[0] if ranked else None

    def as_prompt(self, query: str, *, threshold: float = 0.45) -> str:
        """Grounding block naming the most likely learned concept (empty when unsure).

        Gated on the *absolute* similarity (raw cosine), so an off-topic query — which may
        still pin softmax to 1.0 when only one concept exists — does not falsely fire."""
        ranked = self._ranked(query)
        if not ranked:
            return ""
        label, raw, conf = ranked[0]
        if raw < threshold:
            return ""
        proto = self._protos.get(label)
        seen = ("; e.g. " + ", ".join(proto.examples[:2])) if proto and proto.examples else ""
        return (f"\n\nLearned concept (recognised from few-shot teaching): this looks like "
                f"{label!r} (confidence {conf:.2f}{seen}).")

    # ---- introspection ---- #
    def labels(self) -> List[str]:
        self._hydrate()
        return sorted(self._protos)

    def __len__(self) -> int:
        self._hydrate()
        return len(self._protos)

    # ---- persistence ---- #
    def _persist(self, proto: Prototype) -> None:
        if self.store is None:
            return
        try:
            from nyxara.memory.store import MemoryType
            self.store.remember(
                f"concept:{proto.label}",
                mem_type=MemoryType.SEMANTIC,
                importance=min(1.0, 0.5 + 0.05 * proto.support),
                tags=[self.TAG],
                metadata={"prototype": proto.to_dict(), "label": proto.label})
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            pass

    def _hydrate(self) -> None:
        """Load persisted prototypes from the store once, on first use."""
        if self._hydrated:
            return
        self._hydrated = True
        if self.store is None:
            return
        try:
            hits = self.store.recall("", k=1000, tags=[self.TAG], strengthen=False)
            for rec, _ in hits:
                data = rec.metadata.get("prototype") if isinstance(rec.metadata, dict) else None
                if isinstance(data, dict):
                    proto = Prototype.from_dict(data)
                    # keep the most-supported copy if duplicates exist
                    cur = self._protos.get(proto.label)
                    if cur is None or proto.support >= cur.support:
                        self._protos[proto.label] = proto
        except Exception:  # noqa: BLE001 — hydration is best-effort
            pass


# --------------------------------------------------------------------------- #
# One-shot episodic memory
# --------------------------------------------------------------------------- #
@dataclass
class _Pair:
    cue: str
    response: str
    vec: List[float]
    pair_id: str = field(default_factory=lambda: "oneshot-" + uuid.uuid4().hex[:8])
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"pair_id": self.pair_id, "cue": self.cue, "response": self.response,
                "vec": [round(x, 6) for x in self.vec], "created_at": self.created_at}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_Pair":
        return cls(cue=str(d.get("cue", "")), response=str(d.get("response", "")),
                   vec=[float(x) for x in d.get("vec", [])],
                   pair_id=str(d.get("pair_id", "oneshot-" + uuid.uuid4().hex[:8])),
                   created_at=float(d.get("created_at", time.time())))


class OneShotEpisodicMemory:
    """Remember a single (cue → response) pair and reproduce it on a similar cue."""

    TAG = "oneshot"

    def __init__(self, embedder: Any, *, store: Any = None, threshold: float = 0.6) -> None:
        if embedder is None or not hasattr(embedder, "embed"):
            raise ValueError("OneShotEpisodicMemory needs an embedder exposing .embed")
        self.embedder = embedder
        self.store = store
        self.threshold = threshold
        self._pairs: List[_Pair] = []
        self._hydrated = False

    def remember(self, cue: str, response: str) -> _Pair:
        """Bind a single cue to a response. One exposure is enough."""
        self._hydrate()
        cue, response = str(cue).strip(), str(response).strip()
        pair = _Pair(cue=cue, response=response, vec=self.embedder.embed(cue))
        # de-dup: if we already bound an (almost) identical cue, overwrite its response
        for existing in self._pairs:
            if cosine(existing.vec, pair.vec) >= 0.97:
                existing.response = response
                return existing
        self._pairs.append(pair)
        self._persist(pair)
        return pair

    def recall(self, cue: str, *, threshold: Optional[float] = None) -> Optional[str]:
        """Return the bound response for the most-similar remembered cue, if close enough."""
        self._hydrate()
        if not self._pairs:
            return None
        qvec = self.embedder.embed(str(cue))
        best, best_sim = None, 0.0
        for pair in self._pairs:
            s = cosine(qvec, pair.vec)
            if s > best_sim:
                best, best_sim = pair, s
        thr = self.threshold if threshold is None else threshold
        return best.response if (best is not None and best_sim >= thr) else None

    def as_prompt(self, cue: str) -> str:
        hit = self.recall(cue)
        return f"\n\nYou were told once: {hit}" if hit else ""

    def __len__(self) -> int:
        self._hydrate()
        return len(self._pairs)

    def _persist(self, pair: _Pair) -> None:
        if self.store is None:
            return
        try:
            from nyxara.memory.store import MemoryType
            self.store.remember(
                pair.cue or pair.response, mem_type=MemoryType.EPISODIC, importance=0.7,
                tags=[self.TAG], metadata={"oneshot": pair.to_dict()})
        except Exception:  # noqa: BLE001
            pass

    def _hydrate(self) -> None:
        if self._hydrated:
            return
        self._hydrated = True
        if self.store is None:
            return
        try:
            hits = self.store.recall("", k=1000, tags=[self.TAG], strengthen=False)
            seen = set()
            for rec, _ in hits:
                data = rec.metadata.get("oneshot") if isinstance(rec.metadata, dict) else None
                if isinstance(data, dict):
                    pair = _Pair.from_dict(data)
                    if pair.pair_id not in seen:
                        seen.add(pair.pair_id)
                        self._pairs.append(pair)
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.memory.store import MemoryStore, make_embedder

    print("=" * 70)
    print("NYXARA few-shot / one-shot self-test")
    print("=" * 70)

    embedder = make_embedder()
    # NOTE: with no sentence-transformers installed the store degrades to a *lexical*
    # embedder (token/morphology overlap, not deep paraphrase). The capability is the same;
    # the demo just uses concepts with distinct vocabularies so it holds on that substrate.

    # ---- one-shot & few-shot classification ---- #
    fs = FewShotLearner(embedder)
    # ONE example per class — true one-shot teaching
    fs.learn_one("weather", "the weather is sunny and warm today")
    fs.learn_one("cooking", "i love eating pizza and pasta for dinner")
    fs.learn_one("sport", "football and basketball are fun games to play")
    print(f"\nlearned concepts    : {fs.labels()} (one example each)")
    assert len(fs) == 3

    # held-out queries (never seen verbatim) classify to the right concept
    cases = [("is the weather rainy or sunny today", "weather"),
             ("pizza and pasta are great for dinner", "cooking"),
             ("i want to play football and basketball", "sport")]
    correct = 0
    for text, want in cases:
        got, conf = fs.best(text)
        ok = got == want
        correct += ok
        print(f"  classify {text[:30]:32s} -> {got:9s} ({conf:.2f}) {'✓' if ok else '✗ want ' + want}")
    assert correct >= 2, f"one-shot classification should beat chance, got {correct}/3"

    # a second example sharpens the prototype (few-shot)
    before = fs.best("sunny weather forecast")[1]
    fs.learn("weather", "today the forecast is cloudy with rain and wind")
    after = fs.best("sunny weather forecast")[1]
    print(f"\nfew-shot sharpening : conf {before:.2f} -> {after:.2f}")

    # as_prompt surfaces a recognised concept
    block = fs.as_prompt("what is the weather today")
    print(f"as_prompt           : {block.strip()!r}")
    assert "weather" in block

    # ---- one-shot episodic memory ---- #
    epi = OneShotEpisodicMemory(embedder)
    epi.remember("what is my favourite colour", "your favourite colour is teal")
    # a near-paraphrase of the same cue recalls the bound response from a SINGLE exposure
    hit = epi.recall("remind me my favourite colour please")
    print(f"\none-shot recall     : {hit!r}")
    assert hit is not None and "teal" in hit
    # an unrelated cue does not falsely fire
    assert epi.recall("what time is it in Tokyo") is None
    print("no false recall     : ✓")

    # ---- persistence through a real MemoryStore ---- #
    store = MemoryStore()
    fs2 = FewShotLearner(store.embedder, store=store)
    fs2.learn_one("frobnicator", "a small brass device that hums when warmed")
    # a fresh learner over the SAME store hydrates the persisted concept
    fs3 = FewShotLearner(store.embedder, store=store)
    assert "frobnicator" in fs3.labels()
    print(f"persisted concept   : {fs3.labels()} ✓")

    print("\nALL SELF-TESTS PASSED ✓")
