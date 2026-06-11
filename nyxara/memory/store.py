"""NYXARA · memory/store.py — the multi-store memory engine.

Four memory systems, mirroring cognitive science, over three backends:

* **WORKING** — tiny (Miller's 7±2), volatile; the current contents of mind. The
  lowest-activation item is evicted when the slots overflow.
* **EPISODIC** — timestamped autobiographical events ("what happened, and when").
* **SEMANTIC** — decontextualised facts and concepts ("what is true").
* **PROCEDURAL** — skills and how-to sequences ("how to do it").

Backends, unified behind one API:

* **KV**     — the record store (id → :class:`MemoryRecord`).
* **Vector** — embeddings + cosine similarity for associative recall
  (numpy-accelerated when present, exact pure-python otherwise; pluggable embedder).
* **Graph**  — lightweight typed links between memories (the rich knowledge graph
  lives in ``memory/graph.py``; here we keep direct associations + spreading).

Every memory carries a :class:`~nyxara.memory.provenance.Provenance` (Rule 6), so
recall can be filtered by trust and reported with a source chain. Recall ranks by a
blend of **similarity × recency × importance × trust**, and accessing a memory
*strengthens* it (recency + use), just like a brain.

Depends on :mod:`config`, :mod:`errors`, :mod:`memory.provenance`. numpy/faiss optional.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union,
)

from nyxara.kernel.config import NyxaraSettings, get_settings
from nyxara.kernel.errors import MemoryError_
from nyxara.memory.provenance import Provenance, SourceType

try:  # optional acceleration
    import numpy as _np  # type: ignore
    _HAS_NP = True
except Exception:  # pragma: no cover
    _np = None  # type: ignore
    _HAS_NP = False

__all__ = [
    "MemoryType",
    "MemoryRecord",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "make_embedder",
    "VectorIndex",
    "MemoryStore",
    "has_numpy",
]

_DAY = 86400.0


def has_numpy() -> bool:
    return _HAS_NP


# --------------------------------------------------------------------------- #
# Memory types
# --------------------------------------------------------------------------- #
class MemoryType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
class HashingEmbedder:
    """Deterministic, dependency-free feature-hashing embedder.

    Real models plug in by satisfying the same ``embed(text) -> list[float]`` shape.
    This hashes word/char n-grams into a fixed L2-normalised vector — good enough for
    lexical-overlap similarity until a learned embedder is wired in.
    """

    def __init__(self, dim: int = 128, seed: int = 1469598103934665603) -> None:
        if dim < 8:
            raise MemoryError_("embedding dim too small", context={"dim": dim})
        self.dim = dim
        self._seed = seed

    def _hash(self, token: str) -> int:
        # FNV-1a 64-bit
        h = self._seed
        for ch in token:
            h ^= ord(ch)
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h

    @staticmethod
    def _tokens(text: str) -> List[str]:
        text = text.lower()
        words = "".join(c if c.isalnum() else " " for c in text).split()
        grams: List[str] = list(words)
        # add char trigrams for sub-word robustness
        for w in words:
            padded = f"#{w}#"
            grams.extend(padded[i:i + 3] for i in range(len(padded) - 2))
        return grams or ["∅"]

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in self._tokens(text):
            h = self._hash(tok)
            idx = h % self.dim
            sign = 1.0 if (h >> 1) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class SentenceTransformerEmbedder:
    """A learned semantic embedder (sentence-transformers), import-guarded.

    Satisfies the same ``embed(text) -> list[float]`` shape as :class:`HashingEmbedder`,
    but maps text into a *meaning* space rather than a lexical-overlap one, so recall is
    semantic ("intrusion" finds "unauthorised login") instead of keyword-only. The model
    is loaded lazily; :meth:`available` reports honestly so a bare machine degrades to the
    hashing embedder rather than crashing.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "") -> None:
        from sentence_transformers import SentenceTransformer  # lazy, optional dep
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device or None)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    @staticmethod
    def available() -> bool:
        try:
            import sentence_transformers  # noqa: F401
            return True
        except Exception:  # pragma: no cover - depends on install
            return False

    def embed(self, text: str) -> List[float]:
        vec = self._model.encode([text], normalize_embeddings=True)[0]
        return [float(x) for x in vec]


def make_embedder(settings: Optional[NyxaraSettings] = None) -> Any:
    """Build the configured embedder: learned-semantic when enabled & available, else hashing."""
    s = settings or get_settings()
    mc = s.memory
    if getattr(mc, "semantic_embeddings", False) and SentenceTransformerEmbedder.available():
        try:
            return SentenceTransformerEmbedder(mc.embedding_model, mc.embedding_device)
        except Exception:  # noqa: BLE001 — fall back rather than fail to boot
            pass
    dim = min(256, mc.embedding_dim) if mc.embedding_dim else 128
    return HashingEmbedder(dim=dim)


# --------------------------------------------------------------------------- #
# Vector index
# --------------------------------------------------------------------------- #
def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class VectorIndex:
    """id → unit vector, with cosine top-k search (numpy-accelerated if available)."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vecs: Dict[str, List[float]] = {}
        self._dirty = True
        self._ids: List[str] = []
        self._mat: Any = None  # numpy matrix cache

    def add(self, mem_id: str, vec: Sequence[float]) -> None:
        if len(vec) != self.dim:
            raise MemoryError_("embedding dim mismatch",
                               context={"expected": self.dim, "got": len(vec)})
        self._vecs[mem_id] = list(vec)
        self._dirty = True

    def remove(self, mem_id: str) -> None:
        self._vecs.pop(mem_id, None)
        self._dirty = True

    def __len__(self) -> int:
        return len(self._vecs)

    def _rebuild(self) -> None:
        self._ids = list(self._vecs.keys())
        if _HAS_NP and self._ids:
            self._mat = _np.array([self._vecs[i] for i in self._ids], dtype=float)
        self._dirty = False

    def search(self, query: Sequence[float], k: int = 10,
               allowed: Optional[Iterable[str]] = None) -> List[Tuple[str, float]]:
        if not self._vecs:
            return []
        if self._dirty:
            self._rebuild()
        allow = set(allowed) if allowed is not None else None

        if _HAS_NP and self._mat is not None and allow is None:
            q = _np.array(query, dtype=float)
            qn = q / (_np.linalg.norm(q) or 1.0)
            mat = self._mat
            norms = _np.linalg.norm(mat, axis=1)
            norms[norms == 0] = 1.0
            sims = (mat @ qn) / norms
            order = _np.argsort(-sims)[:k]
            return [(self._ids[i], float(sims[i])) for i in order]

        # pure-python (or filtered) path
        scored: List[Tuple[str, float]] = []
        for mem_id, vec in self._vecs.items():
            if allow is not None and mem_id not in allow:
                continue
            scored.append((mem_id, _cosine(query, vec)))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:k]


# --------------------------------------------------------------------------- #
# Memory record
# --------------------------------------------------------------------------- #
@dataclass
class MemoryRecord:
    content: Any
    mem_type: MemoryType
    provenance: Provenance
    mem_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    embedding: Optional[List[float]] = None
    importance: float = 0.5
    tags: frozenset = field(default_factory=frozenset)
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    links: Dict[str, List[str]] = field(default_factory=dict)  # relation -> [mem_id]
    metadata: Dict[str, Any] = field(default_factory=dict)
    half_life_days: float = 7.0

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, dict):
            return " ".join(f"{k}:{v}" for k, v in self.content.items())
        return str(self.content)

    def recency_factor(self, now: Optional[float] = None) -> float:
        age_days = max(0.0, (now or time.time()) - self.last_access) / _DAY
        return math.exp(-age_days / self.half_life_days)

    def activation(self, now: Optional[float] = None) -> float:
        """Brain-like activation: importance × recency × use × trust."""
        use = 1.0 + math.log1p(self.access_count)
        return (self.importance
                * self.recency_factor(now)
                * use
                * self.provenance.trust(now))

    def touch(self, now: Optional[float] = None) -> None:
        self.last_access = now or time.time()
        self.access_count += 1

    def add_link(self, relation: str, target_id: str) -> None:
        self.links.setdefault(relation, [])
        if target_id not in self.links[relation]:
            self.links[relation].append(target_id)

    def to_dict(self, now: Optional[float] = None) -> Dict[str, Any]:
        return {
            "mem_id": self.mem_id,
            "type": self.mem_type.value,
            "content": self.content,
            "importance": round(self.importance, 4),
            "activation": round(self.activation(now), 5),
            "trust": round(self.provenance.trust(now), 4),
            "tags": sorted(self.tags),
            "access_count": self.access_count,
            "created_at": self.created_at,
            "links": self.links,
            "provenance": self.provenance.to_dict(now),
        }


# --------------------------------------------------------------------------- #
# The memory store
# --------------------------------------------------------------------------- #
@dataclass
class _ScoreWeights:
    similarity: float = 1.0
    recency: float = 0.4
    importance: float = 0.5
    trust: float = 0.6


class MemoryStore:
    """Unified KV + vector + graph memory across the four memory systems."""

    def __init__(
        self,
        *,
        embedder: Optional[Any] = None,
        settings: Optional[NyxaraSettings] = None,
        weights: Optional[_ScoreWeights] = None,
        on_evict: Optional[Callable[[MemoryRecord], None]] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.embedder = embedder or make_embedder(self._settings)
        self._index = VectorIndex(self.embedder.dim)
        self._kv: Dict[str, MemoryRecord] = {}
        self.weights = weights or _ScoreWeights()
        self.on_evict = on_evict
        self.working_slots = self._settings.memory.working_memory_slots
        self._stats = {"remembered": 0, "recalled": 0, "evicted": 0, "forgotten": 0}

    def __len__(self) -> int:
        return len(self._kv)

    # ---- write ---- #
    def remember(
        self,
        content: Any,
        *,
        mem_type: MemoryType = MemoryType.EPISODIC,
        provenance: Optional[Provenance] = None,
        importance: float = 0.5,
        tags: Sequence[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
        half_life_days: Optional[float] = None,
        embed: bool = True,
    ) -> MemoryRecord:
        prov = provenance or Provenance(SourceType.SELF_REFLECTION, confidence=0.6,
                                        method="stored")
        rec = MemoryRecord(
            content=content,
            mem_type=mem_type,
            provenance=prov,
            importance=max(0.0, min(1.0, importance)),
            tags=frozenset(t.lower() for t in tags),
            metadata=metadata or {},
            half_life_days=half_life_days if half_life_days is not None
            else (1.0 if mem_type is MemoryType.WORKING
                  else self._settings.memory.forgetting_half_life_days),
        )
        if embed:
            rec.embedding = self.embedder.embed(rec.text())
            self._index.add(rec.mem_id, rec.embedding)
        self._kv[rec.mem_id] = rec
        self._stats["remembered"] += 1

        if mem_type is MemoryType.WORKING:
            self._enforce_working_capacity()
        return rec

    def _enforce_working_capacity(self) -> None:
        working = [r for r in self._kv.values() if r.mem_type is MemoryType.WORKING]
        while len(working) > self.working_slots:
            victim = min(working, key=lambda r: r.activation())
            self.forget(victim.mem_id, _evicted=True)
            working.remove(victim)

    # ---- read ---- #
    def get(self, mem_id: str, *, touch: bool = False) -> MemoryRecord:
        try:
            rec = self._kv[mem_id]
        except KeyError:
            raise MemoryError_(f"no memory {mem_id!r}") from None
        if touch:
            rec.touch()
        return rec

    def recall(
        self,
        query: Union[str, Sequence[float]],
        *,
        mem_type: Optional[MemoryType] = None,
        k: Optional[int] = None,
        min_trust: float = 0.0,
        tags: Sequence[str] = (),
        strengthen: bool = True,
        now: Optional[float] = None,
    ) -> List[Tuple[MemoryRecord, float]]:
        """Associative recall ranked by similarity × recency × importance × trust."""
        now = now or time.time()
        k = k or self._settings.memory.retrieval_top_k
        qvec = self.embedder.embed(query) if isinstance(query, str) else list(query)

        allowed = None
        if mem_type is not None or tags:
            tagset = {t.lower() for t in tags}
            allowed = [
                r.mem_id for r in self._kv.values()
                if (mem_type is None or r.mem_type is mem_type)
                and (not tagset or tagset & r.tags)
            ]

        # over-fetch from the vector index, then re-rank with the full blend
        hits = self._index.search(qvec, k=max(k * 4, 16), allowed=allowed)
        results: List[Tuple[MemoryRecord, float]] = []
        for mem_id, sim in hits:
            rec = self._kv.get(mem_id)
            if rec is None:
                continue
            trust = rec.provenance.trust(now)
            if trust < min_trust:
                continue
            score = (
                self.weights.similarity * sim
                + self.weights.recency * rec.recency_factor(now)
                + self.weights.importance * rec.importance
                + self.weights.trust * trust
            )
            results.append((rec, score))

        results.sort(key=lambda rs: rs[1], reverse=True)
        results = results[:k]
        self._stats["recalled"] += len(results)
        if strengthen:
            for rec, _ in results:
                rec.touch(now)
        return results

    def recall_one(self, query: Union[str, Sequence[float]], **kw: Any) -> Optional[MemoryRecord]:
        res = self.recall(query, k=1, **kw)
        return res[0][0] if res else None

    # ---- graph / associations ---- #
    def link(self, a: str, b: str, relation: str = "related",
             *, bidirectional: bool = True) -> None:
        ra, rb = self.get(a), self.get(b)
        ra.add_link(relation, b)
        if bidirectional:
            rb.add_link(relation, a)

    def neighbors(self, mem_id: str, relation: Optional[str] = None) -> List[MemoryRecord]:
        rec = self.get(mem_id)
        out: List[MemoryRecord] = []
        for rel, targets in rec.links.items():
            if relation is not None and rel != relation:
                continue
            for t in targets:
                if t in self._kv:
                    out.append(self._kv[t])
        return out

    def spread(self, mem_id: str, *, hops: int = 2, decay: Optional[float] = None,
               now: Optional[float] = None) -> Dict[str, float]:
        """Spreading activation from a seed memory over the link graph."""
        decay = decay if decay is not None else self._settings.memory.spread_decay
        now = now or time.time()
        activations: Dict[str, float] = {mem_id: 1.0}
        frontier = [(mem_id, 1.0, 0)]
        while frontier:
            cur, energy, depth = frontier.pop(0)
            if depth >= hops:
                continue
            for nb in self.neighbors(cur):
                e = energy * decay * nb.activation(now)
                if e <= 0.001:
                    continue
                if e > activations.get(nb.mem_id, 0.0):
                    activations[nb.mem_id] = e
                    frontier.append((nb.mem_id, e, depth + 1))
        return activations

    # ---- maintenance ---- #
    def forget(self, mem_id: str, *, _evicted: bool = False) -> bool:
        rec = self._kv.pop(mem_id, None)
        if rec is None:
            return False
        self._index.remove(mem_id)
        # scrub dangling links
        for other in self._kv.values():
            for rel in list(other.links):
                if mem_id in other.links[rel]:
                    other.links[rel].remove(mem_id)
        if _evicted:
            self._stats["evicted"] += 1
            if self.on_evict:
                try:
                    self.on_evict(rec)
                except Exception:  # noqa: BLE001
                    pass
        else:
            self._stats["forgotten"] += 1
        return True

    def promote(self, mem_id: str, to: MemoryType) -> MemoryRecord:
        """Move a memory between systems (e.g. episodic → semantic in consolidation)."""
        rec = self.get(mem_id)
        rec.mem_type = to
        if to is not MemoryType.WORKING:
            rec.half_life_days = max(rec.half_life_days,
                                     self._settings.memory.forgetting_half_life_days)
        return rec

    def consolidation_candidates(self, *, min_access: int = 3,
                                 now: Optional[float] = None) -> List[MemoryRecord]:
        """Episodic memories accessed often enough to deserve semantic abstraction."""
        return [r for r in self._kv.values()
                if r.mem_type is MemoryType.EPISODIC and r.access_count >= min_access]

    def by_type(self, mem_type: MemoryType) -> List[MemoryRecord]:
        return [r for r in self._kv.values() if r.mem_type is mem_type]

    def working_set(self, now: Optional[float] = None) -> List[MemoryRecord]:
        ws = self.by_type(MemoryType.WORKING)
        ws.sort(key=lambda r: r.activation(now), reverse=True)
        return ws

    # ---- persistence ---- #
    def save(self, path: str) -> str:
        data = {
            "dim": self.embedder.dim,
            "records": [
                {
                    "mem_id": r.mem_id, "content": r.content, "type": r.mem_type.value,
                    "importance": r.importance, "tags": sorted(r.tags),
                    "created_at": r.created_at, "last_access": r.last_access,
                    "access_count": r.access_count, "links": r.links,
                    "metadata": r.metadata, "half_life_days": r.half_life_days,
                    "embedding": r.embedding,
                    "provenance": {
                        "source": r.provenance.source.value,
                        "confidence": r.provenance.confidence,
                        "detail": r.provenance.detail, "method": r.provenance.method,
                        "acquired_at": r.provenance.acquired_at,
                        "half_life_days": r.provenance.half_life_days,
                    },
                }
                for r in self._kv.values()
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, default=str)
        return path

    def load(self, path: str) -> int:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        count = 0
        for d in data["records"]:
            pv = d["provenance"]
            prov = Provenance(
                source=SourceType(pv["source"]), confidence=pv["confidence"],
                detail=pv["detail"], method=pv["method"],
                acquired_at=pv["acquired_at"], half_life_days=pv["half_life_days"],
            )
            rec = MemoryRecord(
                content=d["content"], mem_type=MemoryType(d["type"]), provenance=prov,
                mem_id=d["mem_id"], importance=d["importance"],
                tags=frozenset(d["tags"]), created_at=d["created_at"],
                last_access=d["last_access"], access_count=d["access_count"],
                links=d.get("links", {}), metadata=d.get("metadata", {}),
                half_life_days=d["half_life_days"], embedding=d.get("embedding"),
            )
            self._kv[rec.mem_id] = rec
            if rec.embedding:
                self._index.add(rec.mem_id, rec.embedding)
            count += 1
        return count

    def stats(self) -> Dict[str, Any]:
        by_type = {t.value: len(self.by_type(t)) for t in MemoryType}
        return {**self._stats, "total": len(self._kv), "by_type": by_type,
                "indexed": len(self._index), "numpy": has_numpy()}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile, os
    print("=" * 70)
    print("NYXARA memory-store self-test")
    print("=" * 70)
    print(f"numpy accelerated   : {has_numpy()}")

    ms = MemoryStore()

    # store some semantic facts + episodic events
    ms.remember("The owner's name is Jaypal Khoja, called JP.", mem_type=MemoryType.SEMANTIC,
                provenance=Provenance(SourceType.OWNER, confidence=1.0), importance=1.0,
                tags=["owner", "identity"])
    ms.remember("A suspicious SSH login was detected on port 22 at 3am.",
                mem_type=MemoryType.EPISODIC,
                provenance=Provenance(SourceType.SENSOR, confidence=0.9),
                importance=0.9, tags=["security", "network"])
    ms.remember("To harden SSH: disable root login, use keys, change port, fail2ban.",
                mem_type=MemoryType.PROCEDURAL,
                provenance=Provenance(SourceType.TOOL, confidence=0.85),
                importance=0.7, tags=["security", "howto"])

    # associative recall
    hits = ms.recall("who is my master?", k=2)
    print(f"\nrecall 'who is my master?' -> {[h[0].text()[:40] for h in hits]}")
    assert "Jaypal" in hits[0][0].text()

    sec = ms.recall("network intrusion threat", k=2)
    print(f"recall 'network intrusion' -> {[h[0].text()[:40] for h in sec]}")
    assert any("SSH" in h[0].text() for h in sec)

    # working memory capacity
    ms.working_slots = 3
    ids = [ms.remember(f"thought {i}", mem_type=MemoryType.WORKING,
                       importance=0.1 + 0.1 * i).mem_id for i in range(6)]
    ws = ms.working_set()
    print(f"\nworking set ({len(ws)} slots): {[r.text() for r in ws]}")
    assert len(ws) == 3  # only 3 survive
    assert ms.stats()["evicted"] == 3

    # links + spreading activation
    a = ms.remember("chess", mem_type=MemoryType.SEMANTIC, importance=0.8)
    b = ms.remember("strategy", mem_type=MemoryType.SEMANTIC, importance=0.8)
    c = ms.remember("warfare", mem_type=MemoryType.SEMANTIC, importance=0.8)
    ms.link(a.mem_id, b.mem_id, "related")
    ms.link(b.mem_id, c.mem_id, "related")
    spread = ms.spread(a.mem_id, hops=2)
    print(f"\nspread from 'chess' : {len(spread)} nodes reached")
    assert b.mem_id in spread and c.mem_id in spread

    # consolidation candidates
    for _ in range(4):
        ms.recall("network intrusion threat")
    cands = ms.consolidation_candidates(min_access=3)
    print(f"consolidation cands : {[r.text()[:30] for r in cands]}")

    # persistence
    path = os.path.join(tempfile.gettempdir(), "nyx_mem.json")
    ms.save(path)
    ms2 = MemoryStore()
    n = ms2.load(path)
    print(f"\npersisted/loaded    : {n} records")
    assert n == len(ms)
    reloaded = ms2.recall("who is my master?", k=1)
    assert "Jaypal" in reloaded[0][0].text()
    os.remove(path)

    print(f"\nstats              : {ms.stats()}")
    print("\nALL SELF-TESTS PASSED ✓")
