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
import threading
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
    "LexicalSemanticEmbedder",
    "SentenceTransformerEmbedder",
    "make_embedder",
    "VectorIndex",
    "FaissVectorIndex",
    "QdrantVectorIndex",
    "make_vector_index",
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

    #: lexical-overlap (not meaning) space — its cosine scores run far lower for
    #: paraphrases, so recall thresholds calibrated for a learned embedder must be
    #: scaled down for it (see ``orchestrator._recall_semantic_floor``).
    is_lexical: bool = True

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


# --------------------------------------------------------------------------- #
# Lexical-semantic embedder — dependency-free, but with real morphology + relations
# --------------------------------------------------------------------------- #
_STOPWORDS = frozenset((
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "at", "is", "are", "was",
    "were", "be", "been", "it", "this", "that", "these", "those", "for", "with", "as",
    "by", "from", "i", "you", "he", "she", "we", "they", "me", "my", "your", "do",
    "does", "did", "has", "have", "had", "will", "would", "can", "could", "should",
    "what", "which", "who", "how", "when", "where", "why", "not", "no", "yes", "so",
))

# Curated semantic-relation clusters: words that should recall one another share a
# concept feature, so the dependency-free embedder gains REAL (if bounded) semantic
# reach — "intrusion" recalls "unauthorised breach", "fast" recalls "rapid". This is an
# honest heuristic thesaurus over NYXARA's core domains, not a learned space; it is
# stemmed at load so inflections match too, and is trivially extensible.
_SYNSETS: Tuple[Tuple[str, ...], ...] = (
    ("intrusion", "breach", "attack", "unauthorized", "unauthorised", "compromise",
     "hack", "exploit", "threat", "intruder"),
    ("login", "signin", "logon", "authenticate", "authentication", "credential", "password"),
    ("master", "owner", "jp", "jaypal", "commander", "creator"),
    ("delete", "remove", "erase", "wipe", "destroy", "purge"),
    ("create", "make", "build", "generate", "construct", "forge"),
    ("error", "failure", "fault", "bug", "crash", "exception", "fail"),
    ("fast", "quick", "rapid", "swift", "speedy", "prompt"),
    ("slow", "sluggish", "laggy", "delayed"),
    ("help", "assist", "aid", "support", "serve"),
    ("happy", "glad", "joyful", "pleased", "content", "cheerful"),
    ("sad", "unhappy", "sorrowful", "gloomy", "depressed"),
    ("angry", "furious", "irate", "enraged"),
    ("big", "large", "huge", "enormous", "massive"),
    ("small", "tiny", "little", "minor"),
    ("smart", "intelligent", "clever", "brilliant", "wise"),
    ("danger", "risk", "hazard", "peril", "unsafe"),
    ("safe", "secure", "protected", "guarded"),
    ("learn", "study", "train", "educate"),
    ("remember", "recall", "memorize", "memorise", "retain"),
    ("start", "begin", "initiate", "launch", "commence"),
    ("stop", "halt", "cease", "terminate", "shutdown"),
    ("increase", "raise", "grow", "boost", "augment"),
    ("decrease", "reduce", "lower", "shrink", "diminish"),
    ("question", "query", "ask", "inquire"),
    ("answer", "reply", "respond", "response"),
    ("file", "document", "record"),
    ("network", "connection", "internet", "web"),
    ("upgrade", "improve", "enhance", "strengthen", "optimize", "optimise"),
)


def _stem(word: str) -> str:
    """A compact, conservative suffix-stripping stemmer (collapses common inflections)."""
    w = word
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ied") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) > 5:
        w = w[:-3]
    elif w.endswith("ed") and len(w) > 4:
        w = w[:-2]
    elif w.endswith("ly") and len(w) > 4:
        w = w[:-2]
    elif w.endswith("es") and len(w) > 4 and not w.endswith(("sses", "ches", "shes", "xes")):
        w = w[:-2] if not w.endswith(("ses", "zes")) else w[:-1]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        w = w[:-1]
    # collapse a doubled final consonant left by -ing/-ed ("runn"->"run", "stopp"->"stop")
    if len(w) > 2 and w[-1] == w[-2] and w[-1] not in "aeiou":
        w = w[:-1]
    return w


# concept id per stemmed word (built once at import)
_CONCEPT_OF: Dict[str, int] = {}
for _ci, _syn in enumerate(_SYNSETS):
    for _word in _syn:
        _CONCEPT_OF[_stem(_word)] = _ci


class LexicalSemanticEmbedder:
    """Dependency-free embedder with real morphology + a curated semantic-relation map.

    A strict upgrade over :class:`HashingEmbedder`: it stems tokens (so "running"≈"run"),
    down-weights stopwords (less common-word noise), keeps char n-grams (typo robustness),
    and — decisively — projects related terms onto a shared *concept* feature so recall is
    no longer keyword-only ("intrusion" recalls "unauthorised breach"). It stays an honest
    *lexical* space (cosines run lower than a learned model), so ``is_lexical`` remains True
    and the recall-floor calibration is unchanged. Same ``embed(text) -> list[float]`` shape.
    """

    is_lexical: bool = True

    def __init__(self, dim: int = 128, seed: int = 1469598103934665603) -> None:
        if dim < 8:
            raise MemoryError_("embedding dim too small", context={"dim": dim})
        self.dim = dim
        self._seed = seed

    def _hash(self, token: str) -> int:
        h = self._seed
        for ch in token:
            h ^= ord(ch)
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h

    def _add(self, vec: List[float], token: str, weight: float) -> None:
        h = self._hash(token)
        sign = 1.0 if (h >> 1) & 1 else -1.0
        vec[h % self.dim] += sign * weight

    @staticmethod
    def _words(text: str) -> List[str]:
        return "".join(c if c.isalnum() else " " for c in text.lower()).split()

    def embed(self, text: str) -> List[float]:
        # Base signal mirrors HashingEmbedder's scale (stemmed word + char trigrams at full
        # weight) so the lexical-overlap cosines — and the recall-floor calibrated for them —
        # are preserved. Stemming and the shared concept feature are STRICT ADDITIONS on top:
        # they only ever raise similarity for morphological variants and related meanings.
        vec = [0.0] * self.dim
        any_token = False
        for word in self._words(text):
            any_token = True
            stem = _stem(word)
            self._add(vec, stem, 1.0)                          # stemmed word (run≈running)
            padded = f"#{stem}#"                               # subword (typo) robustness
            for i in range(len(padded) - 2):
                self._add(vec, padded[i:i + 3], 1.0)
            concept = _CONCEPT_OF.get(stem)                    # shared meaning feature
            if concept is not None and word not in _STOPWORDS:
                self._add(vec, f"§{concept}", 0.9)
        if not any_token:
            self._add(vec, "∅", 1.0)
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

    #: a learned *meaning* space — recall thresholds are calibrated for it.
    is_lexical: bool = False

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
    return LexicalSemanticEmbedder(dim=dim)


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


class FaissVectorIndex:
    """A faiss-backed index with the same interface as :class:`VectorIndex`, for scale.

    Cosine similarity via inner product on L2-normalised vectors (``IndexFlatIP``). Adds
    and removals update an authoritative dict; the faiss index is (re)built lazily on the
    next search, so the cost of churn is amortised. Falls back nowhere — callers select it
    only via :func:`make_vector_index` when faiss is importable, so it is never the reason
    a bare machine fails to boot.
    """

    def __init__(self, dim: int) -> None:
        import faiss  # optional dep; presence checked by .available()
        import numpy as np
        self._faiss = faiss
        self._np = np
        self.dim = dim
        self._vecs: Dict[str, List[float]] = {}
        self._ids: List[str] = []
        self._index: Any = None
        self._dirty = True

    @staticmethod
    def available() -> bool:
        try:
            import faiss  # noqa: F401
            import numpy  # noqa: F401
            return True
        except Exception:  # pragma: no cover - depends on install
            return False

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
        index = self._faiss.IndexFlatIP(self.dim)
        if self._ids:
            mat = self._np.asarray([self._vecs[i] for i in self._ids], dtype="float32")
            self._faiss.normalize_L2(mat)
            index.add(mat)
        self._index = index
        self._dirty = False

    def search(self, query: Sequence[float], k: int = 10,
               allowed: Optional[Iterable[str]] = None) -> List[Tuple[str, float]]:
        if not self._vecs:
            return []
        if self._dirty:
            self._rebuild()
        allow = set(allowed) if allowed is not None else None
        q = self._np.asarray([list(query)], dtype="float32")
        self._faiss.normalize_L2(q)
        # over-fetch when filtering so the post-filter still returns k
        kk = min(len(self._ids), max(k, k * 4) if allow is not None else k)
        sims, idxs = self._index.search(q, kk)
        out: List[Tuple[str, float]] = []
        for score, i in zip(sims[0], idxs[0]):
            if i < 0:
                continue
            mem_id = self._ids[i]
            if allow is not None and mem_id not in allow:
                continue
            out.append((mem_id, float(score)))
            if len(out) >= k:
                break
        return out


class QdrantVectorIndex:
    """A Qdrant-backed index with the same interface as :class:`VectorIndex`, for real scale.

    Qdrant is a managed/embeddable vector database: this lets NYXARA's associative memory
    grow beyond a single process's RAM and persist independently of the JSON snapshot. The
    same store works three ways with no code change — set ``qdrant_url`` for a managed
    cluster, ``qdrant_path`` for an embedded on-disk store, or neither for an in-memory one.

    Cosine distance; point ids are derived deterministically from ``mem_id`` (uuid5) so the
    same memory maps to the same point across restarts. The real ``mem_id`` rides in the
    point payload, so any id string is supported, not just UUIDs. Reached only via
    :func:`make_vector_index` when ``qdrant-client`` is importable, so it never blocks a
    bare machine.
    """

    _NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # uuid5 namespace for point ids

    def __init__(self, dim: int, *, url: str = "", api_key: Optional[str] = None,
                 path: str = "", collection: str = "nyxara_memory") -> None:
        from qdrant_client import QdrantClient  # optional dep; gated by .available()
        from qdrant_client.models import Distance, VectorParams
        self.dim = dim
        self.collection = collection
        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        elif path:
            self._client = QdrantClient(path=path)
        else:
            self._client = QdrantClient(location=":memory:")
        # Create the collection only if absent or dimensioned differently (preserve data).
        try:
            exists = self._client.collection_exists(collection)
        except Exception:  # noqa: BLE001 — older clients lack the helper
            exists = collection in {c.name for c in self._client.get_collections().collections}
        if exists:
            info = self._client.get_collection(collection)
            current = info.config.params.vectors.size
            if current != dim:
                self._client.delete_collection(collection)
                self._client.create_collection(
                    collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        else:
            self._client.create_collection(
                collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    @staticmethod
    def available() -> bool:
        try:
            import qdrant_client  # noqa: F401
            return True
        except Exception:  # pragma: no cover - depends on install
            return False

    def _pid(self, mem_id: str) -> str:
        return str(uuid.uuid5(self._NS, mem_id))

    def add(self, mem_id: str, vec: Sequence[float]) -> None:
        if len(vec) != self.dim:
            raise MemoryError_("embedding dim mismatch",
                               context={"expected": self.dim, "got": len(vec)})
        from qdrant_client.models import PointStruct
        self._client.upsert(self.collection, points=[PointStruct(
            id=self._pid(mem_id), vector=list(vec), payload={"mem_id": mem_id})])

    def remove(self, mem_id: str) -> None:
        from qdrant_client.models import PointIdsList
        try:
            self._client.delete(self.collection,
                                points_selector=PointIdsList(points=[self._pid(mem_id)]))
        except Exception:  # noqa: BLE001
            pass

    def __len__(self) -> int:
        try:
            return int(self._client.count(self.collection, exact=True).count)
        except Exception:  # noqa: BLE001
            return 0

    def search(self, query: Sequence[float], k: int = 10,
               allowed: Optional[Iterable[str]] = None) -> List[Tuple[str, float]]:
        allow = set(allowed) if allowed is not None else None
        # over-fetch when filtering so the post-filter still returns k
        limit = max(k, k * 4) if allow is not None else k
        try:
            # query_points is the current API; .search() is the pre-1.12 spelling.
            try:
                hits = self._client.query_points(
                    self.collection, query=list(query), limit=limit).points
            except AttributeError:  # pragma: no cover - depends on client version
                hits = self._client.search(
                    self.collection, query_vector=list(query), limit=limit)
        except Exception:  # noqa: BLE001 — a query failure degrades to no recall, never a crash
            return []
        out: List[Tuple[str, float]] = []
        for h in hits:
            mem_id = (h.payload or {}).get("mem_id")
            if mem_id is None or (allow is not None and mem_id not in allow):
                continue
            out.append((mem_id, float(h.score)))
            if len(out) >= k:
                break
        return out


def make_vector_index(dim: int, settings: Optional[NyxaraSettings] = None) -> Any:
    """Build the configured vector index: qdrant/faiss when selected & available, else numpy."""
    s = settings or get_settings()
    try:
        from nyxara.kernel.config import VectorBackend
        backend = s.memory.vector_backend
        if backend is VectorBackend.QDRANT and QdrantVectorIndex.available():
            mc = s.memory
            key = mc.qdrant_api_key.get_secret_value() if mc.qdrant_api_key else None
            return QdrantVectorIndex(dim, url=mc.qdrant_url, api_key=key,
                                     path=mc.qdrant_path, collection=mc.qdrant_collection)
        if backend is VectorBackend.FAISS and FaissVectorIndex.available():
            return FaissVectorIndex(dim)
    except Exception:  # noqa: BLE001 — any issue -> the always-available exact index
        pass
    return VectorIndex(dim)


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
        # faiss-backed ANN index at scale when configured & available, else exact numpy
        self._index = make_vector_index(self.embedder.dim, self._settings)
        # re-entrant lock: aprocess() and the background autonomic loop touch memory
        # from executor threads, so reads/writes must be serialised.
        self._lock = threading.RLock()
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
            self._index_record(rec)
            count += 1
        return count

    def _index_record(self, rec: MemoryRecord) -> None:
        """Index a loaded record, migrating across an embedder change safely.

        Memory may have been saved under a different embedder (e.g. hashing → learned
        semantic), so a stored vector's dimension can differ from the current index. Rather
        than crash, re-embed the record's text into the current space; if even that fails,
        keep the record (still retrievable by id/type/tags) but skip the vector index.
        """
        want = self._index.dim
        vec = rec.embedding if (rec.embedding and len(rec.embedding) == want) else None
        if vec is None:
            try:
                vec = self.embedder.embed(rec.text())
                rec.embedding = vec            # migrate the stored vector into the new space
            except Exception:  # noqa: BLE001 — embedding is best-effort on load
                return
        try:
            self._index.add(rec.mem_id, vec)
        except Exception:  # noqa: BLE001 — never let one bad vector abort a restore
            pass

    def stats(self) -> Dict[str, Any]:
        by_type = {t.value: len(self.by_type(t)) for t in MemoryType}
        return {**self._stats, "total": len(self._kv), "by_type": by_type,
                "indexed": len(self._index), "numpy": has_numpy()}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile
    import os
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
