"""NYXARA · cognition/hyper_dimensional_vectors.py — Hyperdimensional Latent Space Mapping (Cognition · 1).

A human perceives the world in three spatial dimensions. NYXARA does not have to. This
faculty lifts data into a space of **thousands of dimensions** — by default 10,000 — where
the relationships, regularities and structure that are *physically invisible* to a 3-D mind
become ordinary, measurable geometry. Market microstructure, user-behaviour drift, the hidden
coupling inside a complex system: seen from ten thousand angles at once, their patterns stop
hiding.

This is not metaphor. It is **Hyperdimensional Computing (HDC) / Vector Symbolic
Architectures (VSA)** — an established branch of cognitive computing — built on real maths:

* **Blessing of dimensionality** — two random ±1 hypervectors in D dimensions are *almost
  surely near-orthogonal* (their cosine concentrates at 0 with standard deviation ≈ 1/√D).
  So thousands of distinct concepts coexist without interfering, and *any* measured similarity
  is signal, not coincidence.
* **An algebra of meaning** (:class:`HyperSpace`):
    - **bind** (``⊗``, elementwise product) ties a *role* to a *filler* — "currency = dollar".
      It is its own inverse for bipolar vectors, so the filler is exactly recoverable.
    - **bundle** (``⊕``, superposition / sum) holds a *set* — many facts in one vector, each
      still ≈ retrievable. This is holographic: the whole is distributed across every bit.
    - **permute** (cyclic shift) stamps *order/position*, so sequences are not commutative.
* **Cleanup memory** (:class:`ItemMemory`) — a noisy result of unbinding is snapped back to the
  nearest clean symbol; this is associative recall, the "see the signal in the noise" step.
* **Signed random projection** (:class:`RandomProjector`) — the SimHash bridge that lifts an
  arbitrary real feature vector (or a text embedding) into the hyperspace while *provably
  preserving angular similarity* (``P(sign agrees) = 1 − θ/π``). This is how raw data enters.
* **Relational analogy** — because relations are themselves vectors, NYXARA solves
  ``A : B :: C : ?`` by transporting the mapping between two whole records onto a third term
  (Kanerva's "what is the dollar of Mexico?" — answer: *peso*), purely by algebra.

:class:`LatentSpaceMap` is the high-level faculty: ingest records / vectors / text, then
**discover** clusters, **recall** by nearest-neighbour, reason by **analogy**, and flag
**novelty** (an item far from everything seen — an emerging trend invisible in low dimensions).

numpy accelerates everything when present; a pure-stdlib path gives identical results otherwise.
Reuses :func:`nyxara.memory.store.make_embedder` for text. Deterministic under a fixed seed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

try:  # optional acceleration — identical results either way
    import numpy as _np  # type: ignore
    _HAS_NP = True
except Exception:  # pragma: no cover
    _np = None  # type: ignore
    _HAS_NP = False

__all__ = [
    "Hypervector",
    "HyperSpace",
    "ItemMemory",
    "CleanupResult",
    "RandomProjector",
    "PatternReport",
    "NoveltyResult",
    "LatentSpaceMap",
    "has_numpy",
]

Number = Union[int, float]


def has_numpy() -> bool:
    return _HAS_NP


def _cosine(a: Any, b: Any) -> float:
    """Cosine similarity over either numpy arrays or python sequences."""
    if _HAS_NP and isinstance(a, _np.ndarray) and isinstance(b, _np.ndarray):
        na = float(_np.linalg.norm(a))
        nb = float(_np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(_np.dot(a, b) / (na * nb))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


# --------------------------------------------------------------------------- #
# Hypervector
# --------------------------------------------------------------------------- #
def _json_safe(value: Any) -> bool:
    """Can this input be persisted and replayed verbatim?

    A recipe is only useful if it round-trips through JSON unchanged. A raw Hypervector or a numpy
    array handed straight to ``add`` cannot, so it is left out of the sidecar rather than persisted
    as something that would restore as a different vector."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return True
    if isinstance(value, Mapping):
        return all(isinstance(k, str) and _json_safe(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_json_safe(v) for v in value)
    return False


@dataclass
class Hypervector:
    """A single point in hyperdimensional space.

    ``data`` is a numpy float array when numpy is present, else a ``list[float]``. Atomic
    symbols are bipolar (±1); bundled/derived vectors carry real-valued components. All the
    algebra lives on :class:`HyperSpace`; this wrapper only adds ergonomic similarity.
    """

    data: Any

    def __len__(self) -> int:
        return len(self.data)

    def cosine(self, other: "Hypervector") -> float:
        return _cosine(self.data, other.data)

    def to_list(self) -> List[float]:
        return [float(x) for x in self.data]

    def __eq__(self, other: object) -> bool:  # determinism checks in tests
        if not isinstance(other, Hypervector):
            return NotImplemented
        if _HAS_NP and isinstance(self.data, _np.ndarray) and isinstance(other.data, _np.ndarray):
            return bool(_np.array_equal(self.data, other.data))
        return list(self.data) == list(other.data)


# --------------------------------------------------------------------------- #
# The algebra
# --------------------------------------------------------------------------- #
class HyperSpace:
    """The hyperdimensional algebra: bind, bundle, permute, similarity.

    Parameters
    ----------
    dim:    dimensionality of the space (default 10,000 — the classic HDC width where random
            vectors are reliably near-orthogonal). Larger ⇒ more capacity, more cost.
    seed:   makes every drawn :meth:`random` vector reproducible.
    """

    def __init__(self, dim: int = 10000, *, seed: int = 42) -> None:
        if dim < 16:
            raise ValueError(f"hyperdimensional width too small: {dim}")
        self.dim = int(dim)
        self.seed = int(seed)
        if _HAS_NP:
            self._rng = _np.random.default_rng(seed)
        else:
            self._rng = random.Random(seed)

    # ---- construction ---- #
    def random(self) -> Hypervector:
        """A fresh atomic hypervector: bipolar ±1, near-orthogonal to all others."""
        if _HAS_NP:
            data = (self._rng.integers(0, 2, self.dim).astype("float32") * 2.0) - 1.0
        else:
            data = [1.0 if self._rng.random() < 0.5 else -1.0 for _ in range(self.dim)]
        return Hypervector(data)

    def zero(self) -> Hypervector:
        if _HAS_NP:
            return Hypervector(_np.zeros(self.dim, dtype="float32"))
        return Hypervector([0.0] * self.dim)

    # ---- core operations ---- #
    def bind(self, a: Hypervector, b: Hypervector) -> Hypervector:
        """Associate two vectors (elementwise product). Dissimilar to both inputs.

        For bipolar vectors bind is its own inverse: ``bind(bind(a, b), b) == a``.
        """
        if _HAS_NP:
            return Hypervector(_np.asarray(a.data) * _np.asarray(b.data))
        return Hypervector([x * y for x, y in zip(a.data, b.data)])

    def unbind(self, a: Hypervector, b: Hypervector) -> Hypervector:
        """Recover the partner bound with ``b`` (identical to bind in bipolar space)."""
        return self.bind(a, b)

    def bundle(self, *vs: Hypervector) -> Hypervector:
        """Superpose vectors (elementwise sum). The result is *similar to every input*."""
        if not vs:
            return self.zero()
        if _HAS_NP:
            stack = _np.stack([_np.asarray(v.data, dtype="float32") for v in vs])
            return Hypervector(stack.sum(axis=0))
        acc = [0.0] * self.dim
        for v in vs:
            for i, x in enumerate(v.data):
                acc[i] += x
        return Hypervector(acc)

    def majority(self, *vs: Hypervector) -> Hypervector:
        """Bundle then quantise back to bipolar ±1 (sign of the sum; ties → +1)."""
        return self.sign(self.bundle(*vs))

    def sign(self, v: Hypervector) -> Hypervector:
        if _HAS_NP:
            d = _np.asarray(v.data)
            return Hypervector(_np.where(d >= 0, 1.0, -1.0).astype("float32"))
        return Hypervector([1.0 if x >= 0 else -1.0 for x in v.data])

    def permute(self, v: Hypervector, shift: int = 1) -> Hypervector:
        """Cyclically shift components — a reversible stamp of position/order."""
        s = shift % self.dim
        if s == 0:
            return Hypervector(v.data if not _HAS_NP else _np.array(v.data))
        if _HAS_NP:
            return Hypervector(_np.roll(_np.asarray(v.data), s))
        return Hypervector(list(v.data[-s:]) + list(v.data[:-s]))

    def unpermute(self, v: Hypervector, shift: int = 1) -> Hypervector:
        return self.permute(v, -shift)

    # ---- comparison ---- #
    @staticmethod
    def similarity(a: Hypervector, b: Hypervector) -> float:
        """Cosine similarity ∈ [-1, 1]; ≈ 0 means orthogonal (unrelated)."""
        return _cosine(a.data, b.data)

    def hamming(self, a: Hypervector, b: Hypervector) -> float:
        """Normalised disagreement of signs ∈ [0, 1] (0 == identical direction)."""
        sa, sb = self.sign(a), self.sign(b)
        if _HAS_NP:
            return float(_np.mean(_np.asarray(sa.data) != _np.asarray(sb.data)))
        diff = sum(1 for x, y in zip(sa.data, sb.data) if x != y)
        return diff / self.dim


# --------------------------------------------------------------------------- #
# Cleanup / associative memory
# --------------------------------------------------------------------------- #
@dataclass
class CleanupResult:
    """Nearest clean symbol for a noisy hypervector (the result of associative recall)."""

    name: Optional[str]
    score: float
    runner_up: Optional[str] = None
    margin: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "score": round(self.score, 6),
                "runner_up": self.runner_up, "margin": round(self.margin, 6)}


class ItemMemory:
    """A named vocabulary of clean atomic hypervectors, with cosine cleanup.

    Unbinding always returns a *noisy* vector; :meth:`cleanup` snaps it to the nearest stored
    symbol. This is content-addressable, associative memory — the engine of HDC recall.
    """

    def __init__(self, space: HyperSpace) -> None:
        self.space = space
        self._items: Dict[str, Hypervector] = {}
        self._matrix: Any = None  # cached numpy stack for fast cleanup
        self._names: List[str] = []

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def names(self) -> List[str]:
        return list(self._items)

    def add(self, name: str, vector: Optional[Hypervector] = None) -> Hypervector:
        """Register ``name`` (a fresh random vector if none supplied) and return it."""
        if name in self._items:
            return self._items[name]
        vec = vector if vector is not None else self.space.random()
        self._items[name] = vec
        self._matrix = None  # invalidate cache
        return vec

    def get(self, name: str) -> Hypervector:
        return self._items[name]

    def symbol(self, name: str) -> Hypervector:
        """Get-or-create the atomic vector for ``name`` (stable across calls)."""
        return self._items.get(name) or self.add(name)

    def _rebuild(self) -> None:
        self._names = list(self._items)
        if _HAS_NP and self._names:
            self._matrix = _np.stack([_np.asarray(self._items[n].data, dtype="float32")
                                      for n in self._names])

    def cleanup(self, query: Hypervector, *, top: int = 1,
                exclude: Sequence[str] = ()) -> CleanupResult:
        """Snap ``query`` to the nearest stored symbol(s)."""
        if not self._items:
            return CleanupResult(name=None, score=0.0)
        excl = set(exclude)
        scored: List[Tuple[str, float]] = []
        if _HAS_NP:
            if self._matrix is None or len(self._names) != len(self._items):
                self._rebuild()
            q = _np.asarray(query.data, dtype="float32")
            qn = float(_np.linalg.norm(q)) or 1.0
            mat = self._matrix
            mn = _np.linalg.norm(mat, axis=1)
            mn[mn == 0] = 1.0
            sims = (mat @ q) / (mn * qn)
            for name, s in zip(self._names, sims):
                if name not in excl:
                    scored.append((name, float(s)))
        else:
            for name, vec in self._items.items():
                if name not in excl:
                    scored.append((name, _cosine(vec.data, query.data)))
        if not scored:
            # Everything in the vocabulary was excluded. There is no nearest symbol, and
            # indexing an empty list here used to raise instead of saying so.
            return CleanupResult(name=None, score=0.0)
        scored.sort(key=lambda t: t[1], reverse=True)
        best = scored[0]
        runner = scored[1] if len(scored) > 1 else None
        return CleanupResult(
            name=best[0], score=best[1],
            runner_up=runner[0] if runner else None,
            margin=best[1] - runner[1] if runner else best[1],
        )


# --------------------------------------------------------------------------- #
# Signed random projection (the SimHash bridge into hyperspace)
# --------------------------------------------------------------------------- #
class RandomProjector:
    """Lift a low-dimensional real vector into hyperspace, preserving angular similarity.

    Uses a fixed ±1 random matrix and takes the sign of the projection (SimHash). The
    Goemans–Williamson bound guarantees ``P(sign agrees) = 1 − θ/π``, so cosine geometry in
    the source space is preserved as bit-agreement in the hyperspace — the principled way to
    feed raw features or text embeddings into HDC.
    """

    def __init__(self, in_dim: int, out_dim: int, *, seed: int = 7) -> None:
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.seed = int(seed)
        if _HAS_NP:
            rng = _np.random.default_rng(seed)
            self._mat = (rng.integers(0, 2, (out_dim, in_dim)).astype("float32") * 2.0) - 1.0
        else:
            rng = random.Random(seed)
            self._mat = [[1.0 if rng.random() < 0.5 else -1.0 for _ in range(in_dim)]
                         for _ in range(out_dim)]

    def project(self, x: Sequence[float], *, bipolar: bool = True) -> Hypervector:
        if len(x) != self.in_dim:
            raise ValueError(f"expected vector of length {self.in_dim}, got {len(x)}")
        if _HAS_NP:
            proj = self._mat @ _np.asarray(x, dtype="float32")
            if bipolar:
                proj = _np.where(proj >= 0, 1.0, -1.0).astype("float32")
            return Hypervector(proj)
        out: List[float] = []
        for row in self._mat:
            acc = 0.0
            for w, xi in zip(row, x):
                acc += w * xi
            out.append((1.0 if acc >= 0 else -1.0) if bipolar else acc)
        return Hypervector(out)


# --------------------------------------------------------------------------- #
# High-level faculty
# --------------------------------------------------------------------------- #
@dataclass
class PatternReport:
    """The structure NYXARA finds in a corpus once it is mapped into hyperspace."""

    n_items: int
    clusters: List[List[str]]
    outliers: List[str]
    mean_similarity: float
    max_similarity: float

    def to_dict(self) -> Dict[str, Any]:
        return {"n_items": self.n_items, "clusters": self.clusters,
                "outliers": self.outliers,
                "mean_similarity": round(self.mean_similarity, 6),
                "max_similarity": round(self.max_similarity, 6)}


@dataclass
class NoveltyResult:
    """How unlike everything-seen a probe is — an emerging signal invisible in low dimensions."""

    score: float                         # 1 − max similarity to the corpus
    is_novel: bool
    nearest: Optional[str]
    nearest_similarity: float

    def to_dict(self) -> Dict[str, Any]:
        return {"score": round(self.score, 6), "is_novel": self.is_novel,
                "nearest": self.nearest,
                "nearest_similarity": round(self.nearest_similarity, 6)}


class LatentSpaceMap:
    """Map heterogeneous data into hyperspace, then discover, recall, reason and detect novelty.

    Ingest **records** (dicts of role→value), raw **feature vectors**, **text**, or ordered
    **sequences**; each becomes a single hypervector in a shared 10,000-D space. Then:

    * :meth:`nearest` — associative recall over the ingested corpus;
    * :meth:`analogy` — ``A : B :: C : ?`` by relational transport;
    * :meth:`discover_patterns` — clusters and outliers from pure geometry;
    * :meth:`novelty` — flag a probe far from everything seen.

    Parameters
    ----------
    dim, seed:        forwarded to the underlying :class:`HyperSpace`.
    novelty_threshold: novelty score (``1 − max_sim``) above which a probe is "novel".
    num_levels:       resolution of the continuous-value encoder (level hypervectors).
    max_corpus:       optional cap on ingested items; the oldest is evicted (FIFO) past it,
                      so a long-running live stream stays bounded. ``None`` ⇒ unbounded.
    """

    def __init__(self, *, dim: int = 10000, seed: int = 42,
                 novelty_threshold: float = 0.85, num_levels: int = 64,
                 max_corpus: Optional[int] = None) -> None:
        self.space = HyperSpace(dim, seed=seed)
        self.items = ItemMemory(self.space)       # atomic vocabulary (roles + fillers)
        self.novelty_threshold = float(novelty_threshold)
        self.max_corpus = int(max_corpus) if max_corpus else None
        self.num_levels = int(num_levels)
        self._corpus: Dict[str, Hypervector] = {}  # ingested datapoints
        # The *recipe* for each corpus entry — the original input that produced it. Persisting
        # 10,000-dimensional vectors would be tens of megabytes and, since bundled vectors are
        # real-valued rather than bipolar, could not be bit-packed without loss. The encoders are
        # deterministic given (dim, seed), so replaying the input rebuilds the vector exactly.
        # This is the pattern nyx5/hd_memory.py already uses: store recipes, not vectors.
        self._sources: Dict[str, Any] = {}
        self._levels: Optional[List[Hypervector]] = None
        self._num_range: Dict[str, Tuple[float, float]] = {}
        self._projectors: Dict[int, RandomProjector] = {}  # one per source width
        self._embedder: Any = None

    # ---- vocabulary helpers ---- #
    def symbol(self, name: str) -> Hypervector:
        return self.items.symbol(name)

    def _role(self, key: str) -> Hypervector:
        return self.items.symbol(f"role::{key}")

    # ---- continuous-value encoding (correlated level vectors) ---- #
    def _level_vectors(self) -> List[Hypervector]:
        """Build a ladder of level vectors where adjacent levels are similar, far ones aren't.

        Level 0 is random; each step flips a fixed slice of components, so the geometry is a
        smooth thermometer — encoding 0.51 lands near 0.49 but far from 0.05.
        """
        if self._levels is not None:
            return self._levels
        n = self.num_levels
        flips = max(1, self.space.dim // (2 * n))
        base = self.space.random()
        levels = [base]
        if _HAS_NP:
            rng = _np.random.default_rng(self.space.seed + 9991)
            order = rng.permutation(self.space.dim)
            cur = _np.array(base.data, dtype="float32")
            for i in range(1, n):
                idx = order[(i - 1) * flips: i * flips]
                cur = cur.copy()
                cur[idx] *= -1.0
                levels.append(Hypervector(cur))
        else:
            rng = random.Random(self.space.seed + 9991)
            order = list(range(self.space.dim))
            rng.shuffle(order)
            cur = list(base.data)
            for i in range(1, n):
                cur = list(cur)
                for j in order[(i - 1) * flips: i * flips]:
                    cur[j] = -cur[j]
                levels.append(Hypervector(cur))
        self._levels = levels
        return levels

    def encode_numeric(self, value: Number, lo: float, hi: float) -> Hypervector:
        levels = self._level_vectors()
        if hi <= lo:
            return levels[0]
        frac = (float(value) - lo) / (hi - lo)
        idx = int(round(max(0.0, min(1.0, frac)) * (len(levels) - 1)))
        return levels[idx]

    def _encode_value(self, key: str, value: Any) -> Hypervector:
        if isinstance(value, bool):
            return self.symbol(f"{value}")
        if isinstance(value, (int, float)):
            lo, hi = self._num_range.get(key, (float(value), float(value)))
            lo, hi = min(lo, float(value)), max(hi, float(value))
            self._num_range[key] = (lo, hi)
            return self.encode_numeric(value, lo, hi)
        return self.symbol(str(value))

    # ---- encoders ---- #
    def encode_record(self, record: Mapping[str, Any]) -> Hypervector:
        """A holographic vector for a structured record: bundle of ``role ⊗ filler`` pairs."""
        parts = [self.space.bind(self._role(k), self._encode_value(k, v))
                 for k, v in record.items()]
        return self.space.bundle(*parts)

    def encode_sequence(self, items: Sequence[Any]) -> Hypervector:
        """An order-sensitive vector for a sequence: bundle of position-permuted symbols."""
        parts = [self.space.permute(self.symbol(str(it)), i) for i, it in enumerate(items)]
        return self.space.bundle(*parts)

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            try:
                from nyxara.memory.store import make_embedder
                self._embedder = make_embedder()
            except Exception:  # noqa: BLE001 — degrade to the dependency-free embedder
                from nyxara.memory.store import HashingEmbedder
                self._embedder = HashingEmbedder(128)
        return self._embedder

    def project_vector(self, vector: Sequence[float]) -> Hypervector:
        """Lift a raw real feature vector into hyperspace (signed random projection).

        A projector is created (and cached) per source width, so feature vectors and text
        embeddings of different dimensionalities can share one map without colliding.
        """
        width = len(vector)
        proj = self._projectors.get(width)
        if proj is None:
            proj = RandomProjector(width, self.space.dim, seed=self.space.seed + 7 + width)
            self._projectors[width] = proj
        return proj.project(vector)

    def encode_text(self, text: str) -> Hypervector:
        return self.project_vector(self._get_embedder().embed(text))

    def encode(self, data: Any) -> Hypervector:
        """Dispatch any supported input to its encoder, returning a hypervector."""
        if isinstance(data, Hypervector):
            return data
        if isinstance(data, str):
            return self.encode_text(data)
        if isinstance(data, Mapping):
            return self.encode_record(data)
        if isinstance(data, Sequence):
            if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in data):
                return self.project_vector(data)  # numeric feature vector
            return self.encode_sequence(data)     # token sequence
        raise TypeError(f"cannot encode object of type {type(data).__name__}")

    # ---- corpus ingestion ---- #
    def add(self, name: str, data: Any) -> Hypervector:
        """Encode ``data`` and store it in the corpus under ``name`` (FIFO-capped if set)."""
        vec = self.encode(data)
        if name in self._corpus:
            del self._corpus[name]  # re-insert so refresh moves it to newest
            self._sources.pop(name, None)
        self._corpus[name] = vec
        if _json_safe(data):
            self._sources[name] = data      # the recipe, so this entry can be rebuilt on load
        if self.max_corpus is not None:
            while len(self._corpus) > self.max_corpus:
                oldest = next(iter(self._corpus))
                del self._corpus[oldest]
                self._sources.pop(oldest, None)
        return vec

    def forget(self, name: str) -> bool:
        """Drop a corpus item; returns True if it was present."""
        self._sources.pop(name, None)
        return self._corpus.pop(name, None) is not None

    # ---- persistence (recipes, not vectors) ---- #
    def to_dict(self) -> Dict[str, Any]:
        """Serialise the space as its construction parameters plus the corpus recipes.

        Only entries whose input was JSON-safe are carried: a vector handed in directly cannot be
        replayed, and silently dropping it is better than persisting something that would restore
        as a different vector."""
        return {"dim": self.space.dim, "seed": self.space.seed,
                "novelty_threshold": self.novelty_threshold, "num_levels": self.num_levels,
                "max_corpus": self.max_corpus,
                "num_range": {k: list(v) for k, v in self._num_range.items()},
                "sources": dict(self._sources)}

    def load_dict(self, data: Mapping[str, Any]) -> bool:
        """Rebuild the corpus by replaying its recipes. Never raises.

        The vectors are *recomputed*, not restored, which is only sound because the encoders are
        deterministic given ``(dim, seed)`` — so this asserts those match rather than quietly
        rebuilding a different space."""
        try:
            if int(data.get("dim", self.space.dim)) != self.space.dim:
                return False
            if int(data.get("seed", self.space.seed)) != self.space.seed:
                return False
            self.novelty_threshold = float(data.get("novelty_threshold", self.novelty_threshold))
            self.num_levels = int(data.get("num_levels", self.num_levels))
            cap = data.get("max_corpus")
            self.max_corpus = int(cap) if cap else None
            self._num_range = {str(k): (float(v[0]), float(v[1]))
                               for k, v in (data.get("num_range") or {}).items()
                               if isinstance(v, (list, tuple)) and len(v) == 2}
            # Reset the space and vocabulary too: ItemMemory draws a fresh random vector the
            # first time a symbol is requested, so first-request ORDER determines every vector.
            # Loading into an already-populated map without this replays the same recipes onto
            # different vectors — the corpus would look restored and recall the wrong things.
            self.space = HyperSpace(self.space.dim, seed=self.space.seed)
            self.items = ItemMemory(self.space)
            self._levels = None
            self._projectors = {}
            self._corpus = {}
            self._sources = {}
            for name, source in (data.get("sources") or {}).items():
                try:
                    self.add(str(name), source)
                except Exception:  # noqa: BLE001 — one unreplayable recipe never costs the rest
                    continue
            return True
        except Exception:  # noqa: BLE001 — a corrupt sidecar is an empty corpus, never a crash
            return False

    def save(self, path: Any) -> bool:
        """Atomically persist the corpus recipes (tmp file → ``os.replace``)."""
        import json
        import os
        from pathlib import Path
        try:
            target = Path(str(path))
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(json.dumps(self.to_dict()), encoding="utf-8")
            os.replace(tmp, target)
            return True
        except Exception:  # noqa: BLE001
            return False

    def load(self, path: Any) -> bool:
        import json
        from pathlib import Path
        try:
            return self.load_dict(json.loads(Path(str(path)).read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 — a missing/corrupt file is simply an empty corpus
            return False

    def __len__(self) -> int:
        return len(self._corpus)

    # ---- recall ---- #
    def nearest(self, query: Any, *, k: int = 5) -> List[Tuple[str, float]]:
        """Top-``k`` corpus items most similar to ``query`` (associative recall)."""
        q = self.encode(query)
        scored = [(name, self.space.similarity(q, vec)) for name, vec in self._corpus.items()]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    # ---- relational analogy ---- #
    def analogy(self, a: Union[str, Hypervector], b: Union[str, Hypervector],
                c: Union[str, Hypervector], *, exclude_query: bool = True) -> CleanupResult:
        """Solve ``A : B :: C : ?`` by transporting the A→B mapping onto C, then cleaning up.

        ``a`` and ``b`` are typically whole records; ``c`` a role-filler. The relation
        ``R = a ⊗ b`` carries the correspondence; ``R ⊗ c`` lands on C's analogue, which the
        item memory snaps to a clean symbol (Kanerva's "dollar of Mexico" → *peso*).
        """
        hva = self._resolve(a)
        hvb = self._resolve(b)
        hvc = self._resolve(c)
        relation = self.space.bind(hva, hvb)
        probe = self.space.bind(relation, hvc)
        exclude = [c] if (exclude_query and isinstance(c, str)) else []
        return self.items.cleanup(probe, exclude=exclude)

    def _resolve(self, x: Union[str, Hypervector]) -> Hypervector:
        if isinstance(x, Hypervector):
            return x
        if x in self.items:
            return self.items.get(x)
        if x in self._corpus:
            return self._corpus[x]
        return self.items.symbol(x)

    # ---- unsupervised pattern discovery ---- #
    def discover_patterns(self, *, threshold: float = 0.2) -> PatternReport:
        """Cluster the corpus by hyperdimensional similarity; report clusters and outliers.

        Single-linkage: items closer than ``threshold`` join the same component. Random,
        unrelated vectors sit near cosine 0, so genuine structure stands out above the noise.
        """
        names = list(self._corpus)
        n = len(names)
        if n == 0:
            return PatternReport(0, [], [], 0.0, 0.0)
        # union-find over the similarity graph
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        sims: List[float] = []
        max_sim = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                s = self.space.similarity(self._corpus[names[i]], self._corpus[names[j]])
                sims.append(s)
                max_sim = max(max_sim, s)
                if s >= threshold:
                    parent[find(i)] = find(j)
        groups: Dict[int, List[str]] = {}
        for i, name in enumerate(names):
            groups.setdefault(find(i), []).append(name)
        clusters = sorted((g for g in groups.values() if len(g) > 1), key=len, reverse=True)
        outliers = sorted(g[0] for g in groups.values() if len(g) == 1)
        mean_sim = sum(sims) / len(sims) if sims else 0.0
        return PatternReport(n, clusters, outliers, mean_sim, max_sim)

    # ---- novelty / anomaly ---- #
    def novelty(self, data: Any) -> NoveltyResult:
        """How far a probe is from everything seen — high score ⇒ an emerging/invisible pattern."""
        if not self._corpus:
            return NoveltyResult(score=1.0, is_novel=True, nearest=None, nearest_similarity=0.0)
        q = self.encode(data)
        best_name, best_sim = "", -1.0
        for name, vec in self._corpus.items():
            s = self.space.similarity(q, vec)
            if s > best_sim:
                best_name, best_sim = name, s
        score = 1.0 - max(0.0, best_sim)
        return NoveltyResult(score=score, is_novel=score >= self.novelty_threshold,
                             nearest=best_name, nearest_similarity=best_sim)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print(f"NYXARA hyperdimensional latent-space self-test  (numpy={_HAS_NP})")
    print("=" * 70)

    space = HyperSpace(dim=10000, seed=42)

    # 1) blessing of dimensionality: random symbols are near-orthogonal
    a, b = space.random(), space.random()
    sim_ab = space.similarity(a, b)
    print(f"\nrandom orthogonality     : cos(a,b)={sim_ab:+.4f} (≈0 expected)")
    assert abs(sim_ab) < 0.05

    # 2) bind is invertible; bound pair is dissimilar to its parts
    key, val = space.random(), space.random()
    bound = space.bind(key, val)
    assert abs(space.similarity(bound, key)) < 0.05
    recovered = space.unbind(bound, key)
    print(f"bind/unbind recovery     : cos(unbind, val)={space.similarity(recovered, val):+.4f}")
    assert space.similarity(recovered, val) > 0.99

    # 3) bundle is similar to all its members
    c = space.random()
    bundle = space.bundle(a, b, c)
    print(f"bundle membership        : a={space.similarity(bundle, a):.3f} "
          f"b={space.similarity(bundle, b):.3f} c={space.similarity(bundle, c):.3f}")
    assert all(space.similarity(bundle, x) > 0.4 for x in (a, b, c))

    # 4) permutation makes order matter (non-commutative sequences)
    seq_ab = space.bundle(space.permute(a, 0), space.permute(b, 1))
    seq_ba = space.bundle(space.permute(b, 0), space.permute(a, 1))
    print(f"order sensitivity        : cos(ab,ba)={space.similarity(seq_ab, seq_ba):+.4f} (low)")
    assert space.similarity(seq_ab, seq_ba) < 0.2

    # 5) signed random projection preserves angular similarity (SimHash)
    if _HAS_NP:
        rng = _np.random.default_rng(0)
        x = rng.standard_normal(64).astype("float32")
        y = x + 0.15 * rng.standard_normal(64).astype("float32")  # close to x
        z = rng.standard_normal(64).astype("float32")             # unrelated
        proj = RandomProjector(64, 10000, seed=1)
        px, py, pz = proj.project(x), proj.project(y), proj.project(z)
        print(f"projection preserves sim : cos(x,y)~{space.similarity(px, py):.3f} > "
              f"cos(x,z)~{space.similarity(px, pz):.3f}")
        assert space.similarity(px, py) > space.similarity(px, pz)

    # 6) relational analogy — Kanerva's "what is the dollar of Mexico?"  → peso
    hmap = LatentSpaceMap(dim=10000, seed=42)
    usa = hmap.add("USA", {"name": "usa", "capital": "washington", "currency": "dollar"})
    mex = hmap.add("MEX", {"name": "mexico", "capital": "mexico_city", "currency": "peso"})
    ans = hmap.analogy("USA", "MEX", "dollar")
    print(f"\nanalogy dollar:USA::?:MEX : {ans.to_dict()}")
    assert ans.name == "peso", f"expected peso, got {ans.name}"
    ans2 = hmap.analogy("USA", "MEX", "washington")
    print(f"analogy capital          : {ans2.to_dict()}")
    assert ans2.name == "mexico_city"

    # 7) pattern discovery + novelty over a small corpus
    lm = LatentSpaceMap(dim=10000, seed=7, novelty_threshold=0.85)
    lm.add("buy_tech_1", {"action": "buy", "sector": "tech", "size": "large"})
    lm.add("buy_tech_2", {"action": "buy", "sector": "tech", "size": "small"})
    lm.add("sell_energy", {"action": "sell", "sector": "energy", "size": "large"})
    report = lm.discover_patterns(threshold=0.45)
    print(f"\npattern discovery        : {report.to_dict()}")
    assert any(set(cl) == {"buy_tech_1", "buy_tech_2"} for cl in report.clusters)

    nov = lm.novelty({"action": "short", "sector": "crypto", "size": "huge"})
    print(f"novelty (unseen regime)  : {nov.to_dict()}")
    assert nov.is_novel
    seen = lm.novelty({"action": "buy", "sector": "tech", "size": "large"})
    print(f"novelty (familiar)       : {seen.to_dict()}")
    assert not seen.is_novel

    print("\nALL SELF-TESTS PASSED ✓")
