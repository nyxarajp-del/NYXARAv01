"""NYXARA · growth/frontier.py — open-ended discovery (🌌, Pillar F · Edge 1, Rule 4).

In a world where raw intelligence is *commoditised* — where every mind has AGI — being "smart"
is the floor, not the edge. The edge belongs to the mind that keeps **expanding the frontier of
what has ever been explored** instead of only getting better at what is already known. That is
what this module gives NYXARA: an **open-ended discovery engine** built on a Quality-Diversity
archive (MAP-Elites + novelty search).

The idea, concretely:

* Every discovery (a question she posed, an experiment she ran, a concept she formed) is reduced
  to a small **behaviour descriptor** — *which niche* it lives in (domain × difficulty ×
  structure), not how good it is.
* The :class:`NoveltyArchive` keeps the single best-quality discovery per niche (the MAP-Elites
  *elite*) and scores how **novel** each new behaviour is (distance to the nearest behaviours
  she has already seen).
* The :class:`FrontierEngine` reads the archive and points the next effort at the **sparsest,
  least-explored niche** — so curiosity (``selfplay``) and the autonomous scientist manufacture
  experience at the *edge* of the explored region, not in its crowded centre. This is the engine
  of genuinely open-ended growth: she does not merely learn existing knowledge, she keeps walking
  off the map.

Like the Intelligence Index, the archive rides NYXARA's existing long-term memory (one protected
SEMANTIC record, tag ``frontier-archive``) — no new persistence format, no schema migration. It
is **pure stdlib** (``math`` / ``dataclasses`` / ``collections``), degrades to in-process state
when memory is absent, and only ever *measures and directs*: it touches no source, no weights,
and no gate. Character-locked like all of growth/.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "DOMAINS",
    "BehaviorDescriptor",
    "Discovery",
    "NoveltyArchive",
    "FrontierEngine",
]

# The coarse domain axis of the behaviour space. The descriptor's first coordinate places a
# discovery on this axis; ``next_direction`` reads it back to name a concrete topic to explore.
DOMAINS: Tuple[str, ...] = (
    "reasoning", "mathematics", "code", "science", "language",
    "knowledge", "planning", "ethics",
)

_ARCHIVE_TAG = "frontier-archive"


# --------------------------------------------------------------------------- #
# A point in behaviour space — *where* a discovery lives, not how good it is
# --------------------------------------------------------------------------- #
@dataclass
class BehaviorDescriptor:
    """A small normalised behaviour vector (each coordinate in ``[0, 1]``).

    The default space is three coordinates — ``(domain, difficulty, structure)`` — but the archive
    works for any fixed length, so the space can be widened later without touching the maths.
    """

    dims: Tuple[float, ...]
    labels: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.dims = tuple(_clamp01(d) for d in self.dims)

    def cell(self, resolution: int) -> Tuple[int, ...]:
        """Bucket each coordinate into ``resolution`` bins — the discrete niche key (the grid cell)."""
        r = max(1, int(resolution))
        return tuple(min(r - 1, int(d * r)) for d in self.dims)

    def distance(self, other: "BehaviorDescriptor") -> float:
        """Euclidean distance, normalised to ``[0, 1]`` by the diagonal of the unit cube."""
        n = min(len(self.dims), len(other.dims))
        if n == 0:
            return 0.0
        sq = sum((self.dims[i] - other.dims[i]) ** 2 for i in range(n))
        return math.sqrt(sq) / math.sqrt(n)

    def to_dict(self) -> Dict[str, Any]:
        return {"dims": list(self.dims), "labels": list(self.labels)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BehaviorDescriptor":
        if not isinstance(d, dict):
            return cls(dims=())
        return cls(dims=tuple(float(x) for x in d.get("dims", []) or []),
                   labels=tuple(str(x) for x in d.get("labels", []) or []))


# --------------------------------------------------------------------------- #
# A thing she discovered
# --------------------------------------------------------------------------- #
@dataclass
class Discovery:
    """One discovery placed in behaviour space, with a scalar quality and provenance."""

    content: str
    descriptor: BehaviorDescriptor
    quality: float = 1.0
    provenance: str = "frontier"
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"content": self.content, "descriptor": self.descriptor.to_dict(),
                "quality": round(float(self.quality), 6), "provenance": self.provenance,
                "at": self.at}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Discovery":
        return cls(content=str(d.get("content", "")),
                   descriptor=BehaviorDescriptor.from_dict(d.get("descriptor", {})),
                   quality=float(d.get("quality", 1.0) or 0.0),
                   provenance=str(d.get("provenance", "frontier")),
                   at=float(d.get("at", time.time()) or time.time()))


# --------------------------------------------------------------------------- #
# The Quality-Diversity archive (MAP-Elites grid + novelty search)
# --------------------------------------------------------------------------- #
class NoveltyArchive:
    """Keep the best-quality discovery per niche, and score the novelty of new behaviours."""

    def __init__(self, *, resolution: int = 8, k: int = 5, ndims: int = 3,
                 learned_novelty: bool = True) -> None:
        self.resolution = max(1, int(resolution))
        self.k = max(1, int(k))
        self.ndims = max(1, int(ndims))
        self._grid: Dict[Tuple[int, ...], Discovery] = {}
        self._behaviors: List[Tuple[float, ...]] = []   # every behaviour seen — for kNN novelty
        self.total_seen: int = 0
        # a learned (Random Network Distillation) novelty signal, trained by gradient descent, that
        # blends with the kNN distance floor. None without numpy ⇒ pure kNN, exactly as before.
        self.learned_novelty = bool(learned_novelty)
        self._rnd: Any = None
        if self.learned_novelty:
            self._rnd = self._build_rnd()

    def _build_rnd(self) -> Any:
        try:
            from nyxara.growth.curiosity_nets import RNDNovelty, has_numpy
            if has_numpy():
                return RNDNovelty(self.ndims)
        except Exception:  # noqa: BLE001 — a missing learner never blocks the kNN floor
            pass
        return None

    # ---- the core operation: place a discovery, report novelty + whether it advanced ---- #
    def add(self, discovery: Discovery) -> Dict[str, Any]:
        """Place ``discovery``; return whether it opened a *new* niche or *improved* an old one."""
        desc = discovery.descriptor
        novelty = self.novelty(desc)                    # measured BEFORE we add it
        cell = desc.cell(self.resolution)
        elite = self._grid.get(cell)
        is_novel = elite is None
        improved = (elite is not None) and (discovery.quality > elite.quality)
        if is_novel or improved:
            self._grid[cell] = discovery
        self._behaviors.append(tuple(desc.dims))
        self.total_seen += 1
        if self._rnd is not None:                        # one real gradient step toward the RND target
            try:
                self._rnd.update(desc.dims)
            except Exception:  # noqa: BLE001 — a flaky learner never blocks placement
                pass
        return {"novel": bool(is_novel), "improved": bool(improved),
                "novelty": round(novelty, 6), "cell": cell,
                "coverage": self.coverage()}

    def _knn_novelty(self, descriptor: BehaviorDescriptor) -> float:
        """Mean distance to the ``k`` nearest behaviours already seen (``1.0`` on an empty archive)."""
        if not self._behaviors:
            return 1.0
        dists = sorted(descriptor.distance(BehaviorDescriptor(dims=b)) for b in self._behaviors)
        nearest = dists[: self.k]
        return sum(nearest) / len(nearest)

    def novelty(self, descriptor: BehaviorDescriptor) -> float:
        """How novel is this behaviour? A blend of the kNN distance floor and the **learned** RND
        novelty (gradient-descent-trained) when numpy is present; pure kNN otherwise. ``1.0`` on an
        empty archive."""
        if not self._behaviors:                          # cold archive → maximally novel (invariant)
            return 1.0
        knn = self._knn_novelty(descriptor)
        if self._rnd is not None:
            rnd = self._rnd.score(descriptor.dims)
            if rnd is not None:
                return 0.5 * knn + 0.5 * float(rnd)
        return knn

    def coverage(self) -> int:
        """How many distinct niches have been reached (filled cells of the grid)."""
        return len(self._grid)

    def capacity(self) -> int:
        """Total number of niches in the grid (``resolution ** ndims``)."""
        return self.resolution ** self.ndims

    def elites(self) -> List[Discovery]:
        """The current best-per-niche discoveries, strongest first."""
        return sorted(self._grid.values(), key=lambda d: d.quality, reverse=True)

    def sparsest_bins(self) -> Tuple[int, ...]:
        """For each coordinate, the bin with the fewest filled cells — the under-explored direction."""
        counts = [[0] * self.resolution for _ in range(self.ndims)]
        for cell in self._grid:
            for axis in range(min(self.ndims, len(cell))):
                counts[axis][cell[axis]] += 1
        return tuple(min(range(self.resolution), key=lambda b: counts[axis][b])
                     for axis in range(self.ndims))

    # ---- persistence helpers (the engine rides these onto long-term memory) ---- #
    def to_dict(self) -> Dict[str, Any]:
        d = {"resolution": self.resolution, "k": self.k, "ndims": self.ndims,
             "total_seen": self.total_seen,
             "grid": {",".join(map(str, cell)): disc.to_dict()
                      for cell, disc in self._grid.items()},
             "behaviors": [list(b) for b in self._behaviors]}
        if self._rnd is not None:
            try:
                d["rnd"] = self._rnd.to_dict()
            except Exception:  # noqa: BLE001 — a flaky learner never blocks persistence
                pass
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any], *, learned_novelty: bool = True) -> "NoveltyArchive":
        if not isinstance(d, dict):
            return cls(learned_novelty=learned_novelty)
        arc = cls(resolution=int(d.get("resolution", 8) or 8),
                  k=int(d.get("k", 5) or 5), ndims=int(d.get("ndims", 3) or 3),
                  learned_novelty=learned_novelty)
        arc.total_seen = int(d.get("total_seen", 0) or 0)
        for key, disc in (d.get("grid", {}) or {}).items():
            try:
                cell = tuple(int(x) for x in str(key).split(","))
            except Exception:  # noqa: BLE001
                continue
            arc._grid[cell] = Discovery.from_dict(disc)
        arc._behaviors = [tuple(float(x) for x in b) for b in (d.get("behaviors", []) or [])]
        rd = d.get("rnd")
        if rd and arc._rnd is not None:
            try:
                from nyxara.growth.curiosity_nets import RNDNovelty
                arc._rnd = RNDNovelty.from_dict(rd)
            except Exception:  # noqa: BLE001 — a bad learner never blocks a restore
                pass
        return arc


# --------------------------------------------------------------------------- #
# The engine: characterise discoveries, drive the next exploration, report progress
# --------------------------------------------------------------------------- #
class FrontierEngine:
    """Turn the QD archive into a *driver*: where on the frontier should NYXARA push next?

    Parameters
    ----------
    memory:      a :class:`~nyxara.memory.store.MemoryStore` for cross-restart persistence
                 (optional — falls back to in-process state when absent), exactly like
                 :class:`~nyxara.growth.intelligence.IntelligenceIndex`.
    settings:    :class:`~nyxara.kernel.config.NyxaraSettings` (pulled from ``get_settings`` if
                 not supplied); only its memory tag is read.
    """

    def __init__(self, *, memory: Any = None, settings: Any = None,
                 resolution: int = 8, k: int = 5) -> None:
        from nyxara.kernel.config import get_settings
        self.settings = settings or get_settings()
        self.memory = memory
        self.resolution = max(1, int(resolution))
        self.k = max(1, int(k))
        self._archive: Optional[NoveltyArchive] = None
        self._recent_novelty: List[float] = []

    # ---- the archive (loaded lazily from memory) ---- #
    def archive(self) -> NoveltyArchive:
        if self._archive is None:
            self._archive = self._load()
        return self._archive

    # ------------------------------------------------------------------ #
    # Characterisation — reduce any content to a behaviour descriptor.
    # Deterministic + pure stdlib, so the same discovery always lands in the same niche.
    # ------------------------------------------------------------------ #
    def characterize(self, content: str, *, domain: Optional[str] = None) -> BehaviorDescriptor:
        text = content or ""
        # domain coordinate: a known domain maps to its slot; an unknown one hashes into the axis.
        if domain and domain.lower() in DOMAINS:
            idx = DOMAINS.index(domain.lower())
            dlabel = domain.lower()
        else:
            idx = (abs(hash(domain)) if domain else _domain_guess(text)) % len(DOMAINS)
            dlabel = domain or DOMAINS[idx]
        domain_coord = (idx + 0.5) / len(DOMAINS)
        # difficulty coordinate: longer / more clause-heavy content sits higher (log-scaled).
        words = max(1, len(text.split()))
        difficulty = _clamp01(math.log2(words + 1) / math.log2(129))   # ~128 words -> 1.0
        # structure coordinate: fraction of non-alphabetic, non-space chars (numbers / symbols / code).
        dense = sum(1 for c in text if not c.isalpha() and not c.isspace())
        structure = _clamp01(dense / max(1, len(text)))
        return BehaviorDescriptor(dims=(domain_coord, difficulty, structure),
                                  labels=(dlabel, "difficulty", "structure"))

    # ------------------------------------------------------------------ #
    # Ingest a discovery — characterise, place, persist.
    # ------------------------------------------------------------------ #
    def ingest(self, content: str, *, domain: Optional[str] = None, quality: float = 1.0,
               provenance: str = "frontier", descriptor: Optional[BehaviorDescriptor] = None,
               save: bool = True) -> Dict[str, Any]:
        desc = descriptor or self.characterize(content, domain=domain)
        disc = Discovery(content=content, descriptor=desc, quality=float(quality),
                         provenance=provenance)
        result = self.archive().add(disc)
        self._recent_novelty = (self._recent_novelty + [result["novelty"]])[-32:]
        if save:
            self.save()
        return result

    # ------------------------------------------------------------------ #
    # Direct the next exploration — point at the sparsest, least-explored niche.
    # ------------------------------------------------------------------ #
    def next_direction(self) -> Dict[str, Any]:
        """Name a concrete topic at the under-explored edge of the frontier."""
        arc = self.archive()
        bins = arc.sparsest_bins()
        # bin -> coordinate centre
        centre = tuple((b + 0.5) / arc.resolution for b in bins)
        domain = DOMAINS[min(len(DOMAINS) - 1, bins[0] * len(DOMAINS) // arc.resolution)]
        difficulty = ("introductory", "intermediate", "hard", "frontier")[
            min(3, bins[1] * 4 // arc.resolution)]
        topic = f"{difficulty} problems in {domain}"
        return {"topic": topic, "domain": domain, "difficulty": difficulty,
                "descriptor": BehaviorDescriptor(dims=centre).to_dict(),
                "coverage": arc.coverage(), "capacity": arc.capacity(),
                "rationale": f"least-explored niche on axes (domain, difficulty, structure)={bins}"}

    def topics(self, n: int = 4) -> List[str]:
        """A short list of under-explored topics to seed curiosity (``selfplay``)."""
        arc = self.archive()
        counts = [0] * len(DOMAINS)
        for cell in arc._grid:
            counts[min(len(DOMAINS) - 1, cell[0] * len(DOMAINS) // arc.resolution)] += 1
        order = sorted(range(len(DOMAINS)), key=lambda i: counts[i])
        diff = self.next_direction()["difficulty"]
        return [f"{diff} problems in {DOMAINS[i]}" for i in order[: max(1, n)]]

    # ------------------------------------------------------------------ #
    # Report — coverage, novelty trend, frontier growth.
    # ------------------------------------------------------------------ #
    def report(self) -> Dict[str, Any]:
        arc = self.archive()
        mean_nov = (sum(self._recent_novelty) / len(self._recent_novelty)
                    if self._recent_novelty else 0.0)
        return {"coverage": arc.coverage(), "capacity": arc.capacity(),
                "coverage_frac": round(arc.coverage() / max(1, arc.capacity()), 6),
                "total_seen": arc.total_seen,
                "recent_novelty": round(mean_nov, 6),
                "top_elites": [d.content[:80] for d in arc.elites()[:3]]}

    # ------------------------------------------------------------------ #
    # Persistence — one protected SEMANTIC record (mirrors IntelligenceIndex).
    # ------------------------------------------------------------------ #
    def _load(self) -> NoveltyArchive:
        learned = bool(getattr(getattr(self.settings, "explorer", None),
                               "frontier_learned_novelty", True))
        rec = self._find_record()
        if rec is not None:
            return NoveltyArchive.from_dict(rec.metadata.get("archive", {}),
                                            learned_novelty=learned)
        return NoveltyArchive(resolution=self.resolution, k=self.k, learned_novelty=learned)

    def save(self) -> bool:
        if self.memory is None or self._archive is None:
            return False
        try:
            from nyxara.memory.provenance import Provenance, SourceType
            from nyxara.memory.store import MemoryType
        except Exception:  # noqa: BLE001
            return False
        tag = str(getattr(self.settings.memory, "deep_synapse_tag", "deep-synapse"))
        arc = self._archive
        content = f"[frontier-archive] coverage {arc.coverage()}/{arc.capacity()}"
        meta = {"archive": arc.to_dict()}
        try:
            existing = self._find_record()
            if existing is not None:
                self.memory.forget(existing.mem_id)
            self.memory.remember(
                content, mem_type=MemoryType.SEMANTIC,
                provenance=Provenance(SourceType.SELF_REFLECTION, confidence=0.9),
                importance=1.0, tags=[_ARCHIVE_TAG, tag], metadata=meta)
            return True
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            return False

    def _find_record(self) -> Any:
        if self.memory is None:
            return None
        try:
            for rec in self.memory._kv.values():
                if _ARCHIVE_TAG in rec.tags and "archive" in rec.metadata:
                    return rec
        except Exception:  # noqa: BLE001
            return None
        return None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _clamp01(x: Any) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:  # noqa: BLE001
        return 0.0


_DOMAIN_HINTS: Dict[str, Sequence[str]] = {
    "mathematics": ("solve", "equation", "+", "*", "integral", "prime", "%", "="),
    "code": ("def ", "function", "code", "python", "compile", "bug", "();", "{}"),
    "science": ("hypothesis", "experiment", "physics", "chemistry", "biology", "why does"),
    "language": ("translate", "summarise", "summarize", "grammar", "essay", "poem"),
    "planning": ("plan", "schedule", "goal", "steps", "strategy"),
    "ethics": ("should", "ought", "fair", "right", "wrong", "moral"),
    "knowledge": ("who", "what is", "when did", "where", "fact"),
}


def _domain_guess(text: str) -> int:
    """Best-effort domain bucket from keywords; defaults to 'reasoning'."""
    low = (text or "").lower()
    best = "reasoning"
    best_hits = 0
    for dom, hints in _DOMAIN_HINTS.items():
        hits = sum(1 for h in hints if h in low)
        if hits > best_hits:
            best, best_hits = dom, hits
    return DOMAINS.index(best) if best in DOMAINS else 0


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA frontier (open-ended discovery) self-test")
    print("=" * 70)

    eng = FrontierEngine()
    samples = [
        ("What is 17 * 23 + 4?", "mathematics"),
        ("Write a function that reverses a linked list.", "code"),
        ("Why does ice float on water?", "science"),
        ("Is it ever right to lie to protect a friend?", "ethics"),
    ]
    for content, dom in samples:
        r = eng.ingest(content, domain=dom, save=False)
        print(f"  +{content[:42]:42s}  novel={r['novel']} novelty={r['novelty']:.3f}")

    print(f"\nreport          : {eng.report()}")
    print(f"next direction  : {eng.next_direction()['topic']}")
    print(f"seed topics     : {eng.topics(4)}")
