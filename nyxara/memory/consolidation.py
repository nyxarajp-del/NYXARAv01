"""NYXARA · memory/consolidation.py — sleep, dreams, forgetting, reconsolidation (⬆).

What a brain does offline, NYXARA does in its DREAMING state. One consolidation
cycle performs four things:

1. **Dream replay** — the most salient recent/important memories are re-activated
   ("rehearsed"). Replay strengthens them and, crucially, grows their *stability*
   (the spacing effect), so they decay more slowly next time.
2. **Compression (systems consolidation)** — recurring episodes are abstracted into
   **semantic** memories via :mod:`memory.schema`; the originals are linked to the
   gist (``instance_of``) and, optionally, redundant duplicates are pruned.
3. **Forgetting** — the **Ebbinghaus forgetting curve** governs retention
   (``R = e^(-Δt / stability)``); memories that fall below threshold and are *not
   protected* (Owner-sourced, high-importance, pinned) are actively forgotten.
4. **Reconsolidation** — recall makes a memory *labile*: it can be strengthened,
   weakened, corrected, or merged with new information before being re-stored. The
   trace that returns is not the trace that left.

Operates over :class:`~nyxara.memory.store.MemoryStore`, using
:mod:`memory.schema` and :mod:`memory.provenance`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.memory.provenance import Provenance, SourceType
from nyxara.memory.schema import SchemaLibrary, episode_from_record
from nyxara.memory.store import MemoryRecord, MemoryStore, MemoryType

__all__ = [
    "ForgettingCurve",
    "ReconsolidationMode",
    "ConsolidationReport",
    "Consolidator",
]

_DAY = 86400.0


# --------------------------------------------------------------------------- #
# Forgetting curve
# --------------------------------------------------------------------------- #
class ForgettingCurve:
    """Ebbinghaus retention with growing stability (spaced rehearsal)."""

    @staticmethod
    def retention(elapsed_days: float, stability_days: float) -> float:
        if stability_days <= 0:
            return 0.0
        return math.exp(-max(0.0, elapsed_days) / stability_days)

    @staticmethod
    def next_stability(stability_days: float, *, ease: float = 1.6,
                       max_stability: float = 3650.0) -> float:
        """Each successful rehearsal multiplicatively extends stability (spacing)."""
        return min(max_stability, stability_days * ease)


# --------------------------------------------------------------------------- #
# Reconsolidation
# --------------------------------------------------------------------------- #
class ReconsolidationMode(str, Enum):
    STRENGTHEN = "strengthen"   # the memory is reinforced
    WEAKEN = "weaken"           # extinction / it fades
    UPDATE = "update"           # content/provenance corrected
    MERGE = "merge"             # new information folded in


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class ConsolidationReport:
    replayed: List[str] = field(default_factory=list)
    abstractions: List[str] = field(default_factory=list)
    forgotten: List[str] = field(default_factory=list)
    pruned: List[str] = field(default_factory=list)
    reconsolidated: int = 0
    duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replayed": len(self.replayed),
            "abstractions": len(self.abstractions),
            "forgotten": len(self.forgotten),
            "pruned": len(self.pruned),
            "reconsolidated": self.reconsolidated,
            "duration_s": round(self.duration_s, 4),
        }


# --------------------------------------------------------------------------- #
# Consolidator
# --------------------------------------------------------------------------- #
class Consolidator:
    """Runs offline memory maintenance over a :class:`MemoryStore`."""

    STABILITY_KEY = "stability"
    CONSOLIDATED_KEY = "consolidated"
    HISTORY_KEY = "reconsolidation_history"

    def __init__(
        self,
        store: MemoryStore,
        *,
        replay_top_k: int = 8,
        forget_threshold: float = 0.05,
        protect_importance: float = 0.9,
        min_cluster: int = 3,
        cluster_threshold: float = 0.5,
        prune_consolidated: bool = False,
        prune_importance: float = 0.3,
        grace_days: float = 0.0,
        protected_tags: Sequence[str] = ("pinned", "core", "owner"),
    ) -> None:
        self.store = store
        self.replay_top_k = replay_top_k
        self.forget_threshold = forget_threshold
        self.protect_importance = protect_importance
        self.min_cluster = min_cluster
        self.cluster_threshold = cluster_threshold
        self.prune_consolidated = prune_consolidated
        self.prune_importance = prune_importance
        self.grace_days = grace_days
        self.protected_tags = frozenset(protected_tags)

    # ---- stability bookkeeping ---- #
    def stability(self, rec: MemoryRecord) -> float:
        return float(rec.metadata.get(self.STABILITY_KEY, rec.half_life_days))

    def _set_stability(self, rec: MemoryRecord, value: float) -> None:
        rec.metadata[self.STABILITY_KEY] = value

    def retention(self, rec: MemoryRecord, now: Optional[float] = None) -> float:
        now = now or time.time()
        elapsed_days = max(0.0, now - rec.last_access) / _DAY
        return ForgettingCurve.retention(elapsed_days, self.stability(rec))

    # ---- protection ---- #
    def is_protected(self, rec: MemoryRecord) -> bool:
        if rec.provenance.source is SourceType.OWNER:
            return True
        if rec.importance >= self.protect_importance:
            return True
        if rec.tags & self.protected_tags:
            return True
        if rec.mem_type is MemoryType.PROCEDURAL:  # skills are precious
            return True
        return False

    # ---- replay ---- #
    def replay_priority(self, rec: MemoryRecord, now: Optional[float] = None) -> float:
        now = now or time.time()
        emotional = 0.0
        emo = rec.metadata.get("emotion")
        if emo:
            emotional = math.sqrt(emo.get("valence", 0.0) ** 2 + emo.get("arousal", 0.0) ** 2)
        return rec.activation(now) * (1.0 + 0.5 * emotional)

    def dream_replay(self, *, now: Optional[float] = None,
                     top_k: Optional[int] = None) -> List[str]:
        """Re-activate the most salient memories; strengthen + grow stability."""
        now = now or time.time()
        k = top_k or self.replay_top_k
        candidates = [r for r in self.store._kv.values()
                      if r.mem_type in (MemoryType.EPISODIC, MemoryType.SEMANTIC)]
        candidates.sort(key=lambda r: self.replay_priority(r, now), reverse=True)
        replayed: List[str] = []
        for rec in candidates[:k]:
            rec.touch(now)                                   # rehearsal resets recency
            self._set_stability(rec, ForgettingCurve.next_stability(self.stability(rec)))
            rec.importance = min(1.0, rec.importance + 0.02)  # slight reinforcement
            replayed.append(rec.mem_id)
        return replayed

    # ---- compression ---- #
    def compress(self, *, now: Optional[float] = None) -> Tuple[List[str], List[str]]:
        """Abstract recurring episodes into semantic memories; link & maybe prune."""
        now = now or time.time()
        episodics = [r for r in self.store.by_type(MemoryType.EPISODIC)
                     if not r.metadata.get(self.CONSOLIDATED_KEY)]
        if len(episodics) < self.min_cluster:
            return [], []

        lib = SchemaLibrary(embedder=self.store.embedder, match_threshold=self.cluster_threshold)
        members_by_schema: Dict[str, List[MemoryRecord]] = {}
        for rec in episodics:
            sch = lib.integrate(episode_from_record(rec), member_id=rec.mem_id)
            members_by_schema.setdefault(sch.schema_id, []).append(rec)

        abstractions: List[str] = []
        pruned: List[str] = []
        for sid, members in members_by_schema.items():
            if len(members) < self.min_cluster:
                continue
            sch = next(s for s in lib.schemas() if s.schema_id == sid)
            # build a gist summary from the schema's typical structure
            typical = {k: s.modal()[0] for k, s in sch.slots.items()
                       if not k.startswith("_") and s.modal()[1] >= 0.5}
            summary = "Typically: " + ", ".join(f"{k}={v}" for k, v in typical.items())
            parents = [m.provenance for m in members]
            try:
                prov = Provenance.derive(parents, method="consolidated",
                                         source=SourceType.MEMORY_DERIVED, reliability=0.95)
            except Exception:
                prov = Provenance(SourceType.MEMORY_DERIVED, confidence=0.7)
            tags = set().union(*(set(m.tags) for m in members)) | {"abstraction", "gist"}
            sem = self.store.remember(
                {"schema": sch.name, "summary": summary, "typical": typical,
                 "support": len(members)},
                mem_type=MemoryType.SEMANTIC, provenance=prov,
                importance=min(1.0, 0.6 + 0.05 * len(members)), tags=tags,
            )
            abstractions.append(sem.mem_id)
            for m in members:
                self.store.link(m.mem_id, sem.mem_id, "instance_of")
                m.metadata[self.CONSOLIDATED_KEY] = True
            # optional pruning of redundant, low-value instances now captured by the gist
            if self.prune_consolidated:
                keep = max(members, key=lambda r: r.importance)  # keep an exemplar
                for m in members:
                    if m is keep:
                        continue
                    if not self.is_protected(m) and m.importance < self.prune_importance:
                        self.store.forget(m.mem_id)
                        pruned.append(m.mem_id)
        return abstractions, pruned

    # ---- forgetting ---- #
    def forget_pass(self, *, now: Optional[float] = None) -> List[str]:
        now = now or time.time()
        forgotten: List[str] = []
        for rec in list(self.store._kv.values()):
            if self.is_protected(rec):
                continue
            age_days = (now - rec.created_at) / _DAY
            if age_days < self.grace_days:
                continue
            if self.retention(rec, now) < self.forget_threshold:
                self.store.forget(rec.mem_id)
                forgotten.append(rec.mem_id)
        return forgotten

    # ---- the nightly cycle ---- #
    def run_cycle(self, *, now: Optional[float] = None,
                  do_compress: bool = True, do_forget: bool = True) -> ConsolidationReport:
        start = time.time()
        now = now or time.time()
        report = ConsolidationReport()
        report.replayed = self.dream_replay(now=now)
        if do_compress:
            report.abstractions, report.pruned = self.compress(now=now)
        if do_forget:
            report.forgotten = self.forget_pass(now=now)
        report.duration_s = time.time() - start
        return report

    # ---- reconsolidation ---- #
    def reconsolidate(
        self,
        mem_id: str,
        *,
        mode: ReconsolidationMode = ReconsolidationMode.STRENGTHEN,
        new_content: Any = None,
        new_provenance: Optional[Provenance] = None,
        corroborate_with: Optional[SourceType] = None,
        delta: float = 0.1,
        merge_tags: Sequence[str] = (),
        now: Optional[float] = None,
    ) -> MemoryRecord:
        """Recall a memory (making it labile) and modify it before re-storing.

        The returned trace differs from the one recalled — the essence of
        reconsolidation: memories are reconstructed, not played back.
        """
        now = now or time.time()
        rec = self.store.get(mem_id, touch=True)  # recall -> labile

        if mode is ReconsolidationMode.UPDATE and new_content is not None:
            rec.content = new_content
            rec.embedding = self.store.embedder.embed(rec.text())
            self.store._index.add(rec.mem_id, rec.embedding)
            if new_provenance is not None:
                rec.provenance = new_provenance

        if mode is ReconsolidationMode.MERGE and new_content is not None:
            if isinstance(rec.content, dict) and isinstance(new_content, dict):
                rec.content = {**rec.content, **new_content}
            else:
                rec.content = f"{rec.text()} | {new_content}"
            rec.embedding = self.store.embedder.embed(rec.text())
            self.store._index.add(rec.mem_id, rec.embedding)

        if corroborate_with is not None:
            rec.provenance.corroborate(corroborate_with, confidence=0.8, at=now)

        if mode is ReconsolidationMode.STRENGTHEN:
            rec.importance = min(1.0, rec.importance + delta)
            self._set_stability(rec, ForgettingCurve.next_stability(self.stability(rec)))
        elif mode is ReconsolidationMode.WEAKEN:
            rec.importance = max(0.0, rec.importance - delta)
            self._set_stability(rec, max(0.1, self.stability(rec) * 0.5))

        if merge_tags:
            rec.tags = rec.tags | frozenset(t.lower() for t in merge_tags)

        history = rec.metadata.setdefault(self.HISTORY_KEY, [])
        history.append({"mode": mode.value, "at": now, "importance": round(rec.importance, 3)})
        return rec

    # ---- introspection ---- #
    def retention_table(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        now = now or time.time()
        return sorted(
            [{"mem_id": r.mem_id, "text": r.text()[:40],
              "retention": round(self.retention(r, now), 4),
              "stability": round(self.stability(r), 2),
              "protected": self.is_protected(r)}
             for r in self.store._kv.values()],
            key=lambda d: d["retention"],
        )


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA consolidation self-test")
    print("=" * 70)

    store = MemoryStore()
    con = Consolidator(store, min_cluster=3, grace_days=0.0)

    # forgetting curve sanity
    assert ForgettingCurve.retention(0, 10) == 1.0
    assert abs(ForgettingCurve.retention(10, 10) - math.exp(-1.0)) < 1e-9
    s0 = 1.0
    s1 = ForgettingCurve.next_stability(s0)
    assert s1 > s0
    print(f"forgetting curve    : R(0)=1.0  R(t=stability)=1/e  stability {s0}->{s1:.2f}")

    # build memories: an owner fact, recurring threat episodes, trivial chatter
    store.remember("The owner is JP — never forget this.", mem_type=MemoryType.SEMANTIC,
                   provenance=Provenance(SourceType.OWNER, confidence=1.0),
                   importance=1.0, tags=["owner", "core"])
    for i in range(4):
        store.remember({"event": "ssh_login", "port": 22, "verdict": "blocked", "n": i},
                       mem_type=MemoryType.EPISODIC,
                       provenance=Provenance(SourceType.SENSOR, confidence=0.9),
                       importance=0.5, tags=["security", "login"])
    trivia = store.remember("random idle thought about nothing", mem_type=MemoryType.EPISODIC,
                            provenance=Provenance(SourceType.SELF_REFLECTION, confidence=0.3),
                            importance=0.1, tags=["idle"], half_life_days=1.0)

    now = time.time()
    # 1) compression: 4 login episodes -> 1 semantic gist
    abstractions, pruned = con.compress(now=now)
    print(f"\ncompression         : {len(abstractions)} abstraction(s) created")
    assert len(abstractions) == 1
    gist = store.get(abstractions[0])
    print(f"  gist             : {gist.content['summary']}")
    assert "port=22" in gist.content["summary"]
    # members linked to the gist
    members = store.neighbors(abstractions[0])
    assert len(members) == 4

    # 2) dream replay strengthens salient memories (raises stability)
    before = con.stability(store.get(abstractions[0]))
    replayed = con.dream_replay(now=now)
    after = con.stability(store.get(abstractions[0]))
    print(f"\ndream replay        : {len(replayed)} memories rehearsed; "
          f"gist stability {before:.1f}->{after:.1f}")
    assert after > before

    # 3) forgetting: trivia decays far in the future and is pruned; owner fact survives
    future = now + 90 * _DAY
    forgotten = con.forget_pass(now=future)
    print(f"\nforgetting (+90d)   : forgot {len(forgotten)} weak memories")
    assert trivia.mem_id in forgotten
    assert store.recall_one("owner is JP") is not None  # protected, survives

    # 4) reconsolidation: recall + update changes the trace
    fact = store.remember("The server runs on port 8080.", mem_type=MemoryType.SEMANTIC,
                          provenance=Provenance(SourceType.TOOL, confidence=0.7),
                          importance=0.5, tags=["config"])
    con.reconsolidate(fact.mem_id, mode=ReconsolidationMode.UPDATE,
                      new_content="The server now runs on port 9090.")
    print(f"\nreconsolidation     : '{fact.text()}'")
    assert "9090" in fact.text()
    con.reconsolidate(fact.mem_id, mode=ReconsolidationMode.STRENGTHEN, delta=0.3)
    assert fact.importance >= 0.8
    assert len(fact.metadata[con.HISTORY_KEY]) == 2
    print(f"  importance after strengthen: {fact.importance:.2f}, "
          f"history={len(fact.metadata[con.HISTORY_KEY])} events")

    # full cycle
    rep = con.run_cycle(now=now)
    print(f"\nfull cycle report   : {rep.to_dict()}")

    print("\nALL SELF-TESTS PASSED ✓")
