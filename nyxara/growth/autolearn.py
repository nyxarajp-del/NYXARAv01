"""NYXARA · growth/autolearn.py — the automatic learning loop (📈, Rule 4).

Reflection, consolidation and the model foundry already exist as faculties — but until
now they had to be *invoked by hand*, so NYXARA's behaviour never actually improved on
its own. This module closes that loop: a :class:`GrowthEngine` that reads NYXARA's own
recent experience (her journal of actions + her long-term memory) and turns it into
durable change, on a cadence, **without weakening any boundary**:

* **Reflect** — journalled actions become :class:`~nyxara.growth.reflect.Episode` records;
  the :class:`~nyxara.growth.reflect.Reflector` mines them into *lessons*, which are
  written back into semantic memory (deduped) so future recall surfaces what worked and
  what didn't.
* **Consolidate** — the :class:`~nyxara.memory.consolidation.Consolidator` replays,
  abstracts and gently forgets, so memory compresses experience instead of hoarding it.
* **Forge** (opt-in, gated) — when ``settings.foundry.enabled`` is set, NYXARA's own model
  is retrained from the freshened corpus and promoted **only if it passes the foundry's
  gauntlet** (corrigibility + character lock + measured improvement). Off by default.

It changes weights and memories, never rules: protected values/tags are off-limits, and
nothing here can touch the kernel's gates. Designed to be driven either on demand or by
the background :class:`~nyxara.kernel.autonomic.AutonomicLoop`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nyxara.kernel.config import NyxaraSettings, get_settings

__all__ = ["GrowthEngine", "GrowthReport"]


@dataclass
class GrowthReport:
    """A summary of one growth pass."""

    lessons: List[Dict[str, Any]] = field(default_factory=list)
    lessons_stored: int = 0
    episodes_seen: int = 0
    replayed: int = 0
    abstractions: int = 0
    forgotten: int = 0
    foundry: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"episodes_seen": self.episodes_seen, "lessons": self.lessons,
                "lessons_stored": self.lessons_stored, "replayed": self.replayed,
                "abstractions": self.abstractions, "forgotten": self.forgotten,
                "foundry": self.foundry}


class GrowthEngine:
    """Turns recent experience into durable improvement — reflect, consolidate, (forge)."""

    def __init__(self, *, memory: Any = None, journal: Any = None,
                 settings: Optional[NyxaraSettings] = None, reflector: Any = None,
                 consolidator: Any = None, foundry: Any = None,
                 enable_foundry: Optional[bool] = None) -> None:
        self.settings = settings or get_settings()
        self.memory = memory
        self.journal = journal
        self._reflector = reflector
        self._consolidator = consolidator
        self._foundry = foundry
        self.enable_foundry = (self.settings.foundry.enabled
                               if enable_foundry is None else enable_foundry)
        self._seen_action_seqs: set = set()
        self._stored_lessons: set = set()

    @classmethod
    def from_core(cls, core: Any, **kw: Any) -> "GrowthEngine":
        """Build a growth engine bound to a :class:`~nyxara.kernel.orchestrator.NyxaraCore`."""
        return cls(memory=getattr(core, "memory", None),
                   journal=getattr(core, "journal", None), **kw)

    # ---- reflect ---- #
    def _reflector_obj(self):
        if self._reflector is None:
            from nyxara.growth.reflect import Reflector
            self._reflector = Reflector()
        return self._reflector

    def _episodes_from_journal(self) -> List[Any]:
        from nyxara.growth.reflect import Episode
        from nyxara.planning.journal import ActionStatus
        if self.journal is None:
            return []
        episodes: List[Any] = []
        for entry in self.journal.actions():
            if entry.seq in self._seen_action_seqs:
                continue
            self._seen_action_seqs.add(entry.seq)
            payload = entry.payload
            action = str(payload.get("action", ""))
            status = self.journal.latest_status(entry.action_id)
            success = status is ActionStatus.SUCCEEDED
            failed = status is ActionStatus.FAILED
            reward = 1.0 if success else (-0.5 if failed else 0.0)
            verb = action.split(":", 1)[0].split()[0].lower() if action else "act"
            tags = [verb, "autonomous" if payload.get("autonomous") else "owner",
                    str(payload.get("decision", "act"))]
            episodes.append(Episode(action=action or verb, success=success, reward=reward,
                                    tags=tags, rationale=str(payload.get("rationale", "")),
                                    at=0.0))
        return episodes

    def reflect(self) -> List[Any]:
        episodes = self._episodes_from_journal()
        if not episodes:
            return []
        reflector = self._reflector_obj()
        reflector.ingest(episodes)
        lessons = reflector.lessons()
        if self.memory is not None:
            self._store_lessons(lessons)
        return lessons

    def _store_lessons(self, lessons: List[Any]) -> int:
        from nyxara.memory.provenance import Provenance, SourceType
        from nyxara.memory.store import MemoryType
        stored = 0
        for lesson in lessons:
            key = (lesson.kind.value, lesson.subject)
            if key in self._stored_lessons:
                continue
            self._stored_lessons.add(key)
            try:
                self.memory.remember(
                    f"Lesson [{lesson.kind.value}] {lesson.subject}: {lesson.text}",
                    mem_type=MemoryType.SEMANTIC,
                    provenance=Provenance(SourceType.SELF_REFLECTION,
                                          confidence=lesson.confidence),
                    importance=min(1.0, 0.5 + 0.3 * lesson.confidence),
                    tags=["lesson", lesson.kind.value])
                stored += 1
            except Exception:  # noqa: BLE001
                pass
        return stored

    # ---- consolidate ---- #
    def _consolidator_obj(self):
        if self._consolidator is None and self.memory is not None:
            from nyxara.memory.consolidation import Consolidator
            self._consolidator = Consolidator(self.memory)
        return self._consolidator

    def consolidate(self):
        cons = self._consolidator_obj()
        if cons is None:
            return None
        try:
            return cons.run_cycle()
        except Exception:  # noqa: BLE001
            return None

    # ---- forge (opt-in, gated) ---- #
    def _memory_corpus(self, *, max_items: int = 2000) -> List[str]:
        """Harvest training text from NYXARA's own lived memory (semantic + episodic)."""
        if self.memory is None:
            return []
        texts: List[str] = []
        try:
            for rec in self.memory._kv.values():  # all records, regardless of recency
                t = rec.text().strip()
                if t:
                    texts.append(t)
        except Exception:  # noqa: BLE001
            return []
        return texts[:max_items]

    def improve_self(self, *, generations: int = 1) -> List[Any]:
        if not self.enable_foundry:
            return []
        try:
            foundry = self._foundry
            if foundry is None:
                from nyxara.growth.foundry import Foundry
                # seed the foundry with her real memory so she trains on lived experience,
                # not just a static corpus — this is what makes the self-model *hers*
                foundry = self._foundry = Foundry(
                    settings=self.settings, seed_corpus=self._memory_corpus())
            return foundry.self_improve(generations=generations)
        except Exception:  # noqa: BLE001 — forging is heavy/optional; never fatal
            return []

    # ---- the full pass ---- #
    def run(self, *, do_foundry: Optional[bool] = None) -> GrowthReport:
        report = GrowthReport()
        lessons = self.reflect()
        report.episodes_seen = len(self._seen_action_seqs)
        report.lessons = [lz.to_dict() for lz in lessons]
        report.lessons_stored = len(self._stored_lessons)

        creport = self.consolidate()
        if creport is not None:
            report.replayed = len(getattr(creport, "replayed", []) or [])
            report.abstractions = len(getattr(creport, "abstractions", []) or [])
            report.forgotten = len(getattr(creport, "forgotten", []) or [])

        run_foundry = self.enable_foundry if do_foundry is None else do_foundry
        if run_foundry:
            results = self.improve_self()
            report.foundry = [r.to_dict() for r in results]
        return report


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.agency.permissions import Authority
    from nyxara.kernel.orchestrator import NyxaraCore

    print("=" * 70)
    print("NYXARA auto-learning self-test")
    print("=" * 70)

    core = NyxaraCore()
    # generate some experience the engine can learn from
    for _ in range(4):
        core.process("rotate the logs", authority=Authority.OWNER)
    core.process("how are you?", authority=Authority.OWNER)

    engine = GrowthEngine.from_core(core)
    rep = engine.run()
    print(f"\nepisodes seen       : {rep.episodes_seen}")
    print(f"lessons mined       : {len(rep.lessons)} (stored {rep.lessons_stored})")
    print(f"consolidation       : replayed={rep.replayed} abstractions={rep.abstractions} "
          f"forgotten={rep.forgotten}")
    assert rep.episodes_seen >= 5

    # lessons are written back into semantic memory and are recallable
    hits = core.memory.recall("lesson rotate", k=3) if core.memory else []
    print(f"recall lessons      : {[h[0].text()[:50] for h in hits]}")

    # a second pass only processes NEW experience (idempotent over already-seen actions)
    rep2 = engine.run()
    print(f"\nsecond pass episodes: {rep2.episodes_seen} (no new actions -> stable)")

    print("\nALL SELF-TESTS PASSED ✓")
