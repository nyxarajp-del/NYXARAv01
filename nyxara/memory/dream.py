"""NYXARA · memory/dream.py — Dreaming System (Level 12).

During idle time, NYXARA replays memories, skills, past reasoning, and failures — like
human sleep consolidation. Four replay passes run in sequence:

    1. memory replay    — Consolidator.dream_replay(): strengthen salient memories
    2. skill replay     — re-activate top-N SkillMemory procedural skills, bump use counts
    3. reasoning replay — re-run N past MindScope INFERENCE thoughts through self-critique
    4. failure replay   — surface recent FAILED actions and ask "what should I have done?"

All replay is internal — no external effects, no gate bypass. DreamSession is safe to run
on any thread that has read access to the core's subsystems.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

__all__ = ["DreamSession", "DreamReport"]


# --------------------------------------------------------------------------- #
# DreamReport
# --------------------------------------------------------------------------- #
@dataclass
class DreamReport:
    """Outcome of one dream session."""
    duration_s: float = 0.0
    memory_replayed: int = 0
    skills_replayed: int = 0
    reasoning_replayed: int = 0
    failures_reviewed: int = 0
    # --- Dream State: data distillation + log pruning + Deep Memory Synapses --- #
    principles_distilled: int = 0
    logs_deleted: int = 0
    synapses_fixed: int = 0
    deep_state: bool = False
    insights: List[str] = field(default_factory=list)
    principles: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_s": round(self.duration_s, 3),
            "memory_replayed": self.memory_replayed,
            "skills_replayed": self.skills_replayed,
            "reasoning_replayed": self.reasoning_replayed,
            "failures_reviewed": self.failures_reviewed,
            "principles_distilled": self.principles_distilled,
            "logs_deleted": self.logs_deleted,
            "synapses_fixed": self.synapses_fixed,
            "deep_state": self.deep_state,
            "insights": self.insights,
            "principles": self.principles,
        }


# --------------------------------------------------------------------------- #
# DreamSession
# --------------------------------------------------------------------------- #
class DreamSession:
    """Run four replay passes as a unified dream cycle.

    Parameters
    ----------
    consolidator:   Consolidator instance (memory replay).
    skill_memory:   SkillMemory instance (skill replay).
    mind:           MindScope instance (inference thought retrieval).
    journal:        Journal instance (failure action retrieval).
    reflector:      Reflector instance (self-critique of failures).
    top_k_memories: how many memories to replay per session.
    top_k_skills:   how many skills to re-activate per session.
    top_k_inferences: how many INFERENCE thoughts to re-run per session.
    """

    def __init__(self, consolidator: Any = None, skill_memory: Any = None,
                 mind: Any = None, journal: Any = None, reflector: Any = None,
                 memory: Any = None, top_k_memories: int = 10, top_k_skills: int = 5,
                 top_k_inferences: int = 5, deep_synapse_tag: str = "deep-synapse",
                 distill_min_support: int = 2) -> None:
        self.consolidator = consolidator
        self.skill_memory = skill_memory
        self.mind = mind
        self.journal = journal
        self.reflector = reflector
        self.memory = memory
        self.top_k_memories = max(1, int(top_k_memories))
        self.top_k_skills = max(1, int(top_k_skills))
        self.top_k_inferences = max(1, int(top_k_inferences))
        self.deep_synapse_tag = deep_synapse_tag
        self.distill_min_support = max(1, int(distill_min_support))
        self._sessions: int = 0
        self._last_report: Any = None

    # ---------------------------------------------------------------------- #
    def dream(self, duration_s: float = 30.0, *, deep: bool = True) -> DreamReport:
        """Run one dream session. Always returns a DreamReport.

        ``duration_s`` is advisory — the session runs all passes regardless, but if any pass
        takes too long, it degrades gracefully. The first four passes *replay* experience; the
        last three (``deep``) consolidate the day's computational logs — distilling core
        principles, deleting useless logs, and fixing the principles into Deep Memory Synapses.
        """
        t0 = time.monotonic()
        report = DreamReport(deep_state=deep)
        try:
            # Pass 1: memory replay
            report.memory_replayed = self._memory_replay()

            # Pass 2: skill replay
            report.skills_replayed = self._skill_replay()

            # Pass 3: reasoning replay
            report.reasoning_replayed, insights = self._reasoning_replay()
            report.insights.extend(insights)

            # Pass 4: failure replay
            report.failures_reviewed, failure_insights = self._failure_replay()
            report.insights.extend(failure_insights)

            if deep:
                # Pass 5: data distillation — mine recurring principles from the day's logs
                principles = self._distill_principles()
                report.principles = principles
                report.principles_distilled = len(principles)

                # Pass 6: delete useless logs (only unprotected, low-salience, transient ones)
                report.logs_deleted = self._delete_useless_logs()

                # Pass 7: fix distilled principles into Deep Memory Synapses (protected, durable)
                report.synapses_fixed = self._fix_synapses(principles)

            self._sessions += 1
        except Exception:  # noqa: BLE001 — dreaming is best-effort
            pass

        report.duration_s = time.monotonic() - t0
        self._last_report = report
        return report

    def dream_state(self, *, deep: bool = True) -> DreamReport:
        """Enter a heavier "Dream State" for prolonged idleness — all replay + consolidation
        passes with wider top-k. Alias-style entry point used by idle maintenance."""
        saved = (self.top_k_memories, self.top_k_inferences)
        self.top_k_memories = max(self.top_k_memories, 20)
        self.top_k_inferences = max(self.top_k_inferences, 10)
        try:
            return self.dream(duration_s=30.0, deep=deep)
        finally:
            self.top_k_memories, self.top_k_inferences = saved

    @property
    def sessions_count(self) -> int:
        return self._sessions

    @property
    def last_report(self) -> Any:
        return self._last_report

    def deep_synapse_count(self) -> int:
        """How many Deep Memory Synapses are currently fixed in long-term memory."""
        if self.memory is None:
            return 0
        try:
            return sum(1 for r in self.memory._kv.values()
                       if self.deep_synapse_tag in r.tags)
        except Exception:  # noqa: BLE001
            return 0

    # ---------------------------------------------------------------------- #
    # Pass 1 — memory replay
    # ---------------------------------------------------------------------- #
    def _memory_replay(self) -> int:
        if self.consolidator is None:
            return 0
        try:
            replayed = self.consolidator.dream_replay(top_k=self.top_k_memories)
            return len(replayed)
        except Exception:  # noqa: BLE001
            return 0

    # ---------------------------------------------------------------------- #
    # Pass 2 — skill replay
    # ---------------------------------------------------------------------- #
    def _skill_replay(self) -> int:
        if self.skill_memory is None:
            return 0
        try:
            all_skills = self.skill_memory.all()
            # sort by usage count — most-used skills are most worth rehearsing
            top = sorted(all_skills, key=lambda s: getattr(s, "uses", 0), reverse=True)
            count = 0
            for skill in top[:self.top_k_skills]:
                try:
                    skill.reinforce()    # bump use count (training signal)
                    count += 1
                except Exception:  # noqa: BLE001
                    pass
            return count
        except Exception:  # noqa: BLE001
            return 0

    # ---------------------------------------------------------------------- #
    # Pass 3 — reasoning replay
    # ---------------------------------------------------------------------- #
    def _reasoning_replay(self):
        """Re-examine recent INFERENCE thoughts to reinforce good reasoning patterns."""
        replayed = 0
        insights: List[str] = []
        if self.mind is None:
            return replayed, insights
        try:
            from nyxara.observe.mindscope import ThoughtKind
            thoughts = self.mind.by_kind(ThoughtKind.INFERENCE)[-self.top_k_inferences * 3:]
            for t in thoughts[:self.top_k_inferences]:
                try:
                    # self-reflection: record a dream-tagged version back to MindScope
                    text = str(getattr(t, "content", "") or "")[:80]
                    if not text:
                        continue
                    self.mind.record(
                        ThoughtKind.INFERENCE,
                        f"[dream-replay] {text}",
                        salience=0.3, confidence=getattr(t, "confidence", 0.5) * 0.9)
                    replayed += 1
                    if replayed == 1:
                        insights.append(f"rehearsed: {text[:50]}")
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        return replayed, insights

    # ---------------------------------------------------------------------- #
    # Pass 4 — failure replay
    # ---------------------------------------------------------------------- #
    def _failure_replay(self):
        """Surface recent failed actions and self-critique them."""
        reviewed = 0
        insights: List[str] = []
        if self.journal is None or self.reflector is None:
            return reviewed, insights
        try:
            from nyxara.planning.journal import ActionStatus
            from nyxara.growth.reflect import Episode
            # get recent actions; find failures
            actions = list(self.journal.actions())[-50:]
            for entry in actions:
                status = self.journal.latest_status(entry.action_id)
                if status is not ActionStatus.FAILED:
                    continue
                payload = getattr(entry, "payload", {})
                action_text = str(payload.get("action", "unknown"))[:60]
                ep = Episode(action=action_text, success=False, reward=-0.5,
                             tags=["failure", "dream"])
                try:
                    critique = self.reflector.critique(ep)
                    suggestion = getattr(critique, "suggestion", "")
                    if suggestion:
                        insights.append(f"failure insight: {suggestion[:60]}")
                    reviewed += 1
                    if reviewed >= 5:
                        break
                except Exception:  # noqa: BLE001
                    reviewed += 1
        except Exception:  # noqa: BLE001
            pass
        return reviewed, insights

    # ---------------------------------------------------------------------- #
    # Pass 5 — data distillation (mine recurring principles from the day's logs)
    # ---------------------------------------------------------------------- #
    def _distill_principles(self) -> List[str]:
        """Cluster recurring content across the day's MindScope thoughts + journal actions into
        a small set of core principles. Deterministic and offline (token-frequency over phrases),
        so distillation works with no LLM."""
        from collections import Counter

        phrases: List[str] = []
        # MindScope inference/decision thoughts are where lessons live
        if self.mind is not None:
            try:
                from nyxara.observe.mindscope import ThoughtKind
                for kind in (ThoughtKind.INFERENCE, ThoughtKind.DECISION):
                    for t in self.mind.by_kind(kind):
                        c = str(getattr(t, "content", "") or "").strip()
                        if c and not c.startswith("[dream"):
                            phrases.append(c)
            except Exception:  # noqa: BLE001
                pass
        # journalled action verbs are recurring behavioural patterns
        if self.journal is not None:
            try:
                for entry in list(self.journal.actions())[-100:]:
                    a = str(getattr(entry, "payload", {}).get("action", "")).strip()
                    if a:
                        phrases.append(a)
            except Exception:  # noqa: BLE001
                pass

        if not phrases:
            return []
        # a "principle" is a normalised phrase that recurs at least `distill_min_support` times
        counts = Counter(p.lower()[:120] for p in phrases)
        principles = [f"Recurring pattern (x{n}): {p}"
                      for p, n in counts.most_common(8)
                      if n >= self.distill_min_support]
        return principles

    # ---------------------------------------------------------------------- #
    # Pass 6 — delete useless logs (unprotected, low-salience, transient only)
    # ---------------------------------------------------------------------- #
    def _delete_useless_logs(self) -> int:
        """Delete low-value computational logs: low-salience MindScope thoughts and transient,
        low-importance, *unprotected* SEMANTIC memories (dream-replay / idle scaffolding). Owner,
        high-importance and deep-synapse memories are never touched."""
        deleted = 0
        # prune low-salience thoughts from the cognitive trace
        if self.mind is not None:
            try:
                deleted += self.mind.prune(min_salience=0.25, keep_last=200)
            except Exception:  # noqa: BLE001
                pass
        # forget transient, low-importance, unprotected memory scaffolding
        if self.memory is not None:
            try:
                from nyxara.memory.consolidation import Consolidator
                cons = self.consolidator if isinstance(self.consolidator, Consolidator) \
                    else Consolidator(self.memory)
                transient = {"dream-replay", "idle", "scratch"}
                for rec in list(self.memory._kv.values()):
                    if cons.is_protected(rec):
                        continue
                    if (rec.tags & transient) and rec.importance < 0.3:
                        if self.memory.forget(rec.mem_id):
                            deleted += 1
            except Exception:  # noqa: BLE001
                pass
        return deleted

    # ---------------------------------------------------------------------- #
    # Pass 7 — fix distilled principles into Deep Memory Synapses (durable)
    # ---------------------------------------------------------------------- #
    def _fix_synapses(self, principles: List[str]) -> int:
        """Write distilled principles into long-term memory as high-importance, protected
        Deep Memory Synapses. Deduped (search-then-skip) so repeated dreams revise, not pile up."""
        if self.memory is None or not principles:
            return 0
        try:
            from nyxara.memory.provenance import Provenance, SourceType
            from nyxara.memory.store import MemoryType
        except Exception:  # noqa: BLE001
            return 0
        # existing synapse texts (avoid duplicates across nights)
        existing = set()
        try:
            for rec in self.memory._kv.values():
                if self.deep_synapse_tag in rec.tags:
                    existing.add(str(rec.content)[:120])
        except Exception:  # noqa: BLE001
            pass
        fixed = 0
        for principle in principles:
            text = f"[Deep Memory Synapse] {principle}"
            if text[:120] in existing:
                continue
            try:
                self.memory.remember(
                    text, mem_type=MemoryType.SEMANTIC,
                    provenance=Provenance(SourceType.SELF_REFLECTION, confidence=0.9),
                    importance=1.0, tags=[self.deep_synapse_tag, "principle"])
                existing.add(text[:120])
                fixed += 1
            except Exception:  # noqa: BLE001
                continue
        return fixed


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA dream-session self-test")
    print("=" * 70)

    # Run with no subsystems — all passes degrade gracefully
    ds = DreamSession()
    report = ds.dream(duration_s=1.0)
    print("\nno subsystems:")
    print(f"  memory_replayed    : {report.memory_replayed}")
    print(f"  skills_replayed    : {report.skills_replayed}")
    print(f"  reasoning_replayed : {report.reasoning_replayed}")
    print(f"  failures_reviewed  : {report.failures_reviewed}")
    print(f"  duration_s         : {report.duration_s:.3f}")
    print(f"  insights           : {report.insights}")
    assert isinstance(report.to_dict(), dict)
    assert ds.sessions_count == 1

    # Run with SkillMemory populated
    from nyxara.growth.skill_memory import SkillMemory
    sm = SkillMemory()
    sm.learn("rotate logs", "run logrotate", tools=["shell"])
    sm.learn("backup db", "pg_dump ...", tools=["shell"])

    ds2 = DreamSession(skill_memory=sm, top_k_skills=2)
    ds2._sessions = 0
    for a in sm.all():
        a.uses = 0  # reset
    report2 = ds2.dream()
    print("\nwith skill_memory:")
    print(f"  skills_replayed : {report2.skills_replayed}")
    assert report2.skills_replayed == 2
    # reinforce incremented uses
    assert all(s.uses >= 1 for s in sm.all())

    # Dream State: distillation + log deletion + Deep Memory Synapses
    from nyxara.observe.mindscope import MindScope, ThoughtKind
    from nyxara.memory.store import MemoryStore

    ms = MindScope()
    for _ in range(3):
        ms.record(ThoughtKind.INFERENCE, "rotating logs keeps the disk healthy", salience=0.6)
    for i in range(250):                       # noise that should be distilled away
        ms.record(ThoughtKind.PERCEPTION, f"tick {i}", salience=0.05)
    store = MemoryStore()

    ds3 = DreamSession(mind=ms, memory=store, distill_min_support=2)
    rep3 = ds3.dream_state(deep=True)
    print("\nDream State:")
    print(f"  principles_distilled : {rep3.principles_distilled}")
    print(f"  logs_deleted         : {rep3.logs_deleted}")
    print(f"  synapses_fixed       : {rep3.synapses_fixed}")
    assert rep3.deep_state
    assert rep3.principles_distilled >= 1, "a recurring principle should be distilled"
    assert rep3.synapses_fixed >= 1, "principles should be fixed into deep synapses"
    assert rep3.logs_deleted >= 1, "low-salience logs should be deleted"
    assert ds3.deep_synapse_count() == rep3.synapses_fixed

    # synapses survive forgetting; a second dream revises (no duplicate piling)
    from nyxara.memory.consolidation import Consolidator
    import time as _t
    Consolidator(store).forget_pass(now=_t.time() + 86400 * 3650)
    assert ds3.deep_synapse_count() == rep3.synapses_fixed, "synapses must survive forgetting"
    rep4 = ds3.dream_state(deep=True)
    assert rep4.synapses_fixed == 0, "already-known principles must not duplicate"

    print("\nALL SELF-TESTS PASSED ✓")
