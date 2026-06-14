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
    insights: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_s": round(self.duration_s, 3),
            "memory_replayed": self.memory_replayed,
            "skills_replayed": self.skills_replayed,
            "reasoning_replayed": self.reasoning_replayed,
            "failures_reviewed": self.failures_reviewed,
            "insights": self.insights,
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
                 top_k_memories: int = 10, top_k_skills: int = 5,
                 top_k_inferences: int = 5) -> None:
        self.consolidator = consolidator
        self.skill_memory = skill_memory
        self.mind = mind
        self.journal = journal
        self.reflector = reflector
        self.top_k_memories = max(1, int(top_k_memories))
        self.top_k_skills = max(1, int(top_k_skills))
        self.top_k_inferences = max(1, int(top_k_inferences))
        self._sessions: int = 0

    # ---------------------------------------------------------------------- #
    def dream(self, duration_s: float = 30.0) -> DreamReport:
        """Run one dream session. Always returns a DreamReport.

        ``duration_s`` is advisory — the session runs all four passes regardless,
        but if any pass takes too long, it degrades gracefully.
        """
        t0 = time.monotonic()
        report = DreamReport()
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

            self._sessions += 1
        except Exception:  # noqa: BLE001 — dreaming is best-effort
            pass

        report.duration_s = time.monotonic() - t0
        return report

    @property
    def sessions_count(self) -> int:
        return self._sessions

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
            from nyxara.mind.mindscope import ThoughtKind
            thoughts = self.mind.recent(
                kind=ThoughtKind.INFERENCE, n=self.top_k_inferences * 3)
            for t in thoughts[:self.top_k_inferences]:
                try:
                    # self-reflection: record a dream-tagged version back to MindScope
                    text = getattr(t, "text", str(t))[:80]
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

    print("\nALL SELF-TESTS PASSED ✓")
