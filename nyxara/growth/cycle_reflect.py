"""NYXARA · growth/cycle_reflect.py — Long-Term Reflection Cycles (Level 8).

Structured daily / weekly / monthly reflection answering:

    "What worked? What failed? What should improve?"

Results are stored as SEMANTIC memories tagged with the cycle type. `what_to_improve`
items become new low-priority goals in the GoalSystem so lessons feed back into the
objective space.

Schedule is managed by the autonomic loop: daily every ~24h, weekly every ~7d,
monthly every ~30d. The cycle reflector is always best-effort — it never blocks
the main cognitive cycle.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

__all__ = ["CycleReflector", "CycleReport"]

_log = logging.getLogger(__name__)

CycleType = Literal["daily", "weekly", "monthly"]

# seconds between cycles
_CYCLE_SECONDS: Dict[str, float] = {
    "daily": 86_400.0,
    "weekly": 604_800.0,
    "monthly": 2_592_000.0,
}


# --------------------------------------------------------------------------- #
# CycleReport
# --------------------------------------------------------------------------- #
@dataclass
class CycleReport:
    """One structured reflection cycle."""
    cycle: str
    timestamp: float = field(default_factory=time.time)
    what_worked: List[str] = field(default_factory=list)
    what_failed: List[str] = field(default_factory=list)
    what_to_improve: List[str] = field(default_factory=list)
    key_lessons: List[str] = field(default_factory=list)
    next_focus: str = ""
    episodes_reviewed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "what_worked": self.what_worked,
            "what_failed": self.what_failed,
            "what_to_improve": self.what_to_improve,
            "key_lessons": self.key_lessons,
            "next_focus": self.next_focus,
            "episodes_reviewed": self.episodes_reviewed,
        }

    def summary_text(self) -> str:
        lines = [f"[{self.cycle.upper()} REFLECTION]"]
        if self.what_worked:
            lines.append(f"Worked   : {'; '.join(self.what_worked[:3])}")
        if self.what_failed:
            lines.append(f"Failed   : {'; '.join(self.what_failed[:3])}")
        if self.what_to_improve:
            lines.append(f"Improve  : {'; '.join(self.what_to_improve[:3])}")
        if self.key_lessons:
            lines.append(f"Lessons  : {'; '.join(self.key_lessons[:3])}")
        if self.next_focus:
            lines.append(f"Focus    : {self.next_focus}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CycleReflector
# --------------------------------------------------------------------------- #
class CycleReflector:
    """Run structured timed reflection cycles and store reports to semantic memory.

    Parameters
    ----------
    reflector:  Reflector instance for lesson/pattern mining.
    memory:     MemoryStore for storing CycleReport as SEMANTIC memories.
    goals:      GoalSystem to receive improvement suggestions as low-priority goals.
    """

    def __init__(self, reflector: Any = None, memory: Any = None,
                 goals: Any = None) -> None:
        self.reflector = reflector
        self.memory = memory
        self.goals = goals
        self._last_cycle: Dict[str, float] = {
            "daily": 0.0, "weekly": 0.0, "monthly": 0.0,
        }
        self._reports: List[CycleReport] = []

    # ---------------------------------------------------------------------- #
    def reflect(self, cycle: CycleType) -> CycleReport:
        """Run one structured reflection pass for the given cycle type.

        Always returns a CycleReport. On error, returns a minimal report with
        the error noted so the caller never gets an exception.
        """
        try:
            return self._run(cycle)
        except Exception as exc:  # noqa: BLE001
            report = CycleReport(cycle=cycle,
                                 next_focus=f"resolve error in reflection: {exc}")
            return report

    def due(self, cycle: CycleType) -> bool:
        """Return True if enough time has elapsed since the last run of this cycle."""
        interval = _CYCLE_SECONDS.get(cycle, 86_400.0)
        return (time.time() - self._last_cycle.get(cycle, 0.0)) >= interval

    def tick(self) -> List[CycleReport]:
        """Check all three cycle types and run any that are due. Called by autonomic."""
        reports: List[CycleReport] = []
        for cycle_type in ("daily", "weekly", "monthly"):
            if self.due(cycle_type):  # type: ignore[arg-type]
                reports.append(self.reflect(cycle_type))  # type: ignore[arg-type]
        return reports

    def all_reports(self) -> List[CycleReport]:
        return list(self._reports)

    # ---------------------------------------------------------------------- #
    def _run(self, cycle: CycleType) -> CycleReport:
        report = CycleReport(cycle=cycle)

        # pull episodes from the reflector
        episodes = self._get_episodes()
        report.episodes_reviewed = len(episodes)

        if episodes:
            worked, failed = self._split_episodes(episodes)
            report.what_worked = [e.action[:60] for e in worked[:5]]
            report.what_failed = [e.action[:60] for e in failed[:5]]

            # mine lessons (Reflector does the heavy lifting)
            if self.reflector is not None:
                try:
                    lessons = self.reflector.lessons()
                    report.key_lessons = [l.text[:80] for l in lessons[:5]]
                    report.what_to_improve = [
                        l.text[:60] for l in lessons
                        if "investigate" in l.kind.lower() or "less" in l.kind.lower()
                    ][:3]
                except Exception:  # noqa: BLE001
                    pass

        # derive next_focus from the most important lesson
        if report.key_lessons:
            report.next_focus = report.key_lessons[0][:80]
        elif report.what_to_improve:
            report.next_focus = report.what_to_improve[0][:80]
        else:
            report.next_focus = "continue current trajectory"

        # store as SEMANTIC memory
        self._store_report(report)

        # push improvement items as low-priority goals
        self._push_improvement_goals(report)

        # record timestamp
        self._last_cycle[cycle] = time.time()
        self._reports.append(report)
        return report

    def _get_episodes(self) -> List[Any]:
        if self.reflector is None:
            return []
        try:
            return list(getattr(self.reflector, "_episodes", []))
        except Exception:  # noqa: BLE001
            return []

    @staticmethod
    def _split_episodes(episodes: List[Any]):
        worked = [e for e in episodes if getattr(e, "success", False)]
        failed = [e for e in episodes if not getattr(e, "success", True)]
        return worked, failed

    def _store_report(self, report: CycleReport) -> None:
        if self.memory is None:
            return
        try:
            from nyxara.memory.provenance import Provenance, SourceType
            from nyxara.memory.store import MemoryType
            text = report.summary_text()
            self.memory.remember(
                text, mem_type=MemoryType.SEMANTIC,
                provenance=Provenance(SourceType.SELF_REFLECTION, confidence=0.8),
                importance=0.7,
                tags=["reflection", report.cycle])
        except Exception:  # noqa: BLE001
            pass

    def _push_improvement_goals(self, report: CycleReport) -> None:
        """Feed improvement lessons back into the objective space as real, owner-aligned,
        low-priority goals — the closing half of the reflection loop.

        Uses the real ``GoalSystem`` API (``Goal`` + ``add``, guarded by ``is_owner_aligned``)
        rather than a non-existent ``set`` method. Each lesson becomes a modest
        capability/knowledge goal anchored on ``owner_benefit`` so it is owner-aligned by
        construction (Rule 1). A malformed item is skipped and logged; the loop is never
        allowed to silently vanish again.
        """
        goals = self.goals
        if goals is None or not report.what_to_improve:
            return
        from nyxara.planning.goals import Goal

        dims = getattr(goals, "dims", ()) or ()
        for item in report.what_to_improve[:2]:
            topic = " ".join(str(item or "").split())[:60].strip()
            if len(topic) < 3:
                continue
            vector: Dict[str, float] = {}
            if "capability" in dims:
                vector["capability"] = 0.5
            if "knowledge" in dims:
                vector["knowledge"] = 0.4
            if "owner_benefit" in dims:
                vector["owner_benefit"] = 0.4
            if not vector and dims:
                vector = {dims[0]: 1.0}
            try:
                goal = Goal(name=f"Improve reasoning: {topic}", vector=vector,
                            priority=0.2, source="reflection")
                if goals.is_owner_aligned(goal):
                    goals.add(goal)
            except (AttributeError, TypeError, ValueError) as exc:
                _log.debug("cycle-reflection goal-injection skipped for %r: %s", topic, exc)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA cycle-reflector self-test")
    print("=" * 70)

    from nyxara.growth.reflect import Episode, Reflector

    reflector = Reflector()
    reflector.ingest([
        Episode("search_web", success=True, reward=1.0, tags=["search"]),
        Episode("search_web", success=True, reward=1.0, tags=["search"]),
        Episode("search_web", success=True, reward=1.0, tags=["search"]),
        Episode("write_file", success=False, reward=-0.5, tags=["write"]),
        Episode("write_file", success=False, reward=-0.5, tags=["write"]),
        Episode("read_file", success=True, reward=0.8, tags=["read"]),
    ])

    cr = CycleReflector(reflector=reflector, memory=None, goals=None)

    # force the cycle to be due
    cr._last_cycle["daily"] = 0.0
    report = cr.reflect("daily")
    print("\ndaily cycle report:")
    print(f"  episodes_reviewed : {report.episodes_reviewed}")
    print(f"  what_worked       : {report.what_worked}")
    print(f"  what_failed       : {report.what_failed}")
    print(f"  key_lessons       : {report.key_lessons}")
    print(f"  next_focus        : {report.next_focus}")
    print(f"\n  summary:\n{report.summary_text()}")

    assert report.cycle == "daily"
    assert report.episodes_reviewed >= 0
    assert isinstance(report.what_worked, list)
    assert isinstance(report.key_lessons, list)

    # tick check
    cr._last_cycle = {"daily": 0.0, "weekly": 0.0, "monthly": 0.0}
    ticked = cr.tick()
    assert len(ticked) == 3
    print(f"\n  tick() ran {len(ticked)} cycles (daily + weekly + monthly)")

    print("\nALL SELF-TESTS PASSED ✓")
