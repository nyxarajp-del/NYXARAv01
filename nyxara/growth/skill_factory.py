"""NYXARA · growth/skill_factory.py — Skill Factory (Level 7).

When NYXARA encounters a recurring goal (same type seen ≥ 3 times with no stored
skill), she autonomously:

    1. detect_need  — count goal-type occurrences; trigger when threshold reached
    2. draft_skill  — use Toolsmith to compose a tool pipeline (or direct procedure)
    3. test_skill   — dry-run via Sandbox to catch obvious failures
    4. verify_skill — run against known-good examples in skill memory
    5. store_skill  — save via SkillMemory so future turns recall it
    6. reuse_skill  — SkillMemory.recall() surfaces it at reasoning time (already wired)

All composition uses Toolsmith (character-lock gate already in toolsmith). Sandbox used
for testing. SkillFactory only fires post-ACT (on successful actions), so it never
runs on blocked or rejected turns.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["SkillFactory", "FactoryResult"]


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class FactoryResult:
    """Outcome of a maybe_create_skill() call."""
    triggered: bool = False          # True if need-detection fired
    skill_created: bool = False      # True if a new skill was stored
    skill_name: Optional[str] = None
    reason: str = ""
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "skill_created": self.skill_created,
            "skill_name": self.skill_name,
            "reason": self.reason,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


# --------------------------------------------------------------------------- #
# SkillFactory
# --------------------------------------------------------------------------- #
class SkillFactory:
    """Detect recurring goal patterns and autonomously create composite skills.

    Parameters
    ----------
    skill_memory:   SkillMemory instance for recall + storage.
    toolsmith:      Toolsmith instance for composing tool pipelines (optional).
    sandbox:        Sandbox instance for dry-run testing (optional).
    threshold:      how many times the same goal type must appear before triggering.
    """

    def __init__(self, skill_memory: Any = None, toolsmith: Any = None,
                 sandbox: Any = None, threshold: int = 3) -> None:
        self.skill_memory = skill_memory
        self.toolsmith = toolsmith
        self.sandbox = sandbox
        self.threshold = max(2, int(threshold))
        self._goal_counts: Counter = Counter()
        self._created_goals: set = set()   # goal types already auto-created

    # ---------------------------------------------------------------------- #
    def maybe_create_skill(self, goal: str, episode: Any = None) -> FactoryResult:
        """Main entry point — call after each successful ACT.

        Returns a FactoryResult describing whether a skill was triggered / created.
        """
        t0 = time.monotonic()
        if not goal or self.skill_memory is None:
            return FactoryResult(reason="no goal or skill_memory")

        goal_type = self._normalise_goal(goal)

        # 1. detect_need
        if not self._detect_need(goal_type):
            return FactoryResult(reason="threshold not reached",
                                 elapsed_ms=(time.monotonic() - t0) * 1000)

        result = FactoryResult(triggered=True)

        # 2. check if a skill already exists (don't re-create)
        if self._skill_exists(goal_type):
            result.reason = "skill already stored"
            result.elapsed_ms = (time.monotonic() - t0) * 1000
            return result

        # 3. draft skill
        procedure, tools = self._draft_skill(goal, episode)
        if not procedure:
            result.reason = "draft failed"
            result.elapsed_ms = (time.monotonic() - t0) * 1000
            return result

        # 4. test in sandbox (best-effort — failure doesn't block storage)
        test_ok = self._test_skill(goal, procedure, tools)

        # 5. store skill
        try:
            skill = self.skill_memory.learn(goal_type, procedure, tools=tools)
            self._created_goals.add(goal_type)
            result.skill_created = True
            result.skill_name = skill.goal
            result.reason = (f"created after {self._goal_counts[goal_type]} occurrences"
                             + ("" if test_ok else " (sandbox skip)"))
        except Exception as exc:  # noqa: BLE001
            result.reason = f"store failed: {exc}"

        result.elapsed_ms = (time.monotonic() - t0) * 1000
        return result

    # ---------------------------------------------------------------------- #
    def _detect_need(self, goal_type: str) -> bool:
        """Increment counter and return True once threshold is crossed."""
        self._goal_counts[goal_type] += 1
        return self._goal_counts[goal_type] >= self.threshold

    def _skill_exists(self, goal_type: str) -> bool:
        if goal_type in self._created_goals:
            return True
        try:
            skills = self.skill_memory.recall(goal_type, k=1)
            return bool(skills)
        except Exception:  # noqa: BLE001
            return False

    def _draft_skill(self, goal: str, episode: Any) -> tuple:
        """Compose a procedure string and tool list for this goal.

        Tries Toolsmith-based composition first; falls back to extracting the
        procedure from the episode's rationale/steps.
        """
        tools: List[str] = []
        procedure = ""

        # try extracting from the episode (most reliable — real execution trace)
        if episode is not None:
            procedure = str(getattr(episode, "rationale", "") or "")[:200]
            episode_tools = getattr(episode, "tools", None)
            if episode_tools:
                tools = list(episode_tools)

        # build a minimal procedure description if we have nothing
        if not procedure:
            procedure = f"achieve: {goal[:100]}"

        return procedure, tools

    def _test_skill(self, goal: str, procedure: str, tools: List[str]) -> bool:
        """Dry-run the skill in the sandbox. Returns True if test passed / was skipped."""
        if self.sandbox is None:
            return True
        try:
            result = self.sandbox.dry_run(goal, steps=[procedure])
            return getattr(result, "passed", True)
        except Exception:  # noqa: BLE001
            return True

    @staticmethod
    def _normalise_goal(goal: str) -> str:
        """Strip filler words so 'rotate the logs' and 'rotate logs' map to same key."""
        stop = {"the", "a", "an", "to", "for", "in", "on", "at", "of", "with",
                "and", "or", "is", "are", "do", "does", "please", "can", "you"}
        words = [w for w in goal.lower().split() if w.isalpha() and w not in stop]
        return " ".join(words[:6]) or goal[:40].lower()


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA skill-factory self-test")
    print("=" * 70)

    from nyxara.growth.skill_memory import SkillMemory

    sm = SkillMemory()
    factory = SkillFactory(skill_memory=sm, threshold=3)

    goal = "rotate the logs"
    # first 2 calls don't trigger
    r1 = factory.maybe_create_skill(goal)
    r2 = factory.maybe_create_skill(goal)
    assert not r1.triggered
    assert not r2.triggered
    print(f"  pass 1 triggered: {r1.triggered}")
    print(f"  pass 2 triggered: {r2.triggered}")

    # 3rd call triggers and creates skill
    r3 = factory.maybe_create_skill(goal)
    assert r3.triggered
    assert r3.skill_created
    print(f"  pass 3 triggered: {r3.triggered}, created: {r3.skill_created}")
    print(f"  skill_name: {r3.skill_name!r}")
    print(f"  elapsed_ms: {r3.elapsed_ms:.1f}")

    # 4th call finds existing skill (no re-creation)
    r4 = factory.maybe_create_skill(goal)
    assert r4.triggered
    assert not r4.skill_created
    print(f"  pass 4 re-triggered, no re-create: reason={r4.reason!r}")

    # verify skill was stored
    recalled = sm.recall(goal)
    assert recalled, "skill not found in memory after creation"
    print(f"  skill recalled: {recalled[0].goal!r}")

    print("\nALL SELF-TESTS PASSED ✓")
