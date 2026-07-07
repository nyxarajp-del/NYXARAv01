"""The reflection→goal feedback loop is live (capability 4 — improve from mistakes).

`CycleReflector._push_improvement_goals` used to call a non-existent `GoalSystem.set(...)`; the
`AttributeError` was swallowed by a bare `except`, so improvement lessons silently never became
goals. These tests prove the loop now closes: a failing-episode reflection creates real,
owner-aligned, low-priority goals via the actual `GoalSystem` API.
"""

from __future__ import annotations

from nyxara.growth.cycle_reflect import CycleReflector
from nyxara.growth.reflect import Episode, Reflector
from nyxara.planning.goals import GoalSystem


def _reflector_with_failures() -> Reflector:
    r = Reflector()
    r.ingest([Episode("write_file", success=False, reward=-0.5, tags=["write"]) for _ in range(3)]
             + [Episode("search_web", success=True, reward=1.0, tags=["search"]) for _ in range(3)])
    return r


def test_reflection_creates_real_goals():
    goals = GoalSystem()
    assert len(goals.goals()) == 0
    cr = CycleReflector(reflector=_reflector_with_failures(), goals=goals)
    report = cr.reflect("daily")
    assert report.what_to_improve                       # there is something to improve
    created = goals.goals()
    assert created, "improvement lessons must become real goals"
    for g in created:
        assert g.priority == 0.2                        # low-priority, as documented
        assert g.source == "reflection"
        assert goals.is_owner_aligned(g)                # never points against the Master (Rule 1)


def test_reflection_goal_loop_does_not_raise_or_swallow():
    # a real GoalSystem has no `.set()`; the fix must not depend on it and must not silently die.
    goals = GoalSystem()
    cr = CycleReflector(reflector=_reflector_with_failures(), goals=goals)
    assert not hasattr(goals, "set")
    cr.reflect("weekly")
    assert goals.goals()                                # the loop produced goals, didn't no-op


def test_reflection_without_goalsystem_is_safe():
    cr = CycleReflector(reflector=_reflector_with_failures(), goals=None)
    report = cr.reflect("monthly")                      # must not raise when no goal system
    assert report.cycle == "monthly"
