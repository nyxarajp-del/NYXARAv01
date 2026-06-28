"""P2-A — grand-plan leaves are goal-specific, not decorative templates.

The DAG and topological order were always real, but every leaf used to read the generic
"Implement materials (step 3) for <goal>". Phase labels are now refined *before* fan-out, so a
reasoner-refined, goal-specific phase name propagates into every descendant step; offline, leaves
still carry the concrete phase label rather than a bare template (capabilities #6, #58).
"""

from __future__ import annotations

from nyxara.planning.grand_plan import GrandPlanner


class _FakeResult:
    def __init__(self, response: str) -> None:
        self.response = response


class _FakeCore:
    """A stand-in core whose reasoner returns concrete, goal-specific phase names."""

    def __init__(self, names):
        self._names = names

    def process(self, prompt, *, authority=None):       # noqa: ARG002
        return _FakeResult("\n".join(f"{i + 1}. {n}" for i, n in enumerate(self._names)))


def test_offline_leaves_reference_goal_and_phase():
    plan = GrandPlanner().decompose("build a powered exosuit", target_steps=200, max_depth=4)
    # every leaf is goal-specific (not a bare "step N") and names its phase
    assert all("exosuit" in s.description for s in plan.steps)
    assert plan.is_acyclic()
    assert len(plan.topological_order()) == plan.leaf_count()


def test_reasoner_refined_phase_names_propagate_into_leaves():
    names = ["Survey biomechanics", "CAD the frame", "Source alloys", "Forge actuators",
             "Bench test", "Field trial", "Tune servos", "Deploy to pilot"]
    core = _FakeCore(names)
    plan = GrandPlanner(core=core).decompose("build a powered exosuit",
                                             target_steps=200, max_depth=4)
    # the refined names reached the *leaves*, not just the top-level phase row
    leaf_text = " ".join(s.description for s in plan.steps)
    assert "CAD the frame" in leaf_text
    assert "Forge actuators" in leaf_text


def test_dependencies_still_respected_after_specialization():
    plan = GrandPlanner().decompose("build a robot", target_steps=200)
    rank = {sid: i for i, sid in enumerate(plan.topological_order())}
    # every dependency is emitted before the leaf that depends on it
    for leaf in plan.steps:
        for dep in leaf.deps:
            assert rank[dep] < rank[leaf.id]
