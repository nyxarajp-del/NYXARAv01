"""NYXARA · planning/grand_plan.py — deep, connected, ~1000-step planning (🗺, long horizon).

:class:`~nyxara.agency.mission.MissionExecutive` advances a goal that has *already* been broken
into milestones, but its own decomposition is shallow — a single numbered list, re-split only
when a step stalls. A genuinely large undertaking ("design, build and deploy a powered exosuit")
is not a flat list: it is a **tree** — phases (research → design → materials → manufacturing →
testing → redesign → optimization → deployment), each fanned out into stages, tasks and concrete
steps — and the phases are **connected**: manufacturing cannot start before materials and design
are done; testing follows manufacturing; redesign consumes test results; and so on.

This module builds that tree. :class:`GrandPlanner` recursively decomposes a goal into a
:class:`GrandPlan` of :class:`PlanNode`\\ s whose leaves number up to a target (≈1000), then wires
a **cross-phase dependency DAG** so the plan is one connected thing rather than a thousand
disconnected lines. The construction is acyclic *by invariant* — every dependency points only at
an earlier-emitted leaf — and :meth:`GrandPlan.topological_order` / :meth:`GrandPlan.layers`
expose the execution order and the parallelizable layers.

The decomposition has the same dual path the rest of NYXARA uses: a **deterministic offline
template** (the engineering-lifecycle skeleton, fanned out to hit the step budget) so the plan is
always reproducible without a model, plus an **optional reasoner-assisted** refinement that
rewrites the generic phase/stage/task labels into goal-specific ones when a ``core`` is supplied.
:meth:`GrandPlan.to_milestones` flattens the leaves into ordered
:class:`~nyxara.agency.mission.Milestone`\\ s — carrying their deps — so the whole plan executes
through the existing gated :class:`MissionExecutive`; autonomy buys horizon, never extra power.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "PlanLevel",
    "PlanNode",
    "GrandPlan",
    "GrandPlanner",
    "DEFAULT_PHASES",
]


# --------------------------------------------------------------------------- #
# Levels + the default engineering-lifecycle phase skeleton
# --------------------------------------------------------------------------- #
class PlanLevel(str, Enum):
    GOAL = "goal"
    PHASE = "phase"
    STAGE = "stage"
    TASK = "task"
    STEP = "step"


# Each phase: (kind, label, [upstream kinds it depends on]). The dependency map is what makes
# the plan *connected* — manufacturing waits on materials+design, testing on manufacturing, etc.
DEFAULT_PHASES: List[Tuple[str, str, List[str]]] = [
    ("research", "Research & first-principles analysis", []),
    ("design", "Design & specification", ["research"]),
    ("materials", "Materials & components sourcing", ["design"]),
    ("manufacturing", "Manufacturing & assembly", ["materials", "design"]),
    ("testing", "Testing & validation", ["manufacturing"]),
    ("redesign", "Redesign from test results", ["testing"]),
    ("optimization", "Optimization & tuning", ["redesign", "testing"]),
    ("deployment", "Deployment & operations", ["optimization"]),
]

_STAGE_VERBS = ["plan", "analyze", "build", "verify", "refine", "integrate", "review", "finalize"]
_TASK_VERBS = ["define", "draft", "implement", "measure", "iterate", "document", "harden", "ship"]


# --------------------------------------------------------------------------- #
# Plan tree
# --------------------------------------------------------------------------- #
@dataclass
class PlanNode:
    """One node of the plan tree (goal / phase / stage / task / step)."""

    id: str
    description: str
    level: PlanLevel
    kind: str = ""                                  # phase kind it belongs to
    children: List["PlanNode"] = field(default_factory=list)
    deps: List[str] = field(default_factory=list)   # ids of leaves this leaf depends on

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def leaves(self) -> List["PlanNode"]:
        if self.is_leaf:
            return [self]
        out: List[PlanNode] = []
        for c in self.children:
            out.extend(c.leaves())
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "description": self.description, "level": self.level.value,
                "kind": self.kind, "deps": list(self.deps),
                "children": [c.to_dict() for c in self.children]}


@dataclass
class GrandPlan:
    """A complete plan tree plus its flattened, dependency-wired leaves."""

    goal: str
    root: PlanNode
    steps: List[PlanNode] = field(default_factory=list)   # leaves, in emission order

    # ---- structure queries ---- #
    def leaf_count(self) -> int:
        return len(self.steps)

    def node_count(self) -> int:
        def count(n: PlanNode) -> int:
            return 1 + sum(count(c) for c in n.children)
        return count(self.root)

    def dag(self) -> Dict[str, List[str]]:
        """Adjacency: leaf id → the ids it depends on."""
        return {s.id: list(s.deps) for s in self.steps}

    def topological_order(self) -> List[str]:
        """A dependency-respecting order of leaf ids (raises on a cycle — should never happen)."""
        graph = self.dag()
        indeg = {nid: 0 for nid in graph}
        for nid, deps in graph.items():
            for d in deps:
                if d in indeg:
                    indeg[nid] += 1
        # Kahn's algorithm, preserving emission order among ready nodes
        order_index = {s.id: i for i, s in enumerate(self.steps)}
        ready = sorted((nid for nid, dg in indeg.items() if dg == 0), key=lambda n: order_index[n])
        out: List[str] = []
        dependents: Dict[str, List[str]] = {nid: [] for nid in graph}
        for nid, deps in graph.items():
            for d in deps:
                if d in dependents:
                    dependents[d].append(nid)
        while ready:
            nid = ready.pop(0)
            out.append(nid)
            for m in dependents[nid]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    # insert preserving emission order
                    idx = 0
                    while idx < len(ready) and order_index[ready[idx]] < order_index[m]:
                        idx += 1
                    ready.insert(idx, m)
        if len(out) != len(graph):
            raise ValueError("grand plan dependency graph has a cycle")
        return out

    def is_acyclic(self) -> bool:
        try:
            self.topological_order()
            return True
        except ValueError:
            return False

    def layers(self) -> List[List[str]]:
        """Parallelizable layers: each layer's leaves depend only on earlier layers."""
        graph = self.dag()
        depth: Dict[str, int] = {}
        for nid in self.topological_order():
            d = 0
            for dep in graph[nid]:
                d = max(d, depth.get(dep, 0) + 1)
            depth[nid] = d
        layers: Dict[int, List[str]] = {}
        order_index = {s.id: i for i, s in enumerate(self.steps)}
        for nid, d in depth.items():
            layers.setdefault(d, []).append(nid)
        return [sorted(layers[d], key=lambda n: order_index[n]) for d in sorted(layers)]

    def to_milestones(self) -> List[Any]:
        """Flatten leaves into ordered :class:`~nyxara.agency.mission.Milestone`\\ s with deps."""
        from nyxara.agency.mission import Milestone, MilestoneStatus
        order = self.topological_order()
        rank = {nid: i for i, nid in enumerate(order)}
        ordered_leaves = sorted(self.steps, key=lambda s: rank[s.id])
        return [Milestone(id=s.id, description=s.description, status=MilestoneStatus.PENDING,
                          deps=list(s.deps)) for s in ordered_leaves]

    def summary(self, max_lines: int = 40) -> str:
        lines = [f"GRAND PLAN — {self.goal}",
                 f"  nodes={self.node_count()} leaves={self.leaf_count()} "
                 f"layers={len(self.layers())} acyclic={self.is_acyclic()}"]
        shown = 0
        for phase in self.root.children:
            lines.append(f"  ▸ {phase.description} ({len(phase.leaves())} steps)")
            shown += 1
            if shown >= max_lines:
                break
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"goal": self.goal, "root": self.root.to_dict(),
                "leaf_count": self.leaf_count(), "node_count": self.node_count(),
                "layers": len(self.layers()), "acyclic": self.is_acyclic()}


# --------------------------------------------------------------------------- #
# The planner
# --------------------------------------------------------------------------- #
class GrandPlanner:
    """Recursively decompose a goal into a connected ~N-step :class:`GrandPlan`."""

    def __init__(self, core: Any = None) -> None:
        self.core = core   # optional NyxaraCore for reasoner-assisted refinement

    def decompose(self, goal: str, *, target_steps: int = 1000, max_depth: int = 4,
                  phases: Optional[List[Tuple[str, str, List[str]]]] = None) -> GrandPlan:
        """Build the plan tree, fan it out to ≈ ``target_steps`` leaves, then wire the DAG."""
        goal = (goal or "").strip() or "unspecified goal"
        target_steps = max(1, min(int(target_steps), 20_000))
        max_depth = max(2, min(int(max_depth), 5))
        phases = phases or DEFAULT_PHASES

        # branch factor: leaves ≈ n_phases · b^(branch_levels), solve for b
        branch_levels = max_depth - 1                       # stage, task, step, …
        n_phases = len(phases)
        b = max(1, round((target_steps / n_phases) ** (1.0 / max(1, branch_levels))))

        root = PlanNode(id="goal", description=goal, level=PlanLevel.GOAL)
        phase_kind_to_node: Dict[str, PlanNode] = {}
        for pi, (kind, label, _deps) in enumerate(phases):
            phase = PlanNode(id=f"p{pi}", description=f"{label} — {goal}",
                             level=PlanLevel.PHASE, kind=kind)
            self._expand(phase, level_idx=1, max_depth=max_depth, branch=b, goal=goal, kind=kind)
            root.children.append(phase)
            phase_kind_to_node[kind] = phase

        plan = GrandPlan(goal=goal, root=root, steps=root.leaves())
        self._wire_dependencies(plan, phases, phase_kind_to_node)
        if self.core is not None:
            self._refine_with_reasoner(plan, goal)
        return plan

    # ---- recursive fan-out ---- #
    def _expand(self, node: PlanNode, *, level_idx: int, max_depth: int, branch: int,
                goal: str, kind: str) -> None:
        if level_idx >= max_depth:
            return
        level = [PlanLevel.GOAL, PlanLevel.PHASE, PlanLevel.STAGE, PlanLevel.TASK,
                 PlanLevel.STEP][min(level_idx + 1, 4)]
        prefix = {PlanLevel.STAGE: "s", PlanLevel.TASK: "t", PlanLevel.STEP: "k"}[level]
        verbs = _STAGE_VERBS if level is PlanLevel.STAGE else _TASK_VERBS
        for i in range(branch):
            verb = verbs[i % len(verbs)]
            child = PlanNode(
                id=f"{node.id}.{prefix}{i}",
                description=f"{verb.capitalize()} {kind} ({level.value} {i + 1}) for {goal}"[:200],
                level=level, kind=kind)
            node.children.append(child)
            self._expand(child, level_idx=level_idx + 1, max_depth=max_depth, branch=branch,
                         goal=goal, kind=kind)

    # ---- dependency wiring (acyclic by the earlier-emitted invariant) ---- #
    def _wire_dependencies(self, plan: GrandPlan,
                           phases: List[Tuple[str, str, List[str]]],
                           phase_nodes: Dict[str, PlanNode]) -> None:
        # group leaves by phase, preserving emission order
        leaves_by_phase: Dict[str, List[PlanNode]] = {}
        for leaf in plan.steps:
            leaves_by_phase.setdefault(leaf.kind, []).append(leaf)

        phase_exit: Dict[str, str] = {}   # last leaf id of each phase
        for kind, _label, upstream in phases:
            leaves = leaves_by_phase.get(kind, [])
            if not leaves:
                continue
            # cross-phase entry deps: the first leaf of this phase waits on each upstream phase's exit
            entry_deps = [phase_exit[u] for u in upstream if u in phase_exit]
            # chain stages within the phase, run tasks within a stage in parallel
            prev_stage_exit: Optional[str] = None
            cur_stage: Optional[str] = None
            stage_leaves: List[PlanNode] = []

            def flush_stage(prev: Optional[str], group: List[PlanNode]) -> Optional[str]:
                # group = all leaves of one stage; chain steps within each task, tasks parallel
                by_task: Dict[str, List[PlanNode]] = {}
                for lf in group:
                    task_id = lf.id.rsplit(".", 1)[0]
                    by_task.setdefault(task_id, []).append(lf)
                stage_entry = [prev] if prev else list(entry_deps)
                last_id = group[-1].id if group else prev
                for _tid, steps in by_task.items():
                    for k, lf in enumerate(steps):
                        if k == 0:
                            lf.deps = [d for d in stage_entry if d]
                        else:
                            lf.deps = [steps[k - 1].id]
                return last_id

            for leaf in leaves:
                stage_id = leaf.id.split(".")[1] if "." in leaf.id else leaf.id
                if cur_stage is None:
                    cur_stage = stage_id
                if stage_id != cur_stage:
                    prev_stage_exit = flush_stage(prev_stage_exit, stage_leaves)
                    stage_leaves = []
                    cur_stage = stage_id
                stage_leaves.append(leaf)
            if stage_leaves:
                prev_stage_exit = flush_stage(prev_stage_exit, stage_leaves)
            phase_exit[kind] = leaves[-1].id

    # ---- optional reasoner refinement ---- #
    def _refine_with_reasoner(self, plan: GrandPlan, goal: str) -> None:
        """Rewrite the generic phase labels into goal-specific ones when a core is available."""
        prompt = (
            f"For the GOAL: {goal}\nList exactly {len(plan.root.children)} ordered top-level "
            f"phases (one per line, '1.', '2.', …), each a short concrete phase name. Output only "
            f"the list.")
        try:
            result = self.core.process(prompt, authority=_owner_authority())
            text = getattr(result, "response", "") or ""
        except Exception:  # noqa: BLE001 — refinement is advisory; the template plan stands
            return
        names = [m.group(1).strip() for line in text.splitlines()
                 if (m := re.match(r"^(?:\d+[.)]|[-*•])\s+(.*)$", line.strip()))]
        for phase, name in zip(plan.root.children, names):
            if name:
                phase.description = f"{name} — {goal}"[:200]


def _owner_authority() -> Any:
    from nyxara.agency.permissions import Authority
    return Authority.OWNER
