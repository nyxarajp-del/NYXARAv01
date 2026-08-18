"""NYXARA · njp/goals.py — what she is trying to do, and what it is worth (🎯, NJP V.02).

A flat list of goals is a to-do list. What makes a goal system cognitive is that it carries the
**structure** (this task exists to serve that goal, which serves that mission) and the
**economics** (what this is worth, what it costs, what it risks, and whether it can be undone).
Without the structure she cannot tell that finishing a task advanced a mission; without the
economics she cannot choose between two plans that both sound good.

    Mission → Goal → Subgoal → Task

Each node carries ``priority``, ``progress``, ``deadline``, ``confidence``, ``dependency`` — and,
because a plan is a bet rather than an intention, ``expected_value``, ``cost``, ``risk`` and
``reversibility``.

**Progress rolls up, it is not reported.** A parent's progress is computed from its children, so
"the mission is 60% done" is arithmetic over real completed tasks rather than a number someone
set. A parent with no children has only its own progress, and says so.

**Plans are compared, not picked.** :meth:`choose` scores competing plans on expected value net of
cost and risk, with **irreversibility penalised specifically** rather than folded into risk. Those
are genuinely different things: a risky move you can undo is an experiment, and a safe-looking one
you cannot undo deserves more scrutiny than its risk number suggests.

**Blocked is a real state.** A task whose dependencies are unmet is not "not started" — it is
waiting on something, and conflating the two hides the actual reason nothing is moving.

Owner alignment stays where it already is: :class:`~nyxara.planning.goals.GoalSystem` owns that
envelope, and this defers to it rather than growing a second opinion about what the Master wants.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["Kind", "State", "Node", "Plan", "GoalTree"]


class Kind:
    """Where a node sits in the hierarchy."""

    MISSION = "mission"
    GOAL = "goal"
    SUBGOAL = "subgoal"
    TASK = "task"

    ALL = (MISSION, GOAL, SUBGOAL, TASK)

    @staticmethod
    def child_of(kind: str) -> str:
        try:
            return Kind.ALL[min(Kind.ALL.index(kind) + 1, len(Kind.ALL) - 1)]
        except ValueError:
            return Kind.TASK


class State:
    """What is actually happening to a node."""

    OPEN = "open"
    BLOCKED = "blocked"        # dependencies unmet — waiting, not idle
    ACTIVE = "active"
    DONE = "done"
    ABANDONED = "abandoned"


@dataclass
class Node:
    """One thing she is trying to do, with its structure and its economics."""

    nid: str = ""
    name: str = ""
    kind: str = Kind.TASK
    parent: str = ""
    children: List[str] = field(default_factory=list)

    priority: float = 0.5
    own_progress: float = 0.0          # progress of THIS node, ignoring children
    deadline: Optional[float] = None
    confidence: float = 0.5            # how likely she is to actually achieve it
    dependencies: List[str] = field(default_factory=list)

    expected_value: float = 0.5
    cost: float = 0.2
    risk: float = 0.2
    reversibility: float = 1.0         # 1.0 fully undoable, 0.0 irreversible

    state: str = State.OPEN
    created: float = field(default_factory=time.time)

    @property
    def overdue(self) -> bool:
        return bool(self.deadline is not None and time.time() > self.deadline
                    and self.state != State.DONE)

    @property
    def net_value(self) -> float:
        """What this is worth once cost and risk are paid, and irreversibility is charged for.

        Irreversibility is charged **separately** from risk on purpose. A risky move you can undo
        is an experiment; a safe-looking one you cannot undo deserves more scrutiny than its risk
        number alone implies, and folding the two together loses exactly that distinction.
        """
        gross = self.expected_value * self.confidence
        return gross - self.cost - (self.risk * (2.0 - self.reversibility))

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.nid, "name": self.name, "kind": self.kind, "state": self.state,
                "parent": self.parent, "children": len(self.children),
                "priority": round(self.priority, 4), "confidence": round(self.confidence, 4),
                "expected_value": round(self.expected_value, 4), "cost": round(self.cost, 4),
                "risk": round(self.risk, 4), "reversibility": round(self.reversibility, 4),
                "net_value": round(self.net_value, 4), "overdue": self.overdue,
                "dependencies": list(self.dependencies)}


@dataclass
class Plan:
    """One way of achieving something, priced so it can be compared with another."""

    name: str = ""
    steps: List[str] = field(default_factory=list)
    expected_value: float = 0.5
    cost: float = 0.2
    risk: float = 0.2
    reversibility: float = 1.0
    confidence: float = 0.5
    simulated: bool = False            # was the outcome actually rolled out, or just asserted?

    @property
    def score(self) -> float:
        gross = self.expected_value * self.confidence
        return gross - self.cost - (self.risk * (2.0 - self.reversibility))

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "steps": self.steps[:8], "score": round(self.score, 4),
                "expected_value": round(self.expected_value, 4), "cost": round(self.cost, 4),
                "risk": round(self.risk, 4), "reversibility": round(self.reversibility, 4),
                "confidence": round(self.confidence, 4), "simulated": self.simulated}


class GoalTree:
    """The hierarchy, with progress that rolls up and plans that are compared."""

    def __init__(self, *, system: Any = None, capacity: int = 512) -> None:
        # The repo's GoalSystem still owns owner-alignment. Deferring to it rather than growing a
        # second opinion about what the Master wants is the whole reason it is here.
        self.system = system if system is not None else self._build_system()
        self.capacity = max(16, int(capacity))
        self.nodes: Dict[str, Node] = {}
        self._seq = 0
        self.completed = 0
        self.abandoned = 0

    @staticmethod
    def _build_system() -> Any:
        try:
            from nyxara.planning.goals import GoalSystem
            return GoalSystem()
        except Exception:  # noqa: BLE001
            return None

    # ---- building the tree --------------------------------------------------- #
    def add(self, name: str, *, kind: str = "", parent: str = "",
            priority: float = 0.5, expected_value: float = 0.5, cost: float = 0.2,
            risk: float = 0.2, reversibility: float = 1.0, confidence: float = 0.5,
            deadline: Optional[float] = None,
            dependencies: Sequence[str] = ()) -> Optional[Node]:
        """Add a node. Its kind is inferred from its parent unless given."""
        try:
            name = str(name).strip()
            if not name:
                return None
            self._seq += 1
            nid = f"n{self._seq}"
            parent_node = self.nodes.get(parent) if parent else None
            if not kind:
                kind = Kind.child_of(parent_node.kind) if parent_node else Kind.GOAL
            node = Node(nid=nid, name=name, kind=kind,
                        parent=parent_node.nid if parent_node else "",
                        priority=float(priority), expected_value=float(expected_value),
                        cost=float(cost), risk=float(risk),
                        reversibility=float(reversibility), confidence=float(confidence),
                        deadline=deadline, dependencies=list(dependencies))
            self.nodes[nid] = node
            if parent_node is not None:
                parent_node.children.append(nid)
            self._refresh_blocked(node)
            self._evict()
            return node
        except Exception:  # noqa: BLE001
            return None

    def mission(self, name: str, **kw: Any) -> Optional[Node]:
        return self.add(name, kind=Kind.MISSION, **kw)

    # ---- progress that rolls up ----------------------------------------------- #
    def progress(self, nid: str) -> float:
        """Progress of a node, computed from its children where it has any.

        "The mission is 60% done" is arithmetic over real completed tasks, not a number someone
        set on the mission. A leaf has only its own progress, which is the honest floor.
        """
        try:
            node = self.nodes.get(str(nid))
            if node is None:
                return 0.0
            if node.state == State.DONE:
                return 1.0
            children = [c for c in node.children if c in self.nodes]
            if not children:
                return max(0.0, min(1.0, node.own_progress))
            # Weighted by priority: finishing the important child moves the parent more.
            total = sum(max(0.01, self.nodes[c].priority) for c in children)
            return sum(self.progress(c) * max(0.01, self.nodes[c].priority)
                       for c in children) / total
        except Exception:  # noqa: BLE001
            return 0.0

    def advance(self, nid: str, progress: float) -> Optional[Node]:
        """Move a leaf's own progress. Completing it re-checks whatever was waiting on it."""
        try:
            node = self.nodes.get(str(nid))
            if node is None:
                return None
            node.own_progress = max(0.0, min(1.0, float(progress)))
            if node.own_progress >= 1.0 and node.state != State.DONE:
                return self.complete(nid)
            if node.state == State.OPEN and node.own_progress > 0.0:
                node.state = State.ACTIVE
            return node
        except Exception:  # noqa: BLE001
            return None

    def complete(self, nid: str) -> Optional[Node]:
        """Mark done, and unblock anything that was waiting on it."""
        try:
            node = self.nodes.get(str(nid))
            if node is None:
                return None
            node.state = State.DONE
            node.own_progress = 1.0
            self.completed += 1
            for other in self.nodes.values():
                if nid in other.dependencies or node.name in other.dependencies:
                    self._refresh_blocked(other)
            return node
        except Exception:  # noqa: BLE001
            return None

    def abandon(self, nid: str, why: str = "") -> Optional[Node]:
        node = self.nodes.get(str(nid))
        if node is not None:
            node.state = State.ABANDONED
            self.abandoned += 1
        return node

    def _refresh_blocked(self, node: Node) -> None:
        """A node whose dependencies are unmet is *blocked*, which is not the same as not started."""
        try:
            if node.state in (State.DONE, State.ABANDONED):
                return
            unmet = [d for d in node.dependencies if not self._satisfied(d)]
            node.state = State.BLOCKED if unmet else (
                State.ACTIVE if node.own_progress > 0.0 else State.OPEN)
        except Exception:  # noqa: BLE001
            pass

    def _satisfied(self, dependency: str) -> bool:
        node = self.nodes.get(dependency)
        if node is None:
            node = next((n for n in self.nodes.values() if n.name == dependency), None)
        return bool(node is not None and node.state == State.DONE)

    # ---- choosing -------------------------------------------------------------- #
    def ready(self) -> List[Node]:
        """Everything actionable right now: not done, not blocked, dependencies met."""
        return [n for n in self.nodes.values()
                if n.state in (State.OPEN, State.ACTIVE)]

    def top(self) -> Optional[Node]:
        """The single most worthwhile thing to do now, by net value then priority.

        Overdue work is surfaced ahead of a marginally better-scoring alternative — a deadline is
        a constraint she agreed to, not another term to trade off.
        """
        try:
            actionable = self.ready()
            if not actionable:
                return None
            overdue = [n for n in actionable if n.overdue]
            pool = overdue or actionable
            return max(pool, key=lambda n: (n.net_value, n.priority))
        except Exception:  # noqa: BLE001
            return None

    def choose(self, plans: Sequence[Plan]) -> Optional[Plan]:
        """Compare plans on what they are actually worth. The best score wins, ties go to safer.

        A tie broken by reversibility rather than arbitrarily: when two plans are worth the same,
        the one that can be undone is strictly the better bet.
        """
        try:
            live = [p for p in plans if p is not None]
            if not live:
                return None
            return max(live, key=lambda p: (round(p.score, 6), p.reversibility, p.confidence))
        except Exception:  # noqa: BLE001
            return None

    def attention_bias(self) -> Dict[str, float]:
        """Top-down bias for :mod:`nyxara.njp.attention` — what she is doing makes things salient."""
        out: Dict[str, float] = {}
        try:
            for node in self.ready()[:8]:
                for token in str(node.name).lower().split():
                    if len(token) > 2:
                        out[token] = max(out.get(token, 0.0), node.priority)
            return out
        except Exception:  # noqa: BLE001
            return out

    def owner_aligned(self, node: Node) -> Optional[bool]:
        """Defer to the repo's own owner-alignment envelope. ``None`` when it cannot say."""
        try:
            system = self.system
            if system is None or not hasattr(system, "is_owner_aligned"):
                return None
            from nyxara.planning.goals import Goal

            return bool(system.is_owner_aligned(
                Goal(name=node.name, vector={"owner_benefit": node.expected_value})))
        except Exception:  # noqa: BLE001
            return None

    def _evict(self) -> None:
        try:
            if len(self.nodes) <= self.capacity:
                return
            done = [n for n in self.nodes.values()
                    if n.state in (State.DONE, State.ABANDONED) and not n.children]
            for node in done[: len(self.nodes) - self.capacity]:
                parent = self.nodes.get(node.parent)
                if parent is not None and node.nid in parent.children:
                    parent.children.remove(node.nid)
                self.nodes.pop(node.nid, None)
        except Exception:  # noqa: BLE001
            pass

    # ---- reporting -------------------------------------------------------------- #
    def tree(self) -> List[Dict[str, Any]]:
        """The hierarchy with rolled-up progress, missions first."""
        return [dict(n.to_dict(), progress=round(self.progress(n.nid), 4))
                for n in sorted(self.nodes.values(), key=lambda n: Kind.ALL.index(n.kind)
                                if n.kind in Kind.ALL else 9)]

    def stats(self) -> Dict[str, Any]:
        by_state: Dict[str, int] = {}
        for node in self.nodes.values():
            by_state[node.state] = by_state.get(node.state, 0) + 1
        missions = [n for n in self.nodes.values() if n.kind == Kind.MISSION]
        return {"nodes": len(self.nodes), "by_state": by_state,
                "completed": self.completed, "abandoned": self.abandoned,
                "blocked": by_state.get(State.BLOCKED, 0),
                "overdue": sum(1 for n in self.nodes.values() if n.overdue),
                "missions": {m.name: round(self.progress(m.nid), 4) for m in missions},
                "top": self.top().to_dict() if self.top() else None}

    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": [dict(n.to_dict(), own_progress=n.own_progress,
                               deadline=n.deadline, nid=n.nid, parent=n.parent,
                               child_ids=list(n.children))
                          for n in self.nodes.values()],
                "seq": self._seq,
                "counters": {"completed": self.completed, "abandoned": self.abandoned}}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            for row in (d.get("nodes") or []):
                nid = str(row.get("nid") or row.get("id") or "")
                if not nid:
                    continue
                self.nodes[nid] = Node(
                    nid=nid, name=str(row.get("name", "")),
                    kind=str(row.get("kind", Kind.TASK)),
                    parent=str(row.get("parent", "")),
                    children=list(row.get("child_ids") or []),
                    priority=float(row.get("priority", 0.5)),
                    own_progress=float(row.get("own_progress", 0.0)),
                    deadline=row.get("deadline"),
                    confidence=float(row.get("confidence", 0.5)),
                    dependencies=list(row.get("dependencies") or []),
                    expected_value=float(row.get("expected_value", 0.5)),
                    cost=float(row.get("cost", 0.2)), risk=float(row.get("risk", 0.2)),
                    reversibility=float(row.get("reversibility", 1.0)),
                    state=str(row.get("state", State.OPEN)))
            self._seq = int(d.get("seq", len(self.nodes)))
            counters = d.get("counters") or {}
            self.completed = int(counters.get("completed", 0))
            self.abandoned = int(counters.get("abandoned", 0))
        except Exception:  # noqa: BLE001
            pass
