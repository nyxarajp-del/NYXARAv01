"""NYXARA · observe/mindscope.py — introspection & explainability (🔬, a debuggable mind).

A mind the Master cannot see into is a mind he cannot trust. This module is the window: it
records NYXARA's cognition as a **causal graph of thoughts** — each percept, recall,
inference, goal, decision and action a node, wired to the thoughts that caused it — so that
*every* conclusion can be traced back to its reasons.

From that graph it answers the questions a curious or worried Master asks:

* **"Why did you think / do that?"** — :meth:`MindScope.why` walks the causal chain from a
  thought back to its roots, and :meth:`MindScope.explain` renders it as a readable line of
  reasoning ("because the disk was at 92% and the goal is to protect your system, I decided
  to rotate the logs").
* **"What are you focused on?"** — :meth:`MindScope.attention` ranks the live thoughts by
  salience, exposing where the mind's spotlight is.
* **"What changed in your head?"** — :meth:`MindScope.snapshot` and :meth:`MindScope.diff`
  capture internal state and show exactly what moved between two moments.

It is pure instrumentation: it observes and explains, it never decides. Together with
:mod:`observe.honesty` and :mod:`observe.self_report` it makes NYXARA's inner life fully
auditable — Rule 6, all the way down.

Pure standard library.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set

__all__ = [
    "ThoughtKind",
    "Thought",
    "MindScope",
]


# --------------------------------------------------------------------------- #
# Thought
# --------------------------------------------------------------------------- #
class ThoughtKind(str, Enum):
    PERCEPTION = "perception"
    RECALL = "recall"
    INFERENCE = "inference"
    GOAL = "goal"
    EMOTION = "emotion"
    ATTENTION = "attention"
    DECISION = "decision"
    ACTION = "action"


@dataclass
class Thought:
    id: str
    kind: ThoughtKind
    content: str
    causes: List[str] = field(default_factory=list)   # ids of thoughts that fed into this
    salience: float = 0.5
    confidence: float = 1.0
    at: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind.value, "content": self.content,
                "causes": list(self.causes), "salience": round(self.salience, 3),
                "confidence": round(self.confidence, 3), "meta": self.meta}


# --------------------------------------------------------------------------- #
# MindScope
# --------------------------------------------------------------------------- #
class MindScope:
    """Records cognition as a causal graph and explains it on demand."""

    def __init__(self, *, clock=None, max_thoughts: int = 10_000) -> None:
        self._clock = clock or time.time
        self.max_thoughts = max_thoughts
        self._thoughts: Dict[str, Thought] = {}
        self._order: List[str] = []
        self._seq = 0
        self._snapshots: Dict[str, Dict[str, Any]] = {}

    # ---- recording the trace ---- #
    def record(self, kind: ThoughtKind, content: str, *, causes: Sequence[str] = (),
               salience: float = 0.5, confidence: float = 1.0, **meta: Any) -> str:
        for c in causes:
            if c not in self._thoughts:
                raise KeyError(f"cause {c!r} is not a recorded thought")
        tid = f"t{self._seq}"
        self._seq += 1
        self._thoughts[tid] = Thought(id=tid, kind=kind, content=content,
                                      causes=list(causes), salience=salience,
                                      confidence=confidence, at=self._clock(), meta=dict(meta))
        self._order.append(tid)
        if len(self._order) > self.max_thoughts:
            old = self._order.pop(0)
            self._thoughts.pop(old, None)
        return tid

    def get(self, tid: str) -> Optional[Thought]:
        return self._thoughts.get(tid)

    def prune(self, *, min_salience: float = 0.25, keep_last: int = 200) -> int:
        """Delete low-salience thoughts during consolidation (data distillation).

        Always keeps the most recent ``keep_last`` thoughts (recent context is precious) and any
        thought another retained thought still depends on (no dangling causal references). Returns
        how many were removed."""
        if len(self._order) <= keep_last:
            return 0
        protected = set(self._order[-keep_last:])
        # keep anything still referenced as a cause by a protected/retained thought
        for tid in list(protected):
            t = self._thoughts.get(tid)
            if t:
                protected.update(t.causes)
        removed = 0
        for tid in list(self._order):
            if tid in protected:
                continue
            t = self._thoughts.get(tid)
            if t is None or t.salience >= min_salience:
                continue
            self._thoughts.pop(tid, None)
            self._order.remove(tid)
            removed += 1
        return removed

    def thoughts(self) -> List[Thought]:
        return [self._thoughts[i] for i in self._order if i in self._thoughts]

    def by_kind(self, kind: ThoughtKind) -> List[Thought]:
        return [t for t in self.thoughts() if t.kind is kind]

    def __len__(self) -> int:
        return len(self._thoughts)

    # ---- causal navigation ---- #
    def direct_causes(self, tid: str) -> List[Thought]:
        t = self._require(tid)
        return [self._thoughts[c] for c in t.causes if c in self._thoughts]

    def direct_effects(self, tid: str) -> List[Thought]:
        return [t for t in self.thoughts() if tid in t.causes]

    def ancestors(self, tid: str) -> Set[str]:
        self._require(tid)
        seen: Set[str] = set()
        stack = list(self._thoughts[tid].causes)
        while stack:
            c = stack.pop()
            if c in seen or c not in self._thoughts:
                continue
            seen.add(c)
            stack.extend(self._thoughts[c].causes)
        return seen

    def descendants(self, tid: str) -> Set[str]:
        self._require(tid)
        seen: Set[str] = set()
        frontier = deque([tid])
        while frontier:
            cur = frontier.popleft()
            for t in self.thoughts():
                if cur in t.causes and t.id not in seen:
                    seen.add(t.id)
                    frontier.append(t.id)
        return seen

    # ---- "why did I think that?" ---- #
    def why(self, tid: str) -> List[Thought]:
        """The causal chain that led to ``tid`` — its ancestors plus itself, in order."""
        relevant = self.ancestors(tid) | {tid}
        # recording order is a valid topological order (causes always precede effects)
        return [self._thoughts[i] for i in self._order if i in relevant]

    def explain(self, tid: str) -> str:
        """A readable line of reasoning for a thought/decision."""
        chain = self.why(tid)
        target = self._require(tid)
        reasons = [t for t in chain if t.id != tid]
        if not reasons:
            return f"I {self._verb(target.kind)} {target.content!r} (no prior cause recorded)."
        premises = "; ".join(f"{self._verb(t.kind)} {t.content!r}" for t in reasons)
        return (f"Because I {premises}, I {self._verb(target.kind)} {target.content!r} "
                f"(confidence {target.confidence:.2f}).")

    @staticmethod
    def _verb(kind: ThoughtKind) -> str:
        return {ThoughtKind.PERCEPTION: "perceived", ThoughtKind.RECALL: "recalled",
                ThoughtKind.INFERENCE: "inferred", ThoughtKind.GOAL: "held the goal",
                ThoughtKind.EMOTION: "felt", ThoughtKind.ATTENTION: "attended to",
                ThoughtKind.DECISION: "decided", ThoughtKind.ACTION: "did"}[kind]

    def reasoning_chain(self, tid: str) -> List[Thought]:
        """The inference/decision steps (filtering out raw inputs) leading to ``tid``."""
        return [t for t in self.why(tid)
                if t.kind in (ThoughtKind.INFERENCE, ThoughtKind.DECISION, ThoughtKind.GOAL)]

    # ---- attention / focus ---- #
    def attention(self, n: int = 5) -> List[Thought]:
        return sorted(self.thoughts(), key=lambda t: (-t.salience, -t.at))[:n]

    def focus(self) -> Optional[Thought]:
        att = self.attention(1)
        return att[0] if att else None

    # ---- internal-state snapshot / diff ---- #
    def snapshot(self, label: str, state: Dict[str, Any]) -> None:
        self._snapshots[label] = dict(state)

    def diff(self, label_a: str, label_b: str) -> Dict[str, Any]:
        a = self._snapshots.get(label_a, {})
        b = self._snapshots.get(label_b, {})
        added = {k: b[k] for k in b if k not in a}
        removed = {k: a[k] for k in a if k not in b}
        changed = {k: {"from": a[k], "to": b[k]} for k in a if k in b and a[k] != b[k]}
        return {"added": added, "removed": removed, "changed": changed}

    # ---- rendering ---- #
    def render_trace(self, n: int = 30) -> str:
        lines = ["MindScope trace:"]
        for t in self.thoughts()[-n:]:
            causes = f" <- {','.join(t.causes)}" if t.causes else ""
            lines.append(f"  {t.id} [{t.kind.value}] {t.content} "
                         f"(sal {t.salience:.2f}){causes}")
        return "\n".join(lines)

    def render_explanation(self, tid: str) -> str:
        out = [f"Why {tid}: {self._require(tid).content!r}"]
        for t in self.why(tid):
            mark = "→" if t.id == tid else " "
            out.append(f"  {mark} {t.id} [{t.kind.value}] {t.content}")
        out.append("")
        out.append(self.explain(tid))
        return "\n".join(out)

    def to_dict(self) -> Dict[str, Any]:
        return {"thoughts": [t.to_dict() for t in self.thoughts()],
                "snapshots": list(self._snapshots)}

    def report(self) -> Dict[str, Any]:
        by_kind: Dict[str, int] = {}
        for t in self.thoughts():
            by_kind[t.kind.value] = by_kind.get(t.kind.value, 0) + 1
        return {"thoughts": len(self._thoughts), "by_kind": by_kind,
                "snapshots": len(self._snapshots),
                "focus": self.focus().content if self.focus() else None}

    def _require(self, tid: str) -> Thought:
        t = self._thoughts.get(tid)
        if t is None:
            raise KeyError(f"no such thought {tid!r}")
        return t


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA mindscope self-test")
    print("=" * 70)

    ms = MindScope()

    # NYXARA's cognition, recorded as a causal graph
    p1 = ms.record(ThoughtKind.PERCEPTION, "disk usage is 92%", salience=0.7)
    p2 = ms.record(ThoughtKind.PERCEPTION, "an anomaly on port 22", salience=0.9)
    g1 = ms.record(ThoughtKind.GOAL, "protect the Master's system", salience=0.8)
    r1 = ms.record(ThoughtKind.INFERENCE, "the disk will fill within hours",
                   causes=[p1], salience=0.6, confidence=0.8)
    d1 = ms.record(ThoughtKind.DECISION, "rotate the logs now",
                   causes=[r1, g1], salience=0.75, confidence=0.9)
    a1 = ms.record(ThoughtKind.ACTION, "rotated the logs", causes=[d1], salience=0.5)

    print("\n" + ms.render_trace())

    # WHY did NYXARA decide to rotate the logs?
    print("\n" + ms.render_explanation(d1))
    chain_ids = {t.id for t in ms.why(d1)}
    assert chain_ids == {p1, g1, r1, d1}      # the full causal chain, not p2 (unrelated)
    assert p2 not in chain_ids

    # natural-language explanation
    exp = ms.explain(d1)
    print(f"\nexplain             : {exp}")
    assert "disk will fill" in exp and "protect the Master" in exp and "rotate the logs" in exp

    # causal navigation
    assert {t.id for t in ms.direct_causes(d1)} == {r1, g1}
    assert {t.id for t in ms.direct_effects(d1)} == {a1}
    assert ms.ancestors(a1) == {p1, g1, r1, d1}
    assert d1 in ms.descendants(p1)
    print("\ncausal navigation   : ancestors/descendants/effects resolved ✓")

    # reasoning chain (just the inferential steps)
    rc = [t.content for t in ms.reasoning_chain(d1)]
    print(f"reasoning chain     : {rc}")
    assert "rotate the logs now" in rc and "the disk will fill within hours" in rc

    # ATTENTION: what is the mind focused on? (the port-22 anomaly, highest salience)
    focus = ms.focus()
    print(f"\nfocus               : {focus.content} (salience {focus.salience})")
    assert focus.id == p2

    # internal-state SNAPSHOT + DIFF
    ms.snapshot("before", {"posture": "normal", "disk": 0.92, "open_ports": [22, 443]})
    ms.snapshot("after", {"posture": "heightened", "disk": 0.40, "open_ports": [443]})
    d = ms.diff("before", "after")
    print(f"\nstate diff          : {d}")
    assert d["changed"]["posture"] == {"from": "normal", "to": "heightened"}
    assert d["changed"]["disk"]["to"] == 0.40

    # honest about gaps: a thought with no recorded cause says so
    free = ms.record(ThoughtKind.INFERENCE, "a hunch")
    assert "no prior cause" in ms.explain(free)
    print("\nhonest gaps         : an uncaused thought is explained as such ✓")

    # a dangling cause is refused (the graph stays consistent)
    try:
        ms.record(ThoughtKind.ACTION, "x", causes=["t999"])
        raise SystemExit("ERROR: dangling cause should be refused")
    except KeyError:
        print("graph integrity     : a dangling cause reference is refused ✓")

    print(f"\nreport              : {ms.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
