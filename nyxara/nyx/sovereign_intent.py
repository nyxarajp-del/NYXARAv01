"""NYXARA · nyx/sovereign_intent.py — one command, decomposed and actually run (🗺, NYX V.02).

:mod:`nyxara.nyx.agenda` keeps goals *she* chose and advances them one beat at a time.
:mod:`nyxara.nyx.hands` runs one tool. :mod:`nyxara.nyx.author` writes one function. What has
never existed is the thing between them: **the Master says one sentence, and it becomes a plan
with parts, an order, a risk class per part, a way back per part, and an execution that keeps
going after a part fails.**

:class:`GoalTree` is that:

* **Decomposition into a DAG.** :mod:`nyxara.nyx.intent` already returns actions, ordering
  (``before → after``), constraints and conditions. Those become nodes and edges — the plan is
  read out of the instruction, not invented beside it.
* **A risk class per node, before anything runs.** :class:`~nyxara.nyx.consequence.ConsequenceGate`
  is asked about every node at *plan* time: reversibility, blast radius, tail risk. A node whose
  class is irreversible is marked before the first node executes, not when it is reached.
* **A rollback strategy per node, decided in advance.** A git branch, a workspace snapshot, or
  an omega rollback point — chosen from the node's own risk class. Deciding this after the fact
  is how systems lose work.
* **Execution that re-plans.** Each node runs through :mod:`~nyxara.nyx.hands` (tools),
  :mod:`~nyxara.nyx.author` (code), :mod:`~nyxara.nyx.theorem_prover` (verification) or
  :mod:`~nyxara.nyx.dialectic` (self-review). A failure asks :mod:`~nyxara.nyx.causal_engine`
  for a root cause and, when it finds one, inserts a repair node rather than stopping.
* **Progress that survives a restart.** The tree, its per-node state and its findings persist to
  the ``nyx.json`` sidecar; :meth:`resume` picks up at the first unfinished node.

Honest framing (the rule this repo keeps, see ``docs/CAPABILITIES.md``):

* **Her reach is exactly** :mod:`~nyxara.nyx.author`'s **bounded synthesis and the registry's
  tools** — nothing here widens either. Ask for a 206k-line C++ port and what you get is a real
  DAG, real attempts on the nodes she can reach, real measurements, and an honest report of
  where it stopped. **Silently shipping broken code is the one thing this repo does not do**, so
  a node she cannot complete is marked ``blocked`` and says why, and the tree reports itself
  *incomplete* rather than *done*.
* **Decomposition is structural, not semantic.** It splits on the conjunctions, orderings and
  conditions :mod:`~nyxara.nyx.intent` actually found. It does not know how to break down a
  problem it has never seen — an instruction with one clause becomes a one-node tree, and that
  is reported rather than padded out.
* **Autonomy runs through the unchanged pipeline.** Every node acts via the registry and the
  consequence gate, so permissions, capability bounds, the audit log and ``/scram`` all apply
  exactly as they did before. Nothing here is a side door.
* Progress is **measured** — nodes done over nodes total — never asserted.

Pure standard library. Every public method is fail-soft.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

__all__ = ["Node", "Plan", "Run", "GoalTree", "ROLLBACKS", "NODE_KINDS"]

#: What a node is asking for, which decides which faculty is spent on it.
NODE_KINDS = ("tool", "code", "verify", "review", "think")

#: The ways back, worst-case first. Chosen at plan time from the node's risk class.
ROLLBACKS = ("git-branch", "workspace-snapshot", "omega-rollback", "none-needed", "none-available")

_PENDING, _RUNNING, _DONE, _FAILED, _BLOCKED, _SKIPPED = (
    "pending", "running", "done", "failed", "blocked", "skipped")

#: Verbs that mean "write something new" rather than "run something that exists".
_CODE_WORDS = ("write", "implement", "code", "function", "port", "generate", "synthesise",
               "synthesize", "likh", "banao", "bana")
_VERIFY_WORDS = ("verify", "prove", "check", "test", "validate", "benchmark", "chala", "jaanch")
_REVIEW_WORDS = ("review", "critique", "audit", "assess", "judge", "dekh")

#: "…and then verify it" — a sequencer is the Master stating an order out loud, so it splits a
#: clause *and* becomes an edge. :mod:`~nyxara.nyx.intent` splits on conjunctions; this is the
#: one thing it leaves inside a clause.
_SEQUENCER = re.compile(r"\s*(?:,\s*)?\b(?:and\s+then|then|after\s+that|afterwards|"
                        r"uske\s+baad|phir|फिर)\b\s*", re.I)


@dataclass
class Node:
    """One part of the plan: what it is, what it costs, and how she gets back if it goes wrong."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    action: str = ""
    kind: str = "think"
    after: List[str] = field(default_factory=list)      # node ids that must finish first
    status: str = _PENDING

    # Decided at plan time, before anything runs.
    reversibility: str = "unknown"
    blast_radius: int = 0
    tail_risk: float = 0.0
    allowed: bool = True
    rollback: str = "none-available"
    risk_why: str = ""

    # Filled in by execution.
    finding: str = ""
    error: str = ""
    attempts: int = 0
    repaired_by: str = ""        # id of a node inserted to fix this one
    at: float = 0.0

    @property
    def finished(self) -> bool:
        return self.status in (_DONE, _SKIPPED)

    @property
    def open(self) -> bool:
        return self.status in (_PENDING, _RUNNING)

    def render(self) -> str:
        mark = {_DONE: "✓", _FAILED: "✗", _BLOCKED: "⛔", _SKIPPED: "–"}.get(self.status, "·")
        line = (f"{mark} [{self.id}] {self.kind:6} {self.action[:56]}"
                f"   ({self.reversibility}, rollback: {self.rollback})")
        if self.finding:
            line += f"\n      → {self.finding[:100]}"
        if self.error:
            line += f"\n      ! {self.error[:100]}"
        return line

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "action": self.action, "kind": self.kind,
                "after": list(self.after), "status": self.status,
                "reversibility": self.reversibility, "blast_radius": self.blast_radius,
                "tail_risk": round(self.tail_risk, 4), "allowed": self.allowed,
                "rollback": self.rollback, "risk_why": self.risk_why,
                "finding": self.finding, "error": self.error, "attempts": self.attempts,
                "repaired_by": self.repaired_by, "at": round(self.at, 3)}


@dataclass
class Plan:
    """A whole instruction, decomposed. Nothing has run yet."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    request: str = ""
    goal: str = ""
    nodes: List[Node] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    question: str = ""            # what she would ask before starting, when it matters
    note: str = ""
    at: float = field(default_factory=time.time)

    @property
    def size(self) -> int:
        return len(self.nodes)

    @property
    def done(self) -> int:
        return sum(1 for n in self.nodes if n.status == _DONE)

    @property
    def progress(self) -> float:
        """Measured: nodes finished over nodes total. Never asserted."""
        return (sum(1 for n in self.nodes if n.finished) / len(self.nodes)) if self.nodes else 0.0

    @property
    def complete(self) -> bool:
        return bool(self.nodes) and all(n.finished for n in self.nodes)

    @property
    def riskiest(self) -> Optional[Node]:
        order = {"irreversible": 3, "recoverable": 2, "unknown": 1, "undoable": 0}
        return max(self.nodes, key=lambda n: order.get(n.reversibility, 1), default=None)

    def node(self, node_id: str) -> Optional[Node]:
        for candidate in self.nodes:
            if candidate.id == node_id:
                return candidate
        return None

    def ready(self) -> List[Node]:
        """Nodes whose dependencies have all finished — the DAG's next front."""
        finished = {n.id for n in self.nodes if n.finished}
        return [n for n in self.nodes
                if n.status == _PENDING and all(dep in finished for dep in n.after)]

    def render(self) -> str:
        lines = [f"plan [{self.id}] {self.request[:80]}",
                 f"  goal      : {self.goal or '(none stated)'}",
                 f"  progress  : {self.done}/{self.size} done ({self.progress:.0%})"]
        for node in self.nodes:
            lines.append("  " + node.render())
        if self.constraints:
            lines.append("  constraints: " + "; ".join(self.constraints[:4]))
        if self.question:
            lines.append(f"  would ask  : {self.question}")
        if self.note:
            lines.append(f"  note       : {self.note}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "request": self.request, "goal": self.goal,
                "nodes": [n.to_dict() for n in self.nodes],
                "constraints": list(self.constraints), "conditions": list(self.conditions),
                "question": self.question, "note": self.note,
                "size": self.size, "done": self.done,
                "progress": round(self.progress, 4), "complete": self.complete,
                "at": round(self.at, 3)}


@dataclass
class Run:
    """What one execution pass actually did. The honest report, not a claim of success."""

    plan_id: str = ""
    executed: List[str] = field(default_factory=list)
    succeeded: int = 0
    skipped: int = 0             # finished with nothing to show — not a success, not a failure
    failed: int = 0
    blocked: int = 0
    repairs: int = 0
    complete: bool = False
    produced: int = 0            # nodes that actually returned a result
    progress: float = 0.0
    stopped_because: str = ""
    elapsed_ms: float = 0.0

    def render(self) -> str:
        head = (f"ran {len(self.executed)} node(s): {self.succeeded} done, "
                f"{self.skipped} nothing-to-do, {self.failed} failed, {self.blocked} blocked, "
                f"{self.repairs} repair(s) inserted")
        tail = (f"  → {'complete' if self.complete else 'INCOMPLETE'} "
                f"at {self.progress:.0%}, {self.produced} node(s) with a result"
                + (f" — {self.stopped_because}" if self.stopped_because else ""))
        return head + "\n" + tail

    def to_dict(self) -> Dict[str, Any]:
        return {"plan_id": self.plan_id, "executed": list(self.executed),
                "succeeded": self.succeeded, "skipped": self.skipped, "failed": self.failed,
                "blocked": self.blocked, "repairs": self.repairs, "complete": self.complete,
                "produced": self.produced, "progress": round(self.progress, 4),
                "stopped_because": self.stopped_because,
                "elapsed_ms": round(self.elapsed_ms, 2)}


class GoalTree:
    """Decompose one instruction into a DAG, price every node's risk, then run it honestly."""

    def __init__(self, brain: Any = None, *, max_nodes: int = 24, max_steps: int = 12,
                 max_repairs: int = 3, execute: bool = True) -> None:
        self.brain = brain
        self.max_nodes = max(1, int(max_nodes))
        self.max_steps = max(1, int(max_steps))
        self.max_repairs = max(0, int(max_repairs))
        self.execute = bool(execute)

        self.plans: Dict[str, Plan] = {}
        self.order: List[str] = []          # newest last
        self.planned = 0
        self.ran = 0
        self.completed = 0
        self.abandoned = 0
        self.repairs = 0

    # ---- decomposition ---------------------------------------------------------- #
    def plan(self, request: str, *, intent: Any = None) -> Plan:
        """Read the instruction's own structure into a DAG. Nothing runs here."""
        out = Plan(request=str(request or "").strip())
        try:
            if not out.request:
                out.note = "there was nothing to decompose"
                return out

            intent = intent if intent is not None else self._read(out.request)
            actions, out.goal = self._parts(out.request, intent)
            if intent is not None:
                out.constraints = [str(c) for c in (getattr(intent, "constraints", []) or [])][:8]
                out.conditions = [str(c) for c in (getattr(intent, "conditions", []) or [])][:8]
                try:
                    out.question = str(intent.clarifying_question() or "")
                except Exception:  # noqa: BLE001 — a missing question is not a failure
                    out.question = ""

            forced: List[Tuple[int, int]] = []
            for position, (action, sequential) in enumerate(actions[: self.max_nodes]):
                out.nodes.append(Node(action=action, kind=self._kind_of(action)))
                if sequential and position:
                    # "…and THEN verify it" is an ordering the Master stated out loud.
                    forced.append((position - 1, position))

            self._wire_order(out, intent, forced)
            for node in out.nodes:
                self._price(node)

            if out.size == 1:
                out.note = ("one clause in, one node out — I split on the orderings and "
                            "conjunctions actually present, and there were none")
            else:
                out.note = f"decomposed into {out.size} node(s) from the instruction's structure"

            self.plans[out.id] = out
            self.order.append(out.id)
            self.planned += 1
            return out
        except Exception as exc:  # noqa: BLE001 — a failed plan is an empty plan, not a crash
            out.note = f"decomposition failed ({type(exc).__name__})"
            return out

    def _read(self, request: str) -> Any:
        try:
            reader = getattr(self.brain, "intent", None)
            return reader.read(request) if reader is not None else None
        except Exception:  # noqa: BLE001
            return None

    def _parts(self, request: str, intent: Any) -> Tuple[List[Tuple[str, bool]], str]:
        """The clauses to do, in the order they were said, plus the stated goal.

        A node is a **clause**, not a bare verb: ``"write a doubler"`` is something that can be
        attempted, while ``"write"`` is not. Constraints, conditions and anything the Master said
        *not* to do are excluded here — a "don't" narrows the plan, it never becomes a step in it.

        Each part carries whether it was introduced by a sequencer (*"and then…"*), because that
        is an ordering the Master stated out loud and it belongs in the DAG.
        """
        goal = ""
        parts: List[Tuple[str, bool]] = []
        try:
            if intent is not None:
                goal = str(getattr(intent, "goal", "") or "")
            excluded = {str(c).strip().lower()
                        for c in (list(getattr(intent, "constraints", []) or [])
                                  + list(getattr(intent, "conditions", []) or []))}
            verbs = {str(a).strip().lower() for a in (getattr(intent, "actions", []) or [])}
            polarity = dict(getattr(intent, "polarity", {}) or {})
            forbidden = {v for v in verbs if not polarity.get(v, True)}

            for clause, sequential in self._clauses(request):
                low = clause.strip().lower()
                if not low or low in excluded:
                    continue
                words = set(low.replace(",", " ").split())
                # A clause carrying a forbidden verb is a "don't". It constrains the plan; it is
                # never a step in it, whatever else it happens to contain.
                if forbidden & words:
                    continue
                parts.append((clause.strip(), sequential))
        except Exception:  # noqa: BLE001
            parts = []
        if not parts:
            parts = [(request.strip(), False)]
        seen: Set[str] = set()
        unique: List[Tuple[str, bool]] = []
        for part, sequential in parts:
            key = part.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append((part.strip(), sequential))
        return unique[: self.max_nodes], goal

    @staticmethod
    def _clauses(request: str) -> List[Tuple[str, bool]]:
        """Split the way :mod:`~nyxara.nyx.intent` splits, then once more on sequencers.

        Agreeing with ``intent`` on what a clause is matters — the ordering pairs it returns are
        clause fragments, and two different splitters would never line up.
        """
        out: List[Tuple[str, bool]] = []
        try:
            from nyxara.nyx.intent import _clauses as split
            for clause in split(request):
                if not clause.strip():
                    continue
                pieces = [p for p in _SEQUENCER.split(clause) if p and p.strip()]
                for position, piece in enumerate(pieces):
                    out.append((piece.strip(" ,.;:"), position > 0))
        except Exception:  # noqa: BLE001
            out = []
        return out or [(request.strip(), False)]

    @staticmethod
    def _kind_of(action: str) -> str:
        """Which faculty this clause is asking for — decided by the verb that comes *first*."""
        low = action.lower()
        best, at = "tool", len(low) + 1
        for kind, words in (("verify", _VERIFY_WORDS), ("code", _CODE_WORDS),
                            ("review", _REVIEW_WORDS)):
            for word in words:
                found = low.find(word)
                if 0 <= found < at:
                    best, at = kind, found
        return best

    def _wire_order(self, out: Plan, intent: Any,
                    forced: Optional[List[Tuple[int, int]]] = None) -> None:
        """Edges come from the ordering the instruction stated, not from node position."""
        wired = False
        try:
            for first_at, second_at in (forced or []):
                if 0 <= first_at < len(out.nodes) and 0 <= second_at < len(out.nodes):
                    first, second = out.nodes[first_at], out.nodes[second_at]
                    if first.id not in second.after:
                        second.after.append(first.id)
                        wired = True
            for before, after in (getattr(intent, "ordering", []) or []):
                first = self._match(out, str(before))
                second = self._match(out, str(after))
                if first is not None and second is not None and first.id != second.id:
                    if first.id not in second.after:
                        second.after.append(first.id)
                        wired = True
        except Exception:  # noqa: BLE001
            wired = False
        if wired:
            self._break_cycles(out)
            return
        # No stated ordering: a verification depends on what it verifies, and nothing else is
        # forced to be sequential — parallel-safe nodes stay parallel.
        producers = [n for n in out.nodes if n.kind in ("code", "tool")]
        for node in out.nodes:
            if node.kind in ("verify", "review"):
                node.after = [p.id for p in producers if p.id != node.id][:4]
        self._break_cycles(out)

    @staticmethod
    def _match(out: Plan, phrase: str) -> Optional[Node]:
        """Find the node an ordering phrase is talking about. Overlap, not identity."""
        low = phrase.strip().lower().strip(" ,.;:")
        if not low:
            return None
        for node in out.nodes:                       # the clause is the phrase, or contains it
            body = node.action.lower()
            if low == body or low in body or body in low:
                return node
        wanted = set(low.split())
        best, score = None, 0
        for node in out.nodes:
            overlap = len(wanted & set(node.action.lower().split()))
            if overlap > score:
                best, score = node, overlap
        return best if score >= 2 else None

    @staticmethod
    def _break_cycles(out: Plan) -> None:
        """A DAG or nothing: an edge that would close a cycle is dropped, not tolerated.

        Position in the sentence is *not* the test — ``"fix the code but first run the tests"``
        states an order that runs backwards through the text, and that ordering is the whole
        reason this module reads intent rather than reading left to right.
        """
        try:
            known = {n.id for n in out.nodes}
            kept: Dict[str, List[str]] = {n.id: [] for n in out.nodes}

            def depends_on(start: str, target: str) -> bool:
                seen: Set[str] = set()
                stack = [start]
                while stack:
                    current = stack.pop()
                    if current == target:
                        return True
                    if current in seen:
                        continue
                    seen.add(current)
                    stack.extend(kept.get(current, ()))
                return False

            for node in out.nodes:
                for dep in node.after:
                    if dep not in known or dep == node.id:
                        continue
                    if depends_on(dep, node.id):
                        continue            # this edge would close a cycle
                    kept[node.id].append(dep)
            for node in out.nodes:
                node.after = kept[node.id]
        except Exception:  # noqa: BLE001
            for node in out.nodes:
                node.after = []

    # ---- risk, priced before anything runs ------------------------------------- #
    def _price(self, node: Node) -> None:
        """Ask the gate about this node now, so an irreversible step is known up front."""
        try:
            gate = getattr(self.brain, "consequence", None)
            if gate is None:
                node.reversibility = "unknown"
                node.risk_why = "no consequence gate is attached; risk is unpriced"
                node.rollback = self._rollback_for(node)
                return
            verdict = gate.check(node.action, {})
            node.reversibility = str(getattr(verdict, "reversibility", "unknown"))
            node.blast_radius = int(getattr(verdict, "blast_radius", 0))
            node.allowed = bool(getattr(verdict, "allowed", True))
            node.risk_why = str(getattr(verdict, "why", ""))
            fore = getattr(verdict, "foresight", None)
            node.tail_risk = float(getattr(fore, "tail_risk", 0.0) or 0.0)
            if not node.allowed:
                node.status = _BLOCKED
                node.error = node.risk_why or "the consequence gate declined this step"
        except Exception:  # noqa: BLE001 — an unpriced node is unknown-risk, never silently safe
            node.reversibility = "unknown"
            node.risk_why = "pricing this step failed; treating it as unknown risk"
        node.rollback = self._rollback_for(node)

    def _rollback_for(self, node: Node) -> str:
        """The way back, decided now. Deciding it after the fact is how work gets lost."""
        try:
            if node.kind in ("verify", "review", "think"):
                return "none-needed"
            if node.reversibility == "undoable" and node.blast_radius == 0:
                return "none-needed"
            if node.kind == "code":
                # Authored code is loaded through the gauntlet, which keeps its own way back.
                return "omega-rollback" if getattr(self.brain, "omega", None) is not None \
                    else "workspace-snapshot"
            if self._has_git():
                return "git-branch"
            if node.blast_radius == 0:
                return "workspace-snapshot"
            return "none-available"
        except Exception:  # noqa: BLE001
            return "none-available"

    def _has_git(self) -> bool:
        try:
            hands = getattr(self.brain, "hands", None)
            names = {str(t.get("name", "")) for t in (hands.catalog() if hands else [])}
            return any(n.startswith("git.") or n == "git" for n in names)
        except Exception:  # noqa: BLE001
            return False

    # ---- execution ------------------------------------------------------------- #
    def run(self, plan: Any = None, *, oversight: Any = None, steps: int = 0,
            dry_run: bool = False) -> Run:
        """Walk the DAG's front, spend a faculty per node, and re-plan around failures."""
        target = self._resolve(plan)
        out = Run(plan_id=getattr(target, "id", ""))
        t0 = time.perf_counter()
        try:
            if target is None or not target.nodes:
                out.stopped_because = "there was no plan to run"
                return out
            if not self.execute or dry_run:
                out.progress = target.progress
                out.stopped_because = ("execution is off — this is the plan and its priced "
                                       "risks, and nothing was run")
                return out

            self.ran += 1
            budget = self.max_steps if steps <= 0 else min(int(steps), self.max_steps)
            repairs = 0
            for _ in range(budget):
                if self._stopped(oversight):
                    out.stopped_because = "oversight paused or scrammed her mid-plan"
                    break
                front = target.ready()
                if not front:
                    break
                node = front[0]
                node.status = _RUNNING
                node.attempts += 1
                node.at = time.time()
                out.executed.append(node.id)

                status, finding, error = self._execute(node, target, oversight=oversight)
                node.finding, node.error = finding, error
                if status == _DONE:
                    node.status = _DONE
                    out.succeeded += 1
                    continue
                if status == _SKIPPED:
                    # Nothing here was hers to do, or nothing here was decidable. That is not a
                    # success and it is not a failure — and it must not be counted as either.
                    node.status = _SKIPPED
                    out.skipped += 1
                    continue

                node.status = _FAILED
                out.failed += 1
                if repairs < self.max_repairs:
                    repair = self._repair(node, target)
                    if repair is not None:
                        repairs += 1
                        self.repairs += 1
                        out.repairs += 1

            out.blocked = sum(1 for n in target.nodes if n.status == _BLOCKED)
            out.progress = target.progress
            out.produced = target.done
            out.complete = target.complete
            if out.complete:
                self.completed += 1
                if not out.produced and not out.stopped_because:
                    out.stopped_because = ("every node finished with nothing to show — this was "
                                           "understood, not acted on, and calling that done "
                                           "would be the wrong word")
            elif not out.stopped_because:
                out.stopped_because = self._why_incomplete(target)
                if out.blocked or out.failed:
                    self.abandoned += 1
            return out
        except Exception as exc:  # noqa: BLE001 — a failed run reports; it never eats the turn
            out.stopped_because = f"the run failed ({type(exc).__name__})"
            return out
        finally:
            out.elapsed_ms = (time.perf_counter() - t0) * 1000.0

    @staticmethod
    def _why_incomplete(target: Plan) -> str:
        stuck = [n for n in target.nodes if n.status in (_FAILED, _BLOCKED)]
        if stuck:
            worst = stuck[0]
            return (f"{len(stuck)} node(s) did not complete — [{worst.id}] "
                    f"{worst.action[:48]}: {worst.error[:80] or 'no reason recorded'}. "
                    f"I am reporting this incomplete rather than calling it done")
        if any(n.open for n in target.nodes):
            return "the step budget ran out with nodes still open — call run() again to continue"
        return "the plan did not finish"

    def _execute(self, node: Node, plan: Plan, *,
                 oversight: Any = None) -> Tuple[str, str, str]:
        """Spend exactly one faculty on one node. Returns ``(status, finding, error)``."""
        try:
            if node.kind == "code":
                return self._do_code(node, oversight=oversight)
            if node.kind == "verify":
                return self._do_verify(node, plan)
            if node.kind == "review":
                return self._do_review(node)
            return self._do_tool(node, oversight=oversight)
        except Exception as exc:  # noqa: BLE001
            return _FAILED, "", f"the step raised {type(exc).__name__}"

    def _do_code(self, node: Node, *, oversight: Any = None) -> Tuple[str, str, str]:
        author = getattr(self.brain, "author", None)
        if author is None:
            return _FAILED, "", "no author is attached; I cannot write code from here"
        got = author.write(node.action, oversight=oversight)
        if getattr(got, "ok", False):
            return _DONE, f"wrote and verified: {getattr(got, 'family', '')} → {got.value!r}", ""
        return _FAILED, "", (str(getattr(got, "refusal", "")) or
                           f"authoring stopped at {getattr(got, 'stopped_at', '?')}")

    def _do_tool(self, node: Node, *, oversight: Any = None) -> Tuple[str, str, str]:
        hands = getattr(self.brain, "hands", None)
        if hands is None:
            return _FAILED, "", "no hands are attached; I cannot reach a tool from here"
        got = hands.reach(node.action, oversight=oversight)
        if getattr(got, "ok", False):
            return _DONE, f"{got.tool}: {str(got.value)[:120]}", ""
        if not getattr(got, "wanted", False):
            # Not every clause is an action. A node that needs no tool is honestly done.
            return _SKIPPED, "nothing here needed a tool; I read it as understood, not as acted on", ""
        return _FAILED, "", (str(getattr(got, "refused", "")) or str(getattr(got, "error", "")) or
                           str(getattr(got, "why", "")) or "the tool did not run")

    def _do_verify(self, node: Node, plan: Plan) -> Tuple[str, str, str]:
        prover = getattr(self.brain, "prover", None)
        if prover is None:
            return _FAILED, "", "no prover is attached; I cannot verify from here"
        claim = node.action
        for dep in node.after:
            source = plan.node(dep)
            if source is not None and source.finding:
                claim = f"{node.action} ({source.finding})"
                break
        got = prover.prove(claim)
        verdict = str(getattr(got, "verdict", "unknown"))
        if verdict == "proved":
            return _DONE, f"proved: {getattr(got, 'note', '')[:100]}", ""
        if verdict == "refuted":
            return _FAILED, "", f"refuted: {getattr(got, 'note', '')[:100]}"
        # Inexpressible is not a failure of the plan; it is a limit of the prover, said plainly.
        return _SKIPPED, f"not decidable here ({verdict}) — {getattr(got, 'note', '')[:80]}", ""

    def _do_review(self, node: Node) -> Tuple[str, str, str]:
        dialectic = getattr(self.brain, "dialectic", None)
        if dialectic is None:
            return _FAILED, "", "no dialectic is attached; I cannot self-review from here"
        got = dialectic.debate(node.action)
        if getattr(got, "abstained", False):
            return _FAILED, "", f"my own review broke it: {getattr(got, 'note', '')[:90]}"
        objections = len(getattr(got, "objections", []) or [])
        return _DONE, f"reviewed, {objections} objection(s) survived", ""

    # ---- re-planning ------------------------------------------------------------ #
    def _repair(self, node: Node, plan: Plan) -> Optional[Node]:
        """A failure asks for a root cause, and a cause found becomes a node — not a stop."""
        try:
            if len(plan.nodes) >= self.max_nodes:
                return None
            cause = self._root_cause(node)
            if not cause:
                return None
            repair = Node(action=cause, kind=self._kind_of(cause), after=list(node.after))
            self._price(repair)
            plan.nodes.append(repair)
            node.repaired_by = repair.id
            # The failed node goes back to pending, behind its own repair.
            node.status = _PENDING
            if repair.id not in node.after:
                node.after.append(repair.id)
            return repair
        except Exception:  # noqa: BLE001
            return None

    def _root_cause(self, node: Node) -> str:
        try:
            causal = getattr(self.brain, "causal", None)
            if causal is None or not node.error:
                return ""
            from nyxara.nyx.graph import concepts_in
            words = concepts_in(node.error, cap=4)
            subject = concepts_in(node.action, cap=1)
            if not words or not subject:
                return ""
            answer = causal.ask(words[0], subject[0], question=node.error[:80])
            if not getattr(answer, "causal", False):
                return ""
            return f"address {words[0]} before {subject[0]}"
        except Exception:  # noqa: BLE001
            return ""

    # ---- the seam the brain uses ------------------------------------------------ #
    def pursue(self, request: str, *, oversight: Any = None,
               dry_run: bool = False) -> Tuple[Plan, Run]:
        """Plan it, then run it. The one call a caller usually wants."""
        made = self.plan(request)
        got = self.run(made, oversight=oversight, dry_run=dry_run)
        return made, got

    def resume(self, *, oversight: Any = None) -> Optional[Run]:
        """Pick up the newest unfinished plan where it stopped — the point of persisting it."""
        try:
            for plan_id in reversed(self.order):
                plan = self.plans.get(plan_id)
                if plan is not None and plan.nodes and not plan.complete:
                    return self.run(plan, oversight=oversight)
            return None
        except Exception:  # noqa: BLE001
            return None

    def latest(self) -> Optional[Plan]:
        return self.plans.get(self.order[-1]) if self.order else None

    @staticmethod
    def _stopped(oversight: Any) -> bool:
        if oversight is None:
            return False
        try:
            for attr in ("scrammed", "is_scrammed", "paused", "is_paused"):
                probe = getattr(oversight, attr, None)
                if probe is None:
                    continue
                if bool(probe() if callable(probe) else probe):
                    return True
            return False
        except Exception:  # noqa: BLE001 — an unreadable oversight is treated as STOP
            return True

    def _resolve(self, plan: Any) -> Optional[Plan]:
        if isinstance(plan, Plan):
            return plan
        if isinstance(plan, str) and plan in self.plans:
            return self.plans[plan]
        return self.latest()

    # ---- introspection ---------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        try:
            open_plans = sum(1 for p in self.plans.values() if not p.complete)
            return {
                "plans": len(self.plans), "planned": self.planned, "runs": self.ran,
                "completed": self.completed, "abandoned": self.abandoned,
                "repairs": self.repairs, "open": open_plans,
                "max_nodes": self.max_nodes, "max_steps": self.max_steps,
                "executes": self.execute,
                "note": ("MY REACH IS EXACTLY author's bounded synthesis and the registry's "
                         "tools — nothing here widens either. A large port becomes a real DAG, "
                         "real attempts on the nodes I can reach, and an honest report of where "
                         "it stopped; a node I cannot finish is marked blocked with its reason "
                         "and the plan reports itself INCOMPLETE rather than done. Decomposition "
                         "is STRUCTURAL — it splits on the orderings and conjunctions intent "
                         "actually found, so a one-clause instruction is a one-node tree. Every "
                         "node runs through the registry and the consequence gate, so "
                         "permissions, capability bounds, the audit log and /scram all still "
                         "apply"),
            }
        except Exception:  # noqa: BLE001
            return {}

    def describe(self) -> str:
        try:
            plan = self.latest()
            if plan is None:
                return "no plan yet — give me an instruction and I will decompose it first."
            return plan.render()
        except Exception:  # noqa: BLE001
            return ""

    # ---- persistence ------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {"planned": self.planned, "ran": self.ran, "completed": self.completed,
                "abandoned": self.abandoned, "repairs": self.repairs,
                "order": list(self.order[-8:]),
                "plans": [self.plans[i].to_dict() for i in self.order[-8:] if i in self.plans]}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.planned = int(data.get("planned", 0))
            self.ran = int(data.get("ran", 0))
            self.completed = int(data.get("completed", 0))
            self.abandoned = int(data.get("abandoned", 0))
            self.repairs = int(data.get("repairs", 0))
            self.plans, self.order = {}, []
            for raw in data.get("plans", []) or []:
                if not isinstance(raw, dict):
                    continue
                plan = Plan(id=str(raw.get("id") or uuid.uuid4().hex[:8]),
                            request=str(raw.get("request", "")),
                            goal=str(raw.get("goal", "")),
                            constraints=[str(c) for c in (raw.get("constraints") or [])],
                            conditions=[str(c) for c in (raw.get("conditions") or [])],
                            question=str(raw.get("question", "")),
                            note=str(raw.get("note", "")),
                            at=float(raw.get("at", time.time())))
                for rawnode in raw.get("nodes", []) or []:
                    if not isinstance(rawnode, dict):
                        continue
                    plan.nodes.append(Node(
                        id=str(rawnode.get("id") or uuid.uuid4().hex[:8]),
                        action=str(rawnode.get("action", "")),
                        kind=str(rawnode.get("kind", "think")),
                        after=[str(d) for d in (rawnode.get("after") or [])],
                        status=str(rawnode.get("status", _PENDING)),
                        reversibility=str(rawnode.get("reversibility", "unknown")),
                        blast_radius=int(rawnode.get("blast_radius", 0)),
                        tail_risk=float(rawnode.get("tail_risk", 0.0)),
                        allowed=bool(rawnode.get("allowed", True)),
                        rollback=str(rawnode.get("rollback", "none-available")),
                        risk_why=str(rawnode.get("risk_why", "")),
                        finding=str(rawnode.get("finding", "")),
                        error=str(rawnode.get("error", "")),
                        attempts=int(rawnode.get("attempts", 0)),
                        repaired_by=str(rawnode.get("repaired_by", "")),
                        at=float(rawnode.get("at", 0.0))))
                self.plans[plan.id] = plan
            self.order = [i for i in (data.get("order") or []) if i in self.plans]
            for plan_id in self.plans:
                if plan_id not in self.order:
                    self.order.append(plan_id)
            return True
        except Exception:  # noqa: BLE001 — a corrupt sidecar must never block boot
            return False
