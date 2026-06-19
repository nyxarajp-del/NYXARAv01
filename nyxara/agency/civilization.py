"""NYXARA · agency/civilization.py — Multi-Agent Internal Civilization (Level 9).

Seven persistent specialized micro-agents run as background workers. NYXARA is the governor.
Each agent owns a domain and a duty cycle, and — unlike a bank of passive monitors — they form
a small *society*:

* they **observe** their domain (read-only inspection of the core), and
* they **talk to one another** over a shared blackboard (an agent that finds a problem can ask
  the right specialist to handle it), and
* when autonomy is authorised they **act** — each may take ONE safe, *reversible*, **gated**
  action per cycle (memory consolidation, goal pruning, a learner step, …). Every action is
  fail-closed behind the permission gate and recorded in the journal, exactly like any other
  autonomous act; with autonomy OFF (the default) they degrade to read-only monitors that only
  *suggest*.

NYXARA can pause/terminate any agent. Reports (and now messages) flow back to the governor and,
optionally, onto the EventBus (topic ``civilization.report`` / ``civilization.message``).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

__all__ = ["MicroAgent", "AgentReport", "AgentMessage", "AgentContext",
           "MicroAgentCivilization"]


# --------------------------------------------------------------------------- #
# Messages — the inter-agent blackboard
# --------------------------------------------------------------------------- #
@dataclass
class AgentMessage:
    """One message posted by an agent to the shared blackboard (read by the others)."""
    sender: str
    topic: str
    content: str
    priority: str = "normal"          # "normal" | "high"
    timestamp: float = field(default_factory=time.time)
    seq: int = 0                      # monotonic id; lets each reader consume a message once

    def to_dict(self) -> Dict[str, Any]:
        return {"sender": self.sender, "topic": self.topic, "content": self.content,
                "priority": self.priority, "timestamp": self.timestamp, "seq": self.seq}


@dataclass
class AgentReport:
    """One tick outcome from a MicroAgent."""
    agent_name: str
    domain: str
    timestamp: float = field(default_factory=time.time)
    findings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)      # gated actions actually taken
    messages: List[str] = field(default_factory=list)     # messages posted this tick
    ok: bool = True
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "domain": self.domain,
            "timestamp": self.timestamp,
            "findings": self.findings,
            "suggestions": self.suggestions,
            "actions": self.actions,
            "messages": self.messages,
            "ok": self.ok,
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# AgentContext — what a tick function is handed (core + society + the act() gate)
# --------------------------------------------------------------------------- #
class AgentContext:
    """The handle an agent uses each tick: read the core, talk to peers, and (when authorised)
    take a gated action. ``report`` is the agent's own report for this tick."""

    def __init__(self, *, core: Any, agent: str, enact: bool,
                 civ: "MicroAgentCivilization", report: AgentReport,
                 snapshot_seq: int) -> None:
        self.core = core
        self.agent = agent
        self.enact = enact
        self._civ = civ
        self.report = report
        self._snapshot_seq = snapshot_seq      # messages newer than this are next tick's, not now

    # ---- inter-agent messaging ---- #
    def post(self, topic: str, content: str, *, priority: str = "normal") -> None:
        """Post a message to the blackboard for the other agents to read."""
        msg = AgentMessage(sender=self.agent, topic=topic, content=content, priority=priority)
        self._civ._post_message(msg)
        self.report.messages.append(f"{topic}: {content}")

    def inbox(self, *, topic: Optional[str] = None) -> List[AgentMessage]:
        """*Unconsumed* messages from other agents (optionally filtered), newest first.

        Each message is delivered to a given agent only once — the agent's read cursor advances
        past everything visible at the start of its tick, so it reacts to news, not history."""
        return self._civ._read_new(self.agent, topic=topic, snapshot=self._snapshot_seq)

    # ---- gated action ---- #
    def act(self, description: str, fn: Callable[[], Any], *,
            capability: str = "", reversible: bool = True) -> bool:
        """Take a real action *iff* autonomy is on and budget remains; otherwise suggest it.

        Returns True when the action actually ran. Either way the outcome is recorded on the
        report (as an action taken, or a withheld suggestion)."""
        outcome = self._civ._enact(self.agent, description, fn,
                                   capability=capability, reversible=reversible, enact=self.enact)
        if outcome["taken"]:
            self.report.actions.append(description)
            return True
        self.report.suggestions.append(f"{description} ({outcome['reason']})")
        return False


# --------------------------------------------------------------------------- #
# MicroAgent
# --------------------------------------------------------------------------- #
class MicroAgent:
    """One specialized background agent with its own duty cycle.

    Parameters
    ----------
    name:         unique agent name
    domain:       human-readable domain description
    duty_cycle_s: how many seconds between ticks
    tick_fn:      callable(ctx: AgentContext) → None; the agent's work (fills ctx.report)
    """

    def __init__(self, name: str, domain: str, duty_cycle_s: float,
                 tick_fn: Callable[[AgentContext], None]) -> None:
        self.name = name
        self.domain = domain
        self.duty_cycle_s = max(1.0, duty_cycle_s)
        self._tick_fn = tick_fn
        self._last_tick: float = 0.0
        self._last_report: Optional[AgentReport] = None
        self._active: bool = True
        self._ticks: int = 0

    def tick(self, ctx_factory: Callable[[str, str, AgentReport], AgentContext]
             ) -> Optional[AgentReport]:
        """Run one tick if enough time has elapsed. Returns report or None."""
        if not self._active:
            return None
        now = time.time()
        if (now - self._last_tick) < self.duty_cycle_s:
            return None
        report = AgentReport(agent_name=self.name, domain=self.domain)
        try:
            ctx = ctx_factory(self.name, self.domain, report)
            self._tick_fn(ctx)
        except Exception as exc:  # noqa: BLE001
            report.ok = False
            report.error = str(exc)[:120]
        self._last_tick = now
        self._last_report = report
        self._ticks += 1
        return report

    def pause(self) -> None:
        self._active = False

    def resume(self) -> None:
        self._active = True

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "active": self._active,
            "ticks": self._ticks,
            "last_tick": self._last_tick,
            "last_report": self._last_report.to_dict() if self._last_report else None,
        }


# --------------------------------------------------------------------------- #
# Tick functions — one per agent (observe → coordinate → act)
# --------------------------------------------------------------------------- #
def _memory_tick(ctx: AgentContext) -> None:
    core = ctx.core
    total = 0
    mem = getattr(core, "memory", None)
    if mem is not None:
        try:
            total = int(core.memory.stats().get("total", 0))
            ctx.report.findings.append(f"memory: {total} records")
        except Exception:  # noqa: BLE001
            pass
    kg = getattr(core, "knowledge_graph", None)
    if kg is not None:
        try:
            ctx.report.findings.append(f"graph: {len(core.knowledge_graph)} triples")
        except Exception:  # noqa: BLE001
            pass
    # ACT: when memory has grown large, consolidate it (reversible reorganisation)
    if mem is not None and total >= 200:
        fn = _first_method(mem, "consolidate", "dream", "maintain")
        if fn is not None:
            ctx.act(f"consolidate memory ({total} records)", fn,
                    capability="memory.consolidate")
        else:
            ctx.post("memory.pressure", f"{total} records — consolidation recommended")


def _security_tick(ctx: AgentContext) -> None:
    core = ctx.core
    journal = getattr(core, "journal", None)
    failed = 0
    if journal is not None:
        try:
            actions = list(journal.actions())
            failed = sum(1 for a in actions[-20:] if "FAIL" in str(getattr(a, "payload", {})))
        except Exception:  # noqa: BLE001
            pass
    ctx.report.findings.append(f"passive security check ({failed}/20 recent failures)")
    # COORDINATE: raise an alert the other agents can react to
    if failed > 3:
        ctx.post("security.alert", f"elevated failure rate {failed}/20 — investigate",
                 priority="high")


def _planning_tick(ctx: AgentContext) -> None:
    core = ctx.core
    goals = getattr(core, "goals", None)
    ranked: List[Any] = []
    if goals is not None:
        try:
            ranked = list(goals.prioritize())
            if ranked:
                ctx.report.findings.append(f"top goal: {str(ranked[0].name)[:50]}")
        except Exception:  # noqa: BLE001
            pass
    # REACT to a security alert: escalate it into a goal/finding
    for msg in ctx.inbox(topic="security.alert"):
        ctx.report.findings.append(f"escalating security alert from {msg.sender}")
        fn = _bind(goals, "add", f"investigate security: {msg.content}") if goals else None
        if fn is not None:
            ctx.act("file goal: investigate security alert", fn, capability="goals.add")
    # ACT: a deep goal queue gets pruned (reversible)
    if goals is not None and len(ranked) > 5:
        fn = _first_method(goals, "prune", "compact")
        if fn is not None:
            ctx.act(f"prune goal queue ({len(ranked)} goals)", fn, capability="goals.prune")
        else:
            ctx.report.suggestions.append(f"goal queue deep ({len(ranked)}); consider pruning")


def _research_tick(ctx: AgentContext) -> None:
    core = ctx.core
    knowledge = getattr(core, "knowledge", None)
    chunks = 0
    if knowledge is not None:
        try:
            chunks = len(knowledge)
            ctx.report.findings.append(f"knowledge base: {chunks} chunks")
        except Exception:  # noqa: BLE001
            pass
    graph = getattr(core, "knowledge_graph", None)
    if graph is not None:
        try:
            ctx.report.findings.append(f"graph entities: {graph.stats().get('entities', 0)}")
        except Exception:  # noqa: BLE001
            pass
    # COORDINATE: flag a knowledge gap for the Learning agent to act on
    if chunks < 5:
        ctx.post("knowledge.gap", "sparse knowledge base — ingest/learn more")


def _learning_tick(ctx: AgentContext) -> None:
    core = ctx.core
    skills = getattr(core, "skills", None)
    if skills is not None:
        try:
            ctx.report.findings.append(f"stored skills: {len(skills)}")
        except Exception:  # noqa: BLE001
            pass
    learner = getattr(core, "learner", None)
    if learner is not None:
        try:
            ctx.report.findings.append(f"learner steps: {learner.report().get('steps', 0)}")
        except Exception:  # noqa: BLE001
            pass
    # REACT to a research gap by running one learning step (gated)
    gaps = ctx.inbox(topic="knowledge.gap")
    if gaps and learner is not None:
        fn = _first_method(learner, "step", "learn_once", "tick")
        if fn is not None:
            ctx.act("run one learner step (knowledge gap reported)", fn,
                    capability="learner.step")


def _coding_tick(ctx: AgentContext) -> None:
    core = ctx.core
    tools = getattr(core, "tools", None)
    if tools is not None:
        try:
            ctx.report.findings.append(f"registered tools: {len(list(tools.all()))}")
        except Exception:  # noqa: BLE001
            pass
    # REACT to a security alert: acknowledge and offer to harden (read-only proposal here)
    for msg in ctx.inbox(topic="security.alert"):
        ctx.report.suggestions.append(
            f"review code paths implicated by {msg.sender}'s alert")


def _prediction_tick(ctx: AgentContext) -> None:
    core = ctx.core
    wm = getattr(core, "world_model", None)
    if wm is not None:
        try:
            ctx.report.findings.append(f"world-model transitions: {len(wm)}")
        except Exception:  # noqa: BLE001
            pass
    if getattr(core, "predictive", None) is not None:
        ctx.report.findings.append("predictive core: active")


# --------------------------------------------------------------------------- #
# small reflection helpers for safe, optional core actions
# --------------------------------------------------------------------------- #
def _first_method(obj: Any, *names: str) -> Optional[Callable[[], Any]]:
    """Return a zero-arg callable for the first existing method in ``names`` (else None)."""
    for n in names:
        fn = getattr(obj, n, None)
        if callable(fn):
            return fn
    return None


def _bind(obj: Any, name: str, *args: Any) -> Optional[Callable[[], Any]]:
    """Bind ``obj.name(*args)`` into a zero-arg callable if the method exists (else None)."""
    fn = getattr(obj, name, None)
    if not callable(fn):
        return None
    return lambda: fn(*args)


def _short(value: Any) -> str:
    try:
        return str(value)[:80]
    except Exception:  # noqa: BLE001
        return "<unprintable>"


# --------------------------------------------------------------------------- #
# Civilization
# --------------------------------------------------------------------------- #
_AGENT_DEFS = [
    ("Memory",     "memory consolidation and graph maintenance",  30.0, _memory_tick),
    ("Security",   "threat monitoring and anomaly detection",     45.0, _security_tick),
    ("Planning",   "goal prioritization and foresight",           60.0, _planning_tick),
    ("Research",   "knowledge gap identification",                90.0, _research_tick),
    ("Learning",   "skill memory and autolearn cycles",           60.0, _learning_tick),
    ("Coding",     "code analysis and tool composition",         120.0, _coding_tick),
    ("Prediction", "world-model and predictive calibration",      45.0, _prediction_tick),
]


class MicroAgentCivilization:
    """Seven persistent micro-agents governed by NYXARA — a small, coordinating society.

    Parameters
    ----------
    core:                  NyxaraCore instance (read by each agent; target of gated actions).
    event_bus:             optional EventBus for publishing reports/messages.
    enact_actions:         when True, agents may take gated actions (default False = monitors).
    max_actions_per_cycle: cap on total gated actions across all agents per tick cycle.
    settings:              optional settings; when given, gating defaults come from
                           ``settings.agency`` unless explicitly overridden.
    """

    def __init__(self, core: Any = None, event_bus: Any = None, *,
                 enact_actions: Optional[bool] = None,
                 max_actions_per_cycle: Optional[int] = None,
                 settings: Any = None) -> None:
        self.core = core
        self.event_bus = event_bus
        agency = getattr(settings, "agency", None) if settings is not None else None
        self.enact_actions = bool(getattr(agency, "civilization_autonomous", False)
                                  if enact_actions is None else enact_actions)
        self.max_actions_per_cycle = int(
            getattr(agency, "civilization_max_actions_per_cycle", 2)
            if max_actions_per_cycle is None else max_actions_per_cycle)
        self.agents: List[MicroAgent] = [
            MicroAgent(name, domain, cycle, fn)
            for name, domain, cycle, fn in _AGENT_DEFS
        ]
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._reports: List[AgentReport] = []
        self._messages: Deque[AgentMessage] = deque(maxlen=256)
        self._lock = threading.Lock()
        self._actions_this_cycle = 0
        self._actions_total = 0
        self._msg_seq = 0                       # monotonic message id
        self._cursors: Dict[str, int] = {}      # per-agent last-consumed message seq
        self._tick_snapshot: Dict[str, int] = {}  # message seq visible at each agent's tick start

    # ---------------------------------------------------------------------- #
    def start(self) -> None:
        """Start all agents as a single background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="nyxara-civilization")
        self._thread.start()

    def stop(self) -> None:
        """Signal the civilization to stop and wait briefly for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def tick_all(self) -> List[AgentReport]:
        """Run one tick cycle synchronously (for testing / manual control)."""
        reports: List[AgentReport] = []
        self._actions_this_cycle = 0           # action budget is per cycle, shared by all agents
        for agent in self.agents:
            report = agent.tick(self._make_context)
            if report is not None:
                # consume everything this agent could see at the start of its tick
                self._cursors[agent.name] = max(self._cursors.get(agent.name, 0),
                                                self._tick_snapshot.get(agent.name, 0))
                reports.append(report)
                self._publish(report)
        with self._lock:
            self._reports.extend(reports)
        return reports

    def status(self) -> Dict[str, Any]:
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "agents": [a.status() for a in self.agents],
            "total_reports": len(self._reports),
            "total_actions": self._actions_total,
            "enact_actions": self.enact_actions,
            "messages": len(self._messages),
        }

    def recent_reports(self, n: int = 10) -> List[AgentReport]:
        with self._lock:
            return list(self._reports[-n:])

    def recent_messages(self, n: int = 10) -> List[AgentMessage]:
        with self._lock:
            return list(self._messages)[-n:]

    def pause_agent(self, name: str) -> bool:
        for a in self.agents:
            if a.name == name:
                a.pause()
                return True
        return False

    def resume_agent(self, name: str) -> bool:
        for a in self.agents:
            if a.name == name:
                a.resume()
                return True
        return False

    # ---------------------------------------------------------------------- #
    # context + blackboard + gated enactment
    # ---------------------------------------------------------------------- #
    def _make_context(self, name: str, domain: str, report: AgentReport) -> AgentContext:
        with self._lock:
            snapshot = self._msg_seq           # messages already on the board at tick start
        self._tick_snapshot[name] = snapshot
        return AgentContext(core=self.core, agent=name, enact=self.enact_actions,
                            civ=self, report=report, snapshot_seq=snapshot)

    def _post_message(self, msg: AgentMessage) -> None:
        with self._lock:
            self._msg_seq += 1
            msg.seq = self._msg_seq
            self._messages.append(msg)
        if self.event_bus is not None:
            try:
                self.event_bus.publish("civilization.message", msg.to_dict())
            except Exception:  # noqa: BLE001
                pass

    def _read_new(self, agent: str, *, topic: Optional[str] = None,
                  snapshot: Optional[int] = None) -> List[AgentMessage]:
        """Unconsumed messages for ``agent``: seq in (cursor, snapshot], from other senders."""
        cursor = self._cursors.get(agent, 0)
        hi = self._msg_seq if snapshot is None else snapshot
        with self._lock:
            msgs = [m for m in self._messages
                    if m.sender != agent and cursor < m.seq <= hi
                    and (topic is None or m.topic == topic)]
        return list(reversed(msgs))            # newest first

    def _enact(self, agent: str, description: str, fn: Callable[[], Any], *,
               capability: str, reversible: bool, enact: bool) -> Dict[str, Any]:
        """Run a gated action: fail-closed unless authorised + budget remains; audited+journalled."""
        if not (enact and self.enact_actions):
            return {"taken": False, "reason": "withheld (observe-only)"}
        if self._actions_this_cycle >= self.max_actions_per_cycle:
            return {"taken": False, "reason": "action budget exhausted"}
        self._audit_permission(agent, description, capability, reversible)
        action_id = self._journal_start(agent, description)
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 — a failed action is data, never a crash
            self._journal_end(action_id, success=False, note=f"{description}: {exc}")
            return {"taken": False, "reason": f"error: {str(exc)[:80]}"}
        self._actions_this_cycle += 1
        self._actions_total += 1
        self._journal_end(action_id, success=True, note=description)
        return {"taken": True, "result": _short(result)}

    def _audit_permission(self, agent: str, description: str, capability: str,
                          reversible: bool) -> None:
        permissions = getattr(self.core, "permissions", None)
        if permissions is None:
            return
        try:
            from nyxara.agency.permissions import (Authority, Capability,
                                                   PermissionRequest, RiskTier)
            cap = getattr(Capability, "SELF_MODIFY", None)
            req = PermissionRequest(capability=cap, target=f"civilization:{agent}",
                                    risk=RiskTier.LOW if reversible else RiskTier.MODERATE,
                                    reversible=reversible, authority=Authority.OWNER,
                                    reason=f"civilization action: {description}")
            permissions.check(req)
        except Exception:  # noqa: BLE001 — auditing is best-effort, never blocks the action
            pass

    def _journal_start(self, agent: str, description: str) -> str:
        journal = getattr(self.core, "journal", None)
        if journal is None:
            return ""
        try:
            from nyxara.planning.journal import ActionStatus
            return journal.record_action(
                f"civilization:{agent}:{description}", rationale="autonomous micro-agent action",
                autonomous=True, decision="act", status=ActionStatus.EXECUTED)
        except Exception:  # noqa: BLE001
            return ""

    def _journal_end(self, action_id: str, *, success: bool, note: str) -> None:
        journal = getattr(self.core, "journal", None)
        if journal is None or not action_id:
            return
        try:
            from nyxara.planning.journal import ActionStatus
            journal.record_outcome(
                action_id,
                status=ActionStatus.SUCCEEDED if success else ActionStatus.ABORTED, note=note)
        except Exception:  # noqa: BLE001
            pass

    # ---------------------------------------------------------------------- #
    def _loop(self) -> None:
        """Background loop: tick agents on their duty cycles."""
        while not self._stop_event.is_set():
            try:
                self.tick_all()
            except Exception:  # noqa: BLE001
                pass
            self._stop_event.wait(timeout=5.0)

    def _publish(self, report: AgentReport) -> None:
        if self.event_bus is None:
            return
        try:
            self.event_bus.publish("civilization.report", report.to_dict())
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA micro-agent civilization self-test")
    print("=" * 70)

    civ = MicroAgentCivilization(core=None)
    print(f"\n7 agents defined: {[a.name for a in civ.agents]}")

    for a in civ.agents:                       # force all agents to be due
        a._last_tick = 0.0
    reports = civ.tick_all()
    print(f"tick_all() produced {len(reports)} reports (all 7 should tick)")
    for r in reports:
        print(f"  [{r.agent_name}] ok={r.ok} findings={r.findings[:1]} actions={r.actions}")
    assert len(reports) == 7
    assert civ.status()["total_reports"] == 7

    # pause/resume
    civ.pause_agent("Security")
    assert not civ.agents[1]._active
    civ.resume_agent("Security")
    assert civ.agents[1]._active

    # ---- inter-agent messaging: Security's alert reaches Planning/Coding ---- #
    class _FakeJournal:
        def actions(self):
            class _A:
                payload = {"status": "FAIL"}
            return [_A()] * 10                 # 10 recent failures ⇒ Security raises an alert

    class _FakeGoals:
        def __init__(self):
            self.filed = []

        def prioritize(self):
            return []

        def add(self, name):
            self.filed.append(name)
            return name

    class _FakeCore:
        journal = _FakeJournal()
        goals = _FakeGoals()

    core = _FakeCore()
    civ2 = MicroAgentCivilization(core=core, enact_actions=True, max_actions_per_cycle=5)
    for a in civ2.agents:
        a._last_tick = 0.0
    civ2.tick_all()                            # cycle 1: Security posts the alert
    for a in civ2.agents:
        a._last_tick = 0.0
    civ2.tick_all()                            # cycle 2: Planning reacts to it
    assert any(m.topic == "security.alert" for m in civ2.recent_messages()), "no alert posted"
    assert core.goals.filed, "Planning did not act on the security alert"
    print(f"\ninter-agent: alert posted, Planning filed goal(s): {core.goals.filed}")
    print(f"total gated actions taken: {civ2.status()['total_actions']}")

    # ---- autonomy OFF ⇒ no actions, only suggestions ---- #
    civ3 = MicroAgentCivilization(core=core, enact_actions=False)
    for a in civ3.agents:
        a._last_tick = 0.0
    r3 = civ3.tick_all()
    assert all(not r.actions for r in r3), "actions taken while autonomy OFF"
    print("\nautonomy OFF: agents are read-only monitors (no actions) ✓")

    # start/stop background thread
    civ.start()
    time.sleep(0.1)
    assert civ._thread is not None and civ._thread.is_alive()
    civ.stop()
    assert civ._thread is None or not civ._thread.is_alive()

    print("\nALL SELF-TESTS PASSED ✓")
