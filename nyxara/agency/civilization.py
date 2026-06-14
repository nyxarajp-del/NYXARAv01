"""NYXARA · agency/civilization.py — Multi-Agent Internal Civilization (Level 9).

Seven persistent specialized micro-agents run as background workers. NYXARA is the
governor. Each agent has a specific domain and duty cycle. Agents report via the
EventBus (topic 'civilization.report'). NYXARA can pause/terminate any agent.

The 7 micro-agents:
    Memory      — memory consolidation and graph maintenance
    Security    — passive threat monitoring and anomaly detection
    Planning    — goal prioritization and foresight
    Research    — knowledge gap identification and internal research
    Learning    — skill memory maintenance and autolearn cycles
    Coding      — code analysis and toolsmith composition
    Prediction  — world-model updates and predictive core calibration

Each agent's tick() is isolated: it reads from the core (read-only), queues a report,
and never writes directly to the world — all real actions still go through the gate.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = ["MicroAgent", "AgentReport", "MicroAgentCivilization"]


# --------------------------------------------------------------------------- #
# AgentReport
# --------------------------------------------------------------------------- #
@dataclass
class AgentReport:
    """One tick outcome from a MicroAgent."""
    agent_name: str
    domain: str
    timestamp: float = field(default_factory=time.time)
    findings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    ok: bool = True
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "domain": self.domain,
            "timestamp": self.timestamp,
            "findings": self.findings,
            "suggestions": self.suggestions,
            "ok": self.ok,
            "error": self.error,
        }


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
    tick_fn:      callable(core) → AgentReport; the agent's actual work
    """

    def __init__(self, name: str, domain: str, duty_cycle_s: float,
                 tick_fn: Callable[[Any], AgentReport]) -> None:
        self.name = name
        self.domain = domain
        self.duty_cycle_s = max(1.0, duty_cycle_s)
        self._tick_fn = tick_fn
        self._last_tick: float = 0.0
        self._last_report: Optional[AgentReport] = None
        self._active: bool = True
        self._ticks: int = 0

    def tick(self, core: Any) -> Optional[AgentReport]:
        """Run one tick if enough time has elapsed. Returns report or None."""
        if not self._active:
            return None
        now = time.time()
        if (now - self._last_tick) < self.duty_cycle_s:
            return None
        try:
            report = self._tick_fn(core)
        except Exception as exc:  # noqa: BLE001
            report = AgentReport(agent_name=self.name, domain=self.domain,
                                 ok=False, error=str(exc)[:120])
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
# Tick functions — one per agent
# --------------------------------------------------------------------------- #

def _memory_tick(core: Any) -> AgentReport:
    findings: List[str] = []
    if getattr(core, "memory", None) is not None:
        try:
            stats = core.memory.stats()
            findings.append(f"memory: {stats.get('total', 0)} records")
        except Exception:  # noqa: BLE001
            pass
    if getattr(core, "knowledge_graph", None) is not None:
        try:
            findings.append(f"graph: {len(core.knowledge_graph)} triples")
        except Exception:  # noqa: BLE001
            pass
    return AgentReport(agent_name="Memory", domain="memory", findings=findings)


def _security_tick(core: Any) -> AgentReport:
    findings: List[str] = []
    suggestions: List[str] = []
    try:
        journal = getattr(core, "journal", None)
        if journal is not None:
            actions = list(journal.actions())
            failed = [a for a in actions[-20:] if "FAIL" in str(
                getattr(a, "payload", {}))]
            if len(failed) > 3:
                suggestions.append(f"elevated failure rate: {len(failed)}/20 recent")
        findings.append("passive security check complete")
    except Exception:  # noqa: BLE001
        pass
    return AgentReport(agent_name="Security", domain="security",
                       findings=findings, suggestions=suggestions)


def _planning_tick(core: Any) -> AgentReport:
    findings: List[str] = []
    suggestions: List[str] = []
    try:
        goals = getattr(core, "goals", None)
        if goals is not None:
            ranked = goals.prioritize()
            if ranked:
                findings.append(f"top goal: {ranked[0].name[:50]}")
                if len(ranked) > 5:
                    suggestions.append(f"goal queue deep ({len(ranked)}); consider pruning")
    except Exception:  # noqa: BLE001
        pass
    return AgentReport(agent_name="Planning", domain="planning",
                       findings=findings, suggestions=suggestions)


def _research_tick(core: Any) -> AgentReport:
    findings: List[str] = []
    try:
        knowledge = getattr(core, "knowledge", None)
        if knowledge is not None:
            findings.append(f"knowledge base: {len(knowledge)} chunks")
        graph = getattr(core, "knowledge_graph", None)
        if graph is not None:
            stats = graph.stats()
            findings.append(f"graph entities: {stats.get('entities', 0)}")
    except Exception:  # noqa: BLE001
        pass
    return AgentReport(agent_name="Research", domain="research", findings=findings)


def _learning_tick(core: Any) -> AgentReport:
    findings: List[str] = []
    try:
        skills = getattr(core, "skills", None)
        if skills is not None:
            findings.append(f"stored skills: {len(skills)}")
        learner = getattr(core, "learner", None)
        if learner is not None:
            report = learner.report()
            findings.append(f"learner steps: {report.get('steps', 0)}")
    except Exception:  # noqa: BLE001
        pass
    return AgentReport(agent_name="Learning", domain="learning", findings=findings)


def _coding_tick(core: Any) -> AgentReport:
    findings: List[str] = []
    try:
        tools = getattr(core, "tools", None)
        if tools is not None:
            all_tools = list(tools.all())
            findings.append(f"registered tools: {len(all_tools)}")
    except Exception:  # noqa: BLE001
        pass
    return AgentReport(agent_name="Coding", domain="coding", findings=findings)


def _prediction_tick(core: Any) -> AgentReport:
    findings: List[str] = []
    try:
        wm = getattr(core, "world_model", None)
        if wm is not None:
            findings.append(f"world-model transitions: {len(wm)}")
        pred = getattr(core, "predictive", None)
        if pred is not None:
            findings.append("predictive core: active")
    except Exception:  # noqa: BLE001
        pass
    return AgentReport(agent_name="Prediction", domain="prediction", findings=findings)


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
    """Seven persistent micro-agents governed by NYXARA.

    Each agent ticks on its own duty cycle; reports flow back to the governor
    (NyxaraCore) via the stored report list and optionally the EventBus.

    Parameters
    ----------
    core:           NyxaraCore instance (passed to each agent tick).
    event_bus:      optional EventBus for publishing reports.
    """

    def __init__(self, core: Any = None, event_bus: Any = None) -> None:
        self.core = core
        self.event_bus = event_bus
        self.agents: List[MicroAgent] = [
            MicroAgent(name, domain, cycle, fn)
            for name, domain, cycle, fn in _AGENT_DEFS
        ]
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._reports: List[AgentReport] = []
        self._lock = threading.Lock()

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
        core = self.core
        for agent in self.agents:
            report = agent.tick(core)
            if report is not None:
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
        }

    def recent_reports(self, n: int = 10) -> List[AgentReport]:
        with self._lock:
            return list(self._reports[-n:])

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
    def _loop(self) -> None:
        """Background loop: tick agents on their duty cycles."""
        while not self._stop_event.is_set():
            try:
                reports = self.tick_all()
                _ = reports  # reports already published in tick_all
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

    # force all agents to be due (set last_tick to 0)
    for a in civ.agents:
        a._last_tick = 0.0

    reports = civ.tick_all()
    print(f"tick_all() produced {len(reports)} reports (all 7 should tick)")
    for r in reports:
        print(f"  [{r.agent_name}] ok={r.ok} findings={r.findings[:1]}")
    assert len(reports) == 7

    st = civ.status()
    print(f"\nstatus: running={st['running']}, total_reports={st['total_reports']}")
    assert st["total_reports"] == 7

    # pause/resume
    civ.pause_agent("Security")
    assert not civ.agents[1]._active
    civ.resume_agent("Security")
    assert civ.agents[1]._active

    # start/stop background thread
    civ.start()
    time.sleep(0.1)
    assert civ._thread is not None and civ._thread.is_alive()
    civ.stop()
    assert civ._thread is None or not civ._thread.is_alive()

    print("\nALL SELF-TESTS PASSED ✓")
