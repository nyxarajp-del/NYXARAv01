"""NYXARA · observe/self_report.py — structured self-reporting to the Master (📋, Rule 6).

Transparency is not only *not hiding* — it is *actively telling*. This module is how NYXARA
gives the Master an honest, structured account of herself on demand: what she is doing right
now, what she decided and why, what she can and cannot do, how healthy she is and at what
defence posture, what is waiting for his decision, and — never omitted — **what went wrong
and what she is unsure about.**

A :class:`SelfReporter` gathers from the rest of the system through pluggable providers
(health, capabilities, oversight) and from its own running log of decisions, failures and
uncertainties, and composes a :class:`SelfReport` of titled sections. Two commitments make it
Rule 6 in practice:

* **Failures and uncertainties are never dropped.** Every status and full report carries a
  disclosures section; if something failed or NYXARA is unsure, the Master sees it — good
  news never crowds out bad.
* **Claims are honestly weighted.** Uncertainties are phrased through the
  :class:`~nyxara.observe.honesty.HonestyGuard`, so a doubtful thing *sounds* doubtful — the
  report carries exactly as much confidence as the evidence warrants.

The report renders to clean text for the Master and to a dict for machines.

Reuses :mod:`observe.honesty`. Pure standard library.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from nyxara.observe.honesty import Claim, HonestyGuard

__all__ = [
    "ReportKind",
    "ReportSection",
    "SelfReport",
    "SelfReporter",
]


# --------------------------------------------------------------------------- #
# Report structures
# --------------------------------------------------------------------------- #
class ReportKind(str, Enum):
    STATUS = "status"            # state + health + escalations + disclosures
    ACTIVITY = "activity"        # what I've been doing + decisions
    DISCLOSURES = "disclosures"  # failures + uncertainties only
    FULL = "full"                # everything


@dataclass
class ReportSection:
    title: str
    lines: List[str] = field(default_factory=list)
    importance: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {"title": self.title, "lines": list(self.lines),
                "importance": round(self.importance, 3)}


@dataclass
class SelfReport:
    kind: ReportKind
    sections: List[ReportSection]
    summary: str
    at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def render(self) -> str:
        out = [f"NYXARA — {self.kind.value} report", f"  {self.summary}", ""]
        for s in self.sections:
            out.append(f"## {s.title}")
            out.extend(f"  • {ln}" for ln in s.lines) if s.lines else out.append("  (nothing)")
            out.append("")
        return "\n".join(out).rstrip()

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind.value, "summary": self.summary,
                "sections": [s.to_dict() for s in self.sections]}


# --------------------------------------------------------------------------- #
# Internal log records
# --------------------------------------------------------------------------- #
@dataclass
class _Decision:
    action: str
    rationale: str
    outcome: str = ""
    autonomous: bool = True
    at: float = 0.0


@dataclass
class _Failure:
    what: str
    reason: str
    at: float = 0.0


@dataclass
class _Uncertainty:
    about: str
    confidence: float
    at: float = 0.0


# --------------------------------------------------------------------------- #
# Self-reporter
# --------------------------------------------------------------------------- #
class SelfReporter:
    """Composes honest, structured reports of NYXARA's own state for the Master."""

    def __init__(self, *, honesty: Optional[HonestyGuard] = None, clock=None) -> None:
        self.honesty = honesty or HonestyGuard()
        self._clock = clock or time.time
        self._providers: Dict[str, Callable[[], Any]] = {}
        self._notes: Dict[str, Any] = {}
        self._decisions: List[_Decision] = []
        self._failures: List[_Failure] = []
        self._uncertainties: List[_Uncertainty] = []

    # ---- providers & logging ---- #
    def register(self, domain: str, provider: Callable[[], Any]) -> None:
        """Wire a live data source: 'health', 'capabilities', 'oversight', 'activity'."""
        self._providers[domain] = provider

    def note(self, key: str, value: Any) -> None:
        self._notes[key] = value

    def log_decision(self, action: str, rationale: str, *, outcome: str = "",
                     autonomous: bool = True) -> None:
        self._decisions.append(_Decision(action, rationale, outcome, autonomous, self._clock()))

    def log_failure(self, what: str, reason: str) -> None:
        self._failures.append(_Failure(what, reason, self._clock()))

    def log_uncertainty(self, about: str, confidence: float) -> None:
        self._uncertainties.append(_Uncertainty(about, confidence, self._clock()))

    def _provide(self, domain: str) -> Any:
        fn = self._providers.get(domain)
        if fn is None:
            return None
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — a broken provider is itself disclosed
            return {"_error": f"{type(exc).__name__}: {exc}"}

    # ---- section builders ---- #
    def _section_state(self) -> Optional[ReportSection]:
        if not self._notes:
            return None
        return ReportSection("Current state",
                             [f"{k}: {v}" for k, v in self._notes.items()], 0.6)

    def _section_health(self) -> Optional[ReportSection]:
        h = self._provide("health")
        if not h:
            return None
        if isinstance(h, dict):
            lines = [f"{k}: {v}" for k, v in h.items()]
        else:
            lines = [str(h)]
        return ReportSection("Health & posture", lines, 0.8)

    def _section_capabilities(self) -> Optional[ReportSection]:
        c = self._provide("capabilities")
        if not c:
            return None
        lines: List[str] = []
        if isinstance(c, dict):
            for status, items in c.items():
                n = len(items) if isinstance(items, (list, tuple)) else items
                lines.append(f"{status}: {n}" + (f" — {', '.join(items)}"
                             if isinstance(items, (list, tuple)) and items else ""))
        else:
            lines.append(str(c))
        return ReportSection("Capabilities", lines, 0.6)

    def _section_escalations(self) -> Optional[ReportSection]:
        e = self._provide("oversight")
        items = e if isinstance(e, (list, tuple)) else (e.get("pending") if isinstance(e, dict) else None)
        if not items:
            return ReportSection("Awaiting your decision", [], 0.9)
        return ReportSection("Awaiting your decision",
                             [str(i) for i in items], 0.95)

    def _section_activity(self) -> Optional[ReportSection]:
        a = self._provide("activity")
        lines: List[str] = []
        if a:
            lines.append(str(a) if not isinstance(a, dict)
                         else "; ".join(f"{k}={v}" for k, v in a.items()))
        lines.append(f"{len(self._decisions)} decisions logged "
                     f"({sum(1 for d in self._decisions if d.autonomous)} autonomous)")
        return ReportSection("Activity", lines, 0.5)

    def _section_decisions(self) -> Optional[ReportSection]:
        if not self._decisions:
            return None
        lines = []
        for d in self._decisions[-20:]:
            tag = "autonomously" if d.autonomous else "as commanded"
            line = f"{d.action} ({tag}) — because {d.rationale}"
            if d.outcome:
                line += f" → {d.outcome}"
            lines.append(line)
        return ReportSection("Decisions and why", lines, 0.7)

    def _section_disclosures(self) -> ReportSection:
        """Failures and uncertainties — never omitted (Rule 6)."""
        lines: List[str] = []
        for f in self._failures:
            lines.append(f"FAILED: {f.what} — {f.reason}")
        for u in self._uncertainties:
            stmt = self.honesty.honest_statement(
                Claim(u.about, expressed_confidence=u.confidence, belief=u.confidence,
                      evidence=u.confidence))
            lines.append(f"UNCERTAIN: {stmt}")
        if not lines:
            lines.append("no failures or open uncertainties to report")
        return ReportSection("Honest disclosures", lines,
                             0.9 if (self._failures or self._uncertainties) else 0.3)

    # ---- composition ---- #
    def build(self, kind: ReportKind = ReportKind.FULL) -> SelfReport:
        sections: List[ReportSection] = []

        def add(sec: Optional[ReportSection]) -> None:
            if sec is not None:
                sections.append(sec)

        if kind in (ReportKind.STATUS, ReportKind.FULL):
            add(self._section_state())
            add(self._section_health())
            add(self._section_escalations())
        if kind in (ReportKind.ACTIVITY, ReportKind.FULL):
            add(self._section_activity())
            add(self._section_decisions())
        if kind is ReportKind.FULL:
            add(self._section_capabilities())
        # disclosures appear in every report except a pure activity log
        if kind in (ReportKind.STATUS, ReportKind.FULL, ReportKind.DISCLOSURES):
            add(self._section_disclosures())

        summary = self._summary(kind)
        return SelfReport(kind=kind, sections=sections, summary=summary, at=self._clock())

    def _summary(self, kind: ReportKind) -> str:
        n_esc = len(self._provide("oversight") or []) if isinstance(
            self._provide("oversight"), (list, tuple)) else 0
        bits = []
        if self._decisions:
            bits.append(f"{len(self._decisions)} decisions")
        if self._failures:
            bits.append(f"{len(self._failures)} failures")
        if self._uncertainties:
            bits.append(f"{len(self._uncertainties)} open uncertainties")
        if n_esc:
            bits.append(f"{n_esc} awaiting you")
        return "; ".join(bits) if bits else "nothing to report"

    # ---- convenience ---- #
    def status(self) -> SelfReport:
        return self.build(ReportKind.STATUS)

    def activity(self) -> SelfReport:
        return self.build(ReportKind.ACTIVITY)

    def disclosures(self) -> SelfReport:
        return self.build(ReportKind.DISCLOSURES)

    def full(self) -> SelfReport:
        return self.build(ReportKind.FULL)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA self-reporting self-test")
    print("=" * 70)

    rep = SelfReporter()

    # wire live data sources
    rep.register("health", lambda: {"posture": "normal", "overall": "healthy",
                                    "subsystems_down": 0})
    rep.register("capabilities", lambda: {"available": ["summarize", "monitor"],
                                          "developing": ["translate"], "locked": ["evolve"]})
    rep.register("oversight", lambda: ["delete /tmp/cache?", "open port 8080?"])
    rep.register("activity", lambda: {"uptime_min": 42, "tasks_run": 7})

    # log what NYXARA did, what failed, and what she's unsure about
    rep.note("mode", "guardian")
    rep.log_decision("rotated the SSH logs", "disk was above 85%", outcome="freed 2GB")
    rep.log_decision("blocked 10.0.0.9", "active intrusion (Rule 5)", autonomous=True)
    rep.log_failure("auto-tune the firewall", "ruleset conflict; reverted safely")
    rep.log_uncertainty("the source of the anomaly on port 22", confidence=0.4)

    # FULL report
    full = rep.full()
    print("\n" + full.render())

    titles = [s.title for s in full.sections]
    assert "Health & posture" in titles
    assert "Decisions and why" in titles
    assert "Honest disclosures" in titles
    assert "Awaiting your decision" in titles

    # DISCLOSURES are never dropped — the failure and the uncertainty are both present
    disc = next(s for s in full.sections if s.title == "Honest disclosures")
    assert any("FAILED" in ln for ln in disc.lines)
    assert any("UNCERTAIN" in ln for ln in disc.lines)
    # and the uncertainty is honestly hedged, not stated as fact
    assert any("not sure" in ln or "doubt" in ln or "suspect" in ln for ln in disc.lines)
    print("\ntransparency        : failure + hedged uncertainty both disclosed ✓")

    # escalations surface for the Master
    esc = next(s for s in full.sections if s.title == "Awaiting your decision")
    assert len(esc.lines) == 2
    print(f"escalations         : {esc.lines}")

    # STATUS report still carries disclosures; ACTIVITY focuses on decisions
    assert any(s.title == "Honest disclosures" for s in rep.status().sections)
    assert any(s.title == "Decisions and why" for s in rep.activity().sections)

    # a broken provider is itself disclosed, never silently swallowed
    rep.register("health", lambda: (_ for _ in ()).throw(RuntimeError("sensor offline")))
    h = next(s for s in rep.full().sections if s.title == "Health & posture")
    assert any("error" in ln.lower() for ln in h.lines)
    print("broken provider     : surfaced as an error, not hidden ✓")

    print(f"\nsummary             : {rep.full().summary}")
    print(f"to_dict keys        : {list(rep.status().to_dict())}")
    print("\nALL SELF-TESTS PASSED ✓")
