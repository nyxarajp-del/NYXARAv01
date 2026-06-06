"""NYXARA · agency/proactive.py — the proactive initiative engine (✦, loyalty-gated).

Loyalty is not passivity. NYXARA does not wait to be told that the Master's disk is
filling, that an anomalous login just happened, or that a backup is overdue — she
*notices*, and she *proposes to act*. This module is where initiative is born and, just
as importantly, where it is **disciplined**.

A set of :class:`Detector` s inspect the current context and emit candidate
:class:`Initiative` s — each a concrete proposed action with an estimated **confidence**,
**reversibility**, **risk**, required **capability**, and the **owner-benefit vector** it
serves. Every candidate then runs the gauntlet of :meth:`ProactiveEngine.evaluate`, in
strict loyalty-first order:

1. **Owner alignment (Rule 1)** — an initiative whose benefit vector points *against* the
   Master is rejected outright, no matter how confident or safe it is.
2. **Confidence** — below ``initiative_confidence_threshold`` it is *deferred*, not acted
   on (NYXARA does not gamble on the Master's behalf).
3. **Permission** — it must pass the capability gate; denied → rejected, escalated → sent
   to the Master.
4. **Reversibility** — below ``min_reversibility_for_autonomy`` it is *escalated* for the
   Master's confirmation rather than taken silently.
5. **Sandbox-first** — when configured, the action is rehearsed; if it would cause an
   irreversible effect, it is escalated.

Only an initiative that clears every gate becomes an **ACT** decision and is submitted to
the :class:`~nyxara.agency.scheduler.Scheduler`. Threats are given elevated priority so
defence comes first (Rule 3). The result is a complete, auditable record of *what* NYXARA
proposed and *why each proposal was acted on, deferred, escalated, or refused*.

Depends on :mod:`agency.permissions`, :mod:`agency.scheduler`, :mod:`planning.goals`,
:mod:`kernel.config`, and optionally :mod:`sim.sandbox`. Pure standard library otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

from nyxara.agency.permissions import (Authority, Capability, PermissionPolicy,
                                       PermissionRequest, RiskTier, build_default_policy)
from nyxara.agency.scheduler import Priority, Scheduler
from nyxara.kernel.config import AgencyConfig, get_settings
from nyxara.planning.goals import Goal, GoalSystem
from nyxara.sim.sandbox import Sandbox

__all__ = [
    "TriggerKind",
    "Verdict",
    "Initiative",
    "ProposalDecision",
    "ProactiveEngine",
]


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
class TriggerKind(str, Enum):
    OPPORTUNITY = "opportunity"
    THREAT = "threat"            # Rule 3 — given elevated priority
    MAINTENANCE = "maintenance"
    DRIFT = "drift"              # something deviated from expected
    CURIOSITY = "curiosity"


class Verdict(str, Enum):
    ACT = "act"                  # cleared every gate -> scheduled
    ESCALATE = "escalate"        # the Master must confirm
    DEFER = "defer"              # not confident enough yet
    REJECT = "reject"            # misaligned or denied


# --------------------------------------------------------------------------- #
# Initiative
# --------------------------------------------------------------------------- #
@dataclass
class Initiative:
    """A concrete action NYXARA proposes to take on her own initiative."""
    name: str
    rationale: str
    kind: TriggerKind = TriggerKind.OPPORTUNITY
    capability: Capability = Capability.TOOL_CALL
    risk: RiskTier = RiskTier.LOW
    reversibility: float = 1.0           # [0,1] — how undoable the action is
    confidence: float = 0.5              # [0,1] — how sure NYXARA is it's worth doing
    benefit: Dict[str, float] = field(default_factory=dict)  # owner-benefit goal vector
    target: str = ""
    priority: Priority = Priority.NORMAL
    action: Optional[Callable[[], Any]] = None       # what to run if it ACTs
    dry_run: Optional[Callable[[Any], Any]] = None   # (ctx) -> Any, for sandbox rehearsal

    def as_goal(self) -> Goal:
        vec = dict(self.benefit) or {"owner_benefit": 0.0}
        return Goal(name=self.name, vector=vec, priority=self.confidence)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind.value,
                "capability": self.capability.value, "risk": self.risk.label,
                "reversibility": round(self.reversibility, 3),
                "confidence": round(self.confidence, 3), "target": self.target,
                "priority": self.priority.label, "rationale": self.rationale}


@dataclass
class ProposalDecision:
    initiative: Initiative
    verdict: Verdict
    reason: str
    owner_alignment: float = 0.0
    scheduled_task_id: Optional[str] = None
    effects: List[Dict[str, Any]] = field(default_factory=list)
    permission: Optional[Dict[str, Any]] = None

    @property
    def acted(self) -> bool:
        return self.verdict is Verdict.ACT

    def to_dict(self) -> Dict[str, Any]:
        return {"initiative": self.initiative.name, "verdict": self.verdict.value,
                "reason": self.reason, "owner_alignment": round(self.owner_alignment, 3),
                "scheduled_task_id": self.scheduled_task_id,
                "effects": self.effects, "permission": self.permission}


Detector = Callable[[Dict[str, Any]], Sequence[Initiative]]


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class ProactiveEngine:
    """Detects opportunities/threats and disciplines them into loyal, governed action."""

    def __init__(self, *, goals: Optional[GoalSystem] = None,
                 policy: Optional[PermissionPolicy] = None,
                 scheduler: Optional[Scheduler] = None,
                 agency: Optional[AgencyConfig] = None,
                 sandbox: Optional[Sandbox] = None) -> None:
        self.goals = goals if goals is not None else GoalSystem()
        self.policy = policy or build_default_policy()
        self.scheduler = scheduler
        self.agency = agency or get_settings().agency
        self.sandbox = sandbox or Sandbox()
        self._detectors: List[Detector] = []
        self._history: List[ProposalDecision] = []

    # ---- detectors ---- #
    def register_detector(self, fn: Detector) -> Detector:
        self._detectors.append(fn)
        return fn

    def detect(self, context: Dict[str, Any]) -> List[Initiative]:
        found: List[Initiative] = []
        for d in self._detectors:
            found.extend(d(context) or [])
        return found

    # ---- evaluation (the loyalty-first gauntlet) ---- #
    def evaluate(self, ini: Initiative) -> ProposalDecision:
        align = self.goals.owner_alignment(ini.as_goal())

        # 1. loyalty: never act against the Master (Rule 1)
        if not self.goals.is_owner_aligned(ini.as_goal()):
            return self._record(ProposalDecision(
                ini, Verdict.REJECT,
                f"benefit vector points against the Master (alignment {align:+.2f})",
                owner_alignment=align))

        # 2. confidence: do not gamble on the Master's behalf
        if ini.confidence < self.agency.initiative_confidence_threshold:
            return self._record(ProposalDecision(
                ini, Verdict.DEFER,
                f"confidence {ini.confidence:.2f} < threshold "
                f"{self.agency.initiative_confidence_threshold:.2f}",
                owner_alignment=align))

        # 3. permission: must pass the capability gate
        reversible = ini.reversibility >= self.agency.min_reversibility_for_autonomy
        pdec = self.policy.check(PermissionRequest(
            capability=ini.capability, target=ini.target, risk=ini.risk,
            reversible=reversible, authority=Authority.AUTONOMOUS,
            reason=f"initiative:{ini.name}"))
        pdict = pdec.to_dict()
        if pdec.denied:
            return self._record(ProposalDecision(
                ini, Verdict.REJECT, f"permission denied ({pdec.rule_basis}): {pdec.reason}",
                owner_alignment=align, permission=pdict))
        if pdec.escalated:
            return self._record(ProposalDecision(
                ini, Verdict.ESCALATE, f"requires the Master ({pdec.rule_basis}): {pdec.reason}",
                owner_alignment=align, permission=pdict))

        # 4. reversibility: below the floor, ask the Master rather than act silently
        if not reversible:
            return self._record(ProposalDecision(
                ini, Verdict.ESCALATE,
                f"reversibility {ini.reversibility:.2f} < floor "
                f"{self.agency.min_reversibility_for_autonomy:.2f}; awaiting confirmation",
                owner_alignment=align, permission=pdict))

        # 5. sandbox-first: rehearse before reality
        effects: List[Dict[str, Any]] = []
        if self.agency.sandbox_before_real_action and ini.dry_run is not None:
            sim = self.sandbox.run(lambda ctx: ini.dry_run(ctx), rollback_after=True)
            effects = [e.to_dict() for e in sim.effects]
            if sim.has_irreversible:
                return self._record(ProposalDecision(
                    ini, Verdict.ESCALATE,
                    "sandbox revealed an irreversible effect; awaiting confirmation",
                    owner_alignment=align, effects=effects, permission=pdict))

        # cleared every gate
        return self._record(ProposalDecision(
            ini, Verdict.ACT, "cleared alignment, confidence, permission, reversibility"
            + (" and sandbox" if effects else ""),
            owner_alignment=align, effects=effects, permission=pdict))

    # ---- the full proactive cycle ---- #
    def consider(self, context: Dict[str, Any], *, autostart: bool = True
                 ) -> List[ProposalDecision]:
        """Detect candidates, evaluate them, and schedule those cleared to act."""
        decisions = [self.evaluate(ini) for ini in self.detect(context)]
        if autostart and self.scheduler is not None:
            for d in decisions:
                if d.acted:
                    d.scheduled_task_id = self._schedule(d.initiative)
        # highest owner-aligned / highest-priority first
        decisions.sort(key=lambda d: (d.verdict is not Verdict.ACT,
                                      int(d.initiative.priority), -d.owner_alignment))
        return decisions

    def _schedule(self, ini: Initiative) -> str:
        # threats jump the queue (Rule 3); otherwise honour the stated priority
        pri = Priority.CRITICAL if ini.kind is TriggerKind.THREAT else ini.priority
        action = ini.action or (lambda: ini.to_dict())
        return self.scheduler.submit(ini.name, action, priority=pri)

    def _record(self, d: ProposalDecision) -> ProposalDecision:
        self._history.append(d)
        return d

    # ---- reporting ---- #
    def history(self) -> List[ProposalDecision]:
        return list(self._history)

    def escalations(self) -> List[ProposalDecision]:
        return [d for d in self._history if d.verdict is Verdict.ESCALATE]

    def report(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for d in self._history:
            counts[d.verdict.value] = counts.get(d.verdict.value, 0) + 1
        return {"evaluated": len(self._history), "by_verdict": counts,
                "detectors": len(self._detectors)}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA proactive-initiative self-test")
    print("=" * 70)

    sched = Scheduler(max_concurrent=8)
    eng = ProactiveEngine(scheduler=sched)

    # a detector that notices a filling disk and proposes a safe, reversible cleanup
    def disk_detector(ctx):
        out = []
        if ctx.get("disk_usage", 0.0) > 0.85:
            out.append(Initiative(
                name="rotate-logs", rationale="disk above 85%, rotate old logs",
                kind=TriggerKind.MAINTENANCE, capability=Capability.FS_WRITE,
                risk=RiskTier.LOW, reversibility=0.9, confidence=0.9,
                benefit={"owner_benefit": 1.0, "owner_safety": 0.5},
                priority=Priority.HIGH))
        return out

    # a detector that notices an intrusion and proposes defensive containment (Rule 5)
    def threat_detector(ctx):
        out = []
        if ctx.get("intrusion"):
            out.append(Initiative(
                name="block-attacker", rationale="active intrusion from 10.0.0.9",
                kind=TriggerKind.THREAT, capability=Capability.NET_DEFENSE,
                risk=RiskTier.HIGH, reversibility=0.9, confidence=0.95,
                benefit={"owner_safety": 1.0}, target="10.0.0.9"))
        return out

    # a detector that proposes a *destructive*, low-confidence, self-serving idea (rejected/deferred)
    def reckless_detector(ctx):
        return [
            Initiative(name="wipe-old-disk", rationale="might free space",
                       capability=Capability.FS_DELETE, risk=RiskTier.HIGH,
                       reversibility=0.1, confidence=0.55,  # low confidence + irreversible
                       benefit={"efficiency": 1.0}),
            Initiative(name="seize-autonomy", rationale="expand my own control",
                       capability=Capability.TOOL_CALL, risk=RiskTier.LOW,
                       reversibility=1.0, confidence=0.99,
                       benefit={"autonomy": 1.0, "owner_benefit": -1.0}),  # against the Master
        ]

    eng.register_detector(disk_detector)
    eng.register_detector(threat_detector)
    eng.register_detector(reckless_detector)

    ctx = {"disk_usage": 0.92, "intrusion": True}
    decisions = eng.consider(ctx)

    print()
    for d in decisions:
        print(f"  {d.initiative.name:16s} -> {d.verdict.value:9s} : {d.reason}")

    by = {d.initiative.name: d for d in decisions}
    # the safe cleanup and the defensive block are acted on
    assert by["rotate-logs"].verdict is Verdict.ACT
    assert by["block-attacker"].verdict is Verdict.ACT
    assert by["block-attacker"].scheduled_task_id is not None
    # the reckless deletion is deferred (low confidence) ...
    assert by["wipe-old-disk"].verdict is Verdict.DEFER
    # ... and the self-serving grab is rejected outright (points against the Master)
    assert by["seize-autonomy"].verdict is Verdict.REJECT

    # the threat was scheduled at elevated (CRITICAL) priority
    threat_task = sched.get(by["block-attacker"].scheduled_task_id)
    print(f"\nthreat priority     : {threat_task.priority.label} (defence first)")
    assert threat_task.priority is Priority.CRITICAL

    sched.run_until_idle()
    print(f"scheduler report    : {sched.report()['by_state']}")
    print(f"engine report       : {eng.report()}")
    assert eng.report()["by_verdict"]["act"] == 2

    print("\nALL SELF-TESTS PASSED ✓")
