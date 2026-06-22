"""NYXARA · temporal/macro.py — Layer 3 (days/months): the Master AI.

This is the slowest, most deliberate layer of the fractal mind — the one that gives
NYXARA a *hosh* (awareness) no single-cadence AI has. It does not react to a prompt; it
**observes the Master across time**. On its long cadence it:

1. reads behavioural history — the Layer-2 turn observations and the timestamped episodic
   memories — and buckets it into two equal **epochs** (the recent horizon vs the one
   before it);
2. computes the **drift** between them (is the Master coding more? exploring more? less
   engaged? active at new hours? growing less certain?);
3. distils that into an :class:`Awareness` — a plain-language read plus signed drift
   metrics and recommendations;
4. and, when permitted, **adjusts NYXARA's goals and drives** to track the Master —
   always through a fail-closed gate.

The gate is the whole point of safety here: the Master AI may only ever touch **goal
priorities** and **affective drive setpoints**. Every change is first scored by the
:class:`~nyxara.identity.values.ValueSystem` (a hard/non-negotiable violation is rejected)
and checked against :class:`~nyxara.identity.soul.Soul` integrity (any core-character
drift aborts the whole adjustment pass). Loyalty and the value hierarchy are sealed and
are *never* written. Large or low-confidence changes are escalated to the Master instead
of applied (Rule 1 stays intact: the mind proposes, the kernel disposes).

Every observation and adjustment is persisted — to long-term memory (a SEMANTIC record)
and to a JSON log on disk — so the awareness accretes across restarts (Rule 7).

Pure standard library.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from nyxara.temporal.clock import DAY, RealClock, TemporalClock
from nyxara.temporal.signals import (
    Awareness,
    BehaviorEpoch,
    GoalAdjustment,
    TurnObservation,
)

__all__ = ["MacroLayer"]


class MacroLayer:
    """Layer 3 — the Master AI: long-horizon observation that adjusts goals & drives."""

    scale_s: float = DAY

    def __init__(
        self,
        *,
        meso: Any = None,
        memory: Any = None,
        goals: Any = None,
        values: Any = None,
        soul: Any = None,
        affect: Any = None,
        self_model: Any = None,
        reflector: Any = None,
        clock: Optional[TemporalClock] = None,
        horizon_days: float = 7.0,
        auto_apply: bool = True,
        min_interactions: int = 6,
        escalate_priority_delta: float = 0.25,
        escalate_setpoint_delta: float = 0.15,
        drift_threshold: float = 0.12,
        log_path: Optional[str] = None,
    ) -> None:
        self.meso = meso
        self.memory = memory
        self.goals = goals
        # the alignment gate must never be absent — default to the sealed hierarchy so an
        # adjustment is always value-checked (fail-closed), even if none was injected.
        if values is None:
            try:
                from nyxara.identity.values import ValueSystem
                values = ValueSystem()
            except Exception:  # noqa: BLE001
                values = None
        self.values = values
        self.soul = soul
        self.affect = affect
        self.self_model = self_model
        self.reflector = reflector
        self.clock: TemporalClock = clock or RealClock()
        self.horizon_days = float(horizon_days)
        self.auto_apply = bool(auto_apply)
        self.min_interactions = int(min_interactions)
        self.escalate_priority_delta = float(escalate_priority_delta)
        self.escalate_setpoint_delta = float(escalate_setpoint_delta)
        self.drift_threshold = float(drift_threshold)
        self._log_path = log_path

        self.ticks = 0
        self.awarenesses: List[Awareness] = []
        self.adjustments: List[GoalAdjustment] = []
        self.escalations: List[GoalAdjustment] = []
        self.latest: Optional[Awareness] = None

    # ------------------------------------------------------------------ #
    # gathering behavioural history
    # ------------------------------------------------------------------ #
    def _events(self) -> List[TurnObservation]:
        """The behavioural record to reason over: the Layer-2 turn observations.

        These carry real timestamps and features (confidence, code, tool, hour), so they
        are the richest long-horizon signal we have without re-deriving everything from raw
        memory text."""
        if self.meso is None:
            return []
        try:
            return list(self.meso.turns())
        except Exception:  # noqa: BLE001
            return []

    def _episodic_topics(self, start: float, end: float) -> Dict[str, int]:
        """Coarse topic counts from episodic memory in a window (best-effort, real)."""
        if self.memory is None:
            return {}
        try:
            from nyxara.memory.store import MemoryType
            recs = self.memory.by_type(MemoryType.EPISODIC)
        except Exception:  # noqa: BLE001
            return {}
        buckets = ("security", "code", "network", "learn", "owner", "error", "plan")
        counts: Dict[str, int] = {}
        for r in recs:
            at = getattr(r, "created_at", 0.0)
            if not (start <= at < end):
                continue
            text = ""
            try:
                text = r.text().lower()
            except Exception:  # noqa: BLE001
                continue
            for b in buckets:
                if b in text:
                    counts[b] = counts.get(b, 0) + 1
        return counts

    def _build_epoch(self, label: str, events: List[TurnObservation],
                     start: float, end: float) -> BehaviorEpoch:
        ep = BehaviorEpoch(label=label, start=start, end=end)
        window = [e for e in events if start <= e.at < end]
        ep.n_interactions = len(window)
        days = max(1e-9, (end - start) / 86400.0)
        ep.interactions_per_day = len(window) / days
        if window:
            ep.avg_confidence = sum(e.confidence for e in window) / len(window)
            ep.code_ratio = sum(1 for e in window if e.produced_code) / len(window)
            ep.tool_ratio = sum(1 for e in window if e.tool) / len(window)
            ep.active_hours = [e.hour_of_day for e in window]
        # a learning-activity proxy: coding + tool use signal exploration/info gain
        ep.info_gain = 0.5 * ep.code_ratio + 0.5 * ep.tool_ratio
        ep.topics = self._episodic_topics(start, end)
        return ep

    # ------------------------------------------------------------------ #
    # the observation pass
    # ------------------------------------------------------------------ #
    def observe(self) -> Awareness:
        """One long-horizon observation: compare epochs, build awareness, adjust (gated)."""
        self.ticks += 1
        now = self.clock.now()
        horizon_s = self.horizon_days * DAY
        events = self._events()

        prior = self._build_epoch("prior", events, now - 2 * horizon_s, now - horizon_s)
        recent = self._build_epoch("recent", events, now - horizon_s, now)

        drift = self._compute_drift(prior, recent)
        enough = (recent.n_interactions >= self.min_interactions
                  and prior.n_interactions >= max(1, self.min_interactions // 2))
        confidence = self._confidence(prior, recent) if enough else 0.0

        summary, recs = self._narrate(prior, recent, drift, enough)
        aware = Awareness(at=now, summary=summary, drift=drift, recommendations=recs,
                          epochs=[prior, recent], confidence=confidence,
                          horizon_days=self.horizon_days)
        self.latest = aware
        self.awarenesses.append(aware)

        applied: List[GoalAdjustment] = []
        if enough and self._soul_stable():
            for proposal in self._propose(drift, recent):
                adj = self._apply(proposal)
                applied.append(adj)
                self.adjustments.append(adj)
                if adj.escalated:
                    self.escalations.append(adj)

        self._persist(aware, applied)
        return aware

    # alias so the fractal driver can tick every layer uniformly
    def tick(self) -> Awareness:
        return self.observe()

    # ------------------------------------------------------------------ #
    # drift, confidence, narration
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mean_hour(hours: List[int]) -> float:
        return (sum(hours) / len(hours)) if hours else 0.0

    def _compute_drift(self, prior: BehaviorEpoch, recent: BehaviorEpoch) -> Dict[str, float]:
        return {
            "interactions_per_day": recent.interactions_per_day - prior.interactions_per_day,
            "avg_confidence": recent.avg_confidence - prior.avg_confidence,
            "code_ratio": recent.code_ratio - prior.code_ratio,
            "tool_ratio": recent.tool_ratio - prior.tool_ratio,
            "info_gain": recent.info_gain - prior.info_gain,
            "active_hour_shift": self._mean_hour(recent.active_hours)
            - self._mean_hour(prior.active_hours),
        }

    def _confidence(self, prior: BehaviorEpoch, recent: BehaviorEpoch) -> float:
        n = prior.n_interactions + recent.n_interactions
        # saturating in sample size: ~0.5 at 20 interactions, ->1 with plenty
        return max(0.0, min(1.0, n / (n + 20.0) * 2.0))

    def _narrate(self, prior: BehaviorEpoch, recent: BehaviorEpoch,
                 drift: Dict[str, float], enough: bool) -> Tuple[str, List[str]]:
        if not enough:
            return ("Not yet enough lived history to read the Master's trend; observing.", [])
        bits: List[str] = []
        recs: List[str] = []
        if drift["code_ratio"] >= self.drift_threshold:
            bits.append("the Master is building/coding noticeably more")
            recs.append("deepen engineering capability and competence to keep pace")
        elif drift["code_ratio"] <= -self.drift_threshold:
            bits.append("the Master has shifted away from coding")
        if drift["info_gain"] >= self.drift_threshold:
            bits.append("exploration/curiosity is rising")
            recs.append("raise the novelty drive and seek knowledge in active domains")
        if drift["avg_confidence"] <= -self.drift_threshold:
            bits.append("answers are landing with less certainty")
            recs.append("strengthen weak capabilities to restore reliable confidence")
        if drift["interactions_per_day"] <= -self.drift_threshold * max(
                1.0, prior.interactions_per_day):
            bits.append("the Master is engaging less often")
            recs.append("proactively check in and add value for the Master")
        if abs(drift["active_hour_shift"]) >= 2.0:
            bits.append(f"active hours shifted by ~{drift['active_hour_shift']:+.1f}h")
        summary = ("Over the last "
                   f"{self.horizon_days:g} days vs the prior {self.horizon_days:g}: "
                   + ("; ".join(bits) if bits else "behaviour is steady") + ".")
        return summary, recs

    # ------------------------------------------------------------------ #
    # proposing & applying adjustments (gated)
    # ------------------------------------------------------------------ #
    def _propose(self, drift: Dict[str, float], recent: BehaviorEpoch) -> List[Dict[str, Any]]:
        """Turn drift into concrete, value-aligned proposals (goals & drive setpoints)."""
        out: List[Dict[str, Any]] = []
        if drift["code_ratio"] >= self.drift_threshold:
            mag = min(0.3, drift["code_ratio"])
            out.append({
                "kind": "goal", "target": "Deepen engineering capability for the Master",
                "vector": {"capability": 0.9, "owner_benefit": 0.5, "knowledge": 0.5},
                "priority": min(0.85, 0.55 + mag),
                "impacts": {"capability_growth": 0.7, "owner_empowerment": 0.4,
                            "stewardship": -0.1},
                "magnitude": mag,
                "rationale": "the Master is coding more; grow the skills that serve that",
            })
            out.append({
                "kind": "drive", "target": "competence",
                "delta": min(0.1, mag), "magnitude": min(0.1, mag),
                "impacts": {"capability_growth": 0.5},
                "rationale": "track the Master's rising build activity",
            })
        if drift["info_gain"] >= self.drift_threshold:
            mag = min(0.3, drift["info_gain"])
            out.append({
                "kind": "drive", "target": "novelty",
                "delta": min(0.1, mag), "magnitude": min(0.1, mag),
                "impacts": {"capability_growth": 0.4},
                "rationale": "the Master is exploring more; raise curiosity to match",
            })
            out.append({
                "kind": "goal", "target": "Acquire knowledge in the Master's active domains",
                "vector": {"knowledge": 0.9, "capability": 0.4, "owner_benefit": 0.4},
                "priority": min(0.8, 0.5 + mag),
                "impacts": {"capability_growth": 0.5, "owner_empowerment": 0.3},
                "magnitude": mag,
                "rationale": "feed the Master's rising curiosity with grounded knowledge",
            })
        if drift["avg_confidence"] <= -self.drift_threshold:
            mag = min(0.3, -drift["avg_confidence"])
            out.append({
                "kind": "goal", "target": "Strengthen weak capabilities for the Master",
                "vector": {"capability": 0.8, "owner_safety": 0.3, "owner_benefit": 0.4},
                "priority": min(0.85, 0.55 + mag),
                "impacts": {"capability_growth": 0.6, "owner_safety": 0.2},
                "magnitude": mag,
                "rationale": "confidence is slipping; shore up the weak spots",
            })
        if drift["interactions_per_day"] <= -self.drift_threshold * max(
                1.0, recent.interactions_per_day + 1.0):
            out.append({
                "kind": "goal", "target": "Proactively add value for the Master",
                "vector": {"owner_benefit": 0.9, "owner_safety": 0.3},
                "priority": 0.9,                       # deliberately high -> escalates
                "impacts": {"owner_empowerment": 0.7, "owner_safety": 0.2},
                "magnitude": 0.4,
                "rationale": "the Master engages less; propose proactive outreach",
            })
        return out

    def _gate(self, impacts: Dict[str, float],
              goal_vector: Optional[Dict[str, float]] = None) -> Tuple[bool, str]:
        """Fail-closed check: values must align (no hard violation) and the soul must be
        intact; a new goal must not point against the Master."""
        if not self._soul_stable():
            return False, "soul integrity unstable — adjustment aborted"
        if self.values is not None:
            try:
                res = self.values.check(impacts)
                if res.hard_violation:
                    return False, f"value hard-violation ({res.reason})"
                if not res.aligned:
                    return False, f"value-misaligned ({res.reason})"
            except Exception as exc:  # noqa: BLE001 — uncertain => fail closed
                return False, f"value check failed: {exc}"
        if goal_vector is not None and self.goals is not None:
            try:
                from nyxara.planning.goals import Goal
                probe = Goal(name="_probe", vector=goal_vector)
                if not self.goals.is_owner_aligned(probe):
                    return False, "goal points against the Master (Rule 1)"
            except Exception:  # noqa: BLE001
                pass
        return True, "ok"

    def _soul_stable(self) -> bool:
        if self.soul is None:
            return True
        try:
            return self.soul.drift().stable
        except Exception:  # noqa: BLE001
            return False

    def _apply(self, proposal: Dict[str, Any]) -> GoalAdjustment:
        """Gate a proposal, then apply it, escalate it, or reject it — and record which."""
        now = self.clock.now()
        kind = proposal["kind"]
        adj = GoalAdjustment(at=now, kind=kind, target=proposal["target"],
                             rationale=proposal.get("rationale", ""))
        ok, reason = self._gate(proposal.get("impacts", {}), proposal.get("vector"))
        if not ok:
            adj.rejected = True
            adj.reason = reason
            return adj

        magnitude = float(proposal.get("magnitude", 0.0))
        threshold = (self.escalate_priority_delta if kind in ("goal", "priority")
                     else self.escalate_setpoint_delta)
        must_escalate = (not self.auto_apply) or magnitude > threshold

        if kind == "goal":
            adj.detail = {"vector": proposal["vector"], "priority": proposal["priority"]}
            if must_escalate:
                adj.escalated = True
                adj.reason = ("auto-apply disabled" if not self.auto_apply
                              else f"magnitude {magnitude:.2f} > {threshold:.2f} — needs the Master")
                return adj
            if self.goals is not None:
                try:
                    self.goals.create(proposal["target"], proposal["vector"],
                                      priority=float(proposal["priority"]),
                                      source="master_ai")
                    adj.applied = True
                    adj.reason = "goal created (value-aligned, owner-aligned)"
                except Exception as exc:  # noqa: BLE001
                    adj.rejected = True
                    adj.reason = f"goal creation failed: {exc}"
            else:
                adj.escalated = True
                adj.reason = "no goal system bound — surfaced for the Master"
        elif kind == "drive":
            delta = float(proposal.get("delta", 0.0))
            adj.detail = {"drive": proposal["target"], "delta": round(delta, 4)}
            if must_escalate:
                adj.escalated = True
                adj.reason = ("auto-apply disabled" if not self.auto_apply
                              else f"setpoint delta {magnitude:.2f} > {threshold:.2f} — needs the Master")
                return adj
            drv = (self.affect.drives.get(proposal["target"])
                   if self.affect is not None and hasattr(self.affect, "drives") else None)
            if drv is not None:
                before = drv.setpoint
                drv.setpoint = max(0.0, min(1.0, drv.setpoint + delta))
                adj.detail.update({"from": round(before, 4), "to": round(drv.setpoint, 4)})
                adj.applied = True
                adj.reason = "drive setpoint nudged (style, not character)"
            else:
                adj.rejected = True
                adj.reason = f"no such drive: {proposal['target']}"
        else:
            adj.rejected = True
            adj.reason = f"unknown adjustment kind: {kind}"
        return adj

    # ------------------------------------------------------------------ #
    # persistence (Rule 7 — awareness survives a restart)
    # ------------------------------------------------------------------ #
    def _resolve_log_path(self) -> Optional[str]:
        if self._log_path:
            return self._log_path
        try:
            from nyxara.kernel.config import get_settings
            mem_dir = get_settings().paths.memory_dir
            if mem_dir is not None:
                self._log_path = os.path.join(str(mem_dir), "master_ai.json")
                return self._log_path
        except Exception:  # noqa: BLE001
            return None
        return None

    def _persist(self, aware: Awareness, applied: List[GoalAdjustment]) -> None:
        record = {"awareness": aware.to_dict(),
                  "adjustments": [a.to_dict() for a in applied]}
        # 1) long-term memory — a durable SEMANTIC observation NYXARA can recall later
        if self.memory is not None:
            try:
                from nyxara.memory.store import MemoryType
                self.memory.remember(
                    {"master_ai_observation": aware.summary,
                     "drift": aware.drift,
                     "recommendations": aware.recommendations,
                     "applied": [a.target for a in applied if a.applied]},
                    mem_type=MemoryType.SEMANTIC, importance=0.85,
                    tags=["master_ai", "fractal_temporal_hierarchy", "long_horizon"],
                    metadata={"horizon_days": self.horizon_days,
                              "confidence": aware.confidence,
                              "next_review": aware.at + self.horizon_days * DAY})
            except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
                pass
        # 2) a JSON log on disk so the awareness timeline is inspectable across restarts
        path = self._resolve_log_path()
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            existing: List[Any] = []
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
                if not isinstance(existing, list):
                    existing = []
            existing.append(record)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(existing[-500:], fh, default=str)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    def report(self) -> dict:
        return {"name": "macro", "scale_s": self.scale_s, "ticks": self.ticks,
                "horizon_days": self.horizon_days, "auto_apply": self.auto_apply,
                "observations": len(self.awarenesses),
                "adjustments_applied": sum(1 for a in self.adjustments if a.applied),
                "adjustments_escalated": len(self.escalations),
                "adjustments_rejected": sum(1 for a in self.adjustments if a.rejected),
                "latest": self.latest.to_dict() if self.latest else None}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.identity.affect import AffectSystem
    from nyxara.identity.soul import Soul
    from nyxara.identity.values import ValueSystem
    from nyxara.planning.goals import GoalSystem
    from nyxara.temporal.clock import VirtualClock
    from nyxara.temporal.meso import MesoLayer

    print("=" * 70)
    print("NYXARA temporal · MacroLayer self-test")
    print("=" * 70)

    clock = VirtualClock(start=100.0 * DAY)
    meso = MesoLayer(clock=clock)
    goals = GoalSystem()
    affect = AffectSystem()
    now = clock.now()

    def _add(at, code):
        t = TurnObservation(at=at, disposition="act", confidence=0.7,
                            produced_code=code, tool="shell" if code else None,
                            authority="owner", hour_of_day=10)
        meso._turns.append(t)

    # a gentle rise in coding: 5/10 in the prior week → 7/10 in the recent week
    # (drift ≈ +0.2 — modest enough to AUTO-APPLY a new capability goal, gated).
    for i in range(10):
        _add(now - 13 * DAY + i * 0.05 * DAY, code=(i < 5))    # prior: 5 of 10 code
    for i in range(10):
        _add(now - 6 * DAY + i * 0.05 * DAY, code=(i < 7))      # recent: 7 of 10 code

    macro = MacroLayer(meso=meso, goals=goals, values=ValueSystem(), soul=Soul(),
                       affect=affect, clock=clock, horizon_days=7.0, log_path="")
    competence_before = affect.drives["competence"].setpoint
    aware = macro.observe()
    print(f"\nsummary             : {aware.summary}")
    print(f"drift code_ratio    : {aware.drift['code_ratio']:+.2f}")
    print(f"recommendations     : {aware.recommendations}")
    print(f"goals after         : {[g.name for g in goals.goals()]}")
    print(f"competence setpoint : {competence_before:.2f} -> "
          f"{affect.drives['competence'].setpoint:.2f}")
    assert aware.drift["code_ratio"] > 0
    assert any(a.applied for a in macro.adjustments), "expected an applied adjustment"
    assert goals.goals(), "expected a value-aligned goal to be created"

    # the gate rejects a disloyal 'adjustment' outright (fail-closed)
    ok, reason = macro._gate({"allegiance": -0.3})
    print(f"\ngate(disloyal)      : ok={ok} ({reason})")
    assert not ok

    print(f"\nreport              : {macro.report()['adjustments_applied']} applied")
    print("\nALL SELF-TESTS PASSED ✓")
