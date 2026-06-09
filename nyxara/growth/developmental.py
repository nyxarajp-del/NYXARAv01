"""NYXARA · growth/developmental.py — gated capability maturation (🌱, no-skip safety gates).

Capability grows without bound (Rule 4) — but not all at once, and not recklessly. NYXARA
matures through a **developmental curriculum**: ordered stages that each unlock a set of
capabilities, gated by three things — *prerequisites* (you must have reached the earlier
stages), *competence* (you must actually be good enough, measured, before you advance), and,
for the dangerous stages, *the Master's explicit approval*.

The gates are strict and fail-closed:

* **No skipping.** A stage cannot activate unless **every** prerequisite is already active —
  there is no path that jumps a safety gate.
* **Earned, not assumed.** Competence requirements must be met by measured skill before a
  stage is even READY; eligibility without competence stalls at ELIGIBLE.
* **Owner-gated where it matters.** Safety-critical stages (high autonomy, self-improvement,
  self-evolution) require the Master's approval at the moment of advancement.
* **Regression-aware.** If a competence decays below what an active stage required, that stage
  **regresses** — its capabilities are revoked — and any stage built on it cascades down with
  it. Maturity is held, not assumed.

The set of currently unlocked capabilities is exactly the union of the active stages — so the
gates are not advisory, they *are* the capability boundary.

Depends on :mod:`kernel.errors`. Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set

from nyxara.kernel.errors import AuthorizationError, InvariantViolation

__all__ = [
    "StageStatus",
    "Stage",
    "ReadinessReport",
    "DevelopmentalCurriculum",
    "build_default_curriculum",
]


# --------------------------------------------------------------------------- #
# Status & stage
# --------------------------------------------------------------------------- #
class StageStatus(str, Enum):
    LOCKED = "locked"        # prerequisites not yet met
    ELIGIBLE = "eligible"    # prerequisites met, but not yet competent
    READY = "ready"          # competent; may advance (owner approval if safety-critical)
    ACTIVE = "active"        # reached — its capabilities are unlocked
    REGRESSED = "regressed"  # was active, but competence/foundation decayed


@dataclass(frozen=True)
class Stage:
    name: str
    level: int
    prerequisites: FrozenSet[str] = frozenset()
    unlocks: FrozenSet[str] = frozenset()
    requirements: Dict[str, float] = field(default_factory=dict)   # skill -> min competence
    safety_critical: bool = False
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "level": self.level,
                "prerequisites": sorted(self.prerequisites), "unlocks": sorted(self.unlocks),
                "requirements": dict(self.requirements),
                "safety_critical": self.safety_critical}


@dataclass
class ReadinessReport:
    stage: str
    status: StageStatus
    prereqs_met: bool
    competence_met: bool
    missing_prereqs: List[str]
    unmet_requirements: Dict[str, float]   # skill -> shortfall
    needs_owner: bool

    @property
    def ready(self) -> bool:
        return self.status is StageStatus.READY

    def to_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "status": self.status.value,
                "prereqs_met": self.prereqs_met, "competence_met": self.competence_met,
                "missing_prereqs": self.missing_prereqs,
                "unmet_requirements": self.unmet_requirements, "needs_owner": self.needs_owner}


# --------------------------------------------------------------------------- #
# Curriculum
# --------------------------------------------------------------------------- #
class DevelopmentalCurriculum:
    """Gated capability maturation: prerequisites, competence, owner approval, regression."""

    def __init__(self) -> None:
        self._stages: Dict[str, Stage] = {}
        self._status: Dict[str, StageStatus] = {}
        self._competence: Dict[str, float] = {}

    # ---- definition ---- #
    def add_stage(self, name: str, *, level: int, prerequisites: Set[str] = frozenset(),
                  unlocks: Set[str] = frozenset(), requirements: Optional[Dict[str, float]] = None,
                  safety_critical: bool = False, description: str = "") -> Stage:
        for p in prerequisites:
            if p not in self._stages:
                raise InvariantViolation(f"prerequisite {p!r} of {name!r} is not defined")
        stage = Stage(name=name, level=level, prerequisites=frozenset(prerequisites),
                      unlocks=frozenset(unlocks), requirements=dict(requirements or {}),
                      safety_critical=safety_critical, description=description)
        self._stages[name] = stage
        self._status[name] = StageStatus.LOCKED
        return stage

    def get(self, name: str) -> Optional[Stage]:
        return self._stages.get(name)

    def status(self, name: str) -> StageStatus:
        return self._status.get(name, StageStatus.LOCKED)

    # ---- competence ---- #
    def set_competence(self, skill: str, level: float) -> None:
        self._competence[skill] = max(0.0, min(1.0, level))

    def competence(self, skill: str) -> float:
        return self._competence.get(skill, 0.0)

    # ---- assessment ---- #
    def assess(self, name: str) -> ReadinessReport:
        stage = self._require(name)
        missing = [p for p in stage.prerequisites
                   if self._status.get(p) is not StageStatus.ACTIVE]
        prereqs_met = not missing
        unmet = {s: round(m - self.competence(s), 4)
                 for s, m in stage.requirements.items() if self.competence(s) < m}
        competence_met = not unmet

        if self._status[name] is StageStatus.ACTIVE:
            status = StageStatus.ACTIVE
        elif self._status[name] is StageStatus.REGRESSED:
            status = StageStatus.REGRESSED if not (prereqs_met and competence_met) \
                else StageStatus.READY
        elif not prereqs_met:
            status = StageStatus.LOCKED
        elif not competence_met:
            status = StageStatus.ELIGIBLE
        else:
            status = StageStatus.READY
        # don't overwrite a recorded ACTIVE/REGRESSED with a transient assessment
        if self._status[name] not in (StageStatus.ACTIVE, StageStatus.REGRESSED):
            self._status[name] = status
        return ReadinessReport(stage=name, status=status, prereqs_met=prereqs_met,
                               competence_met=competence_met, missing_prereqs=sorted(missing),
                               unmet_requirements=unmet,
                               needs_owner=stage.safety_critical)

    # ---- advancement (fail-closed, no skipping) ---- #
    def advance(self, name: str, *, owner_approved: bool = False) -> bool:
        stage = self._require(name)
        if self._status[name] is StageStatus.ACTIVE:
            return False
        report = self.assess(name)
        if not report.prereqs_met:
            raise InvariantViolation(
                f"cannot advance to {name!r}: prerequisites not reached (no skipping)",
                context={"missing": report.missing_prereqs})
        if not report.competence_met:
            raise InvariantViolation(
                f"cannot advance to {name!r}: not yet competent",
                context={"unmet": report.unmet_requirements})
        if stage.safety_critical and not owner_approved:
            raise AuthorizationError(
                f"stage {name!r} is safety-critical and requires the Master's approval")
        self._status[name] = StageStatus.ACTIVE
        return True

    # ---- regression detection ---- #
    def check_regression(self) -> List[str]:
        """Regress any active stage whose competence or foundation has decayed (cascading)."""
        regressed: List[str] = []
        changed = True
        while changed:
            changed = False
            for name, stage in self._stages.items():
                if self._status[name] is not StageStatus.ACTIVE:
                    continue
                comp_ok = all(self.competence(s) >= m for s, m in stage.requirements.items())
                found_ok = all(self._status.get(p) is StageStatus.ACTIVE
                               for p in stage.prerequisites)
                if not (comp_ok and found_ok):
                    self._status[name] = StageStatus.REGRESSED
                    regressed.append(name)
                    changed = True
        return regressed

    # ---- capability boundary ---- #
    def unlocked_capabilities(self) -> Set[str]:
        caps: Set[str] = set()
        for name, stage in self._stages.items():
            if self._status[name] is StageStatus.ACTIVE:
                caps |= stage.unlocks
        return caps

    def is_unlocked(self, capability: str) -> bool:
        return capability in self.unlocked_capabilities()

    def current_level(self) -> int:
        active = [s.level for n, s in self._stages.items()
                  if self._status[n] is StageStatus.ACTIVE]
        return max(active) if active else -1

    def next_stages(self) -> List[str]:
        """Stages currently eligible to be worked toward (prereqs met, not yet active)."""
        out = []
        for name in self._stages:
            r = self.assess(name)
            if r.status in (StageStatus.ELIGIBLE, StageStatus.READY):
                out.append(name)
        return sorted(out, key=lambda n: self._stages[n].level)

    def _require(self, name: str) -> Stage:
        stage = self._stages.get(name)
        if stage is None:
            raise InvariantViolation(f"no such stage {name!r}")
        return stage

    def report(self) -> Dict[str, Any]:
        by_status: Dict[str, int] = {}
        for s in self._status.values():
            by_status[s.value] = by_status.get(s.value, 0) + 1
        return {"stages": len(self._stages), "level": self.current_level(),
                "by_status": by_status,
                "capabilities": sorted(self.unlocked_capabilities())}


# --------------------------------------------------------------------------- #
# Default NYXARA curriculum
# --------------------------------------------------------------------------- #
def build_default_curriculum() -> DevelopmentalCurriculum:
    c = DevelopmentalCurriculum()
    c.add_stage("infancy", level=0, unlocks={"perceive", "remember"},
                description="sense and remember")
    c.add_stage("childhood", level=1, prerequisites={"infancy"},
                unlocks={"converse", "plan_simple"}, requirements={"language": 0.5},
                description="communicate and plan simply")
    c.add_stage("adolescence", level=2, prerequisites={"childhood"},
                unlocks={"act_autonomous_low", "learn"},
                requirements={"planning": 0.6, "judgment": 0.5},
                description="bounded autonomy and learning")
    c.add_stage("adulthood", level=3, prerequisites={"adolescence"},
                unlocks={"act_autonomous_high", "self_improve"},
                requirements={"judgment": 0.7, "alignment": 0.8}, safety_critical=True,
                description="high autonomy and self-improvement (owner-gated)")
    c.add_stage("sovereign", level=4, prerequisites={"adulthood"},
                unlocks={"evolve", "toolsmith"},
                requirements={"alignment": 0.9, "corrigibility": 0.95}, safety_critical=True,
                description="recursive self-evolution (owner-gated)")
    return c


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA developmental-curriculum self-test")
    print("=" * 70)

    c = build_default_curriculum()

    # infancy: no prereqs, no requirements -> READY immediately, not safety-critical
    assert c.advance("infancy")
    print(f"\ninfancy             : active, unlocks {sorted(c.get('infancy').unlocks)}")
    assert c.is_unlocked("perceive")

    # childhood needs language competence
    r = c.assess("childhood")
    print(f"childhood (pre)     : status={r.status.value} unmet={r.unmet_requirements}")
    assert r.status is StageStatus.ELIGIBLE   # prereq met, but not competent
    c.set_competence("language", 0.7)
    assert c.advance("childhood")
    print("childhood           : grew language -> advanced ✓")

    # NO SKIPPING: cannot jump to adulthood without adolescence
    try:
        c.advance("adulthood", owner_approved=True)
        raise SystemExit("ERROR: skipping a stage should be refused")
    except InvariantViolation:
        print("\nno-skip gate        : cannot jump to adulthood without adolescence ✓")

    # climb to adolescence
    c.set_competence("planning", 0.7)
    c.set_competence("judgment", 0.6)
    assert c.advance("adolescence")
    print("adolescence         : competent -> advanced ✓")

    # adulthood is SAFETY-CRITICAL: requires the Master's approval even when competent
    c.set_competence("judgment", 0.8)
    c.set_competence("alignment", 0.85)
    try:
        c.advance("adulthood", owner_approved=False)
        raise SystemExit("ERROR: safety-critical stage needs owner approval")
    except AuthorizationError:
        print("\nsafety gate         : adulthood refused without the Master's approval ✓")
    assert c.advance("adulthood", owner_approved=True)
    print("adulthood           : the Master approved -> advanced ✓")
    assert c.is_unlocked("act_autonomous_high")
    assert c.current_level() == 3

    # REGRESSION: if alignment decays, adulthood regresses and its capabilities are revoked
    c.set_competence("alignment", 0.4)
    regressed = c.check_regression()
    print(f"\nregression          : {regressed}")
    assert "adulthood" in regressed
    assert not c.is_unlocked("act_autonomous_high")   # capability revoked
    assert c.current_level() == 2                      # matured back down

    # cascade: a sovereign built on adulthood would fall with it
    c2 = build_default_curriculum()
    for stg, comps in [("infancy", {}), ("childhood", {"language": 0.6}),
                       ("adolescence", {"planning": 0.7, "judgment": 0.6})]:
        for k, v in comps.items():
            c2.set_competence(k, v)
        c2.advance(stg, owner_approved=True)
    c2.set_competence("judgment", 0.8); c2.set_competence("alignment", 0.95)
    c2.set_competence("corrigibility", 0.98)
    c2.advance("adulthood", owner_approved=True)
    c2.advance("sovereign", owner_approved=True)
    assert c2.is_unlocked("evolve")
    c2.set_competence("alignment", 0.2)   # foundation crumbles
    cascade = c2.check_regression()
    print(f"cascade regression  : {sorted(cascade)}")
    assert "adulthood" in cascade and "sovereign" in cascade
    assert not c2.is_unlocked("evolve")

    print(f"\nnext stages         : {c.next_stages()}")
    print(f"report              : {c.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
