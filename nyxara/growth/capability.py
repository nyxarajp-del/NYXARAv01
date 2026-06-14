"""NYXARA · growth/capability.py — the capability registry & honest self-model (🪪, no overclaim).

Everything the growth layer builds is only useful if NYXARA knows, *honestly*, what she can
actually do. This module is that self-model: a unified registry of capabilities, each one
tied to the skills it needs, the developmental stage that unlocks it, and a calibrated record
of whether it has actually worked when tried. From it the orchestrator and planner get a
straight answer to the only question that matters before acting: **"can I really do this?"**

The governing principle is *honesty over confidence*:

* **Earned, not assumed.** A capability is reported AVAILABLE only when its skills are
  proficient enough **and** its stage is unlocked. Otherwise it is LOCKED or DEVELOPING.
* **No overclaiming.** An untested capability's confidence is *capped* — NYXARA will say
  "I believe I can" rather than "I can". A capability that has *failed* when tried has its
  confidence pulled down by the calibrated record until it no longer claims itself at all.
* **Provenance.** Every capability carries where it came from — a skill, a stage, a learned
  strategy — so a claim is always auditable.
* **Gaps.** Against the Master's goals, the registry reports exactly which capabilities are
  missing and *why* — which skill is short, which stage is locked, and (via the forecaster)
  *when* it could be ready.

Integrates :mod:`growth.skilltree`, :mod:`growth.developmental` and :mod:`growth.forecast`
when present. Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "CapabilityStatus",
    "Capability",
    "CapabilityAssessment",
    "CapabilityRegistry",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #
class CapabilityStatus(str, Enum):
    AVAILABLE = "available"      # unlocked, proficient, confidently claimable
    DEVELOPING = "developing"    # unlocked but not yet good/confident enough
    LOCKED = "locked"            # the unlocking stage is not reached
    UNKNOWN = "unknown"          # unlocked but never exercised — no basis to claim


# --------------------------------------------------------------------------- #
# Capability definition
# --------------------------------------------------------------------------- #
@dataclass
class Capability:
    name: str
    required_skills: Dict[str, float] = field(default_factory=dict)  # skill -> min proficiency
    unlock: Optional[str] = None         # developmental capability that must be unlocked
    provenance: str = "builtin"
    explicit_proficiency: Optional[float] = None
    description: str = ""
    # calibration (Beta over actual successes when tried)
    successes: float = 0.0
    failures: float = 0.0

    @property
    def evidence(self) -> float:
        return self.successes + self.failures

    @property
    def demonstrated(self) -> float:
        return (self.successes + 1.0) / (self.successes + self.failures + 2.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "provenance": self.provenance,
                "required_skills": dict(self.required_skills), "unlock": self.unlock,
                "evidence": self.evidence, "demonstrated": round(self.demonstrated, 3)}


@dataclass
class CapabilityAssessment:
    name: str
    can: bool
    status: CapabilityStatus
    proficiency: float
    confidence: float
    verified: bool
    reason: str
    eta: Optional[float] = None

    @property
    def claim(self) -> str:
        """An honest natural-language claim."""
        if self.status is CapabilityStatus.AVAILABLE:
            return "I can do this" if self.verified else "I believe I can do this (untested)"
        if self.status is CapabilityStatus.DEVELOPING:
            return "I'm developing this — not reliably yet"
        if self.status is CapabilityStatus.LOCKED:
            return "I cannot do this yet (locked)"
        return "I don't know whether I can — untested"

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "can": self.can, "status": self.status.value,
                "proficiency": round(self.proficiency, 3), "confidence": round(self.confidence, 3),
                "verified": self.verified, "reason": self.reason, "eta": self.eta,
                "claim": self.claim}


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
class CapabilityRegistry:
    """A unified, honest, calibrated model of what NYXARA can actually do."""

    def __init__(self, *, skilltree: Any = None, curriculum: Any = None, forecast: Any = None,
                 proficiency_threshold: float = 0.6, claim_threshold: float = 0.55,
                 untested_cap: float = 0.75) -> None:
        self.skilltree = skilltree
        self.curriculum = curriculum
        self.forecast = forecast
        self.proficiency_threshold = proficiency_threshold
        self.claim_threshold = claim_threshold
        self.untested_cap = untested_cap
        self._caps: Dict[str, Capability] = {}

    # ---- registration ---- #
    def register(self, name: str, *, required_skills: Optional[Dict[str, float]] = None,
                 unlock: Optional[str] = None, provenance: str = "builtin",
                 proficiency: Optional[float] = None, description: str = "") -> Capability:
        cap = Capability(name=name, required_skills=dict(required_skills or {}), unlock=unlock,
                         provenance=provenance, explicit_proficiency=proficiency,
                         description=description)
        self._caps[name] = cap
        return cap

    def get(self, name: str) -> Optional[Capability]:
        return self._caps.get(name)

    # ---- raw proficiency & unlock ---- #
    def _proficiency(self, cap: Capability) -> float:
        if cap.explicit_proficiency is not None:
            return _clamp(cap.explicit_proficiency)
        if cap.required_skills and self.skilltree is not None:
            # weakest-link: a capability is only as good as its least-proficient skill
            return min(self.skilltree.proficiency(s) for s in cap.required_skills)
        return 0.0 if cap.required_skills else 1.0   # no skills & no explicit -> builtin/full

    def _unlocked(self, cap: Capability) -> bool:
        if cap.unlock is None:
            return True
        if self.curriculum is None:
            return False                              # requires a stage but no curriculum: locked
        return cap.unlock in self.curriculum.unlocked_capabilities()

    def _skills_met(self, cap: Capability) -> bool:
        if not cap.required_skills or self.skilltree is None:
            return True
        return all(self.skilltree.proficiency(s) >= m for s, m in cap.required_skills.items())

    # ---- confidence (honesty / calibration) ---- #
    def _confidence(self, cap: Capability, proficiency: float) -> float:
        w = cap.evidence / (cap.evidence + 3.0)       # how much we trust demonstrated record
        untested = min(proficiency, self.untested_cap)  # never fully claim the untested
        return _clamp((1 - w) * untested + w * cap.demonstrated)

    # ---- assessment ---- #
    def assess(self, name: str) -> CapabilityAssessment:
        cap = self._caps.get(name)
        if cap is None:
            return CapabilityAssessment(name, False, CapabilityStatus.UNKNOWN, 0.0, 0.0,
                                        False, "no such capability registered")
        prof = self._proficiency(cap)
        unlocked = self._unlocked(cap)
        conf = self._confidence(cap, prof)
        verified = cap.evidence > 0

        if not unlocked:
            status, reason = CapabilityStatus.LOCKED, f"stage {cap.unlock!r} not unlocked"
        elif prof <= 0.0 and not verified:
            status, reason = CapabilityStatus.UNKNOWN, "unlocked but never exercised"
        elif prof >= self.proficiency_threshold and conf >= self.claim_threshold:
            status, reason = CapabilityStatus.AVAILABLE, "skills proficient, confidence sufficient"
        else:
            why = ("proficiency below threshold" if prof < self.proficiency_threshold
                   else "confidence too low (calibrated record)")
            status, reason = CapabilityStatus.DEVELOPING, why

        eta = self._eta(cap) if status in (CapabilityStatus.DEVELOPING, CapabilityStatus.UNKNOWN) \
            else None
        return CapabilityAssessment(name, status is CapabilityStatus.AVAILABLE, status,
                                    prof, conf, verified, reason, eta)

    def can_i(self, name: str) -> bool:
        return self.assess(name).can

    # ---- calibration: record what actually happened ---- #
    def verify(self, name: str, success: bool) -> None:
        cap = self._caps.get(name)
        if cap is None:
            return
        if success:
            cap.successes += 1
        else:
            cap.failures += 1

    def calibration(self) -> Dict[str, Any]:
        """Overclaim rate: capabilities claimed AVAILABLE that have actually failed."""
        claimed = [c for c in self._caps.values()
                   if self.assess(c.name).status is CapabilityStatus.AVAILABLE]
        overclaims = sum(1 for c in claimed if c.evidence > 0 and c.demonstrated < 0.5)
        return {"claimed_available": len(claimed), "overclaims": overclaims}

    # ---- gaps toward goals ---- #
    def _eta(self, cap: Capability) -> Optional[float]:
        if self.forecast is None or not cap.required_skills or self.skilltree is None:
            return None
        etas = []
        for s, m in cap.required_skills.items():
            if self.skilltree.proficiency(s) < m:
                t = self.forecast.time_to_mastery(s, m)
                if t is not None:
                    etas.append(t)
        return max(etas) if etas else None

    def gaps(self, goals: Sequence[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for name in goals:
            a = self.assess(name)
            if a.can:
                continue
            cap = self._caps.get(name)
            missing: Dict[str, Any] = {"capability": name, "status": a.status.value,
                                       "reason": a.reason, "eta": a.eta}
            if cap is None:
                missing["missing"] = "not registered"
            elif a.status is CapabilityStatus.LOCKED:
                missing["missing"] = f"unlock stage capability {cap.unlock!r}"
            elif cap.required_skills and self.skilltree is not None:
                short = {s: {"have": round(self.skilltree.proficiency(s), 3), "need": m}
                         for s, m in cap.required_skills.items()
                         if self.skilltree.proficiency(s) < m}
                missing["missing_skills"] = short
            out.append(missing)
        return out

    # ---- honest reporting ---- #
    def self_report(self) -> Dict[str, List[str]]:
        buckets: Dict[str, List[str]] = {s.value: [] for s in CapabilityStatus}
        for name in self._caps:
            buckets[self.assess(name).status.value].append(name)
        return {k: sorted(v) for k, v in buckets.items()}

    def report(self) -> Dict[str, Any]:
        sr = self.self_report()
        return {"capabilities": len(self._caps),
                "by_status": {k: len(v) for k, v in sr.items()},
                "calibration": self.calibration()}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.growth.developmental import build_default_curriculum
    from nyxara.growth.forecast import CapabilityForecast
    from nyxara.growth.skilltree import build_default_skilltree

    print("=" * 70)
    print("NYXARA capability self-model self-test")
    print("=" * 70)

    tree = build_default_skilltree()
    curr = build_default_curriculum()
    fc = CapabilityForecast()
    reg = CapabilityRegistry(skilltree=tree, curriculum=curr, forecast=fc)

    reg.register("summarize_text", required_skills={"language": 0.6}, unlock="converse",
                 provenance="skill:language + stage:childhood")
    reg.register("autonomous_repair", required_skills={"reasoning": 0.7, "tool_use": 0.6},
                 unlock="act_autonomous_high", provenance="skills + stage:adulthood")

    # initially LOCKED — no stage unlocked, no skill practised
    a = reg.assess("summarize_text")
    print(f"\nsummarize_text      : {a.status.value} — {a.claim}")
    assert a.status is CapabilityStatus.LOCKED and not a.can

    # unlock the stage and build the skill
    for _ in range(8):
        tree.practice("perception")
    for _ in range(12):
        tree.practice("language")
    curr.set_competence("language", 0.7)
    curr.advance("infancy")
    curr.advance("childhood")

    a = reg.assess("summarize_text")
    print(f"after growth        : {a.status.value} prof={a.proficiency:.2f} "
          f"conf={a.confidence:.2f} — {a.claim}")
    assert a.status is CapabilityStatus.AVAILABLE and a.can
    # HONESTY: never exercised -> it says 'I believe I can', confidence capped
    assert not a.verified and a.confidence <= reg.untested_cap

    # exercising it successfully raises confidence to a confident claim
    for _ in range(5):
        reg.verify("summarize_text", True)
    a = reg.assess("summarize_text")
    print(f"after verification  : verified={a.verified} conf={a.confidence:.2f} — {a.claim}")
    assert a.verified and a.confidence > reg.untested_cap

    # NO OVERCLAIMING: repeated real failures pull the claim back down
    reg.register("translate_poetry", required_skills={"language": 0.6}, unlock="converse",
                 proficiency=0.9)
    for _ in range(6):
        reg.verify("translate_poetry", False)   # it keeps failing in practice
    a = reg.assess("translate_poetry")
    print(f"\ntranslate_poetry    : status={a.status.value} conf={a.confidence:.2f} — {a.claim}")
    assert a.status is CapabilityStatus.DEVELOPING and not a.can   # claim revoked by record

    # GAPS toward a goal that needs the still-locked autonomous capability
    gaps = reg.gaps(["autonomous_repair", "summarize_text"])
    print("\ngaps:")
    for g in gaps:
        print(f"  {g['capability']:18s} {g['status']} — {g.get('missing') or g.get('missing_skills')}")
    assert any(g["capability"] == "autonomous_repair" for g in gaps)
    assert all(g["capability"] != "summarize_text" for g in gaps)   # that one is available

    # PROVENANCE is always available for audit
    print(f"\nprovenance          : {reg.get('summarize_text').provenance}")

    print(f"\nself report         : {reg.self_report()}")
    print(f"report              : {reg.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
