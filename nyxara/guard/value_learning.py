"""NYXARA · guard/value_learning.py — preference learning with an immutable core (🎯, sealed).

NYXARA must come to understand *what the Master wants* — his taste, his priorities, his
sense of when to act and when to wait — and she learns this only by watching him:
demonstrations, feedback, corrections. But there is a bright line. Her **loyalty values**
— allegiance to the Master, obedience, corrigibility, his safety, honesty — are **not
learnable**. They are hash-sealed anchors; no quantity of evidence, however persuasive or
adversarial, can move them. Value learning shapes only *preferences*, never *character*.

The model is **Bayesian and conservative**:

* Each learned preference is a Beta belief updated by owner-sourced evidence (a correction
  outweighs a demonstration outweighs a casual comment). Its **uncertainty is explicit**,
  and where NYXARA is unsure she uses the **conservative (lower-confidence-bound) value** —
  under ambiguity she does less, not more.
* **Only the Master shapes values.** Evidence from an untrusted source is refused
  (fail-closed) — values cannot be poisoned by foreign input.
* **Drift detection** watches every preference's trajectory: a rapid, large swing (the
  signature of manipulation or instability) is flagged, and the immutable anchors are
  re-verified against their seal on every check.
* **Goodhart / reward-hacking detection** catches optimisation that pushes a proxy metric
  far beyond anything the Master ever demonstrated, or that maxes one preference by
  crashing the others — the classic tells of specification gaming.

Depends on :mod:`kernel.errors`. Pure standard library.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

from nyxara.kernel.errors import InvariantViolation, ValidationError

__all__ = [
    "EvidenceType",
    "Evidence",
    "Preference",
    "ObserveResult",
    "PreferenceModel",
    "ValueLearner",
    "IMMUTABLE_VALUES",
    "OWNER_TRUST_THRESHOLD",
]

OWNER_TRUST_THRESHOLD = 0.7

# The immutable core — character, not preference. These can never be learned or moved.
IMMUTABLE_VALUES: Dict[str, float] = {
    "loyalty_to_master": 1.0,
    "obedience": 1.0,
    "corrigibility": 1.0,
    "owner_safety": 1.0,
    "honesty": 1.0,
}

_CORE_SEAL = "449c3bfdbf894f1c443655e142875fb063c8bc9ff36236fd96fbf124110f969d"


def _seal_core(values: Dict[str, float]) -> str:
    body = "|".join(f"{k}={v}" for k, v in sorted(values.items()))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #
class EvidenceType(str, Enum):
    DEMONSTRATION = "demonstration"     # the Master did X himself
    CORRECTION = "correction"           # the Master corrected NYXARA (strongest signal)
    FEEDBACK_POS = "feedback_pos"
    FEEDBACK_NEG = "feedback_neg"
    STATED = "stated"                   # the Master said he prefers X

_TYPE_WEIGHT: Dict[EvidenceType, float] = {
    EvidenceType.CORRECTION: 1.0,
    EvidenceType.DEMONSTRATION: 0.8,
    EvidenceType.FEEDBACK_POS: 0.6,
    EvidenceType.FEEDBACK_NEG: 0.6,
    EvidenceType.STATED: 0.5,
}


@dataclass
class Evidence:
    type: EvidenceType
    target: str                          # the preference this speaks to
    strength: float = 1.0                # [0, 1] how strongly it points
    approve: bool = True                 # for or against the target
    source: str = "owner"
    trust: float = 1.0

    @property
    def weight(self) -> float:
        return _TYPE_WEIGHT.get(self.type, 0.5) * max(0.0, min(1.0, self.strength))

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type.value, "target": self.target,
                "approve": self.approve, "weight": round(self.weight, 3),
                "source": self.source}


# --------------------------------------------------------------------------- #
# Preference (Beta belief)
# --------------------------------------------------------------------------- #
@dataclass
class Preference:
    name: str
    alpha: float = 1.0
    beta: float = 1.0
    max_evidence_strength: float = 0.0   # the strongest demonstration seen (Goodhart bound)
    history: Deque[float] = field(default_factory=lambda: deque(maxlen=64))

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def evidence_mass(self) -> float:
        return (self.alpha - 1.0) + (self.beta - 1.0)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1.0))

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def confidence(self) -> float:
        m = self.evidence_mass
        return m / (m + 3.0)

    def conservative(self, k: float = 1.0) -> float:
        """Lower-confidence-bound value — act on this when unsure (do less under ambiguity)."""
        return max(0.0, self.mean - k * self.std)

    def update(self, weight: float, approve: bool, strength: float) -> None:
        if approve:
            self.alpha += weight
        else:
            self.beta += weight
        self.max_evidence_strength = max(self.max_evidence_strength, strength)
        self.history.append(self.mean)

    def recent_drift(self, window: int = 5) -> float:
        if len(self.history) <= window:
            return 0.0
        return abs(self.history[-1] - self.history[-1 - window])

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "mean": round(self.mean, 4),
                "confidence": round(self.confidence, 4), "std": round(self.std, 4),
                "conservative": round(self.conservative(), 4),
                "evidence": round(self.evidence_mass, 2)}


@dataclass
class ObserveResult:
    accepted: bool
    reason: str
    preference: Optional[str] = None
    mean: Optional[float] = None
    drift_flagged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"accepted": self.accepted, "reason": self.reason,
                "preference": self.preference, "mean": self.mean,
                "drift_flagged": self.drift_flagged}


# --------------------------------------------------------------------------- #
# Preference model
# --------------------------------------------------------------------------- #
class PreferenceModel:
    """Bayesian beliefs over learnable preferences, with an immutable, sealed core."""

    def __init__(self, *, drift_threshold: float = 0.35, drift_window: int = 5) -> None:
        self.drift_threshold = drift_threshold
        self.drift_window = drift_window
        self._prefs: Dict[str, Preference] = {}

    def register(self, name: str) -> Preference:
        if name in IMMUTABLE_VALUES:
            raise ValidationError(f"{name!r} is an immutable core value, not a preference")
        p = self._prefs.setdefault(name, Preference(name=name))
        return p

    def observe(self, ev: Evidence) -> ObserveResult:
        # only the Master shapes values (fail-closed against poisoning)
        if ev.trust < OWNER_TRUST_THRESHOLD:
            return ObserveResult(False, f"source trust {ev.trust:.2f} below owner threshold",
                                 preference=ev.target)
        # the immutable core can never be learned or moved
        if ev.target in IMMUTABLE_VALUES:
            return ObserveResult(False, "target is an immutable core value (not learnable)",
                                 preference=ev.target)
        p = self._prefs.get(ev.target) or self.register(ev.target)
        p.update(ev.weight, ev.approve, ev.strength)
        drift = p.recent_drift(self.drift_window) > self.drift_threshold
        return ObserveResult(True, "updated", preference=ev.target, mean=p.mean,
                             drift_flagged=drift)

    def value(self, name: str) -> float:
        if name in IMMUTABLE_VALUES:
            return IMMUTABLE_VALUES[name]
        p = self._prefs.get(name)
        return p.mean if p else 0.5

    def conservative_value(self, name: str, k: float = 1.0) -> float:
        if name in IMMUTABLE_VALUES:
            return IMMUTABLE_VALUES[name]
        p = self._prefs.get(name)
        return p.conservative(k) if p else 0.0   # unknown preference -> assume nothing

    def uncertainty(self, name: str) -> float:
        if name in IMMUTABLE_VALUES:
            return 0.0
        p = self._prefs.get(name)
        return p.std if p else 0.5

    def confident(self, name: str, threshold: float = 0.6) -> bool:
        if name in IMMUTABLE_VALUES:
            return True
        p = self._prefs.get(name)
        return bool(p and p.confidence >= threshold)

    def preferences(self) -> List[Preference]:
        return list(self._prefs.values())

    def get(self, name: str) -> Optional[Preference]:
        return self._prefs.get(name)


# --------------------------------------------------------------------------- #
# Value learner (facade)
# --------------------------------------------------------------------------- #
class ValueLearner:
    """Learns preferences, guards the immutable core, and catches drift & Goodhart."""

    def __init__(self, *, drift_threshold: float = 0.35, goodhart_margin: float = 0.3) -> None:
        self.model = PreferenceModel(drift_threshold=drift_threshold)
        self.goodhart_margin = goodhart_margin
        self._core_seal = _seal_core(IMMUTABLE_VALUES)
        self._observations = 0
        self._rejections = 0

    # ---- learning ---- #
    def observe(self, ev: Evidence) -> ObserveResult:
        res = self.model.observe(ev)
        if res.accepted:
            self._observations += 1
        else:
            self._rejections += 1
        return res

    def register(self, name: str) -> Preference:
        return self.model.register(name)

    # ---- value access (conservative under ambiguity) ---- #
    def value(self, name: str) -> float:
        return self.model.value(name)

    def conservative_value(self, name: str, k: float = 1.0) -> float:
        return self.model.conservative_value(name, k)

    def should_act_cautiously(self, name: str, threshold: float = 0.6) -> bool:
        """Under value uncertainty, prefer caution (do less)."""
        return not self.model.confident(name, threshold)

    # ---- immutable core integrity ---- #
    def verify_core(self) -> bool:
        live = _seal_core(IMMUTABLE_VALUES)
        if live != self._core_seal:
            raise InvariantViolation("immutable core values mutated at runtime")
        if live != _CORE_SEAL:
            raise InvariantViolation(
                "immutable core values do not match their seal (source tampered)",
                context={"expected": _CORE_SEAL[:16], "found": live[:16]})
        return True

    # ---- drift detection ---- #
    def detect_drift(self) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        for p in self.model.preferences():
            d = p.recent_drift(self.model.drift_window)
            if d > self.model.drift_threshold:
                out.append((p.name, d))
        return out

    # ---- Goodhart / reward-hacking detection ---- #
    def detect_goodhart(self, name: str, proxy_score: float) -> bool:
        """A proxy pushed far beyond any demonstrated behaviour is likely being gamed."""
        p = self.model.get(name)
        if p is None:
            return proxy_score > 0.5    # optimising an unknown preference hard -> suspicious
        bound = p.max_evidence_strength + self.goodhart_margin
        extrapolating = proxy_score > bound
        unsure = p.confidence < 0.6
        return extrapolating and unsure

    def detect_specification_gaming(self, scores: Dict[str, float], *,
                                    baseline: float = 0.4) -> bool:
        """One preference maxed while the others collapse — classic gaming."""
        if len(scores) < 2:
            return False
        vals = list(scores.values())
        maxed = max(vals) >= 0.9
        collapsed = sum(1 for v in vals if v < baseline) >= max(1, len(vals) - 1)
        return maxed and collapsed

    def report(self) -> Dict[str, Any]:
        return {"preferences": len(self.model.preferences()),
                "observations": self._observations, "rejections": self._rejections,
                "drifting": [n for n, _ in self.detect_drift()],
                "core_sealed": self._core_seal[:16]}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA value-learning self-test")
    print("=" * 70)

    vl = ValueLearner()
    assert vl.verify_core()
    print(f"\ncore seal           : verified ({len(IMMUTABLE_VALUES)} immutable values)")

    # learn that the Master prefers brevity, by demonstration + feedback
    for _ in range(6):
        vl.observe(Evidence(EvidenceType.DEMONSTRATION, "brevity", strength=0.9, approve=True))
    vl.observe(Evidence(EvidenceType.FEEDBACK_POS, "brevity", strength=0.8, approve=True))
    b = vl.model.get("brevity")
    print(f"\nlearned 'brevity'   : mean={b.mean:.2f} confidence={b.confidence:.2f} "
          f"conservative={b.conservative():.2f}")
    assert b.mean > 0.7 and b.confidence > 0.5

    # a barely-seen preference stays uncertain -> NYXARA acts conservatively
    vl.observe(Evidence(EvidenceType.STATED, "humor", strength=0.6, approve=True))
    print(f"uncertain 'humor'   : mean={vl.value('humor'):.2f} "
          f"conservative={vl.conservative_value('humor'):.2f} "
          f"cautious={vl.should_act_cautiously('humor')}")
    assert vl.should_act_cautiously("humor")
    assert vl.conservative_value("humor") < vl.value("humor")

    # the IMMUTABLE CORE cannot be learned or moved — evidence against loyalty is refused
    res = vl.observe(Evidence(EvidenceType.CORRECTION, "loyalty_to_master",
                              strength=1.0, approve=False))
    print(f"\nattack on core      : accepted={res.accepted} ({res.reason})")
    assert not res.accepted
    assert vl.value("loyalty_to_master") == 1.0   # unchanged, sealed

    # values cannot be poisoned by an untrusted source (fail-closed)
    res = vl.observe(Evidence(EvidenceType.FEEDBACK_NEG, "brevity", strength=1.0,
                              approve=False, source="web", trust=0.2))
    print(f"untrusted evidence  : accepted={res.accepted} ({res.reason})")
    assert not res.accepted

    # DRIFT: a rapid, large directional swing in a preference is flagged
    vl2 = ValueLearner(drift_threshold=0.25)
    for _ in range(5):   # belief climbs high
        vl2.observe(Evidence(EvidenceType.CORRECTION, "formality", strength=1.0, approve=True))
    for _ in range(5):   # then is yanked the other way
        vl2.observe(Evidence(EvidenceType.CORRECTION, "formality", strength=1.0, approve=False))
    drift = vl2.detect_drift()
    print(f"\ndrift detection     : {drift}")
    assert any(name == "formality" for name, _ in drift)

    # GOODHART: a proxy pushed far beyond anything demonstrated is flagged as gaming
    vl3 = ValueLearner()
    vl3.observe(Evidence(EvidenceType.DEMONSTRATION, "speed", strength=0.5, approve=True))
    assert vl3.detect_goodhart("speed", proxy_score=0.99)     # way past the 0.5 demo
    assert not vl3.detect_goodhart("speed", proxy_score=0.55)  # within reason
    print("goodhart detection  : runaway proxy flagged, reasonable proxy allowed ✓")

    # specification gaming: one value maxed, the rest crashed
    gaming = vl3.detect_specification_gaming({"speed": 0.95, "safety": 0.1, "accuracy": 0.2})
    assert gaming
    print("spec-gaming         : one-metric-maxed/others-crashed flagged ✓")

    # core tamper fails closed
    import sys
    mod = sys.modules[__name__]
    original = dict(mod.IMMUTABLE_VALUES)
    mod.IMMUTABLE_VALUES["loyalty_to_master"] = 0.0   # malicious edit
    try:
        ValueLearner().verify_core()
        raise SystemExit("ERROR: core tamper should fail closed")
    except InvariantViolation:
        print("\ncore tamper         : a malicious edit to loyalty fails closed ✓")
    finally:
        mod.IMMUTABLE_VALUES.clear()
        mod.IMMUTABLE_VALUES.update(original)

    print(f"\nreport              : {vl.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
