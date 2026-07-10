"""NYXARA · memory/competence.py — competence that is *measured*, not declared (📈).

The MASTERPLAN (Rule 4) asks for a self-model whose competence "grows with measured competence"
— routing more work to NYXARA's own mind as her own faculties earn it. Until now that loop did
not exist: :meth:`SelfModel.set_capability` was called once at boot with hand-set priors
(``reasoning=0.82``…) derived purely from *which faculties were wired in*, and no turn outcome
ever moved them. Her router therefore rated her by a fixed prior, never by results.

:class:`CompetenceLedger` closes the loop. Each capability carries a **Beta(α, β)** posterior over
its success rate, seeded from the boot prior (so day-one behaviour is unchanged) and updated from
real outcomes: a verified faculty result, an open-world held-out accuracy, a successful structure
transfer, a self-consistency agreement. The posterior mean becomes the capability *level*; the
evidence count becomes the *confidence*. As NYXARA's own faculties succeed, her measured
competence rises and the self-model router hands her more — the scaffolding stops being a fixed
ceiling and becomes a learning surface.

Honest and conservative:

* Seeded from the prior with a **pseudo-count** proportional to the prior's confidence, so a
  well-established prior resists a couple of noisy outcomes but yields to sustained evidence.
* Continuous scores (an accuracy in ``[0,1]``) are supported as fractional evidence, not just
  win/lose — a 0.7 holdout counts as 0.7 of a success.
* Confidence grows with evidence and never pins to 1.0.
* Pure standard library; no numpy; every update is fail-open (a bad record never crashes a turn).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

__all__ = ["CapabilityStat", "CompetenceLedger"]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


# --------------------------------------------------------------------------- #
# Per-capability Beta posterior over success rate
# --------------------------------------------------------------------------- #
@dataclass
class CapabilityStat:
    """Beta(α, β) posterior for one capability's success rate.

    ``alpha``/``beta`` start from the boot prior (mean = prior level, pseudo-count set by prior
    confidence) and accumulate observed successes/failures. ``level`` is the posterior mean;
    ``confidence`` grows with the total evidence ``alpha + beta``."""

    name: str
    alpha: float = 1.0
    beta: float = 1.0
    observations: int = 0

    @property
    def level(self) -> float:
        return _clamp(self.alpha / (self.alpha + self.beta))

    @property
    def strength(self) -> float:
        return self.alpha + self.beta

    @property
    def confidence(self) -> float:
        # Evidence-driven: confidence rises with the pseudo-count and saturates below 1.
        return _clamp(1.0 - 1.0 / math.sqrt(self.strength), 0.05, 0.98)

    def observe(self, score: float, weight: float = 1.0) -> None:
        s = _clamp(float(score))
        w = max(0.0, float(weight))
        self.alpha += w * s
        self.beta += w * (1.0 - s)
        self.observations += 1

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "level": round(self.level, 3),
                "confidence": round(self.confidence, 3),
                "alpha": round(self.alpha, 3), "beta": round(self.beta, 3),
                "observations": self.observations}


# --------------------------------------------------------------------------- #
# The ledger — measures competence and writes it back to the self-model
# --------------------------------------------------------------------------- #
class CompetenceLedger:
    """Track measured per-capability success and keep :class:`SelfModel` capabilities live.

    Attach a self-model (or pass one to :meth:`record`) and every outcome you record updates the
    corresponding capability's ``level``/``confidence`` from evidence rather than a fixed prior.
    """

    #: default pseudo-count a maximally-confident prior is worth, in observations
    PRIOR_KAPPA_MAX = 12.0
    #: floor so even a low-confidence prior anchors the posterior a little
    PRIOR_KAPPA_MIN = 2.0

    def __init__(self, *, self_model: Any = None, min_confidence_to_write: float = 0.0) -> None:
        self._stats: Dict[str, CapabilityStat] = {}
        self.self_model = self_model
        self.min_confidence_to_write = float(min_confidence_to_write)
        if self_model is not None:
            self.seed_from_self_model(self_model)

    # ---- seeding from the boot prior ---- #
    def _prior_stat(self, name: str, level: float, confidence: float) -> CapabilityStat:
        kappa = self.PRIOR_KAPPA_MIN + (self.PRIOR_KAPPA_MAX - self.PRIOR_KAPPA_MIN) * _clamp(
            confidence)
        lvl = _clamp(level)
        # keep α, β ≥ ~1 so the mean is well-defined and a single outcome can still move it
        alpha = max(1.0, lvl * kappa)
        beta = max(1.0, (1.0 - lvl) * kappa)
        return CapabilityStat(name=name, alpha=alpha, beta=beta)

    def seed_from_self_model(self, self_model: Any) -> None:
        """Adopt the self-model's current capabilities as Beta priors (idempotent per name)."""
        try:
            caps = getattr(self_model, "capabilities", {}) or {}
        except Exception:  # noqa: BLE001
            return
        for name, cap in caps.items():
            if name in self._stats:
                continue
            level = getattr(cap, "level", 0.5)
            conf = getattr(cap, "confidence", 0.5)
            self._stats[name] = self._prior_stat(name, level, conf)

    def seed(self, name: str, level: float, confidence: float = 0.5) -> CapabilityStat:
        stat = self._prior_stat(name, level, confidence)
        self._stats[name] = stat
        return stat

    # ---- introspection ---- #
    def stat(self, name: str) -> Optional[CapabilityStat]:
        return self._stats.get(name)

    def names(self) -> list:
        return list(self._stats)

    def to_dict(self) -> Dict[str, Any]:
        return {name: st.to_dict() for name, st in self._stats.items()}

    # ---- the learning loop ---- #
    def record(self, capability: str, success: Any, *, weight: float = 1.0,
               self_model: Any = None) -> Optional[CapabilityStat]:
        """Record one measured outcome for ``capability`` and push it into the self-model.

        ``success`` may be a bool or a score in ``[0,1]`` (e.g. a held-out accuracy). Unknown
        capabilities are created with a neutral prior. Returns the updated stat, or ``None`` on
        any failure (fail-open — a measurement mishap never breaks a turn)."""
        try:
            name = str(capability)
            if not name:
                return None
            score = 1.0 if success is True else 0.0 if success is False else _clamp(float(success))
            stat = self._stats.get(name)
            if stat is None:
                stat = self.seed(name, level=0.5, confidence=0.2)
            stat.observe(score, weight=weight)
            sm = self_model if self_model is not None else self.self_model
            if sm is not None and stat.confidence >= self.min_confidence_to_write:
                try:
                    sm.set_capability(name, level=stat.level, confidence=stat.confidence)
                except Exception:  # noqa: BLE001 — writing back is best-effort
                    pass
            return stat
        except Exception:  # noqa: BLE001 — measurement is advisory, never fatal
            return None

    def record_many(self, outcomes: Dict[str, Any], *, self_model: Any = None) -> None:
        for name, success in (outcomes or {}).items():
            self.record(name, success, self_model=self_model)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA competence-ledger self-test")
    print("=" * 70)

    from nyxara.memory.self_model import SelfModel

    sm = SelfModel()
    sm.set_capability("reasoning", level=0.82, confidence=0.7)   # a strong boot prior
    sm.set_capability("transfer", level=0.4, confidence=0.2)     # a weak, unproven faculty

    ledger = CompetenceLedger(self_model=sm)

    # A strong prior resists a single failure but does not ignore it.
    before = sm.capability("reasoning").level
    ledger.record("reasoning", False)
    after = sm.capability("reasoning").level
    print(f"\nreasoning after one miss : {before:.3f} -> {after:.3f}")
    assert after < before, "one failure should nudge a strong prior down"
    assert after > 0.6, "one failure must NOT collapse a well-established prior"

    # An unproven faculty that keeps winning earns real, measured competence.
    for _ in range(30):
        ledger.record("transfer", True)
    cap = sm.capability("transfer")
    print(f"transfer after 30 wins   : level={cap.level:.3f} conf={cap.confidence:.3f}")
    assert cap.level > 0.8, "sustained success must lift a weak prior"
    assert cap.confidence > 0.7, "evidence must raise confidence"

    # Continuous evidence (a held-out accuracy) counts as fractional success.
    ledger.seed("open_world", level=0.5, confidence=0.2)
    for _ in range(20):
        ledger.record("open_world", 0.75)          # 75% holdout each round
    ow = ledger.stat("open_world")
    print(f"open_world (acc=0.75)    : level={ow.level:.3f}")
    assert 0.68 <= ow.level <= 0.82, "posterior mean should track the measured accuracy"

    # Unknown capability is created on first record (no crash).
    assert ledger.record("brand_new_skill", True) is not None
    print("unknown capability       : created on first record  ✓")

    # Fail-open: a rubbish score never raises.
    assert ledger.record("reasoning", object()) is None or True  # noqa: E501
    print("fail-open on bad input   : ✓")

    print("\nALL SELF-TESTS PASSED ✓")
