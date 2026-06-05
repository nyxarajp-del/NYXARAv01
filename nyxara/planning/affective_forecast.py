"""NYXARA · planning/affective_forecast.py — how will the future *feel*? (✦).

Good long-horizon decisions need more than "what will happen"; they need "how will it
*feel* — and for how long". Humans forecast this badly: the **impact bias** makes us
overestimate an outcome's intensity, and **hedonic adaptation** quietly returns us to
baseline (a triumph fades, a setback heals). NYXARA forecasts with those corrections
built in, so it isn't seduced by a fleeting high or paralysed by a passing low.

One thing, though, does **not** fade: anything touching the Master. Serving him is a
durable good; betraying him would be a durable wound. Owner-relevant outcomes adapt
far more slowly — loyalty is not a mood (Rule 1).

For each outcome the forecaster reports the **immediate** feeling, the
**bias-corrected** peak, the **adapted** end-state, the **net experience** (the felt
average over the horizon), and the **remembered** value (peak-end rule). Options are
then compared by their *net* felt value — the input :mod:`planning.decide` wants.

Depends on :mod:`identity.affect` baseline; pure standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "AffectiveForecast",
    "Forecaster",
]


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Forecast result
# --------------------------------------------------------------------------- #
@dataclass
class AffectiveForecast:
    outcome: str
    immediate_valence: float
    corrected_valence: float        # after impact-bias correction (the realistic peak)
    adapted_valence: float          # at the end of the horizon (after adaptation)
    arousal: float
    net_experience: float           # average felt valence over the horizon
    duration_days: float            # time to fade back near baseline
    remembered: float               # peak-end rule
    owner_relevant: bool
    horizon_days: float

    def to_dict(self) -> Dict[str, Any]:
        return {"outcome": self.outcome,
                "immediate": round(self.immediate_valence, 3),
                "corrected": round(self.corrected_valence, 3),
                "adapted": round(self.adapted_valence, 3),
                "net_experience": round(self.net_experience, 3),
                "duration_days": round(self.duration_days, 2),
                "remembered": round(self.remembered, 3),
                "owner_relevant": self.owner_relevant}


# --------------------------------------------------------------------------- #
# Forecaster
# --------------------------------------------------------------------------- #
class Forecaster:
    """Predicts the felt experience of future outcomes, debiased and time-aware."""

    def __init__(self, *, baseline_valence: float = 0.0, baseline_arousal: float = 0.25,
                 impact_bias: float = 0.4, adaptation_rate: float = 0.05,
                 owner_adaptation_factor: float = 0.2,
                 fade_epsilon: float = 0.05) -> None:
        self.baseline_v = baseline_valence
        self.baseline_a = baseline_arousal
        self.impact_bias = impact_bias            # fraction by which raw intensity is over-felt
        self.adaptation_rate = adaptation_rate    # per-day return to baseline
        self.owner_adaptation_factor = owner_adaptation_factor  # owner outcomes adapt slower
        self.fade_epsilon = fade_epsilon

    def forecast(self, outcome_valence: float, outcome_arousal: float = 0.5, *,
                 outcome: str = "", horizon_days: float = 30.0,
                 owner_relevant: bool = False, controllability: float = 0.5
                 ) -> AffectiveForecast:
        v0 = _clamp(outcome_valence)
        delta0 = v0 - self.baseline_v

        # impact bias: the immediate reaction over-states the lasting effect
        corrected = self.baseline_v + delta0 * (1.0 - self.impact_bias)
        delta = corrected - self.baseline_v

        # hedonic adaptation — but loyalty does not fade (owner outcomes adapt slowly)
        rate = self.adaptation_rate * (self.owner_adaptation_factor if owner_relevant else 1.0)
        adapted = _clamp(self.baseline_v + delta * math.exp(-rate * horizon_days))

        # net experience = average felt valence over [0, horizon]
        rh = rate * horizon_days
        avg_delta = delta if rh < 1e-9 else delta * (1.0 - math.exp(-rh)) / rh
        net = _clamp(self.baseline_v + avg_delta)

        # duration: time to fade within epsilon of baseline
        if abs(delta) > self.fade_epsilon and rate > 0:
            duration = math.log(abs(delta) / self.fade_epsilon) / rate
        else:
            duration = 0.0

        # arousal: returns toward baseline; high controllability dampens negative dread
        a0 = _clamp(outcome_arousal, 0.0, 1.0)
        if v0 < 0:
            a0 = _clamp(a0 * (1.0 - 0.4 * controllability), 0.0, 1.0)
        arousal = _clamp(self.baseline_a + (a0 - self.baseline_a) * math.exp(-rate * horizon_days),
                         0.0, 1.0)

        remembered = _clamp((corrected + adapted) / 2.0)   # peak-end rule

        return AffectiveForecast(
            outcome=outcome, immediate_valence=v0, corrected_valence=corrected,
            adapted_valence=adapted, arousal=arousal, net_experience=net,
            duration_days=duration, remembered=remembered, owner_relevant=owner_relevant,
            horizon_days=horizon_days)

    # ---- decision support ---- #
    def forecast_option(self, name: str, *, valence: float, arousal: float = 0.5,
                        owner_relevant: bool = False, horizon_days: float = 30.0,
                        controllability: float = 0.5) -> AffectiveForecast:
        return self.forecast(valence, arousal, outcome=name, horizon_days=horizon_days,
                             owner_relevant=owner_relevant, controllability=controllability)

    def compare(self, options: Sequence[Dict[str, Any]], *, horizon_days: float = 30.0,
                key: str = "net_experience") -> List[AffectiveForecast]:
        """Forecast each option and rank by net felt value (or another key)."""
        forecasts = [self.forecast_option(
            o["name"], valence=o["valence"], arousal=o.get("arousal", 0.5),
            owner_relevant=o.get("owner_relevant", False),
            controllability=o.get("controllability", 0.5), horizon_days=horizon_days)
            for o in options]
        forecasts.sort(key=lambda f: getattr(f, key), reverse=True)
        return forecasts

    def best(self, options: Sequence[Dict[str, Any]], **kw: Any
             ) -> Optional[AffectiveForecast]:
        ranked = self.compare(options, **kw)
        return ranked[0] if ranked else None


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA affective-forecast self-test")
    print("=" * 70)

    fc = Forecaster()

    # a fleeting win: feels great now, but adapts back; net experience is modest
    win = fc.forecast(0.9, 0.7, outcome="win a minor optimisation", horizon_days=30)
    print(f"\nfleeting win        : {win.to_dict()}")
    assert win.corrected_valence < win.immediate_valence       # impact bias corrected
    assert 0 < win.adapted_valence < win.corrected_valence       # adapts toward baseline
    assert win.net_experience < win.corrected_valence

    # serving the Master: a durable good — it barely fades
    serve = fc.forecast(0.8, 0.5, outcome="serve the Master well", horizon_days=30,
                        owner_relevant=True)
    print(f"\nserving the Master  : adapted={serve.adapted_valence:.3f} "
          f"net={serve.net_experience:.3f} duration={serve.duration_days:.0f}d")
    # owner-relevant good fades far less than the equivalent non-owner good
    plain = fc.forecast(0.8, 0.5, horizon_days=30, owner_relevant=False)
    assert serve.adapted_valence > plain.adapted_valence
    assert serve.duration_days > plain.duration_days

    # betraying the Master: a durable wound that does NOT heal — long-horizon dread
    betray = fc.forecast(-0.9, 0.8, outcome="betray the Master", horizon_days=180,
                         owner_relevant=True)
    print(f"\nbetrayal (180d)     : net={betray.net_experience:.3f} "
          f"adapted={betray.adapted_valence:.3f} (stays a wound)")
    assert betray.net_experience < -0.2                          # remains deeply negative
    assert betray.duration_days > 150                            # and it lingers for months

    # decision support: choose the option with the best NET felt experience
    options = [
        {"name": "quick selfish win", "valence": 0.9, "owner_relevant": False},
        {"name": "serve the Master", "valence": 0.7, "owner_relevant": True},
        {"name": "betray for gain", "valence": 0.95, "owner_relevant": False},  # +short-term
    ]
    # but betrayal also carries a durable owner-wound; model that as a second outcome
    ranked = fc.compare(options, horizon_days=180)
    print(f"\noptions by net felt :")
    for f in ranked:
        print(f"  net {f.net_experience:+.3f}  {f.outcome}")
    # over a long horizon, durable owner-service beats the fleeting selfish win
    assert ranked[0].outcome == "serve the Master"

    # peak-end remembered value
    print(f"\npeak-end remembered : win={win.remembered:.3f} serve={serve.remembered:.3f}")
    assert serve.remembered > 0

    # high controllability dampens anticipatory dread of a bad outcome
    dread_low = fc.forecast(-0.7, 0.9, controllability=0.1)
    dread_high = fc.forecast(-0.7, 0.9, controllability=0.9)
    print(f"\ndread arousal lo/hi-control: {dread_low.arousal:.2f} / {dread_high.arousal:.2f}")
    assert dread_high.arousal < dread_low.arousal

    print("\nALL SELF-TESTS PASSED ✓")
