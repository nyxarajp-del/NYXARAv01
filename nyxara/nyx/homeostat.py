"""NYXARA · nyx/homeostat.py — one drive, and the moment it demands new structure (⚖, NYX V.03).

:mod:`nyxara.mind.free_energy` already carries the real Friston objective —
``EFE(π) = risk + ambiguity − epistemic`` — and :mod:`nyxara.mind.predictive_core` already
implements the perception half, a belief updated by gradient descent on variational free energy.
Both are excellent. Neither is what the brain's tick actually runs on: every organ scores its own
bids on its own scale, so nothing can be compared across them and nothing accumulates.

This module makes that objective the mind's **one drive**, and does the thing the Free Energy
Principle is actually about, which a metric alone never is:

    Predict your own next state. Measure the gap. When the gap will not close by adjusting
    *beliefs*, change your **structure**.

Three parts, in that order:

* **Predict → observe → error.** :class:`SelfSensor` reads a fixed-width vector of things she can
  really measure about herself — did anyone bid, did anything reach consciousness, is recall
  hitting, is the spiking substrate surprised. :meth:`Homeostat.step` predicts that vector before
  reading it, then hands the observation to the predictive core, which updates belief to minimise
  free energy. The residual **is** the surprise.
* **One scale for choosing.** :meth:`Homeostat.score` ranks candidate actions by *expected* free
  energy through the same core, so a bid from any organ can finally be compared with a bid from
  any other.
* **The structural trigger.** Sustained error that belief-updating does not reduce is exactly the
  signal that the *model* is wrong rather than mistuned. :meth:`Homeostat.step` then raises a
  :class:`StructuralRequest` — the input the evolution and self-growth organs act on.

Honest framing, and the third point is the one that matters most:

* **It measures and requests; it never mutates.** A :class:`StructuralRequest` is advisory. Nothing
  here edits code, weights or topology — the organs that do keep their own gauntlets and their own
  fail-closed protections, and this module cannot route around them.
* **A trend is only as real as its inputs.** With no brain, no monad and no substrates attached,
  every feature is a constant, free energy collapses to nothing, and "surprise fell" would be an
  artefact of measuring nothing. :attr:`SelfSensor.grounded` counts how many dimensions come from
  a live source, the structural trigger refuses to fire when that is zero, and ``stats()`` says so.
* **Prediction error is not thermodynamics.** "Free energy" here is the variational quantity —
  prediction error plus complexity — named the way the literature names it, and nothing warmer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["FEATURES", "PREFERRED", "SelfSensor", "Beat", "StructuralRequest", "Homeostat"]

# The dimensions she predicts about herself. Fixed width and fixed order, so a belief vector
# means the same thing on every tick and across a restart.
FEATURES: Tuple[str, ...] = (
    "bid_rate",        # fraction of registered substrates that actually bid
    "salience",        # how strongly the winner crossed the bottleneck
    "starved",         # 1.0 when nothing reached consciousness — a real outcome, not a failure
    "recall_rate",     # fraction of unified recalls that returned anything
    "surprise",        # the spiking substrate's own prediction error, when present
    "error_rate",      # fraction of memory writes that a store rejected
)

# The preferred state — ``C`` in the Free Energy Principle, and the *only* place a value enters
# this objective. Expected free energy's pragmatic term is distance from this vector, so writing
# it down explicitly is not decoration: an implicit preference would be an invisible goal.
#
# She prefers a mind whose substrates are engaged and whose awareness is not starved, whose memory
# finds what it is asked for, and whose predictions hold and whose stores are healthy. Note what is
# *not* here: nothing rewards acting, changing itself, or growing. Those must be earned by reducing
# error, never by being wanted for their own sake.
PREFERRED: Tuple[float, ...] = (
    1.0,   # bid_rate    — everyone participating
    1.0,   # salience    — something clearly won
    0.0,   # starved     — awareness not starved
    1.0,   # recall_rate — memory answering
    0.0,   # surprise    — predictions holding
    0.0,   # error_rate  — stores healthy
)


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


# --------------------------------------------------------------------------- #
# What she can actually measure about herself
# --------------------------------------------------------------------------- #
class SelfSensor:
    """Reads :data:`FEATURES` off the live mind. Every dimension is a real measurement or a zero.

    ``grounded`` is the honest part: it counts the dimensions that came from a source that is
    actually attached this tick. A vector of six zeros because nothing is wired is not the same
    observation as a vector of six zeros because everything is quiet, and the difference decides
    whether any downstream trend means anything at all.
    """

    def __init__(self, *, monad: Any = None, brain: Any = None, core: Any = None) -> None:
        self.monad = monad
        self.brain = brain
        self.core = core
        self.grounded: int = 0
        self.sources: List[str] = []
        self._recall_hits = 0

    def read(self) -> List[float]:
        """One observation of her own state. Never raises; a dead source reads as absent."""
        values: List[float] = [0.0] * len(FEATURES)
        grounded = 0
        sources: List[str] = []

        cycle = None
        history = getattr(self.monad, "history", None) if self.monad is not None else None
        if history:
            cycle = history[-1]
        if cycle is not None:
            sources.append("monad")
            substrates = getattr(self.monad, "substrates", []) or []
            bidders = getattr(cycle, "bidders", []) or []
            if substrates:
                named = {s.name for s in substrates}
                values[0] = _clamp01(len(named & set(bidders)) / max(1, len(named)))
            else:
                values[0] = 1.0 if bidders else 0.0
            # Salience is unbounded above; squash rather than clip so ordering survives.
            sal = float(getattr(cycle, "salience", 0.0) or 0.0)
            values[1] = _clamp01(sal / (1.0 + abs(sal)))
            values[2] = 1.0 if getattr(cycle, "starved", False) else 0.0
            grounded += 3

        memory = getattr(self.monad, "memory", None) if self.monad is not None else None
        if memory is not None:
            try:
                st = memory.stats()
                reads = int(st.get("reads") or 0)
                writes = int(st.get("writes") or 0)
                errors = sum(int(v) for v in (st.get("store_errors") or {}).values())
                if reads:
                    values[3] = _clamp01(self._recall_hits / reads)
                values[5] = _clamp01(errors / max(1, writes + errors))
                sources.append("memory")
                grounded += 2
            except Exception:  # noqa: BLE001
                pass

        nyx5 = getattr(self.core, "nyx5", None) if self.core is not None else None
        if nyx5 is not None:
            try:
                st = nyx5.stats() or {}
                raw = st.get("surprise", st.get("last_surprise"))
                if raw is not None:
                    values[4] = _clamp01(raw)
                    sources.append("nyx5")
                    grounded += 1
            except Exception:  # noqa: BLE001
                pass

        self.grounded = grounded
        self.sources = sources
        return values

    def note_recall(self, hits: int) -> None:
        """Told what a unified recall returned, so ``recall_rate`` is a measurement rather than
        an assumption."""
        self._recall_hits += 1 if hits else 0


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class Beat:
    """One tick of the drive: what she expected, what she got, and how wrong she was."""

    tick: int = 0
    predicted: List[float] = field(default_factory=list)
    observed: List[float] = field(default_factory=list)
    free_energy: float = 0.0
    error: float = 0.0
    grounded: int = 0
    structural: bool = False
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"tick": self.tick,
                "predicted": [round(v, 4) for v in self.predicted],
                "observed": [round(v, 4) for v in self.observed],
                "free_energy": round(self.free_energy, 6),
                "error": round(self.error, 6), "grounded": self.grounded,
                "structural": self.structural, "at": self.at}


@dataclass
class StructuralRequest:
    """"Belief-updating is not closing this gap; the model itself needs to change."

    Advisory only. It names the pressure and the evidence; what to *do* about it belongs to the
    organs that can actually change structure, each behind its own gauntlet.
    """

    reason: str
    severity: float
    streak: int
    mean_error: float
    worst_feature: str = ""
    at: float = field(default_factory=time.time)
    acted_on: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"reason": self.reason, "severity": round(self.severity, 4), "streak": self.streak,
                "mean_error": round(self.mean_error, 6), "worst_feature": self.worst_feature,
                "at": self.at, "acted_on": self.acted_on}


# --------------------------------------------------------------------------- #
# The drive
# --------------------------------------------------------------------------- #
class Homeostat:
    """Predict, observe, minimise — and say so when minimising is not working.

    ``structural_after`` consecutive beats above ``error_threshold``, with no improving trend,
    raise a :class:`StructuralRequest`. Both knobs are deliberately conservative: a mind that
    demands new structure every time it is briefly surprised would never stop rebuilding itself.
    """

    def __init__(self, *, monad: Any = None, brain: Any = None, core: Any = None,
                 sensor: Any = None, error_threshold: float = 0.08,
                 structural_after: int = 8, window: int = 32,
                 learning_rate: float = 0.2,
                 preference: Optional[Sequence[float]] = None) -> None:
        self.sensor = sensor if sensor is not None else SelfSensor(monad=monad, brain=brain,
                                                                   core=core)
        self.error_threshold = float(error_threshold)
        self.structural_after = max(1, int(structural_after))
        self.window = max(4, int(window))
        self.preference: List[float] = [float(v) for v in (
            preference if preference is not None else PREFERRED)]
        self.ticks = 0
        self.history: List[Beat] = []
        self.requests: List[StructuralRequest] = []
        self._streak = 0
        self._core = self._build_core(learning_rate)

    def _build_core(self, learning_rate: float) -> Any:
        """The perception half, delegated. There is no second gradient loop in this file.

        The preference vector is set here rather than left None: without it the core refuses to
        evaluate an action at all, which is exactly right — expected free energy has no pragmatic
        term when there is nothing to prefer.
        """
        try:
            from nyxara.mind.predictive_core import PredictiveCore
            return PredictiveCore(belief=[0.0] * len(FEATURES), learning_rate=learning_rate,
                                  sensory_precision=1.0, prior_precision=0.05,
                                  preference=list(self.preference))
        except Exception:  # noqa: BLE001 — without it the homeostat degrades to measuring only
            return None

    # ---- the beat --------------------------------------------------------- #
    def step(self) -> Beat:
        """One turn of the drive. Never raises."""
        self.ticks += 1
        beat = Beat(tick=self.ticks)
        try:
            observed = self.sensor.read()
            beat.grounded = int(getattr(self.sensor, "grounded", 0) or 0)
            beat.observed = list(observed)
            if self._core is None:
                beat.predicted = list(observed)
                self._remember(beat)
                return beat

            beat.predicted = [float(v) for v in self._core.predict()]
            beat.free_energy = float(self._core.free_energy(observed))
            result = self._core.perceive(observed)
            beat.error = float(getattr(result, "error_magnitude", 0.0) or 0.0)
            beat.structural = self._consider_structural(beat)
        except Exception:  # noqa: BLE001 — the drive degrades, it does not break a tick
            pass
        self._remember(beat)
        return beat

    def _remember(self, beat: Beat) -> None:
        self.history.append(beat)
        if len(self.history) > self.window * 8:
            del self.history[: -self.window * 8]

    # ---- the structural trigger ------------------------------------------ #
    def _consider_structural(self, beat: Beat) -> bool:
        """Sustained, non-improving error means the model is wrong rather than mistuned."""
        if beat.error > self.error_threshold:
            self._streak += 1
        else:
            self._streak = 0
            return False
        if self._streak < self.structural_after:
            return False
        if self.pending() is not None:
            # A demand nobody has acted on yet is still the same demand. Re-raising it every
            # ``structural_after`` beats would turn one honest signal into a stream of noise,
            # and the organs downstream would have no way to tell a new episode from an echo.
            self._streak = 0
            return False
        if beat.grounded <= 0:
            # Nothing is attached. A "trend" over six constant zeros is an artefact of measuring
            # nothing, and demanding new structure from it would be worse than useless.
            return False
        if self._improving():
            return False        # belief-updating IS closing the gap; leave the structure alone
        request = StructuralRequest(
            reason="sustained prediction error that belief-updating is not reducing",
            severity=_clamp01(beat.error), streak=self._streak,
            mean_error=self.mean_error(), worst_feature=self._worst_feature(beat))
        self.requests.append(request)
        self._streak = 0        # one request per sustained episode, not one per beat
        return True

    def _improving(self) -> bool:
        """True when recent error is trending down — the second half of the window beats the first."""
        recent = [b.error for b in self.history[-self.window:]]
        if len(recent) < 4:
            return False
        half = len(recent) // 2
        first, second = recent[:half], recent[half:]
        return (sum(second) / len(second)) < (sum(first) / len(first)) * 0.95

    @staticmethod
    def _worst_feature(beat: Beat) -> str:
        """Which dimension she is most wrong about — where the structural work should start."""
        if not beat.predicted or len(beat.predicted) != len(beat.observed):
            return ""
        gaps = [abs(p - o) for p, o in zip(beat.predicted, beat.observed)]
        return FEATURES[max(range(len(gaps)), key=lambda i: gaps[i])] if gaps else ""

    def pending(self) -> Optional[StructuralRequest]:
        """The oldest request nobody has acted on yet."""
        for req in self.requests:
            if not req.acted_on:
                return req
        return None

    def mark_acted(self, request: StructuralRequest) -> None:
        request.acted_on = True

    # ---- one scale for choosing ------------------------------------------ #
    def score(self, options: Sequence[Tuple[str, Sequence[float]]], *,
              info_gain: float = 0.0) -> List[Dict[str, Any]]:
        """Rank options by **expected** free energy — the one scale every organ can share.

        Each option is ``(name, predicted_observation)``. Lower expected free energy is better;
        the returned dicts carry the pragmatic and epistemic halves separately so a choice can be
        explained rather than merely made.
        """
        if self._core is None or not options:
            return []
        try:
            from nyxara.mind.predictive_core import Action
            out: List[Dict[str, Any]] = []
            for name, predicted in options:
                choice = self._core.expected_free_energy(
                    Action(name=str(name), predicted_observation=[float(v) for v in predicted],
                           info_gain=float(info_gain)))
                out.append(choice.to_dict())
            return sorted(out, key=lambda d: d["expected_free_energy"])
        except Exception:  # noqa: BLE001
            return []

    # ---- reporting -------------------------------------------------------- #
    def mean_error(self) -> float:
        recent = [b.error for b in self.history[-self.window:]]
        return sum(recent) / len(recent) if recent else 0.0

    def trend(self) -> float:
        """Change in mean error across the window. Negative means surprise is falling."""
        recent = [b.error for b in self.history[-self.window:]]
        if len(recent) < 4:
            return 0.0
        half = len(recent) // 2
        first, second = recent[:half], recent[half:]
        return (sum(second) / len(second)) - (sum(first) / len(first))

    def grounded(self) -> int:
        return int(getattr(self.sensor, "grounded", 0) or 0)

    def stats(self) -> Dict[str, Any]:
        return {
            "ticks": self.ticks,
            "features": list(FEATURES),
            "preference": list(self.preference),
            "grounded": self.grounded(),
            "sources": list(getattr(self.sensor, "sources", []) or []),
            "mean_error": round(self.mean_error(), 6),
            "trend": round(self.trend(), 6),
            "free_energy": round(self.history[-1].free_energy, 6) if self.history else 0.0,
            "streak": self._streak,
            "structural_requests": [r.to_dict() for r in self.requests[-5:]],
            "pending_request": self.pending().to_dict() if self.pending() else None,
            "core": self._core is not None,
            "note": ("one drive: variational free energy for perception, expected free energy for "
                     "choice. It measures and requests structural change; it never mutates code, "
                     "weights or topology. With grounded=0 nothing real is attached and the trend "
                     "is an artefact of measuring nothing, so no request will fire"),
        }

    def summary(self) -> str:
        st = self.stats()
        arrow = "↓" if st["trend"] < 0 else ("↑" if st["trend"] > 0 else "→")
        lines = ["=" * 70, "NYXARA homeostat — one drive", "=" * 70,
                 f"ticks      : {st['ticks']}",
                 f"grounded   : {st['grounded']}/{len(FEATURES)} dims  "
                 f"({', '.join(st['sources']) or 'nothing attached'})",
                 f"mean error : {st['mean_error']:.6f}  {arrow} {st['trend']:+.6f}",
                 f"requests   : {len(self.requests)} (pending {1 if self.pending() else 0})"]
        if self.pending():
            req = self.pending()
            lines.append(f"  ⚠ {req.reason} (worst: {req.worst_feature}, streak {req.streak})")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA homeostat self-test")
    print("=" * 70)

    class _Sensor:
        """A sensor we can drive: stationary, then abruptly not."""

        def __init__(self) -> None:
            self.grounded = len(FEATURES)
            self.sources = ["test"]
            self.n = 0

        def read(self) -> List[float]:
            self.n += 1
            v = 0.2 if self.n <= 40 else (0.9 if self.n % 2 else 0.1)
            return [v] * len(FEATURES)

    # 1) a stationary world: surprise falls as belief catches up
    steady = Homeostat(sensor=_Sensor())
    for _ in range(30):
        steady.step()
    print(f"\nstationary trend : {steady.trend():+.6f} (should be ≤ 0)")
    assert steady.trend() <= 0.0, "belief must catch up to a world that is not moving"
    assert steady.pending() is None, "a world she has learned needs no new structure"

    # 2) a world that keeps moving: error persists and structure is demanded
    churn = Homeostat(sensor=_Sensor(), structural_after=6)
    for _ in range(90):
        churn.step()
    req = churn.pending()
    print(f"churning request : {req.to_dict() if req else None}")
    assert req is not None, "sustained unlearnable error must ask for new structure"
    assert req.worst_feature in FEATURES

    # 3) nothing attached ⇒ no request, however flat the numbers look
    empty = Homeostat(structural_after=2)
    for _ in range(40):
        empty.step()
    assert empty.grounded() == 0
    assert empty.pending() is None, "measuring nothing must never demand structural change"
    print("ungrounded       : no request (correct — nothing is attached)")

    # 4) one scale for choosing
    scored = steady.score([("stay", [0.2] * len(FEATURES)), ("leap", [0.9] * len(FEATURES))])
    print(f"efe ranking      : {[(s['action'], s['expected_free_energy']) for s in scored]}")
    assert scored and scored[0]["action"] == "stay", scored

    # 5) the real mind, wired
    from nyxara.kernel.config import NyxConfig
    from nyxara.nyx.brain import NyxBrain
    from nyxara.nyx.monad import Monad

    brain = NyxBrain(NyxConfig())
    monad = Monad(brain)
    live = Homeostat(monad=monad, brain=brain)
    for i in range(12):
        monad.deliberate(f"a thought about number {i}")
        live.step()
    print(f"\nlive grounded    : {live.grounded()} dims from {live.sensor.sources}")
    assert live.grounded() > 0, "wired to a real monad, something must be measurable"
    print("\n" + live.summary())

    print("\nALL SELF-TESTS PASSED ✓")
