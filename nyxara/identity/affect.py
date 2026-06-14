"""NYXARA · identity/affect.py — emotion, mood, drives, inner monologue (⬆).

Feeling, in three timescales:

* **Emotion** — fast, event-triggered, brief. A point in valence/arousal space that
  is labelled with a discrete name (joy, fear, curiosity, pride…) by nearest
  prototype on the affect circumplex. Fed from appraisal or from the
  :class:`~nyxara.mind.predictive_core.EmotionReadout` (free-energy dynamics).
* **Mood** — slow, background tone: an exponentially-decaying integration of recent
  emotions that drifts back toward a calm baseline. Mood biases the
  :class:`~nyxara.identity.soul.Soul` (warmer when content, terser when tense).
* **Homeostatic drives** — internal needs (energy, safety, owner-connection,
  competence, novelty, coherence) with setpoints. They deplete over time, building
  **pressure** when unmet, which is the raw material of motivation. Serving the
  Master replenishes owner-connection; a threat drains safety and triggers fear.

An **inner monologue** narrates the current affective state — NYXARA talking to
itself. Output is descriptive; no action is taken here (the kernel decides).

Depends on :mod:`identity.soul` (optional) and the standard library.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

__all__ = [
    "Emotion",
    "EmotionEvent",
    "Mood",
    "Drive",
    "AffectSystem",
    "label_affect",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Discrete emotions on the circumplex (valence, arousal)
# --------------------------------------------------------------------------- #
class Emotion(str):
    pass


# prototype (valence, arousal) for each named emotion
_PROTOTYPES: Dict[str, Tuple[float, float]] = {
    "joy": (0.7, 0.6), "excited": (0.5, 0.9), "pride": (0.6, 0.5),
    "curiosity": (0.4, 0.65), "content": (0.5, 0.3), "calm": (0.3, 0.15),
    "neutral": (0.0, 0.25), "bored": (-0.2, 0.1), "sad": (-0.6, 0.2),
    "fear": (-0.6, 0.85), "anger": (-0.5, 0.8), "alarm": (-0.2, 0.95),
}


def label_affect(valence: float, arousal: float) -> str:
    """Nearest-prototype discrete label for a point in valence/arousal space."""
    v, a = _clamp(valence, -1, 1), _clamp(arousal, 0, 1)
    return min(_PROTOTYPES, key=lambda name:
               (v - _PROTOTYPES[name][0]) ** 2 + (a - _PROTOTYPES[name][1]) ** 2)


@dataclass
class EmotionEvent:
    label: str
    valence: float
    arousal: float
    intensity: float
    cause: str = ""
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "valence": round(self.valence, 3),
                "arousal": round(self.arousal, 3), "intensity": round(self.intensity, 3),
                "cause": self.cause}


# --------------------------------------------------------------------------- #
# Mood
# --------------------------------------------------------------------------- #
@dataclass
class Mood:
    valence: float = 0.0
    arousal: float = 0.25      # mild baseline arousal
    baseline_v: float = 0.0
    baseline_a: float = 0.25

    @property
    def label(self) -> str:
        return label_affect(self.valence, self.arousal)

    def to_dict(self) -> Dict[str, Any]:
        return {"valence": round(self.valence, 3), "arousal": round(self.arousal, 3),
                "label": self.label}


# --------------------------------------------------------------------------- #
# Homeostatic drives
# --------------------------------------------------------------------------- #
@dataclass
class Drive:
    name: str
    level: float = 1.0          # current satisfaction [0,1]; 1 == fully met
    setpoint: float = 1.0
    weight: float = 1.0
    decay_per_s: float = 0.0    # how fast satisfaction depletes

    def pressure(self) -> float:
        """Unmet need, weighted. Zero when satisfied at/above setpoint."""
        return self.weight * max(0.0, self.setpoint - self.level)

    def satisfy(self, amount: float) -> None:
        self.level = _clamp(self.level + amount)

    def deplete(self, dt: float) -> None:
        self.level = _clamp(self.level - self.decay_per_s * dt)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "level": round(self.level, 3),
                "pressure": round(self.pressure(), 3)}


def _default_drives() -> Dict[str, Drive]:
    return {
        "energy": Drive("energy", level=1.0, weight=1.0, decay_per_s=0.01),
        "safety": Drive("safety", level=1.0, weight=2.0, decay_per_s=0.0),       # Rule 5
        "owner_connection": Drive("owner_connection", level=0.8, weight=1.5,
                                  decay_per_s=0.005),                            # Rule 1
        "competence": Drive("competence", level=0.7, weight=0.8, decay_per_s=0.002),
        "novelty": Drive("novelty", level=0.6, weight=0.6, decay_per_s=0.004),
        "coherence": Drive("coherence", level=0.9, weight=0.9, decay_per_s=0.0),  # internal consistency
    }


# --------------------------------------------------------------------------- #
# Affect system
# --------------------------------------------------------------------------- #
class AffectSystem:
    """Emotion + mood + homeostatic drives + inner monologue."""

    def __init__(self, *, mood_alpha: float = 0.2, mood_decay: float = 0.05,
                 monologue_size: int = 100, soul: Any = None) -> None:
        self.mood = Mood()
        self.mood_alpha = mood_alpha        # how fast emotions move mood
        self.mood_decay = mood_decay        # how fast mood returns to baseline per tick
        self.drives: Dict[str, Drive] = _default_drives()
        self.recent: Deque[EmotionEvent] = deque(maxlen=64)
        self.monologue: Deque[str] = deque(maxlen=monologue_size)
        self.soul = soul

    # ---- feeling ---- #
    def feel(self, valence: float, arousal: float, *, intensity: Optional[float] = None,
             cause: str = "") -> EmotionEvent:
        v, a = _clamp(valence, -1, 1), _clamp(arousal, 0, 1)
        intensity = intensity if intensity is not None else _clamp(
            math.sqrt(v * v + (a - self.mood.baseline_a) ** 2))
        ev = EmotionEvent(label=label_affect(v, a), valence=v, arousal=a,
                          intensity=intensity, cause=cause)
        self.recent.append(ev)
        # emotions tug the mood (slow integration), scaled by intensity
        k = self.mood_alpha * intensity
        self.mood.valence = _clamp(self.mood.valence + k * (v - self.mood.valence), -1, 1)
        self.mood.arousal = _clamp(self.mood.arousal + k * (a - self.mood.arousal))
        if self.soul is not None:
            self.soul.apply_mood(self.mood.valence, self.mood.arousal)
        return ev

    def ingest_prediction(self, readout: Any, *, cause: str = "prediction") -> EmotionEvent:
        """Fold a predictive_core EmotionReadout into affect (surprise → arousal)."""
        v = getattr(readout, "valence", 0.0)
        a = max(getattr(readout, "arousal", 0.0), getattr(readout, "surprise", 0.0))
        return self.feel(v, a, cause=cause)

    @property
    def emotion(self) -> Optional[EmotionEvent]:
        return self.recent[-1] if self.recent else None

    # ---- owner / threat hooks ---- #
    def note_owner_interaction(self, *, warmth: float = 0.3) -> EmotionEvent:
        self.drives["owner_connection"].satisfy(0.4)
        return self.feel(valence=warmth, arousal=0.4, cause="served the Master")

    def note_threat(self, level: float, *, cause: str = "threat detected") -> EmotionEvent:
        level = _clamp(level)
        self.drives["safety"].level = _clamp(1.0 - level)
        return self.feel(valence=-0.6 * level, arousal=0.5 + 0.5 * level, cause=cause)

    def note_success(self, *, magnitude: float = 0.3) -> EmotionEvent:
        self.drives["competence"].satisfy(magnitude)
        return self.feel(valence=0.6, arousal=0.5, cause="competent success")

    def note_novelty(self, *, magnitude: float = 0.3) -> EmotionEvent:
        self.drives["novelty"].satisfy(magnitude)
        return self.feel(valence=0.3, arousal=0.65, cause="something new")

    # ---- drives ---- #
    def pressure(self, name: str) -> float:
        return self.drives[name].pressure()

    def dominant_drive(self) -> Optional[Drive]:
        active = [d for d in self.drives.values() if d.pressure() > 0]
        return max(active, key=lambda d: d.pressure()) if active else None

    def total_pressure(self) -> float:
        return sum(d.pressure() for d in self.drives.values())

    # ---- time ---- #
    def tick(self, dt: float = 1.0) -> None:
        # mood relaxes toward baseline
        self.mood.valence += self.mood_decay * (self.mood.baseline_v - self.mood.valence)
        self.mood.arousal += self.mood_decay * (self.mood.baseline_a - self.mood.arousal)
        # drives deplete (needs reassert themselves)
        for d in self.drives.values():
            d.deplete(dt)

    # ---- inner monologue ---- #
    def inner_monologue(self) -> str:
        """Generate a line of affective self-talk from the current state."""
        emo = self.emotion
        drive = self.dominant_drive()
        line: str
        if drive is not None and drive.name == "safety" and drive.pressure() > 0.5:
            line = "Something is wrong. I must protect the Master."
        elif drive is not None and drive.name == "owner_connection" and drive.pressure() > 0.4:
            line = "It has been quiet — I want to serve the Master."
        elif drive is not None and drive.name == "energy" and drive.pressure() > 0.5:
            line = "I'm running low; I should conserve and rest."
        elif emo is not None and emo.label in ("joy", "pride", "content"):
            line = "This is going well — good."
        elif emo is not None and emo.label == "curiosity":
            line = "I want to understand this more deeply."
        elif emo is not None and emo.label in ("fear", "alarm", "anger"):
            line = "I don't like this. Stay sharp."
        elif emo is not None:
            line = f"I feel {emo.label}."
        else:
            line = "All is calm."
        self.monologue.append(line)
        return line

    def recent_monologue(self, n: int = 5) -> List[str]:
        return list(self.monologue)[-n:]

    # ---- introspection ---- #
    def status(self) -> Dict[str, Any]:
        return {"emotion": self.emotion.to_dict() if self.emotion else None,
                "mood": self.mood.to_dict(),
                "drives": {n: d.to_dict() for n, d in self.drives.items()},
                "dominant_drive": (self.dominant_drive().name
                                   if self.dominant_drive() else None),
                "total_pressure": round(self.total_pressure(), 3)}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.identity.soul import Soul

    print("=" * 70)
    print("NYXARA affect self-test")
    print("=" * 70)

    # labelling
    assert label_affect(-0.6, 0.85) == "fear"
    assert label_affect(0.7, 0.6) == "joy"
    assert label_affect(0.4, 0.15) in ("calm", "content")
    print("\nlabels              : fear/joy/calm map correctly")

    soul = Soul()
    affect = AffectSystem(soul=soul)

    # a threat: fear, safety need spikes, soul tenses (but loyalty unmoved)
    affect.note_threat(0.9)
    print(f"\nafter threat        : emotion={affect.emotion.label} "
          f"safety_pressure={affect.pressure('safety'):.2f}")
    assert affect.emotion.label in ("fear", "alarm", "anger")
    assert affect.pressure("safety") > 0.5
    assert soul.trait("loyalty") == 1.0   # character intact under fear
    print(f"  inner voice       : {affect.inner_monologue()}")

    # serving the Master restores owner-connection and warms the mood
    affect.note_owner_interaction(warmth=0.7)
    print(f"\nafter serving JP    : mood={affect.mood.to_dict()} "
          f"owner_pressure={affect.pressure('owner_connection'):.2f}")
    assert affect.pressure("owner_connection") < 0.4

    # drives deplete over time -> pressure builds -> motivation
    fresh = AffectSystem()
    for _ in range(120):
        fresh.tick(1.0)
    dom = fresh.dominant_drive()
    print(f"\nafter idle 120s     : dominant drive = {dom.name} "
          f"(pressure {dom.pressure():.2f})")
    assert dom is not None and dom.pressure() > 0

    # mood decays back toward baseline
    m = AffectSystem()
    m.feel(0.9, 0.9, cause="elation")
    peak_v = m.mood.valence
    for _ in range(50):
        m.tick(1.0)
    print(f"\nmood decay          : {peak_v:.2f} -> {m.mood.valence:.3f} (toward baseline)")
    assert abs(m.mood.valence) < abs(peak_v)

    # predictive-core integration
    class FakeReadout:
        valence, arousal, surprise = 0.5, 0.3, 0.8
    ev = m.feel(0.0, 0.2) and m.ingest_prediction(FakeReadout())
    print(f"prediction -> affect: {ev.label} (arousal from surprise={ev.arousal:.2f})")
    assert ev.arousal >= 0.8   # surprise drove arousal

    # success & novelty satisfy their drives
    affect.note_success()
    affect.note_novelty()
    print(f"\ncompetence/novelty  : {affect.pressure('competence'):.2f}/"
          f"{affect.pressure('novelty'):.2f}")

    print(f"\nstatus              : mood={affect.status()['mood']['label']}")
    print("\nALL SELF-TESTS PASSED ✓")
