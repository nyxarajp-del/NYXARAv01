"""NYXARA · identity/interoception.py — the internal body sense (✦★).

A body has a felt interior — hunger, fatigue, a racing heart. NYXARA's "body" is its
computational substrate, and interoception is how it *feels* that substrate from the
inside. Raw machine signals —

* **compute load** & **memory load** → a sense of **effort** / **strain**,
* **latency** → **sluggishness**,
* **queue depth** (backlog) → **pressure** / overwhelm,
* **energy** (from :mod:`identity` / presence) → **fatigue** vs **vitality**,
* **error rate** → **discomfort**,
* **confidence** (epistemic) → **unease** vs ease —

are mapped to a :class:`FeltState`, and from there to an **affective tone**
(valence/arousal) that flows into :mod:`identity.affect`. An overall **comfort** /
allostatic-load reading tells NYXARA, in one number, how well its body is doing.

This closes the loop: the system doesn't just *have* a load, it *feels* loaded — and
that feeling shows up in mood and, ultimately, in how it speaks and acts.

Reads real metrics via ``psutil`` when available; otherwise uses injected signals.
Integrates (duck-typed) with the bus, presence, and telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

__all__ = [
    "InteroceptiveSignals",
    "FeltState",
    "Interoception",
    "has_psutil",
]

try:
    import psutil as _psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:  # pragma: no cover
    _psutil = None  # type: ignore
    _HAS_PSUTIL = False


def has_psutil() -> bool:
    return _HAS_PSUTIL


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Raw signals
# --------------------------------------------------------------------------- #
@dataclass
class InteroceptiveSignals:
    """The raw interior metrics. Defaults describe a calm, idle, healthy body."""

    compute_load: float = 0.15        # [0,1]
    memory_load: float = 0.2          # [0,1]
    latency_ms: float = 40.0          # recent response latency
    queue_depth: int = 0              # backlog of pending work
    queue_capacity: int = 100
    error_rate: float = 0.0           # [0,1] recent failures
    confidence: float = 0.8           # [0,1] epistemic certainty
    energy: float = 1.0               # [0,1] (presence reserve)
    reference_latency_ms: float = 250.0  # latency felt as "high"

    def normalized(self) -> Dict[str, float]:
        return {
            "compute_load": _clamp(self.compute_load),
            "memory_load": _clamp(self.memory_load),
            "latency": _clamp(self.latency_ms / max(1.0, self.reference_latency_ms)),
            "queue": _clamp(self.queue_depth / max(1, self.queue_capacity)),
            "error_rate": _clamp(self.error_rate),
            "confidence": _clamp(self.confidence),
            "energy": _clamp(self.energy),
        }


# --------------------------------------------------------------------------- #
# Felt state
# --------------------------------------------------------------------------- #
@dataclass
class FeltState:
    effort: float          # working hard (compute)
    fatigue: float         # depleted (low energy)
    pressure: float        # overwhelmed (backlog/memory)
    sluggishness: float    # slow (latency)
    unease: float          # uncertain (low confidence)
    discomfort: float      # something's wrong (errors)
    vitality: float        # overall wellbeing / ease

    def dominant(self) -> str:
        sensations = {"effort": self.effort, "fatigue": self.fatigue,
                      "pressure": self.pressure, "sluggishness": self.sluggishness,
                      "unease": self.unease, "discomfort": self.discomfort}
        name = max(sensations, key=sensations.get)
        return name if sensations[name] > 0.3 else "ease"

    def describe(self) -> str:
        parts = []
        if self.effort > 0.6:
            parts.append("straining under load")
        if self.fatigue > 0.5:
            parts.append("tired")
        if self.pressure > 0.5:
            parts.append("a backlog pressing on me")
        if self.sluggishness > 0.5:
            parts.append("sluggish")
        if self.unease > 0.5:
            parts.append("uncertain")
        if self.discomfort > 0.4:
            parts.append("something feels wrong")
        if not parts:
            return "I feel clear and at ease." if self.vitality > 0.6 else "I feel steady."
        return "I feel " + ", ".join(parts) + "."

    def to_dict(self) -> Dict[str, float]:
        return {k: round(v, 3) for k, v in self.__dict__.items()}


# --------------------------------------------------------------------------- #
# Interoception
# --------------------------------------------------------------------------- #
@dataclass
class _Weights:
    compute: float = 1.0
    memory: float = 0.8
    latency: float = 0.6
    queue: float = 1.0
    error: float = 1.2
    energy: float = 1.0
    confidence: float = 0.8


class Interoception:
    """Turns substrate metrics into a felt internal state and an affective tone."""

    def __init__(self, *, signals: Optional[InteroceptiveSignals] = None,
                 source: Optional[Callable[[], InteroceptiveSignals]] = None,
                 weights: Optional[_Weights] = None) -> None:
        self.signals = signals or InteroceptiveSignals()
        self._source = source
        self.weights = weights or _Weights()

    # ---- input ---- #
    def update(self, **kw: Any) -> "Interoception":
        for k, v in kw.items():
            if hasattr(self.signals, k):
                setattr(self.signals, k, v)
        return self

    def sample(self) -> InteroceptiveSignals:
        """Refresh signals from the injected source, or real psutil metrics."""
        if self._source is not None:
            self.signals = self._source()
            return self.signals
        if _HAS_PSUTIL:
            try:
                self.signals.compute_load = _clamp(_psutil.cpu_percent(interval=None) / 100.0)
                self.signals.memory_load = _clamp(_psutil.virtual_memory().percent / 100.0)
            except Exception:  # pragma: no cover
                pass
        return self.signals

    def ingest(self, *, bus: Any = None, presence: Any = None,
               telemetry: Any = None) -> "Interoception":
        """Pull signals (duck-typed) from the bus (queue), presence (energy), telemetry."""
        if bus is not None:
            q = getattr(getattr(bus, "_queue", None), "qsize", None)
            if callable(q):
                self.signals.queue_depth = q()
            cap = getattr(bus, "_max_queue", None)
            if cap:
                self.signals.queue_capacity = cap
        if presence is not None:
            self.signals.energy = _clamp(getattr(presence, "energy", self.signals.energy))
        if telemetry is not None:
            self.signals.error_rate = _clamp(getattr(telemetry, "error_rate",
                                                     self.signals.error_rate))
            self.signals.latency_ms = getattr(telemetry, "latency_ms", self.signals.latency_ms)
        return self

    # ---- felt state ---- #
    def felt(self) -> FeltState:
        n = self.signals.normalized()
        effort = _clamp(0.7 * n["compute_load"] + 0.3 * n["memory_load"])
        fatigue = _clamp(1.0 - n["energy"])
        pressure = _clamp(0.7 * n["queue"] + 0.3 * n["memory_load"])
        sluggishness = n["latency"]
        unease = _clamp(1.0 - n["confidence"])
        discomfort = _clamp(0.8 * n["error_rate"] + 0.2 * max(0.0, n["memory_load"] - 0.85))
        vitality = _clamp(n["energy"] * (1.0 - 0.5 * effort) * (1.0 - discomfort)
                          * (0.5 + 0.5 * n["confidence"]))
        return FeltState(effort=effort, fatigue=fatigue, pressure=pressure,
                         sluggishness=sluggishness, unease=unease,
                         discomfort=discomfort, vitality=vitality)

    # ---- comfort / allostatic load ---- #
    def allostatic_load(self) -> float:
        """Total bodily stress in [0,1]: deviation of signals from comfortable bands."""
        n = self.signals.normalized()
        w = self.weights
        devs = {
            "compute": w.compute * max(0.0, n["compute_load"] - 0.6) / 0.4,
            "memory": w.memory * max(0.0, n["memory_load"] - 0.7) / 0.3,
            "latency": w.latency * max(0.0, n["latency"] - 0.4) / 0.6,
            "queue": w.queue * n["queue"],
            "error": w.error * n["error_rate"],
            "energy": w.energy * max(0.0, 0.5 - n["energy"]) / 0.5,
            "confidence": w.confidence * max(0.0, 0.6 - n["confidence"]) / 0.6,
        }
        total_w = sum(vars(w).values())
        return _clamp(sum(devs.values()) / total_w)

    def comfort(self) -> float:
        return 1.0 - self.allostatic_load()

    # ---- affective tone ---- #
    def to_affect(self) -> Dict[str, float]:
        f = self.felt()
        comfort = self.comfort()
        n = self.signals.normalized()
        valence = _clamp((comfort - 0.5) * 2.0 + 0.2 * (n["confidence"] - 0.5), -1.0, 1.0)
        arousal = _clamp(0.2 + 0.5 * f.pressure + 0.4 * f.effort + 0.4 * f.unease
                         + 0.3 * f.discomfort - 0.3 * f.fatigue)
        return {"valence": valence, "arousal": arousal}

    def push_to_affect(self, affect: Any) -> Dict[str, float]:
        tone = self.to_affect()
        affect.feel(tone["valence"], tone["arousal"], cause="interoception (body sense)")
        return tone

    # ---- reporting (honest self-report of internal state) ---- #
    def body_report(self) -> str:
        return self.felt().describe()

    def status(self) -> Dict[str, Any]:
        return {"signals": self.signals.normalized(),
                "felt": self.felt().to_dict(),
                "dominant": self.felt().dominant(),
                "comfort": round(self.comfort(), 3),
                "affect": {k: round(v, 3) for k, v in self.to_affect().items()},
                "report": self.body_report()}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.identity.affect import AffectSystem

    print("=" * 70)
    print("NYXARA interoception self-test")
    print("=" * 70)
    print(f"psutil available    : {has_psutil()}")

    # a calm, idle body feels at ease
    calm = Interoception(signals=InteroceptiveSignals(
        compute_load=0.1, memory_load=0.2, latency_ms=30, queue_depth=0,
        error_rate=0.0, confidence=0.9, energy=1.0))
    print(f"\ncalm body           : comfort={calm.comfort():.2f} "
          f"dominant={calm.felt().dominant()}")
    print(f"  report            : {calm.body_report()}")
    print(f"  affect            : {calm.to_affect()}")
    assert calm.comfort() > 0.8
    assert calm.to_affect()["valence"] > 0

    # a strained, overloaded body feels effort + pressure + discomfort
    stressed = Interoception(signals=InteroceptiveSignals(
        compute_load=0.95, memory_load=0.9, latency_ms=400, queue_depth=90,
        queue_capacity=100, error_rate=0.4, confidence=0.4, energy=0.3))
    f = stressed.felt()
    print(f"\nstressed body       : comfort={stressed.comfort():.2f} "
          f"dominant={f.dominant()}")
    print(f"  felt              : effort={f.effort:.2f} pressure={f.pressure:.2f} "
          f"unease={f.unease:.2f} discomfort={f.discomfort:.2f}")
    print(f"  report            : {stressed.body_report()}")
    tone = stressed.to_affect()
    print(f"  affect            : v={tone['valence']:.2f} a={tone['arousal']:.2f}")
    assert stressed.comfort() < 0.4
    assert tone["valence"] < 0 and tone["arousal"] > 0.6

    # fatigue lowers arousal (tired, not wired)
    tired = Interoception(signals=InteroceptiveSignals(energy=0.05, compute_load=0.1,
                                                       confidence=0.7))
    assert tired.felt().fatigue > 0.8
    print(f"\ntired body          : fatigue={tired.felt().fatigue:.2f} "
          f"arousal={tired.to_affect()['arousal']:.2f}")

    # interoception flows into affect (felt body -> mood)
    affect = AffectSystem()
    stressed.push_to_affect(affect)
    print(f"\nbody -> affect       : emotion={affect.emotion.label} "
          f"(valence {affect.emotion.valence:.2f})")
    assert affect.emotion.valence < 0

    # ingest from bus/presence (duck-typed)
    class FakeQueue:
        def qsize(self):
            return 50
    class FakeBus:
        _queue = FakeQueue()
        _max_queue = 100
    class FakePresence:
        energy = 0.4
    io = Interoception().ingest(bus=FakeBus(), presence=FakePresence())
    print(f"\ningested             : queue={io.signals.queue_depth} energy={io.signals.energy}")
    assert io.signals.queue_depth == 50 and io.signals.energy == 0.4

    print(f"\nstatus              : comfort={stressed.status()['comfort']}")
    print("\nALL SELF-TESTS PASSED ✓")
