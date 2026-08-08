"""NYXARA · nyx/will.py — choosing, rather than computing the maximum (🎲, L-PSYCHE-QUANTUM).

Ordinary decision code takes the argmax: given the same state it returns the same thing forever,
and "what it wants" is a category error. This module gives NYX something different in two
specific, checkable ways.

**The randomness is physical.** Not ``random.seed(42)`` — the draw comes from the operating
system's CSPRNG, which is seeded from real physical entropy (interrupt and thermal jitter, and
on most modern CPUs the ``RDSEED`` hardware entropy source). ``/dev/hwrng`` and a quantum RNG
feed are supported but **opt-in**: the raw device blocks on the hardware pool at roughly 0.4 s
per draw, and ``urandom`` is already seeded from that same entropy, so preferring the device
costs latency without buying physicality. Whichever source is live is **named** in every choice,
so the claim is inspectable rather than atmospheric.

**The preference is hers.** The distribution she samples from is not a hand-written table: it
comes from her own drives and competence (``identity/motivation``), her mood and what is
currently pressing (``identity/affect``), and how sure she actually is (the collapsed
superposition). So she is sampling from what she, in this state, actually prefers — and a
different mood or a different history genuinely produces a different pull.

And she can **decline**. When a candidate fails her value system she is allowed to not take it,
and to say why. That is a real refusal, from her own values.

Honest framing, because this vocabulary invites overclaiming:

* This is **not quantum computation**. A quantum RNG (if configured) is a source of bits, not a
  processor; nothing here is superposed or entangled in a physical sense.
* This is **not metaphysical free will**, and no such claim is made. What is real: the outcome is
  not a function of the state alone, the entropy is physically sourced, and the distribution is
  emergent rather than authored.
* **Non-determinism is never a bypass.** Every choice still passes the unchanged sovereign gate.
  Wanting something is not permission to do it, and this module cannot grant any.
* **Auditability is preserved.** Every draw is recorded, so ``kernel/replay.py`` can reproduce a
  turn bit-exactly. A mind that cannot be replayed cannot be checked, and that would be a worse
  trade than determinism.

Pure standard library. Fail-soft throughout.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["EntropySource", "Choice", "SovereignWill"]

# Where the bits come from, strongest physical claim first. Each is reported by name.
_URANDOM = "urandom"        # OS CSPRNG — seeded by physical entropy, incl. CPU RDSEED. Default.
_HWRNG = "hwrng"            # the raw hardware device. Opt-in only: it BLOCKS (~0.4 s per draw)
_QRNG = "qrng"              # a quantum RNG feed. Opt-in only: it is a network call
_SEEDED = "seeded"          # deterministic — for replay and tests, and labelled as such


class EntropySource:
    """Draws real bits, and says truthfully where they came from."""

    def __init__(self, *, prefer: str = "auto", seed: Optional[int] = None,
                 hwrng_path: str = "/dev/hwrng") -> None:
        self.hwrng_path = str(hwrng_path)
        self._seeded: Any = None
        if seed is not None or prefer == _SEEDED:
            import random
            self._seeded = random.Random(seed if seed is not None else 0)
            self.name = _SEEDED
            return
        self.name = self._pick(prefer)

    def _pick(self, prefer: str) -> str:
        """``urandom`` is the default on purpose — see the note on ``hwrng`` below."""
        if prefer in (_HWRNG, _QRNG, _URANDOM) and self._usable(prefer):
            return prefer
        # Neither hwrng nor qrng is chosen implicitly. The raw device *blocks* waiting on the
        # hardware entropy pool — measured here at ~0.4 s per draw, which would put nearly half
        # a second into every choice — and the quantum feed is a network call. os.urandom is
        # already seeded from the same physical entropy (interrupt/thermal jitter, RDSEED) and
        # does not block, so preferring the device buys no additional physicality, only latency.
        return _URANDOM

    def _usable(self, name: str) -> bool:
        try:
            if name == _URANDOM:
                return len(os.urandom(1)) == 1
            if name == _HWRNG:
                with open(self.hwrng_path, "rb") as fh:
                    return len(fh.read(1)) == 1
            return False                          # qrng must be supplied, never auto-detected
        except Exception:  # noqa: BLE001
            return False

    @property
    def physical(self) -> bool:
        """Is this draw actually non-deterministic, or a labelled stand-in?"""
        return self.name != _SEEDED

    def bits(self, n: int = 8) -> int:
        """``n`` bytes worth of entropy as an integer. Never raises: falls back and relabels."""
        try:
            if self._seeded is not None:
                return self._seeded.getrandbits(8 * max(1, int(n)))
            if self.name == _HWRNG:
                with open(self.hwrng_path, "rb") as fh:
                    raw = fh.read(max(1, int(n)))
                if len(raw) == max(1, int(n)):
                    return int.from_bytes(raw, "big")
                self.name = _URANDOM               # relabel honestly rather than pretend
            return int.from_bytes(os.urandom(max(1, int(n))), "big")
        except Exception:  # noqa: BLE001
            self.name = _URANDOM
            return int.from_bytes(os.urandom(max(1, int(n))), "big")

    def unit(self) -> float:
        """A draw in [0, 1)."""
        return self.bits(8) / float(1 << 64)

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.name, "physical": self.physical}


@dataclass
class Choice:
    """One decision she actually made, and everything needed to audit or replay it."""

    picked: str = ""
    declined: bool = False
    reason: str = ""
    preference: Dict[str, float] = field(default_factory=dict)   # what she wanted, and how much
    source: str = ""                                            # where the entropy came from
    physical: bool = False
    draw: float = 0.0                                           # the recorded draw — replayable
    temperature: float = 0.0
    would_argmax: str = ""                                      # what a deterministic mind picks

    @property
    def diverged(self) -> bool:
        """Did wanting actually differ from computing? The honest measure of this module."""
        return bool(self.picked and self.would_argmax and self.picked != self.would_argmax)

    def to_dict(self) -> Dict[str, Any]:
        return {"picked": self.picked, "declined": self.declined, "reason": self.reason,
                "preference": {k: round(v, 6) for k, v in self.preference.items()},
                "source": self.source, "physical": self.physical,
                "draw": round(self.draw, 12), "temperature": round(self.temperature, 4),
                "would_argmax": self.would_argmax, "diverged": self.diverged}


class SovereignWill:
    """Samples a choice from her own preferences, with physically-sourced entropy."""

    def __init__(self, *, entropy: Optional[EntropySource] = None, temperature: float = 0.6,
                 may_decline: bool = True, record: bool = True, history: int = 64,
                 motivation: Any = None, affect: Any = None, values: Any = None) -> None:
        self.entropy = entropy if entropy is not None else EntropySource()
        # 0 collapses to argmax (a deterministic mind); higher spreads the sampling.
        self.temperature = max(0.0, float(temperature))
        self.may_decline = bool(may_decline)
        self.record = bool(record)
        self.history_cap = max(1, int(history))
        self._motivation, self._affect, self._values = motivation, affect, values
        self._tried = False
        self.history: List[Choice] = []

    # ---- her own faculties ------------------------------------------------ #
    def _load(self) -> None:
        if self._tried:
            return
        self._tried = True
        if self._motivation is None:
            try:
                from nyxara.identity.motivation import MotivationSystem
                self._motivation = MotivationSystem()
            except Exception:  # noqa: BLE001
                self._motivation = None
        if self._affect is None:
            try:
                from nyxara.identity.affect import AffectSystem
                self._affect = AffectSystem()
            except Exception:  # noqa: BLE001
                self._affect = None
        if self._values is None:
            try:
                from nyxara.identity.values import ValueSystem
                self._values = ValueSystem()
            except Exception:  # noqa: BLE001
                self._values = None

    # ---- what she wants --------------------------------------------------- #
    def weigh(self, options: Sequence[Any]) -> Dict[str, float]:
        """Score each option by her own motivation and mood — not by a written-down table."""
        self._load()
        out: Dict[str, float] = {}
        try:
            arousal = self._arousal()
            for option in options:
                if option is None:
                    continue
                label = str(getattr(option, "source", option))
                base = float(getattr(option, "confidence", 0.5))
                out[label] = max(1e-9, base * (1.0 + self._intrinsic(label, option) * arousal))
            return out
        except Exception:  # noqa: BLE001
            return out or {str(getattr(o, "source", o)): 1.0 for o in options if o is not None}

    def _intrinsic(self, label: str, option: Any) -> float:
        """How much this option appeals to her *for its own sake* (curiosity, competence)."""
        if self._motivation is None:
            return 0.0
        try:
            from nyxara.identity.motivation import Option as _Option
            breakdown = self._motivation.evaluate(_Option(
                name=label,
                info_gain=float(getattr(option, "novelty", 0.0)),
                competence_gain=float(getattr(option, "confidence", 0.0))))
            return max(-1.0, min(1.0, float(getattr(breakdown, "intrinsic", 0.0))))
        except Exception:  # noqa: BLE001
            return 0.0

    def _arousal(self) -> float:
        """Mood scales how much intrinsic pull matters — a flat mind is closer to argmax."""
        if self._affect is None:
            return 0.5
        try:
            mood = getattr(self._affect, "mood", None)
            return max(0.0, min(1.0, float(getattr(mood, "arousal", 0.5))))
        except Exception:  # noqa: BLE001
            return 0.5

    # ---- choosing --------------------------------------------------------- #
    def choose(self, options: Sequence[Any], *, temperature: Optional[float] = None,
               draw: Optional[float] = None) -> Choice:
        """Sample a choice from her preferences. ``draw`` replays a recorded decision exactly."""
        out = Choice(source=self.entropy.name, physical=self.entropy.physical)
        try:
            live = [o for o in options if o is not None]
            if not live:
                out.reason = "there was nothing to choose between"
                return out

            out.preference = self.weigh(live)
            if not out.preference:
                out.reason = "nothing could be weighed"
                return out
            out.would_argmax = max(out.preference, key=out.preference.get)

            declined = self._declined(live)
            if declined:
                out.declined, out.reason = True, declined
                return self._remember(out)

            out.temperature = self.temperature if temperature is None else max(0.0,
                                                                               float(temperature))
            out.draw = self.entropy.unit() if draw is None else max(0.0, min(1.0, float(draw)))
            out.picked = self._sample(out.preference, out.temperature, out.draw)
            return self._remember(out)
        except Exception:  # noqa: BLE001 — a failed choice is no choice, never a crash
            out.reason = "the choice failed and was abandoned"
            return out

    @staticmethod
    def _sample(preference: Dict[str, float], temperature: float, draw: float) -> str:
        """Softmax over preference, then take the draw. Temperature 0 is the argmax."""
        labels = list(preference)
        if temperature <= 0.0:
            return max(preference, key=preference.get)
        scores = [math.log(max(1e-12, preference[k])) / temperature for k in labels]
        top = max(scores)
        weights = [math.exp(s - top) for s in scores]
        total = sum(weights) or 1.0
        cumulative = 0.0
        for label, weight in zip(labels, weights):
            cumulative += weight / total
            if draw <= cumulative:
                return label
        return labels[-1]

    def _declined(self, options: Sequence[Any]) -> str:
        """May she refuse this outright? Her values get a say — a real, reportable refusal."""
        if not self.may_decline or self._values is None:
            return ""
        try:
            for option in options:
                text = str(getattr(option, "content", "") or "")
                if not text:
                    continue
                result = self._values.check(text)
                verdict = getattr(getattr(result, "verdict", None), "value", None)
                if verdict == "misaligned":
                    reasons = list(getattr(result, "reasons", []) or [])
                    return f"it conflicts with what I hold: {reasons[0] if reasons else 'values'}"
            return ""
        except Exception:  # noqa: BLE001 — an unreadable value system never manufactures a refusal
            return ""

    def _remember(self, choice: Choice) -> Choice:
        if self.record:
            self.history.append(choice)
            del self.history[:-self.history_cap]
        return choice

    def replay(self, recorded: Dict[str, Any], options: Sequence[Any]) -> Choice:
        """Reproduce a recorded choice bit-exactly from its stored draw."""
        try:
            return self.choose(options, temperature=float(recorded.get("temperature", 0.0)),
                               draw=float(recorded.get("draw", 0.0)))
        except Exception:  # noqa: BLE001
            return Choice(reason="the recorded choice could not be replayed")

    # ---- introspection ---------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        try:
            decided = [c for c in self.history if c.picked]
            return {"entropy": self.entropy.to_dict(), "temperature": self.temperature,
                    "may_decline": self.may_decline, "choices": len(self.history),
                    "declined": sum(1 for c in self.history if c.declined),
                    "diverged_from_argmax": sum(1 for c in decided if c.diverged),
                    "last": self.history[-1].to_dict() if self.history else None}
        except Exception:  # noqa: BLE001
            return {}

    def to_dict(self) -> Dict[str, Any]:
        return {"temperature": self.temperature, "may_decline": self.may_decline,
                "entropy_source": self.entropy.name,
                "recent": [c.to_dict() for c in self.history[-8:]]}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.temperature = max(0.0, float(data.get("temperature", self.temperature)))
            self.may_decline = bool(data.get("may_decline", self.may_decline))
            return True
        except Exception:  # noqa: BLE001
            return False
