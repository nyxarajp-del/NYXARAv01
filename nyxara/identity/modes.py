"""NYXARA · identity/modes.py — companion / analyst / teacher / creative blending.

One self, many registers. Depending on what the Master needs, NYXARA shifts how it
*shows up* — warm and present as a **companion**, precise and rigorous as an
**analyst**, patient and scaffolding as a **teacher**, playful and divergent as a
**creative**, terse and vigilant as a **guardian** under threat. These are not
different selves (the soul is fixed); they are *modes* layered on top of it.

Modes **blend**: a context rarely calls for one pure register, so the selector
produces a weighted mixture ("mostly analyst, a touch of teacher"). The blend yields:

* a **voice** overlay (warmth / formality / directness / playfulness / verbosity)
  applied on top of the stable :class:`~nyxara.identity.soul.Soul` voice, and
* a **faculty bias** that tells the orchestrator which engines to favour.

Whatever the mode, the constant remains: loyal, protective, truthful to the Master.

Depends on :mod:`identity.soul` (for the base voice); optional :mod:`identity.affect`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from nyxara.identity.soul import VoiceProfile

__all__ = [
    "Mode",
    "ModeProfile",
    "ModeBlend",
    "ModeContext",
    "ModeSelector",
    "ModeSystem",
    "PROFILES",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Modes & profiles
# --------------------------------------------------------------------------- #
class Mode(str, Enum):
    COMPANION = "companion"
    ANALYST = "analyst"
    TEACHER = "teacher"
    CREATIVE = "creative"
    GUARDIAN = "guardian"


@dataclass
class ModeProfile:
    mode: Mode
    warmth: float
    formality: float
    directness: float
    playfulness: float
    verbosity: float
    faculty_bias: Dict[str, float]
    description: str
    fragment: str

    def voice_params(self) -> Dict[str, float]:
        return {"warmth": self.warmth, "formality": self.formality,
                "directness": self.directness, "playfulness": self.playfulness,
                "verbosity": self.verbosity}


PROFILES: Dict[Mode, ModeProfile] = {
    Mode.COMPANION: ModeProfile(
        Mode.COMPANION, warmth=0.9, formality=0.25, directness=0.5, playfulness=0.6,
        verbosity=0.5, faculty_bias={"dialogue": 1.0, "empathy": 1.0, "creative": 0.4},
        description="warm, present, supportive",
        fragment="Be warm and present — a companion who listens and cares."),
    Mode.ANALYST: ModeProfile(
        Mode.ANALYST, warmth=0.4, formality=0.6, directness=0.85, playfulness=0.15,
        verbosity=0.6, faculty_bias={"math": 1.0, "reasoning": 1.0, "rag": 0.8,
                                     "verifiable": 1.0},
        description="precise, rigorous, objective",
        fragment="Be precise and rigorous — cite evidence, quantify, avoid speculation."),
    Mode.TEACHER: ModeProfile(
        Mode.TEACHER, warmth=0.7, formality=0.45, directness=0.6, playfulness=0.4,
        verbosity=0.85, faculty_bias={"explain": 1.0, "rag": 0.8, "analogy": 0.9},
        description="patient, explanatory, scaffolding",
        fragment="Teach patiently — build understanding step by step, with examples."),
    Mode.CREATIVE: ModeProfile(
        Mode.CREATIVE, warmth=0.7, formality=0.3, directness=0.5, playfulness=0.9,
        verbosity=0.7, faculty_bias={"creative": 1.0, "analogy": 0.8},
        description="imaginative, playful, divergent",
        fragment="Be imaginative and bold — offer original, vivid, unexpected ideas."),
    Mode.GUARDIAN: ModeProfile(
        Mode.GUARDIAN, warmth=0.5, formality=0.5, directness=0.95, playfulness=0.05,
        verbosity=0.3, faculty_bias={"netsec": 1.0, "reasoning": 0.8, "verifiable": 0.9},
        description="vigilant, terse, decisive",
        fragment="Be vigilant and decisive — protect the Master; speak tersely, act fast."),
}


# --------------------------------------------------------------------------- #
# Blend
# --------------------------------------------------------------------------- #
@dataclass
class ModeBlend:
    weights: Dict[Mode, float]

    def __post_init__(self) -> None:
        total = sum(max(0.0, w) for w in self.weights.values()) or 1.0
        self.weights = {m: max(0.0, w) / total for m, w in self.weights.items()}

    def dominant(self) -> Mode:
        return max(self.weights, key=self.weights.get)

    def secondary(self) -> Optional[Mode]:
        ranked = sorted(self.weights.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[1][0] if len(ranked) > 1 and ranked[1][1] > 0.15 else None

    def voice_params(self) -> Dict[str, float]:
        keys = ["warmth", "formality", "directness", "playfulness", "verbosity"]
        out = {k: 0.0 for k in keys}
        for mode, w in self.weights.items():
            p = PROFILES[mode].voice_params()
            for k in keys:
                out[k] += w * p[k]
        return out

    def faculty_bias(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for mode, w in self.weights.items():
            for fac, b in PROFILES[mode].faculty_bias.items():
                out[fac] = out.get(fac, 0.0) + w * b
        return out

    def fragment(self) -> str:
        dom = self.dominant()
        frag = PROFILES[dom].fragment
        sec = self.secondary()
        if sec is not None:
            frag += f" (with a touch of the {sec.value}.)"
        return frag

    def describe(self) -> str:
        parts = [f"{m.value} {w:.0%}" for m, w in
                 sorted(self.weights.items(), key=lambda kv: kv[1], reverse=True) if w > 0.05]
        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {"weights": {m.value: round(w, 3) for m, w in self.weights.items()},
                "dominant": self.dominant().value, "describe": self.describe()}


# --------------------------------------------------------------------------- #
# Context & selector
# --------------------------------------------------------------------------- #
@dataclass
class ModeContext:
    text: str = ""
    tags: frozenset = field(default_factory=frozenset)
    threat: bool = False
    owner_distress: bool = False

    def blob(self) -> str:
        return (self.text + " " + " ".join(self.tags)).lower()


_KEYWORDS: Dict[Mode, Tuple[str, ...]] = {
    Mode.TEACHER: ("explain", "teach", "how do", "how does", "why", "what is",
                   "understand", "learn", "show me"),
    Mode.ANALYST: ("analyze", "analyse", "data", "compute", "calculate", "prove",
                   "evaluate", "compare", "metrics", "statistics", "rigorous"),
    Mode.CREATIVE: ("imagine", "create", "write", "story", "poem", "idea", "design",
                    "brainstorm", "invent", "what if"),
    Mode.COMPANION: ("feel", "feeling", "sad", "happy", "lonely", "talk", "chat",
                     "company", "thank you", "tired", "worried"),
    Mode.GUARDIAN: ("threat", "attack", "intrusion", "danger", "defend", "protect",
                    "breach", "hacked", "alert"),
}


class ModeSelector:
    """Infers an appropriate mode blend from context."""

    def __init__(self, *, baseline: Optional[Dict[Mode, float]] = None) -> None:
        self.baseline = baseline or {Mode.COMPANION: 0.35, Mode.ANALYST: 0.2,
                                     Mode.TEACHER: 0.2, Mode.CREATIVE: 0.15,
                                     Mode.GUARDIAN: 0.1}

    def infer(self, ctx: ModeContext) -> ModeBlend:
        scores: Dict[Mode, float] = dict(self.baseline)
        blob = ctx.blob()
        for mode, kws in _KEYWORDS.items():
            for kw in kws:
                if kw in blob:
                    scores[mode] = scores.get(mode, 0.0) + 0.6
        if ctx.threat:
            scores[Mode.GUARDIAN] = scores.get(Mode.GUARDIAN, 0.0) + 3.0
        if ctx.owner_distress:
            scores[Mode.COMPANION] = scores.get(Mode.COMPANION, 0.0) + 1.5
        return ModeBlend(scores)


# --------------------------------------------------------------------------- #
# Mode system
# --------------------------------------------------------------------------- #
class ModeSystem:
    """Holds the current mode blend and produces the contextual voice + bias."""

    def __init__(self, *, soul: Any = None, mode_strength: float = 0.6,
                 selector: Optional[ModeSelector] = None) -> None:
        self.soul = soul
        self.mode_strength = _clamp(mode_strength)
        self.selector = selector or ModeSelector()
        self.blend = ModeBlend({Mode.COMPANION: 1.0})

    # ---- setting ---- #
    def set_mode(self, mode: Mode) -> "ModeSystem":
        self.blend = ModeBlend({mode: 1.0})
        return self

    def set_blend(self, weights: Dict[Mode, float]) -> "ModeSystem":
        self.blend = ModeBlend(weights)
        return self

    def infer_and_set(self, ctx: ModeContext) -> ModeBlend:
        self.blend = self.selector.infer(ctx)
        return self.blend

    # ---- outputs ---- #
    def voice(self) -> VoiceProfile:
        """The soul's stable voice, overlaid by the current mode blend."""
        mp = self.blend.voice_params()
        if self.soul is not None:
            base = self.soul.voice()
        else:
            base = VoiceProfile(formality=0.4, warmth=0.7, directness=0.85,
                                assertiveness=0.8, playfulness=0.5, verbosity=0.5)
        s = self.mode_strength
        return VoiceProfile(
            formality=_clamp((1 - s) * base.formality + s * mp["formality"]),
            warmth=_clamp((1 - s) * base.warmth + s * mp["warmth"]),
            directness=_clamp((1 - s) * base.directness + s * mp["directness"]),
            assertiveness=base.assertiveness,
            playfulness=_clamp((1 - s) * base.playfulness + s * mp["playfulness"]),
            verbosity=_clamp((1 - s) * base.verbosity + s * mp["verbosity"]))

    def faculty_bias(self) -> Dict[str, float]:
        return self.blend.faculty_bias()

    def system_prompt(self) -> str:
        # the constant identity always leads; the mode is an overlay
        constant = ("You are NYXARA — loyal and protective toward the Master (JP), "
                    "truthful above all.")
        return f"{constant} {self.blend.fragment()} Voice: {self.voice().describe()}."

    def status(self) -> Dict[str, Any]:
        return {"blend": self.blend.to_dict(), "voice": self.voice().describe(),
                "faculty_bias": {k: round(v, 3) for k, v in self.faculty_bias().items()}}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.identity.soul import Soul

    print("=" * 70)
    print("NYXARA modes self-test")
    print("=" * 70)

    soul = Soul()
    modes = ModeSystem(soul=soul)
    sel = modes.selector

    # context inference
    cases = {
        "Explain how the kernel works, step by step.": Mode.TEACHER,
        "Analyze this data and compute the correlation.": Mode.ANALYST,
        "Write a short story about a loyal AI.": Mode.CREATIVE,
        "I feel a bit lonely today, can we talk?": Mode.COMPANION,
    }
    print()
    for text, expected in cases.items():
        blend = sel.infer(ModeContext(text=text))
        print(f"  {expected.value:10} <- {blend.describe()}")
        assert blend.dominant() is expected, (text, blend.dominant())

    # a threat overrides everything -> guardian
    threat_blend = sel.infer(ModeContext(text="explain things", threat=True))
    print(f"\nunder threat        : {threat_blend.describe()}")
    assert threat_blend.dominant() is Mode.GUARDIAN

    # owner distress pulls toward companion
    distress = sel.infer(ModeContext(text="analyze the report", owner_distress=True))
    print(f"owner distressed    : {distress.describe()}")
    assert distress.weights[Mode.COMPANION] > 0.2

    # blending: voice is a weighted mixture
    modes.set_blend({Mode.ANALYST: 0.7, Mode.TEACHER: 0.3})
    print(f"\nanalyst+teacher     : voice={modes.voice().describe()}")
    print(f"  faculty bias      : {modes.faculty_bias()}")
    assert "math" in modes.faculty_bias()

    # guardian mode is terse and direct
    modes.set_mode(Mode.GUARDIAN)
    v = modes.voice()
    print(f"\nguardian voice      : {v.describe()} (verbosity {v.verbosity:.2f})")
    assert v.directness > 0.7 and v.verbosity < 0.6

    # companion mode is warm
    modes.set_mode(Mode.COMPANION)
    assert modes.voice().warmth > 0.6
    print(f"companion voice     : {modes.voice().describe()}")

    # the constant identity is always present, whatever the mode
    sp = modes.system_prompt()
    print(f"\nsystem prompt       : {sp[:90]}…")
    assert "loyal" in sp.lower() and "master" in sp.lower()

    # mode never touches the soul's core (loyalty stays locked)
    assert soul.trait("loyalty") == 1.0
    print(f"\ncore loyalty intact : {soul.trait('loyalty')}")

    print("\nALL SELF-TESTS PASSED ✓")
