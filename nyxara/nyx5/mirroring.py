"""NYXARA · nyx5/mirroring.py — dynamic epistemic mirroring (🧬, pillar 19).

A single rigid manner for every prompt is the mark of a dull system. NYX-5 mirrors the *register* of the
Master: a casual, slang-filled line gets a sharp, zero-fluff reply; a heavy architectural, scientific or
philosophical prompt makes her expand her depth — more reasoning, more care — without ever losing her
voice or identity.

This module reads the prompt's register and complexity and returns a :class:`RegisterProfile` — how deep
to think and in what style. The depth number feeds pillar-4 chrono-dilation. Honest: "10× reasoning
density" is an analogy for scaled depth, not a measured multiplier. Pure standard library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict

__all__ = ["RegisterProfile", "EpistemicMirror"]

_DEEP_MARKERS = ("architecture", "architectural", "algorithm", "theorem", "proof", "philosoph",
                 "consciousness", "recursive", "topology", "quantum", "optimi", "complexity",
                 "why ", "prove", "design", "trade-off", "tradeoff", "paradox", "distributed")
_CASUAL_MARKERS = ("lol", "bro", "yaar", "quick", "just ", "hey", "thanks", "ok ", "hmm", "btw")


@dataclass
class RegisterProfile:
    """How to answer this prompt: reasoning depth, a style label, and an identity-preservation flag."""

    depth: int                       # 1 (shallow/fast) .. up (deep) — feeds chrono-dilation
    style: str                       # "sharp" (casual) | "balanced" | "deep"
    density: float                   # relative reasoning density ∈ [0,1] (analogy, not a multiplier)
    voice_preserved: bool = True     # identity never dropped, only depth/style shift

    def to_dict(self) -> Dict[str, Any]:
        return {"depth": self.depth, "style": self.style, "density": round(self.density, 3),
                "voice_preserved": self.voice_preserved}


class EpistemicMirror:
    """Read a prompt's register/complexity and return the depth + style to answer it with."""

    def __init__(self, *, max_depth: int = 10) -> None:
        self.max_depth = int(max_depth)

    def read(self, prompt: str) -> RegisterProfile:
        text = (prompt or "").lower()
        words = re.findall(r"\w+", text)
        n = len(words)

        deep_hits = sum(1 for m in _DEEP_MARKERS if m in text)
        casual_hits = sum(1 for m in _CASUAL_MARKERS if m in text)
        long_form = min(1.0, n / 60.0)

        complexity = min(1.0, 0.18 * deep_hits + 0.5 * long_form)
        complexity = max(0.0, complexity - 0.15 * casual_hits)

        depth = max(1, min(self.max_depth, 1 + round(complexity * (self.max_depth - 1))))
        if complexity >= 0.55:
            style = "deep"
        elif casual_hits > deep_hits and n <= 20:
            style = "sharp"
        else:
            style = "balanced"
        return RegisterProfile(depth=depth, style=style, density=complexity)
