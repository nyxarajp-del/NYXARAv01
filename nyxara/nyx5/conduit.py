"""NYXARA · nyx5/conduit.py — the symbiotic consciousness link (🌌, pillar 13).

The bandwidth between a mind and its Master is words, and words are slow. NYX-5 shortens the channel:
she learns the *implicit context* behind the Master's terse commands, so a two-word instruction expands
into its full intent. She keeps a learned phrasebook (short form → full intent) and, for unseen inputs,
falls back to fuzzy hyperdimensional resonance against what she has learned.

Honest ceiling: this is **bounded context-inference**, not the "O(1) telepathy" of the pitch. When a
command is genuinely ambiguous — no confident expansion — she does the honest thing and **abstains**,
asking the Master to clarify rather than guessing wrong. Extends pillar 8's intent history; reuses the
HDC algebra for the fuzzy fallback. Pure standard library on top of the HDC module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from nyxara.cognition.hyper_dimensional_vectors import HyperSpace, ItemMemory

__all__ = ["Expansion", "SymbioticConduit"]


@dataclass
class Expansion:
    """A terse command expanded to its inferred intent, with a confidence and a clarify flag."""

    intent: Optional[str]
    confidence: float
    clarify: bool                    # True => too ambiguous; ask the Master rather than guess

    def to_dict(self) -> Dict[str, Any]:
        return {"intent": self.intent, "confidence": round(self.confidence, 4),
                "clarify": self.clarify}


class SymbioticConduit:
    """Expand terse commands into full intent; abstain (clarify) when not confident."""

    def __init__(self, *, ambiguity_gate: float = 0.5, dim: int = 10000, seed: int = 42) -> None:
        self.ambiguity_gate = float(ambiguity_gate)
        self.space = HyperSpace(dim=dim, seed=seed)
        self.mem = ItemMemory(self.space)            # command anchors only (clean cleanup ranking)
        self._toks = ItemMemory(self.space)          # the token alphabet (kept out of ranking)
        self._phrasebook: Dict[str, str] = {}

    def learn(self, terse: str, full_intent: str) -> None:
        """Teach the conduit that ``terse`` means ``full_intent`` (an exact + a fuzzy HDC anchor)."""
        key = terse.strip().lower()
        self._phrasebook[key] = full_intent
        self.mem.add(f"cmd:{key}", self._embed(key))

    def _embed(self, text: str):
        toks = [t for t in text.replace("-", " ").split() if t]
        if not toks:
            return self.space.zero()
        return self.space.bundle(*[self._toks.symbol(f"tok:{t}") for t in toks])

    def expand(self, terse: str) -> Expansion:
        """Return the inferred full intent, or abstain (clarify=True) if too ambiguous."""
        key = terse.strip().lower()
        if key in self._phrasebook:
            return Expansion(intent=self._phrasebook[key], confidence=1.0, clarify=False)
        if not self._phrasebook:
            return Expansion(intent=None, confidence=0.0, clarify=True)
        result = self.mem.cleanup(self._embed(key))
        if result.name and result.name.startswith("cmd:") and result.score >= self.ambiguity_gate:
            matched = result.name[len("cmd:"):]
            return Expansion(intent=self._phrasebook.get(matched), confidence=result.score,
                             clarify=False)
        return Expansion(intent=None, confidence=result.score if result.name else 0.0, clarify=True)

    def to_dict(self) -> Dict[str, Any]:
        return {"ambiguity_gate": self.ambiguity_gate, "dim": self.space.dim,
                "seed": self.space.seed, "phrasebook": dict(self._phrasebook)}

    def load_dict(self, d: Dict[str, Any]) -> None:
        self.ambiguity_gate = float(d.get("ambiguity_gate", self.ambiguity_gate))
        self.space = HyperSpace(dim=int(d.get("dim", 10000)), seed=int(d.get("seed", 42)))
        self.mem = ItemMemory(self.space)
        self._toks = ItemMemory(self.space)
        self._phrasebook = {}
        for terse, full in d.get("phrasebook", {}).items():
            self.learn(terse, full)
