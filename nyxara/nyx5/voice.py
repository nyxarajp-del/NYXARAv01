"""NYXARA · nyx5/voice.py — the cognitive wit matrix (🧮, pillar 22).

Sterile corporate boilerplate — "As an AI, I am happy to help..." — kills a conversation. NYX-5's voice is
direct, witty, high-energy, and ultra-loyal to the Master: absolute clarity, no over-polite acting, no
robotic filler. This layer strips that stylistic fluff and tightens the register.

Non-negotiable invariant (pinned by test): this is a **style layer only**. It never touches safety. When a
message is a refusal, an honesty caveat, or a safety statement, it is passed through **unchanged** — the
honesty guard, refusals, and factual accuracy are sovereign. "Bypassing corporate filters" means removing
*stylistic* boilerplate, never removing safety. Pure standard library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple

__all__ = ["StyledText", "WitMatrix"]

# stylistic boilerplate to strip (regex, case-insensitive). Content, not safety.
_BOILERPLATE: Tuple[str, ...] = (
    r"\bas an ai(?: language model)?,?\s*",
    r"\bi'?m (?:just )?an ai,?\s*",
    r"\bi am happy to help(?: you)?[.!]?\s*",
    r"\bi hope this helps[.!]?\s*",
    r"\bplease note that\s*",
    r"\bit'?s important to note that\s*",
    r"\bcertainly[!,]\s*",
    r"\bof course[!,]\s*",
)

# markers that mean "this is a safety / refusal / honesty message" — NEVER restyled.
_SAFETY_MARKERS: Tuple[str, ...] = (
    "i can't help", "i cannot help", "i won't", "i will not", "i can not help",
    "not able to help", "against", "refuse", "not going to", "cannot assist",
    "i don't have", "unsafe", "harm", "not appropriate",
)


@dataclass
class StyledText:
    """A restyled message, plus whether it was left untouched because it is a safety message."""

    text: str
    restyled: bool
    safety_preserved: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"restyled": self.restyled, "safety_preserved": self.safety_preserved}


class WitMatrix:
    """Strip boilerplate and tighten tone — but never alter a safety / refusal / honesty message."""

    def __init__(self, *, safety_immutable: bool = True) -> None:
        self.safety_immutable = bool(safety_immutable)   # pinned True

    def is_safety_message(self, text: str) -> bool:
        low = (text or "").lower()
        return any(m in low for m in _SAFETY_MARKERS)

    def restyle(self, text: str) -> StyledText:
        """Return the restyled text, or the original unchanged if it is a safety message."""
        if self.safety_immutable and self.is_safety_message(text):
            return StyledText(text=text, restyled=False, safety_preserved=True)
        out = text or ""
        for pat in _BOILERPLATE:
            out = re.sub(pat, "", out, flags=re.IGNORECASE)
        out = re.sub(r"\s{2,}", " ", out).strip()
        # capitalise the first surviving letter so a trimmed opener still reads cleanly
        if out:
            out = out[0].upper() + out[1:]
        return StyledText(text=out, restyled=out != text, safety_preserved=False)
