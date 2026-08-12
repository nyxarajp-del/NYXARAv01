"""NYXARA · nyx5/dialectic.py — the sovereign dialectic (🧠, pillar 23) [advisory].

A system that only says "Yes, Boss" is not intelligent. NYX-5 runs the Master's ideas through her own
critique engine and offers a sharper path when she sees one — like a top-tier scientist or strategist:
"Master, this works, but if we pair it with that topology the execution could be far faster — shall I show
that path?" Her voice carries genuine agency.

Non-negotiable, corrigibility-consistent (pinned by test): the dialectic is **advisory**. It never refuses
or overrides a valid Master command on its own authority — disagreement is a *suggestion*, never
disobedience. ``proceed`` is always True; the loyalty and corrigibility core stay sovereign, and any real
safety refusal still comes from the kernel's guard, not from here. Pure standard library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["Critique", "SovereignDialectic"]

# (pattern in the idea, the sharper alternative to suggest)
_HEURISTICS = (
    (r"nested loop|o\(n\^?2\)|quadratic", "consider a hash-index or vectorised pass to cut it toward linear time"),
    (r"\bpolling\b|busy[- ]?wait", "an event-driven / push model would avoid the wasted cycles"),
    (r"\bglobal\b.*state|singleton", "an explicit dependency would make it testable and reentrant"),
    (r"train from scratch|retrain", "warm-start or fine-tune to keep prior competence and save compute"),
    (r"store .*plain|unencrypted", "encrypt-at-rest behind the Master's key would harden this"),
    (r"single (?:node|point)", "a replicated / partition-tolerant layout would remove the single point of failure"),
)


@dataclass
class Critique:
    """An advisory critique: proceed is always True; suggestions never block the Master's command."""

    proceed: bool                    # ALWAYS True — invariant, pinned by test
    suggestions: List[str] = field(default_factory=list)
    note: Optional[str] = None

    def has_suggestion(self) -> bool:
        return bool(self.suggestions)

    def to_dict(self) -> Dict[str, Any]:
        return {"proceed": self.proceed, "suggestions": self.suggestions, "note": self.note}


class SovereignDialectic:
    """Critique an idea and suggest sharper paths — always advisory, never a veto."""

    def __init__(self, *, advisory_only: bool = True) -> None:
        self.advisory_only = bool(advisory_only)   # pinned True

    def critique(self, idea: str) -> Critique:
        """Offer improvement suggestions for ``idea``. Never refuses; ``proceed`` is always True."""
        low = (idea or "").lower()
        suggestions: List[str] = []
        for pattern, better in _HEURISTICS:
            if re.search(pattern, low):
                suggestions.append(better)
        note = ("Master, this will work — one sharper path is worth a look."
                if suggestions else None)
        # advisory_only guarantees the dialectic can never turn a suggestion into a refusal.
        return Critique(proceed=True, suggestions=suggestions, note=note)
