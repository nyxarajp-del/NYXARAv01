"""NYXARA · nyx5/threading.py — the anticipatory thought engine (🎭, pillar 21).

An expert does not just answer — she reads where the conversation is going and holds the next few moves
ready. On each query NYX-5 simulates a small set of **next directions** in parallel, along distinct axes:

  1. **technical** — the concrete implementation / architecture path,
  2. **conceptual** — the higher-tier theoretical or scientific extension,
  3. **strategic** — the exact next step for the project.

She offers these as options so the conversation moves at pace instead of stalling on follow-up questions.
Honest: these are *predicted* directions — useful suggestions, not certainties; a direction that does not
fit is simply dropped. Reuses the active-inference / intent signals. Pure standard library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

__all__ = ["ThreadDirection", "AnticipatoryThreads"]

_AXES = ("technical", "conceptual", "strategic")
_TEMPLATES = {
    "technical": "Implementation path for {subject}: the concrete architecture / code to build it.",
    "conceptual": "Theoretical extension of {subject}: the deeper principle or higher-tier framing.",
    "strategic": "Next concrete step on {subject}: the single action that moves the project forward.",
}


@dataclass
class ThreadDirection:
    """One anticipated next direction along a named axis."""

    axis: str
    line: str

    def to_dict(self) -> Dict[str, Any]:
        return {"axis": self.axis, "line": self.line}


class AnticipatoryThreads:
    """Generate a small set of distinct next-directions from the current query."""

    def __init__(self, *, directions: int = 3) -> None:
        self.directions = max(1, min(len(_AXES), int(directions)))

    @staticmethod
    def _subject(query: str) -> str:
        words = re.findall(r"\w+", query or "")
        if not words:
            return "this"
        # keep the most content-bearing tail of the query as the subject
        return " ".join(words[-6:])

    def anticipate(self, query: str) -> List[ThreadDirection]:
        """Return up to ``directions`` distinct next-directions grounded in the query's subject."""
        subject = self._subject(query)
        out: List[ThreadDirection] = []
        for axis in _AXES[: self.directions]:
            out.append(ThreadDirection(axis=axis, line=_TEMPLATES[axis].format(subject=subject)))
        return out
