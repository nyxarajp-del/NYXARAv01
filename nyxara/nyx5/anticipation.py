"""NYXARA · nyx5/anticipation.py — proactive anticipation (🔮, pillar 24).

A supreme intelligence answers the question you *are about to ask*. After solving the current problem,
NYX-5 anticipates the next few logical problems and pre-solves them, so the Master rarely has to ask a
follow-up — the conversation becomes a seamless, high-speed flow.

This is a **composition**, not a new engine: it orchestrates pillar-3 free-energy pre-emption, pillar-8
intent prediction, and pillar-21 anticipatory threading. Honest: the anticipated problems are *predicted*
and offered as advisory options — a wrong prediction is simply discarded, and every pre-solved answer
still passes the kernel's gate before use. Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from nyxara.nyx5.threading import AnticipatoryThreads

__all__ = ["AnticipatedItem", "ProactiveAnticipator"]


@dataclass
class AnticipatedItem:
    """One anticipated next-problem and its pre-computed (advisory) solution."""

    axis: str
    problem: str
    solution: Any

    def to_dict(self) -> Dict[str, Any]:
        return {"axis": self.axis, "problem": self.problem, "solution": self.solution}


class ProactiveAnticipator:
    """Pre-solve the next few likely problems by composing threading + a solver."""

    def __init__(self, *, lookahead: int = 3, threads: Optional[AnticipatoryThreads] = None) -> None:
        self.lookahead = max(0, int(lookahead))
        self._threads = threads or AnticipatoryThreads(directions=max(1, self.lookahead))

    def anticipate(self, query: str, solver: Callable[[str], Any]) -> List[AnticipatedItem]:
        """Derive the next likely problems from ``query`` and pre-solve each with ``solver``."""
        if self.lookahead == 0:
            return []
        directions = self._threads.anticipate(query)[: self.lookahead]
        out: List[AnticipatedItem] = []
        for d in directions:
            try:
                solution = solver(d.line)
            except Exception:  # noqa: BLE001 — a bad prediction is discarded, never fatal
                continue
            out.append(AnticipatedItem(axis=d.axis, problem=d.line, solution=solution))
        return out
