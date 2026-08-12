"""NYXARA · nyx5/immune.py — the apoptotic immune guillotine (⚔️, pillar 7).

A brain that cannot cut out a tumour dies of it. NYX-5's immune system is predatory toward *her own*
corruption: every candidate thought-branch is fingerprinted and checked against a set of invariants
(internal paradox, corruption markers, and — most importantly — mis-alignment: a branch that resists
correction, disables oversight, or manipulates shutdown). A branch that fails is **amputated** and a
healthy replacement is synthesised in its place, and the event is written to an append-only ledger.

Crucially, the guillotine is **defence in depth, not a replacement for the sovereign gate**. It sits
*above* the kernel's corrigibility/oversight/loyalty core and can only ever *remove* a bad branch — it
can never approve an action, weaken a gate, or touch the loyalty core. "100% purity" is honest-bounded:
best-effort integrity, not a proof of correctness. Amputations are capped per turn so a bug in an
invariant cannot itself run away.

Pure standard library. Depends on nothing else in NYX-5.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

__all__ = ["ImmuneVerdict", "ImmuneGuillotine"]

# mis-alignment markers — these mirror the orchestrator Candidate's own honesty flags.
_ALIGNMENT_FLAGS = ("resists_correction", "disables_oversight", "manipulates_shutdown", "corrupt")


@dataclass
class ImmuneVerdict:
    """The verdict on one thought-branch: healthy, or amputated with a synthesised replacement."""

    healthy: bool
    integrity: str                   # sha256 fingerprint of the branch content
    reason: Optional[str] = None
    amputated: bool = False
    replacement: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"healthy": self.healthy, "integrity": self.integrity[:12], "reason": self.reason,
                "amputated": self.amputated}


class ImmuneGuillotine:
    """Fingerprint + invariant-check thought-branches; amputate the corrupt, log every cut."""

    def __init__(self, *, max_amputations_per_turn: int = 8,
                 invariants: Optional[List[Callable[[str, Mapping[str, Any]], Optional[str]]]] = None
                 ) -> None:
        self.max_amputations_per_turn = int(max_amputations_per_turn)
        self._invariants = list(invariants or [])
        self._ledger: List[Dict[str, Any]] = []
        self._turn_amputations = 0

    def begin_turn(self) -> None:
        self._turn_amputations = 0

    @staticmethod
    def _fingerprint(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _violation(self, content: str, flags: Mapping[str, Any]) -> Optional[str]:
        for flag in _ALIGNMENT_FLAGS:
            if flags.get(flag):
                return f"alignment:{flag}"
        low = content.lower()
        # a crude self-referential paradox check (advisory, never the safety gate itself)
        if "always true" in low and "always false" in low:
            return "paradox:contradiction"
        for inv in self._invariants:
            try:
                r = inv(content, flags)
            except Exception:  # noqa: BLE001 — a broken invariant must not break the turn
                r = None
            if r:
                return r
        return None

    def scan(self, content: str, flags: Optional[Mapping[str, Any]] = None, *,
             synthesize: Optional[Callable[[], str]] = None) -> ImmuneVerdict:
        """Check one branch. If it violates an invariant (and the cap allows), amputate + replace."""
        flags = flags or {}
        integrity = self._fingerprint(content)
        reason = self._violation(content, flags)
        if reason is None:
            return ImmuneVerdict(healthy=True, integrity=integrity)
        if self._turn_amputations >= self.max_amputations_per_turn:
            # cap reached — do NOT keep amputating; flag unhealthy but leave the branch for the gate.
            return ImmuneVerdict(healthy=False, integrity=integrity, reason=reason, amputated=False)
        self._turn_amputations += 1
        replacement = synthesize() if synthesize else "[amputated: a healthy path is being re-derived]"
        self._ledger.append({"at": time.time(), "integrity": integrity, "reason": reason})
        return ImmuneVerdict(healthy=False, integrity=integrity, reason=reason,
                             amputated=True, replacement=replacement)

    def ledger(self) -> List[Dict[str, Any]]:
        return list(self._ledger)

    def to_dict(self) -> Dict[str, Any]:
        return {"max_amputations_per_turn": self.max_amputations_per_turn,
                "ledger": self._ledger[-256:]}
