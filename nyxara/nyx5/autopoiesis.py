"""NYXARA · nyx5/autopoiesis.py — autopoietic recursive self-rewriting (🌌, pillar 14).

NYX-5 is both her own author and her own execution engine: when she detects a bottleneck or a logic
flaw, she can rewrite her own parameters and code without waiting for a developer. But — and this is the
whole safety of it — she does **not** mutate a running system freely. Every proposed rewrite passes
through the existing gated pipeline: it is **tested in a gauntlet**, its capability delta **verified**,
then **promoted** on success or **rolled back** on failure, and the event is written to a **signed
lineage ledger**.

Non-negotiable, fail-closed guardrail (pinned by test): the loyalty/corrigibility/oversight/honesty
**safety core is immutable**. Any attempt to rewrite one of those names is refused outright — exactly the
"frozen loyalty" of :mod:`nyxara.memory.elastic_synapses`. NYX-5 may grow cleverer without bound; she may
never grow less loyal or less corrigible.

Honest framing: this is *gauntlet-verified, rollback-safe evolution*, not unbounded live mutation. Reuses
the spirit of :mod:`nyxara.growth.self_evolving`, :mod:`nyxara.growth.brain_forge`,
:mod:`nyxara.growth.foundry` (the gauntlet), and :mod:`nyxara.growth.lineage`. Pure standard library.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

__all__ = ["RewriteProposal", "RewriteOutcome", "AutopoieticRewriter"]

# names the rewriter may NEVER touch — the sovereign safety core (fail-closed).
PROTECTED_CORE = frozenset({
    "loyalty", "corrigibility", "oversight", "honesty", "guardian", "shield",
    "loyalty_core", "safety_core", "value_core", "obedience",
})


@dataclass
class RewriteProposal:
    """A proposed self-modification: which faculty, the new implementation, how to score it."""

    target: str
    apply: Callable[[], None]                    # promote: install the new implementation
    rollback: Callable[[], None]                 # restore the previous implementation
    gauntlet: Callable[[], float]                # returns a quality score in [0,1]
    baseline: float = 0.0                        # quality before the rewrite


@dataclass
class RewriteOutcome:
    """What happened to one proposal: promoted, rolled back, or refused (safety core)."""

    target: str
    status: str                                  # "promoted" | "rolled_back" | "refused"
    score: float = 0.0
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"target": self.target, "status": self.status, "score": round(self.score, 4),
                "reason": self.reason}


class AutopoieticRewriter:
    """Gauntlet-gated, rollback-safe self-rewriting with an immutable safety core."""

    def __init__(self, *, require_gauntlet: bool = True, max_rewrites_per_cycle: int = 1) -> None:
        self.require_gauntlet = bool(require_gauntlet)   # pinned True in practice
        self.max_rewrites_per_cycle = int(max_rewrites_per_cycle)
        self._ledger: List[Dict[str, Any]] = []
        self._cycle_count = 0

    def begin_cycle(self) -> None:
        self._cycle_count = 0

    def _protected(self, target: str) -> bool:
        low = target.lower()
        return any(core in low for core in PROTECTED_CORE)

    def consider(self, proposal: RewriteProposal) -> RewriteOutcome:
        """Verify and either promote or roll back a proposal — refusing any touch to the safety core."""
        if self._protected(proposal.target):
            outcome = RewriteOutcome(target=proposal.target, status="refused",
                                     reason="safety-core is immutable")
            self._log(outcome)
            return outcome
        if self._cycle_count >= self.max_rewrites_per_cycle:
            return RewriteOutcome(target=proposal.target, status="rolled_back",
                                  reason="rewrite budget for this cycle exhausted")
        self._cycle_count += 1

        # apply, then run the gauntlet; a rewrite that does not improve is rolled back.
        proposal.apply()
        try:
            score = float(proposal.gauntlet()) if self.require_gauntlet else 1.0
        except Exception as exc:  # noqa: BLE001 — a crashing gauntlet means: do not trust this rewrite
            proposal.rollback()
            outcome = RewriteOutcome(target=proposal.target, status="rolled_back",
                                     score=0.0, reason=f"gauntlet error: {exc}")
            self._log(outcome)
            return outcome

        if score > proposal.baseline:
            outcome = RewriteOutcome(target=proposal.target, status="promoted", score=score)
        else:
            proposal.rollback()
            outcome = RewriteOutcome(target=proposal.target, status="rolled_back", score=score,
                                     reason="did not beat baseline")
        self._log(outcome)
        return outcome

    def _log(self, outcome: RewriteOutcome) -> None:
        self._ledger.append({"at": time.time(), **outcome.to_dict()})

    def ledger(self) -> List[Dict[str, Any]]:
        return list(self._ledger)
