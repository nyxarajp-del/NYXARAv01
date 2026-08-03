"""NYXARA · growth/neuromod.py — neuromodulated plasticity + REM pruning (🧠, Rule 4, F3).

Biology does not treat every experience alike, and neither should NYXARA. Two neuromodulators gate how
strongly a solve is written into her library prior, and a sleep-time pruner keeps the library lean:

* **Dopamine (novelty / reward)** and **acetylcholine (surprise / attention).** A solve that is *novel*
  (a structure she has not produced before) and *surprising* (it took real search to find) is written
  in **fast** (a high learning rate — strong hippocampal encoding). A *habitual*, low-surprise solve
  barely nudges the prior — **saving memory and compute**, exactly as an over-familiar routine should.
* **REM pruning by utility.** During the sleep phase she prunes learned abstractions whose **utility
  has stayed non-positive** over a rolling window — concepts that stopped earning their keep. Pruning
  is **reversible**: a pruned abstraction is *archived*, never destroyed (:meth:`Library.restore_abstraction`),
  and the base primitives + any abstraction another one depends on are **never** prunable. The brain
  never bloats, yet nothing learned is ever truly lost.

Pure standard library; no LLM; it modulates *how* she learns and *what stays resident*, never *what she
values* — it cannot touch the character core.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List

from nyxara.growth.noesis import Library

__all__ = ["Neuromodulators", "UtilityLedger", "REMPruner", "PruneReport"]


@dataclass
class Neuromodulators:
    """Compute the F3 learning-rate multiplier for a solve from its novelty + surprise."""

    surprise_scale: float = 3000.0     # expansions at which surprise saturates
    min_lr: float = 0.05               # a habitual solve still leaves a faint trace
    max_lr: float = 2.0                # a novel + surprising solve is written in fast

    def weight(self, *, novelty: float, expansions: float) -> float:
        """novelty ∈ [0,1] (dopamine); surprise from search effort (acetylcholine)."""
        novelty = max(0.0, min(1.0, novelty))
        surprise = max(0.0, min(1.0, expansions / self.surprise_scale))
        dopamine = 0.2 + 0.8 * novelty
        acetylcholine = 0.3 + 0.7 * surprise
        lr = 2.0 * dopamine * acetylcholine     # ~[0.06 … 2.0]
        return max(self.min_lr, min(self.max_lr, lr))


@dataclass
class UtilityLedger:
    """Rolling per-abstraction utility: usage benefit minus a small holding cost, over a window."""

    window: int = 5
    holding_cost: float = 0.5
    grace: int = 2                     # cycles before a new abstraction is eligible for pruning
    history: Dict[str, Deque[float]] = field(default_factory=dict)
    age: Dict[str, int] = field(default_factory=dict)

    def record_cycle(self, usage: Dict[str, int]) -> None:
        """Fold one cycle of abstraction usage counts into the rolling utility windows."""
        for name in list(self.history.keys()) + [n for n in usage if n not in self.history]:
            u = float(usage.get(name, 0)) - self.holding_cost
            dq = self.history.setdefault(name, deque(maxlen=self.window))
            dq.append(u)
            self.age[name] = self.age.get(name, 0) + 1

    def utility(self, name: str) -> float:
        dq = self.history.get(name)
        return sum(dq) if dq else 0.0

    def is_stale(self, name: str) -> bool:
        """Non-positive rolling utility past the grace period → a candidate for pruning."""
        return (self.age.get(name, 0) > self.grace
                and len(self.history.get(name, ())) >= min(self.window, self.age.get(name, 0))
                and self.utility(name) <= 0.0)

    def forget(self, name: str) -> None:
        self.history.pop(name, None)
        self.age.pop(name, None)


@dataclass
class PruneReport:
    pruned: List[str]
    kept: int
    archived_total: int

    def to_dict(self) -> Dict[str, Any]:
        return {"pruned": self.pruned, "kept": self.kept, "archived_total": self.archived_total}


class REMPruner:
    """Sleep-time pruning of stale abstractions — reversible, dependency-safe, primitive-safe."""

    def prune(self, library: Library, ledger: UtilityLedger) -> PruneReport:
        pruned: List[str] = []
        for name in list(library.abstractions.keys()):
            if not ledger.is_stale(name):
                continue
            if library.referenced_by_abstractions(name):
                continue  # never prune something another concept still depends on
            if library.prune_abstraction(name):
                ledger.forget(name)
                pruned.append(name)
        return PruneReport(pruned=pruned, kept=len(library.abstractions),
                           archived_total=len(library.archived))
