"""NYXARA · nyx/superpose.py — many answers held at once (⚛, NYX V.01 pillar 4).

Ordinary decision code commits early: it scores the options and takes the best, and everything
else stops existing. This module does not. Every specialist's bid becomes a **hypothesis with a
real amplitude**, all of them are carried forward together, evidence updates them by Bayes, and
the state is **collapsed only when a decision is actually needed** — and even then it reports
honestly when nothing dominated, instead of bluffing the leader.

That is what makes "hold thousands of possibilities at once" a real property rather than a
figure of speech: the runners-up are still in the state, their probabilities are real, and the
entropy of the distribution is a live measure of how undecided she genuinely is.

Honest framing — this matters, because the vocabulary invites overclaiming:

* These are **classical probability amplitudes** on ordinary silicon. There are no qubits, no
  entanglement, and **no quantum speedup**. Candidates are held together; they are not
  *computed* together. Evaluating 1,000 hypotheses costs 1,000 hypotheses' worth of work.
* "Collapse" is a Born-rule *argmax with a threshold*, not a physical measurement.
* Undecided is a real, reportable outcome (``decided=False``): staying superposed and saying so
  is more honest than committing to a leader that never earned it.

Composes the audited :class:`~nyxara.quantum.superposition_states.Superposition`. Fail-soft.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from nyxara.quantum.superposition_states import Superposition

__all__ = ["Collapsed", "SolutionSuperposition"]


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


@dataclass
class Collapsed:
    """The result of measuring the state — including what did *not* win."""

    winner: Any = None                       # the winning Proposal (payload), or None
    label: str = ""
    probability: float = 0.0
    decided: bool = False                    # False == genuinely undecided, not a hidden guess
    entropy: float = 0.0                     # bits — how spread out her belief still is
    margin: float = 0.0
    runners_up: List[Tuple[str, float]] = field(default_factory=list)
    considered: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "probability": round(self.probability, 6),
                "decided": self.decided, "entropy": round(self.entropy, 6),
                "margin": round(self.margin, 6), "considered": self.considered,
                "runners_up": [[lab, round(p, 6)] for lab, p in self.runners_up]}


class SolutionSuperposition:
    """Holds every candidate answer at once; folds in evidence; collapses only on demand."""

    def __init__(self, *, collapse_threshold: float = 0.5, max_candidates: int = 32) -> None:
        self.collapse_threshold = float(collapse_threshold)
        self.max_candidates = max(1, int(max_candidates))
        self._state = Superposition(collapse_threshold=self.collapse_threshold)
        self._by_label: Dict[str, Any] = {}
        self._verifiable: set = set()

    def __len__(self) -> int:
        return len(self._by_label)

    # ---- construction ----------------------------------------------------- #
    @classmethod
    def from_proposals(cls, proposals: Sequence[Any], *, collapse_threshold: float = 0.5,
                       max_candidates: int = 32) -> "SolutionSuperposition":
        """Seed the state from the specialists' bids, amplitude ∝ stated confidence."""
        out = cls(collapse_threshold=collapse_threshold, max_candidates=max_candidates)
        out.add_all(proposals)
        return out

    def add_all(self, proposals: Sequence[Any]) -> "SolutionSuperposition":
        try:
            amplitudes: Dict[str, float] = {}
            for proposal in list(proposals)[: self.max_candidates]:
                if proposal is None:
                    continue
                label = str(getattr(proposal, "source", "?"))
                confidence = _clamp01(float(getattr(proposal, "confidence", 0.0)))
                # amplitude, not probability: the Born rule squares it back out
                amplitudes[label] = max(1e-6, confidence)
                self._by_label[label] = proposal
                if bool(getattr(proposal, "verifiable", False)):
                    self._verifiable.add(label)
            if not amplitudes:
                return self
            # One batched write, then a single renormalisation. Adding one at a time would
            # renormalise after each, repeatedly re-inflating whatever went in first — the
            # order candidates arrived in would then decide the winner.
            self._state.add_many(amplitudes)
            return self
        except Exception:  # noqa: BLE001 — a malformed bid must not break the state
            return self

    # ---- evidence --------------------------------------------------------- #
    def observe(self, likelihoods: Mapping[str, float]) -> "SolutionSuperposition":
        """Fold in evidence: ``likelihoods[label]`` = P(evidence | that answer being right).

        This is where anything that learns *about* the candidates enters — a grounding check, a
        derivation that succeeded, or (later) how many simulated futures each survived.
        """
        try:
            self._state.observe({k: max(0.0, float(v)) for k, v in likelihoods.items()
                                 if k in self._by_label})
        except Exception:  # noqa: BLE001
            pass
        return self

    def probabilities(self) -> Dict[str, float]:
        try:
            return {str(k): float(v) for k, v in self._state.probabilities().items()}
        except Exception:  # noqa: BLE001
            return {}

    def entropy(self) -> float:
        try:
            return float(self._state.entropy())
        except Exception:  # noqa: BLE001
            return 0.0

    # ---- measurement ------------------------------------------------------ #
    def collapse(self, *, force: bool = False) -> Collapsed:
        """Measure. ``decided=False`` means she genuinely has not made up her mind."""
        out = Collapsed(considered=len(self._by_label))
        try:
            if not self._by_label:
                return out
            result = self._state.collapse(force=force)
            if result.label is None:
                return out
            probs = self.probabilities()
            ranked = sorted(probs.items(), key=lambda kv: -kv[1])
            out.label = str(result.label)
            out.winner = self._by_label.get(out.label)
            out.probability = float(result.probability)
            out.decided = bool(result.decided)
            out.entropy = float(result.entropy)
            out.margin = float(result.margin)
            out.runners_up = [(lab, p) for lab, p in ranked[1:4]]
            return out
        except Exception:  # noqa: BLE001 — a failed measurement is an undecided one
            return out

    def stats(self) -> Dict[str, Any]:
        try:
            return {"candidates": len(self._by_label), "entropy": round(self.entropy(), 4),
                    "verifiable": sorted(self._verifiable),
                    "probabilities": {k: round(v, 4) for k, v in self.probabilities().items()}}
        except Exception:  # noqa: BLE001
            return {}
