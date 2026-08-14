"""NYXARA · njp/attention.py — what gets her attention, and what loses (👁, NJP V.02).

Every organ NJP has now wants to speak on every turn. The grounder has an answer, memory has a
recall, the predictor has an expectation, curiosity has a question, the goal system has a priority,
the reasoner has a conclusion. Handed to a caller as a pile, that is not integration — it is six
opinions and no cognition. Something has to **lose**.

So the organs do not report; they **compete**. Each submits a candidate with its salience, one
wins the cycle, and the winner is what the turn is actually about. That is the whole point of a
bottleneck: not throughput, but the fact that attending to one thing means not attending to
another, and a mind that never pays that cost is not focusing at all.

**Salience is five measured terms, not a mood**::

    novelty + relevance + uncertainty + goal_relevance + prediction_error

Every one is read off something real — novelty from the manifold's own familiarity, prediction
error from what the fabric actually got wrong, goal relevance from the live goal set, uncertainty
from the epistemic state, relevance from overlap with what is being asked. Nothing here invents a
number to make a candidate look important.

**Two behaviours that stop it being a leaderboard**, both from the repo's existing
:class:`~nyxara.kernel.workspace.GlobalWorkspace`:

* **habituation** — a signature that keeps arriving gets less salient each time. Without it the
  loudest organ wins forever and attention is just a constant.
* **inhibition of return** — what just won is suppressed next cycle, so she moves on instead of
  re-attending to the thing she has already dealt with.

This module supplies the scoring and the wiring; the arbitration itself is the kernel's workspace,
unchanged. Reusing it matters beyond saving code: it means NJP's attention and the rest of
NYXARA's attention are the same mechanism, so what reaches her awareness is one stream rather
than two that have to be reconciled.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["Candidate", "Focus", "Attention"]


@dataclass
class Candidate:
    """One organ's bid for the turn's attention, with the terms that justify it."""

    payload: Any = None
    source: str = ""
    novelty: float = 0.0
    relevance: float = 0.0
    uncertainty: float = 0.0
    goal_relevance: float = 0.0
    prediction_error: float = 0.0
    tags: Tuple[str, ...] = ()

    @property
    def salience(self) -> float:
        """The five terms, weighted. Deliberately flat-ish — no term dominates by construction.

        ``prediction_error`` carries slightly more because being wrong about something is the one
        signal that reliably means *this needs attention now*; ``novelty`` slightly less because
        every cold-start turn is novel and letting that dominate makes a new brain permanently
        distractible.
        """
        return (self.novelty * 0.8
                + self.relevance * 1.0
                + self.uncertainty * 0.9
                + self.goal_relevance * 1.1
                + self.prediction_error * 1.2) / 5.0

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "salience": round(self.salience, 4),
                "novelty": round(self.novelty, 4), "relevance": round(self.relevance, 4),
                "uncertainty": round(self.uncertainty, 4),
                "goal_relevance": round(self.goal_relevance, 4),
                "prediction_error": round(self.prediction_error, 4),
                "tags": list(self.tags)}


@dataclass
class Focus:
    """What won the cycle, what lost, and why."""

    winner: Optional[Candidate] = None
    source: str = ""
    salience: float = 0.0
    margin: float = 0.0                # how clear the winner was
    considered: int = 0
    losers: List[str] = field(default_factory=list)
    broadcast: bool = False            # did it clear the workspace's access threshold?
    ms: float = 0.0

    @property
    def attended(self) -> bool:
        return self.winner is not None

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "salience": round(self.salience, 4),
                "margin": round(self.margin, 4), "considered": self.considered,
                "losers": self.losers[:8], "broadcast": self.broadcast,
                "attended": self.attended, "ms": round(self.ms, 3),
                "winner": self.winner.to_dict() if self.winner else None}


class Attention:
    """Scores the organs' bids and runs them through the kernel's own bottleneck."""

    def __init__(self, brain: Any = None, *, workspace: Any = None,
                 capacity: int = 1, access_threshold: float = 0.15) -> None:
        self.brain = brain
        # capacity=1 is the bottleneck. Raising it does not make her think about more things at
        # once; it makes "attention" stop meaning anything.
        self.capacity = max(1, int(capacity))
        self.access_threshold = float(access_threshold)
        self.workspace = workspace if workspace is not None else self._build_workspace()

        self.cycles = 0
        self.attended = 0
        self.by_source: Dict[str, int] = {}
        self.last: Optional[Focus] = None

    def _build_workspace(self) -> Any:
        try:
            from nyxara.kernel.workspace import GlobalWorkspace
            return GlobalWorkspace(capacity=self.capacity,
                                   access_threshold=self.access_threshold)
        except Exception:  # noqa: BLE001 — she still ranks, without habituation or IOR
            return None

    # ---- gathering the bids ------------------------------------------------- #
    def gather(self, thought: Any) -> List[Candidate]:
        """Ask every organ that has something to say for its bid, with real numbers.

        An organ that is off, or that has nothing for this turn, simply does not bid. That is
        different from bidding zero: a silent organ should not dilute the field, and a field of
        zeros would hand the turn to whichever organ happened to be listed first.
        """
        out: List[Candidate] = []
        try:
            percept = getattr(thought, "percept", None)
            novelty = float(getattr(percept, "novelty", 1.0)) if percept else 1.0
            surprise = float(getattr(getattr(percept, "settled", None), "surprise", 0.0) or 0.0)
            goals = self._goal_weights()

            grounding = getattr(percept, "grounding", None) if percept else None
            answer = getattr(grounding, "answer", None) if grounding is not None else None
            if answer is not None and getattr(answer, "answered", False):
                out.append(Candidate(
                    payload=answer.text, source="grounding", novelty=novelty,
                    relevance=float(answer.confidence), goal_relevance=goals.get("answer", 0.0),
                    uncertainty=1.0 - float(answer.confidence),
                    prediction_error=surprise, tags=("answer", "grounded")))

            recall = getattr(percept, "recall", None) if percept else None
            best = getattr(recall, "hit", None) if recall is not None else None
            if best is not None and getattr(best, "text", ""):
                out.append(Candidate(
                    payload=best.text, source="memory", novelty=novelty * 0.5,
                    relevance=float(getattr(recall, "score", 0.0) or 0.0),
                    goal_relevance=goals.get("recall", 0.0),
                    prediction_error=surprise, tags=("recall",)))

            anticipated = getattr(percept, "anticipated", None) if percept else None
            if anticipated is not None and getattr(anticipated, "trusted", False):
                out.append(Candidate(
                    payload=anticipated, source="prediction", novelty=novelty,
                    relevance=float(getattr(anticipated, "familiarity", 0.0) or 0.0),
                    prediction_error=surprise, tags=("prediction",)))

            # A wrong prediction is the strongest reason to attend to anything, so it bids on its
            # own rather than only colouring the others.
            if surprise > 0.5:
                out.append(Candidate(
                    payload=f"prediction was wrong by {surprise:.0%}", source="surprise",
                    novelty=novelty, uncertainty=surprise, prediction_error=surprise,
                    tags=("error", "surprise")))

            out.extend(self._organ_bids(thought, novelty=novelty, surprise=surprise, goals=goals))
            return out
        except Exception:  # noqa: BLE001 — a failed gather attends to nothing, never crashes
            return out

    def _organ_bids(self, thought: Any, *, novelty: float, surprise: float,
                    goals: Dict[str, float]) -> List[Candidate]:
        """Bids from the slower organs — curiosity and goals — when they have one."""
        out: List[Candidate] = []
        brain = self.brain
        if brain is None:
            return out
        try:
            curiosity = getattr(brain, "curiosity", None)
            question = curiosity.top() if curiosity is not None else None
            if question is not None:
                out.append(Candidate(
                    payload=question, source="curiosity", novelty=novelty,
                    uncertainty=float(getattr(question, "uncertainty", 0.5)),
                    goal_relevance=float(getattr(question, "value", 0.0)),
                    prediction_error=surprise, tags=("question", "gap")))
        except Exception:  # noqa: BLE001
            pass
        try:
            goal_system = getattr(brain, "goals", None)
            top = goal_system.top() if goal_system is not None else None
            if top is not None:
                out.append(Candidate(
                    payload=top, source="goal", relevance=float(getattr(top, "priority", 0.0)),
                    goal_relevance=1.0, tags=("goal",)))
        except Exception:  # noqa: BLE001
            pass
        return out

    def _goal_weights(self) -> Dict[str, float]:
        """Top-down bias: what she is currently trying to do makes related things more salient."""
        try:
            goal_system = getattr(self.brain, "goals", None) if self.brain else None
            if goal_system is None:
                return {}
            return goal_system.attention_bias()
        except Exception:  # noqa: BLE001
            return {}

    # ---- the competition ----------------------------------------------------- #
    def attend(self, thought: Any = None, *,
               candidates: Optional[Sequence[Candidate]] = None) -> Focus:
        """Run one attention cycle. Exactly one thing wins, or nothing does."""
        out = Focus()
        t0 = time.perf_counter()
        try:
            bids = list(candidates) if candidates is not None else self.gather(thought)
            out.considered = len(bids)
            if not bids:
                return out
            self.cycles += 1

            ranked = sorted(bids, key=lambda c: c.salience, reverse=True)

            winner = self._arbitrate(ranked)
            # Losers are computed against the ACTUAL winner, not against rank. Habituation and
            # inhibition-of-return mean the top-ranked bid frequently does not win, so slicing
            # `ranked[1:]` listed the winner among the things it beat.
            out.losers = [c.source for c in ranked if c is not winner][:8]
            if winner is None:
                return out
            out.winner = winner
            out.source = winner.source
            out.salience = winner.salience
            out.margin = winner.salience - (ranked[1].salience if len(ranked) > 1 else 0.0)
            out.broadcast = winner.salience >= self.access_threshold
            self.attended += 1
            self.by_source[winner.source] = self.by_source.get(winner.source, 0) + 1
            self.last = out
            return out
        except Exception:  # noqa: BLE001
            return out
        finally:
            out.ms = (time.perf_counter() - t0) * 1000.0

    def _arbitrate(self, ranked: Sequence[Candidate]) -> Optional[Candidate]:
        """Hand the field to the kernel's workspace, so habituation and IOR actually apply.

        Falling back to "highest salience wins" when no workspace is available is correct but
        strictly worse: it has no memory of what already won, so the same source can hold her
        attention indefinitely. The fallback exists so the organ still works, not because the two
        are equivalent.
        """
        try:
            if self.workspace is None:
                return ranked[0] if ranked else None
            from nyxara.kernel.workspace import Content

            by_id: Dict[str, Candidate] = {}
            for candidate in ranked:
                content = Content(
                    payload=candidate.payload, source=candidate.source,
                    activation=1.0, novelty=candidate.novelty,
                    urgency=candidate.prediction_error,
                    goal_relevance=candidate.goal_relevance,
                    source_priority=candidate.relevance,
                    tags=frozenset(candidate.tags))
                by_id[content.item_id] = candidate
                self.workspace.submit(content)

            broadcasts = self.workspace.cycle() or []
            for broadcast in broadcasts:
                # `Broadcast.winner` — the Content that reached consciousness. Reading a
                # `.content`/`.lead` attribute that does not exist silently discarded every win,
                # so the workspace arbitrated correctly and this threw the answer away: three
                # candidates considered, nothing ever attended to.
                item_id = getattr(getattr(broadcast, "winner", None), "item_id", "")
                if item_id in by_id:
                    return by_id[item_id]
            # The workspace declined everything — nothing cleared the access threshold. That is a
            # real outcome (a turn where nothing was worth attending to), not a reason to override
            # it with the top of the list.
            return None
        except Exception:  # noqa: BLE001
            return ranked[0] if ranked else None

    # ---- reporting ------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        return {"cycles": self.cycles, "attended": self.attended,
                "by_source": dict(self.by_source),
                "attention_rate": (round(self.attended / self.cycles, 4)
                                   if self.cycles else None),
                "workspace": self.workspace is not None,
                "capacity": self.capacity,
                "last": self.last.to_dict() if self.last else None}
