"""NYXARA · njp/curiosity.py — her own questions, and which one is worth asking (❓, NJP V.02).

Answering what she is asked is the easy half. The half that makes a mind is noticing what it does
**not** know and going after it — and doing that without becoming a machine that asks endless
questions, which is its own kind of useless.

**Questions come from measured gaps, never from a list.** Four sources, each a real hole in
something she actually has:

* a **known-unknown** — a question the reasoner could not answer and recorded as such;
* a **missing relation** — an entity that lacks a property *most* of its peers have. If Ravi and
  Sara both have an employer and Amit does not, that asymmetry is the question, and it was
  discovered rather than declared. A majority rather than a unanimous vote, because peers are
  rarely uniform and demanding that they are is demanding that she never notice anything;
* a **repeated failure** — a prediction that keeps being wrong in the same way, read off the error
  record. Being wrong once is noise; being wrong the same way five times is a gap;
* a **thin region** — somewhere the world model has too little coverage to predict at all.

**Which question to ask is a decision, not an appetite.** This is where
:class:`~nyxara.planning.voi.ValueOfInformation` does the work: expected value of information
against the cost of getting it, so she asks the question that **reduces uncertainty most per unit
cost** rather than the one that occurred to her most recently. It also decides *how* — act on what
she has, ask the Master, or go and investigate — and asking the Master is deliberately expensive,
because a mind that asks about everything has simply moved its work onto him.

**Curiosity that never resolves is just anxiety.** Every question carries what would close it, and
:meth:`resolve` records the answer and retires it. A question that has been asked and answered
does not come back, and one that keeps failing to resolve is escalated rather than silently
re-asked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = ["Gap", "Question", "Curiosity"]


class Gap:
    """Why she does not know something. The kind determines how it can be closed."""

    UNKNOWN = "known_unknown"           # asked, could not answer
    MISSING_RELATION = "missing_relation"   # peers have it, this one does not
    REPEATED_FAILURE = "repeated_failure"   # wrong the same way, repeatedly
    THIN_COVERAGE = "thin_coverage"     # too little data to predict here at all


@dataclass
class Question:
    """One thing she wants to know, with what it would take and what it would buy."""

    text: str = ""
    gap: str = Gap.UNKNOWN
    subject: str = ""
    predicate: str = ""
    uncertainty: float = 0.5           # how unsure she is, 0…1
    stakes: float = 0.3                # what it costs to stay ignorant
    cost: float = 0.2                  # what answering it would cost
    value: float = 0.0                 # EVPI, filled in by the VOI pass
    action: str = ""                   # act | ask | gather
    asked: int = 0
    born: float = field(default_factory=time.time)
    resolved: bool = False
    answer: str = ""

    def key(self) -> Tuple[str, str, str]:
        return (self.gap, self.subject.lower(), self.predicate)

    @property
    def stale(self) -> bool:
        """Asked repeatedly and still open — escalate rather than ask a fourth time."""
        return self.asked >= 3 and not self.resolved

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "gap": self.gap, "subject": self.subject,
                "predicate": self.predicate, "uncertainty": round(self.uncertainty, 4),
                "value": round(self.value, 4), "cost": round(self.cost, 4),
                "action": self.action, "asked": self.asked, "resolved": self.resolved,
                "stale": self.stale}


# Asking the Master is expensive by design: a mind that asks about everything has moved its work
# onto him. Investigating something herself is cheap by comparison.
_COST_ASK = 0.6
_COST_GATHER = 0.15

# Fraction of peers that must hold a property before its absence counts as a gap worth asking
# about. Below a majority the "asymmetry" is just variation; at 1.0 a single atypical peer
# silences every question.
_PEER_MAJORITY = 0.6


class Curiosity:
    """Finds her own gaps, and decides which one is worth closing."""

    def __init__(self, brain: Any = None, *, voi: Any = None,
                 capacity: int = 128, min_value: float = 0.05,
                 min_failures: int = 3) -> None:
        self.brain = brain
        self.voi = voi if voi is not None else self._build_voi()
        self.capacity = max(8, int(capacity))
        # EVPI below which a question is not worth asking at all. Without a floor she would
        # surface every trivial gap, and a mind that asks about everything is not curious, it is
        # exhausting.
        self.min_value = float(min_value)
        self.min_failures = max(2, int(min_failures))

        self.questions: Dict[Tuple[str, str, str], Question] = {}
        self.resolved_count = 0
        self.passes = 0
        self.raised = 0

    @staticmethod
    def _build_voi() -> Any:
        try:
            from nyxara.planning.voi import ValueOfInformation
            return ValueOfInformation()
        except Exception:  # noqa: BLE001 — she still finds gaps, and ranks them more crudely
            return None

    # ---- finding gaps -------------------------------------------------------- #
    def wonder(self) -> List[Question]:
        """One pass over everything she has, looking for holes. Returns what is newly raised."""
        raised: List[Question] = []
        try:
            self.passes += 1
            for question in (self._from_unknowns() + self._from_missing_relations()
                             + self._from_failures() + self._from_thin_coverage()):
                if self._raise(question) is question:
                    raised.append(question)
            self._appraise()
            self._evict()
            return raised
        except Exception:  # noqa: BLE001 — a failed pass wonders about nothing
            return raised

    def _from_unknowns(self) -> List[Question]:
        """Questions the reasoner could not answer and recorded."""
        out: List[Question] = []
        try:
            reasoner = getattr(self.brain, "reasoner", None)
            if reasoner is None:
                return out
            for state in reasoner.problems.values():
                for unknown in state.unknown[:4]:
                    out.append(Question(
                        text=unknown, gap=Gap.UNKNOWN, subject=unknown[:60],
                        uncertainty=0.9, stakes=0.4, cost=_COST_ASK))
        except Exception:  # noqa: BLE001
            pass
        return out

    def _from_missing_relations(self) -> List[Question]:
        """An entity that lacks a property most of its peers have.

        This is the one that is genuinely *discovered*: nothing declares that a person ought to
        have an employer. It comes from noticing that the other entities carrying the same
        relations mostly carry this one too, and that this one does not.
        """
        out: List[Question] = []
        try:
            grounder = getattr(self.brain, "grounder", None)
            if grounder is None:
                return out
            # subject -> the predicates it participates in
            by_subject: Dict[str, Set[str]] = {}
            for (subject, predicate), triples in grounder.facts.items():
                if any(not t.superseded for t in triples):
                    by_subject.setdefault(subject, set()).add(predicate)
            if len(by_subject) < 3:
                return out                    # too few entities for "peers" to mean anything
            for subject, predicates in by_subject.items():
                peers = [p for s, p in by_subject.items()
                         if s != subject and len(p & predicates) >= 1]
                if len(peers) < 2:
                    continue
                # A property MOST peers have and this one does not.
                #
                # Deliberately a majority rather than a universal intersection. Requiring every
                # peer to share the property means one atypical peer silences the question
                # entirely: with Ravi and Sara both employed, Amit not, and the Master (who has a
                # name but no employer) also sharing `located_in`, the intersection collapses to
                # exactly the property Amit already has, and the real asymmetry disappears.
                # Peers are rarely uniform, and demanding that they are is demanding that she
                # never notice anything.
                tally: Dict[str, int] = {}
                for peer in peers:
                    for predicate in peer:
                        tally[predicate] = tally.get(predicate, 0) + 1
                needed = max(2, int(len(peers) * _PEER_MAJORITY + 0.5))
                common = {p for p, n in tally.items() if n >= needed}
                for missing in sorted(common - predicates)[:2]:
                    out.append(Question(
                        text=f"what is {subject}'s {missing.replace('_', ' ')}?",
                        gap=Gap.MISSING_RELATION, subject=subject, predicate=missing,
                        uncertainty=0.7, stakes=0.3, cost=_COST_ASK))
        except Exception:  # noqa: BLE001
            pass
        return out

    def _from_failures(self) -> List[Question]:
        """A prediction that keeps being wrong the same way. Once is noise; five times is a gap."""
        out: List[Question] = []
        try:
            predictor = getattr(self.brain, "predictor", None)
            if predictor is None:
                return out
            by_kind: Dict[str, int] = {}
            for outcome in getattr(predictor, "outcomes", None) or []:
                diagnosis = getattr(outcome, "diagnosis", None)
                kind = str(getattr(diagnosis, "kind", "") or "")
                if kind and not getattr(outcome, "correct", True):
                    by_kind[kind] = by_kind.get(kind, 0) + 1
            for kind, count in by_kind.items():
                if count < self.min_failures:
                    continue
                out.append(Question(
                    text=f"why does my {kind.replace('_', ' ')} keep failing?",
                    gap=Gap.REPEATED_FAILURE, subject=kind, predicate="failure_cause",
                    uncertainty=min(1.0, count / 10.0), stakes=0.7, cost=_COST_GATHER))
        except Exception:  # noqa: BLE001
            pass
        return out

    def _from_thin_coverage(self) -> List[Question]:
        """Regions the world model cannot predict in because it has barely seen them."""
        out: List[Question] = []
        try:
            world = getattr(self.brain, "world", None)
            if world is None:
                return out
            for name, count in (world.thin_regions() or [])[:3]:
                out.append(Question(
                    text=f"what usually follows {name}?",
                    gap=Gap.THIN_COVERAGE, subject=name, predicate="consequence",
                    uncertainty=0.8, stakes=0.4, cost=_COST_GATHER))
        except Exception:  # noqa: BLE001
            pass
        return out

    # ---- deciding what is worth asking ---------------------------------------- #
    def _appraise(self) -> None:
        """Price every open question by expected value of information.

        This is what turns a pile of gaps into a decision. Without it she would surface whichever
        question was generated last, which is not curiosity — it is recency.
        """
        try:
            for question in self.questions.values():
                if question.resolved:
                    continue
                question.value, question.action = self._value_of(question)
        except Exception:  # noqa: BLE001
            pass

    def _value_of(self, question: Question) -> Tuple[float, str]:
        """``(EVPI, action)`` — how much closing this gap is worth, and how to close it."""
        try:
            if self.voi is None:
                # Crude but honest fallback: value scales with what is unknown and what rides on
                # it, minus what it costs. Reported as such rather than dressed up as EVPI.
                return (max(0.0, question.uncertainty * question.stakes - question.cost),
                        "ask" if question.cost >= _COST_ASK else "gather")
            from nyxara.planning.voi import DecisionContext, InfoSource

            context = DecisionContext(
                uncertainty=question.uncertainty, stakes=question.stakes,
                reversibility=1.0,
                about_owner_intent=(question.gap == Gap.UNKNOWN))
            sources = [
                InfoSource(name="ask", kind="ask", reliability=0.95, cost=_COST_ASK),
                InfoSource(name="gather", kind="gather", reliability=0.6, cost=_COST_GATHER),
            ]
            recommendation = self.voi.recommend(context, sources)
            return (float(getattr(recommendation, "evpi", 0.0)),
                    str(getattr(getattr(recommendation, "action", None), "value", "")
                        or getattr(recommendation, "action", "")))
        except Exception:  # noqa: BLE001
            return (0.0, "")

    # ---- the queue -------------------------------------------------------------- #
    def _raise(self, question: Question) -> Optional[Question]:
        """Add a question, unless it is already open or already answered."""
        try:
            key = question.key()
            existing = self.questions.get(key)
            if existing is not None:
                if existing.resolved:
                    return None           # asked and answered; it does not come back
                existing.uncertainty = max(existing.uncertainty, question.uncertainty)
                return existing
            self.questions[key] = question
            self.raised += 1
            return question
        except Exception:  # noqa: BLE001
            return None

    def top(self) -> Optional[Question]:
        """The single question most worth asking right now, or None if none clears the floor."""
        try:
            open_questions = [q for q in self.questions.values()
                              if not q.resolved and q.value >= self.min_value and not q.stale]
            if not open_questions:
                return None
            return max(open_questions, key=lambda q: q.value)
        except Exception:  # noqa: BLE001
            return None

    def ask(self) -> Optional[Question]:
        """Take the top question and mark it asked. Returns None when nothing is worth asking."""
        question = self.top()
        if question is not None:
            question.asked += 1
        return question

    def resolve(self, question: Any, answer: str = "") -> bool:
        """Record that a question was answered. Curiosity that never resolves is just anxiety."""
        try:
            key = question.key() if hasattr(question, "key") else None
            target = self.questions.get(key) if key else None
            if target is None:
                return False
            target.resolved = True
            target.answer = str(answer or "")
            self.resolved_count += 1
            return True
        except Exception:  # noqa: BLE001
            return False

    def open_questions(self) -> List[Question]:
        return sorted((q for q in self.questions.values() if not q.resolved),
                      key=lambda q: q.value, reverse=True)

    def stale_questions(self) -> List[Question]:
        """Asked three times and still open — these should escalate, not be re-asked."""
        return [q for q in self.questions.values() if q.stale]

    def _evict(self) -> None:
        """Cap the queue, dropping the least valuable resolved questions first."""
        try:
            if len(self.questions) <= self.capacity:
                return
            ranked = sorted(self.questions.items(),
                            key=lambda kv: (not kv[1].resolved, kv[1].value))
            for key, _ in ranked[: len(self.questions) - self.capacity]:
                self.questions.pop(key, None)
        except Exception:  # noqa: BLE001
            pass

    # ---- reporting ---------------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        open_questions = self.open_questions()
        by_gap: Dict[str, int] = {}
        for question in open_questions:
            by_gap[question.gap] = by_gap.get(question.gap, 0) + 1
        return {"passes": self.passes, "raised": self.raised,
                "open": len(open_questions), "resolved": self.resolved_count,
                "stale": len(self.stale_questions()), "by_gap": by_gap,
                "voi": self.voi is not None,
                "top": self.top().to_dict() if self.top() else None}

    def to_dict(self) -> Dict[str, Any]:
        return {"questions": [q.to_dict() for q in self.questions.values()],
                "counters": {"passes": self.passes, "raised": self.raised,
                             "resolved": self.resolved_count}}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            for row in (d.get("questions") or []):
                question = Question(
                    text=str(row.get("text", "")), gap=str(row.get("gap", Gap.UNKNOWN)),
                    subject=str(row.get("subject", "")),
                    predicate=str(row.get("predicate", "")),
                    uncertainty=float(row.get("uncertainty", 0.5)),
                    value=float(row.get("value", 0.0)), cost=float(row.get("cost", 0.2)),
                    action=str(row.get("action", "")), asked=int(row.get("asked", 0)),
                    resolved=bool(row.get("resolved", False)))
                if question.text:
                    self.questions[question.key()] = question
            counters = d.get("counters") or {}
            self.passes = int(counters.get("passes", 0))
            self.raised = int(counters.get("raised", 0))
            self.resolved_count = int(counters.get("resolved", 0))
        except Exception:  # noqa: BLE001
            pass
