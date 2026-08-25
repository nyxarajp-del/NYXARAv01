"""NYXARA · njp/blackbox.py — the cognitive black box (📻, §26).

**What she could not answer about herself.** :mod:`nyxara.njp.metareason` keeps one record per
``(kind, strategy)`` pair, so she can say *"simulation beats retrieval on causal problems"*. That
is a real thing to know and it is the only resolution she had. It cannot answer the question this
module exists for::

    "Is condition mein meri strategy historically fail hoti hai."

A kind is not a condition. Two causal questions — one where grounding found the entity and one
where it found nothing, one asked as a plain question and one buried in a greeting, one about
something she has met four hundred times and one entirely novel — are the same ``kind`` and
routinely different problems. Averaging them produces a number that is true of neither, and the
reliable case subsidises the shaky one, which is backwards: the novel ungrounded turn is exactly
where the caller most needs the warning.

**Why nothing already did this.** Three things record fragments and none records the episode:

* :class:`~nyxara.njp.ledger.ErrorRecord` holds ``claim / tokens / verdict / truth`` — what was
  believed and what was true, with no trace of *how* it was reached or under what circumstances.
* ``observe/turn_ledger.py`` records which **language rung** spoke, which is a fact about the LLM
  ladder and says nothing about her reasoning.
* :class:`~nyxara.njp.metareason.Strategy` holds ``wins``/``trials`` per kind, which is the
  average this module exists to disaggregate.

So the material for "my strategy fails under these conditions" was never written down, and the
question was unanswerable rather than unanswered.

**The record.** One :class:`CognitiveEpisode` per graded turn, holding the §26 chain end to end —
input, belief state, strategy, prediction, action, result, error, update — beside the
:class:`Conditions` under which it happened. Conditions are a small **discrete** tuple on purpose.
Continuous features would make every episode unique and nothing would ever have a comparable
neighbour, which is the failure mode of a log pretending to be a memory. Six coarse bands is a
resolution at which episodes actually repeat.

**The one rule that makes it honest.** :meth:`BlackBox.similar` returns ``None`` — not a low
number, not a hedge — until it has :attr:`BlackBox.min_samples` episodes matching the condition.
A rate computed from two episodes is precisely the flattering statistic this repo exists to
refuse, and the difference between *"this strategy fails here"* and *"I have never tried this
strategy here"* is the entire content of the answer.

**It may only ever lower.** :meth:`BlackBox.penalty` returns a non-negative number that
:meth:`~nyxara.njp.metareason.MetaReasoner.choose` subtracts from a candidate's UCB score, and it
is zero wherever there is no record. A condition-level history can demote a strategy that has
measurably failed here; it can never promote one, never invent one, and never reach a strategy
the kind-level pool did not already contain. The reason is the same one
:meth:`~nyxara.njp.brain.NJPBrain._temper_by_novelty` gives for the fabric's seat: a record of her
own past behaviour is not independent evidence about the world, so letting it raise a score would
make "she has done this before" its own justification for doing it again.

Bounded, pure standard library, fail-soft throughout: a failed record is a missing row and never a
broken turn. Persisted through :meth:`to_dict` / :meth:`load_dict` alongside the rest of the brain,
so a condition learned yesterday is still known tomorrow.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

__all__ = ["Conditions", "CognitiveEpisode", "Verdict", "BlackBox", "band"]


def band(value: Optional[float], *, edges: Tuple[float, ...] = (0.34, 0.67)) -> str:
    """A scalar as one of ``low`` / ``mid`` / ``high``, or ``unknown`` when there was none.

    Deliberately coarse. The whole value of a condition is that other episodes share it, and a
    float never repeats — a "condition" keyed on ``confidence=0.8137`` has exactly one member for
    ever and can never accumulate the evidence that would make it worth consulting.

    ``None`` is its own band rather than being folded into ``low``. "She was not confident" and
    "nothing measured her confidence" are different situations, and merging them would let a
    stream of unmeasured turns look like a stream of unconfident ones.
    """
    try:
        if value is None:
            return "unknown"
        v = float(value)
        if v != v:  # NaN is an absent measurement wearing a number's clothes
            return "unknown"
        if v < edges[0]:
            return "low"
        if v < edges[1]:
            return "mid"
        return "high"
    except Exception:  # noqa: BLE001
        return "unknown"


@dataclass(frozen=True)
class Conditions:
    """The circumstances of a turn, at a resolution at which turns repeat.

    Frozen and hashable so it can be a dictionary key directly: the index this class exists to
    support is *"every episode that happened under these circumstances"*, and anything requiring a
    scan to answer that would be consulted on the turn path and quickly stop being consulted.
    """

    kind: str = ""              # the problem kind metareason classified it as
    act: str = ""               # what the Master was doing with the turn — ask, tell, greet
    epistemic: str = ""         # what she was entitled to say: known / believed / unknown
    grounded: bool = False      # did grounding find anything at all
    novelty: str = "unknown"    # how unfamiliar the turn was, banded
    confidence: str = "unknown"  # how sure she was, banded

    def key(self) -> Tuple[str, ...]:
        return (self.kind, self.act, self.epistemic,
                "grounded" if self.grounded else "ungrounded", self.novelty, self.confidence)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Conditions":
        try:
            return Conditions(
                kind=str(d.get("kind", "") or ""), act=str(d.get("act", "") or ""),
                epistemic=str(d.get("epistemic", "") or ""),
                grounded=bool(d.get("grounded", False)),
                novelty=str(d.get("novelty", "unknown") or "unknown"),
                confidence=str(d.get("confidence", "unknown") or "unknown"))
        except Exception:  # noqa: BLE001
            return Conditions()


@dataclass
class CognitiveEpisode:
    """One graded turn, as the chain that produced it rather than as its outcome alone.

    Every field but ``correct`` is context. Keeping them is what separates this from a scoreboard:
    a scoreboard says a strategy is at 0.6, and an episode says which turns made up the 0.4 and
    what they had in common.
    """

    stimulus: str = ""                     # INPUT
    belief: str = ""                       # BELIEF STATE — what she took to be true going in
    strategy: str = ""                     # STRATEGY — which one was chosen
    prediction: str = ""                   # PREDICTION — what she expected to be right
    action: str = ""                       # ACTION — what she actually said or did
    result: str = ""                       # RESULT — what reality turned out to be
    error: float = 0.0                     # ERROR — how wrong, in [0, 1]
    update: str = ""                       # UPDATE — what changed as a consequence
    correct: bool = False
    conditions: Conditions = field(default_factory=Conditions)
    t: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"stimulus": self.stimulus[:200], "belief": self.belief[:200],
                "strategy": self.strategy, "prediction": self.prediction[:200],
                "action": self.action[:200], "result": self.result[:200],
                "error": round(float(self.error), 4), "update": self.update[:200],
                "correct": bool(self.correct), "conditions": self.conditions.to_dict(),
                "t": round(self.t, 3)}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CognitiveEpisode":
        try:
            return CognitiveEpisode(
                stimulus=str(d.get("stimulus", "") or ""), belief=str(d.get("belief", "") or ""),
                strategy=str(d.get("strategy", "") or ""),
                prediction=str(d.get("prediction", "") or ""),
                action=str(d.get("action", "") or ""), result=str(d.get("result", "") or ""),
                error=float(d.get("error", 0.0) or 0.0), update=str(d.get("update", "") or ""),
                correct=bool(d.get("correct", False)),
                conditions=Conditions.from_dict(d.get("conditions") or {}),
                t=float(d.get("t", time.time())))
        except Exception:  # noqa: BLE001
            return CognitiveEpisode()


@dataclass(frozen=True)
class Verdict:
    """What the record says about one strategy under one condition, or that it says nothing.

    There is no "no record" value of ``rate``. :meth:`BlackBox.similar` returns ``None`` in that
    case and this object is not constructed at all, so a caller cannot accidentally read silence
    as a number — which is the mistake the ``min_samples`` rule exists to make impossible rather
    than merely discouraged.
    """

    strategy: str
    trials: int
    rate: float                 # measured successes / trials, under these conditions only
    baseline: Optional[float]   # the same strategy's rate across ALL conditions, when known

    @property
    def underperforming(self) -> bool:
        """Is it doing measurably worse *here* than it does in general?

        ``baseline is None`` means there is nothing to compare against, and an absent comparison
        is not evidence of a deficit.
        """
        return self.baseline is not None and self.rate < self.baseline - 0.1

    def to_dict(self) -> Dict[str, Any]:
        return {"strategy": self.strategy, "trials": self.trials, "rate": round(self.rate, 4),
                "baseline": None if self.baseline is None else round(self.baseline, 4),
                "underperforming": self.underperforming}


class BlackBox:
    """A bounded record of cognitive episodes, indexed by the conditions they happened under.

    Read it through :meth:`similar` for the honest verdict and :meth:`penalty` for the number a
    strategy selector can act on. Both decline where there is not enough evidence, and neither can
    move a score upward.
    """

    def __init__(self, *, capacity: int = 2048, min_samples: int = 5,
                 max_penalty: float = 0.5) -> None:
        #: Episodes are dropped oldest-first. A black box is a *recent* record by construction —
        #: an unbounded one would grow without limit on the turn path, and a condition she has
        #: since stopped meeting is not evidence about the conditions she meets now.
        self.capacity = max(16, int(capacity))
        #: Below this many matching episodes :meth:`similar` returns ``None``. Five is the point
        #: at which a single unlucky turn stops being able to swing a rate by more than 0.2.
        self.min_samples = max(2, int(min_samples))
        self.max_penalty = max(0.0, float(max_penalty))
        self.episodes: List[CognitiveEpisode] = []
        self.recorded = 0
        self.consulted = 0
        self.declined = 0        # consultations that had no record to answer with
        self.acted_on = 0        # consultations that produced a non-zero penalty

    # ---- writing ------------------------------------------------------------ #
    def record(self, episode: CognitiveEpisode) -> bool:
        """Keep one graded episode. Returns whether it was kept."""
        try:
            if not isinstance(episode, CognitiveEpisode) or not episode.strategy:
                # An episode with no strategy cannot answer the question this class exists for,
                # and keeping it would dilute every rate it is averaged into.
                return False
            self.episodes.append(episode)
            self.recorded += 1
            if len(self.episodes) > self.capacity:
                del self.episodes[:len(self.episodes) - self.capacity]
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---- reading ------------------------------------------------------------ #
    def _matching(self, conditions: Conditions, strategy: str) -> List[CognitiveEpisode]:
        key = conditions.key()
        return [e for e in self.episodes
                if e.strategy == strategy and e.conditions.key() == key]

    def _baseline(self, strategy: str) -> Optional[float]:
        got = [e for e in self.episodes if e.strategy == strategy]
        if len(got) < self.min_samples:
            return None
        return sum(1.0 for e in got if e.correct) / float(len(got))

    def similar(self, conditions: Conditions, strategy: str) -> Optional[Verdict]:
        """What has happened when this strategy ran under these conditions.

        ``None`` means she has no record worth reading — either she has never tried it here, or
        she has tried it too few times for the answer to mean anything. Both are "I do not know",
        and reporting them as a number would be the same error as a stated 0.9 in a domain where
        0.9 has meant 0.6.
        """
        try:
            self.consulted += 1
            matching = self._matching(conditions, strategy)
            if len(matching) < self.min_samples:
                self.declined += 1
                return None
            rate = sum(1.0 for e in matching if e.correct) / float(len(matching))
            return Verdict(strategy=strategy, trials=len(matching), rate=rate,
                           baseline=self._baseline(strategy))
        except Exception:  # noqa: BLE001
            self.declined += 1
            return None

    def penalty(self, conditions: Conditions, strategy: str) -> float:
        """How much to subtract from this strategy's score here. Never negative, often zero.

        Scaled by how far below its own baseline it is running, so a strategy that is merely
        mediocre everywhere is not punished twice for being mediocre here too. Where there is no
        baseline to compare against, a rate below one-half is still evidence of a poor fit and is
        charged against the absolute, which is the only reference available.
        """
        try:
            verdict = self.similar(conditions, strategy)
            if verdict is None:
                return 0.0
            reference = verdict.baseline if verdict.baseline is not None else 0.5
            shortfall = reference - verdict.rate
            if shortfall <= 0.0:
                return 0.0
            self.acted_on += 1
            return min(self.max_penalty, shortfall)
        except Exception:  # noqa: BLE001
            return 0.0

    def failing(self, *, limit: int = 5) -> List[Verdict]:
        """The conditions she is measurably worst under — what "I am weak at X" is derived from.

        This is the read that :mod:`nyxara.njp.curriculum` and the self-model want: not a claim
        about her competence, but the specific circumstances in which a specific strategy of hers
        has a worse record than it has in general.
        """
        out: List[Verdict] = []
        try:
            seen = set()
            for episode in self.episodes:
                mark = (episode.conditions.key(), episode.strategy)
                if mark in seen:
                    continue
                seen.add(mark)
                verdict = self.similar(episode.conditions, episode.strategy)
                if verdict is not None and verdict.underperforming:
                    out.append(verdict)
            out.sort(key=lambda v: v.rate)
        except Exception:  # noqa: BLE001
            return out[:limit]
        return out[:limit]

    # ---- reporting and persistence ------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        strategies = sorted({e.strategy for e in self.episodes})
        return {"episodes": len(self.episodes), "capacity": self.capacity,
                "recorded": self.recorded, "strategies": len(strategies),
                "conditions": len({e.conditions.key() for e in self.episodes}),
                "min_samples": self.min_samples,
                "consulted": self.consulted, "declined": self.declined,
                "acted_on": self.acted_on,
                # The share of consultations that had nothing to say. High is the honest early
                # state, not a fault: a black box on its first day knows nothing and reports so.
                "decline_rate": round(self.declined / float(max(self.consulted, 1)), 4)}

    def to_dict(self) -> Dict[str, Any]:
        return {"capacity": self.capacity, "min_samples": self.min_samples,
                "max_penalty": self.max_penalty,
                "counters": {"recorded": self.recorded, "consulted": self.consulted,
                             "declined": self.declined, "acted_on": self.acted_on},
                "episodes": [e.to_dict() for e in self.episodes]}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.capacity = max(16, int(d.get("capacity", self.capacity)))
            self.min_samples = max(2, int(d.get("min_samples", self.min_samples)))
            self.max_penalty = max(0.0, float(d.get("max_penalty", self.max_penalty)))
            self.episodes = [CognitiveEpisode.from_dict(row)
                             for row in (d.get("episodes") or [])][-self.capacity:]
            counters = d.get("counters") or {}
            self.recorded = int(counters.get("recorded", len(self.episodes)))
            self.consulted = int(counters.get("consulted", 0))
            self.declined = int(counters.get("declined", 0))
            self.acted_on = int(counters.get("acted_on", 0))
        except Exception:  # noqa: BLE001 — a corrupt sidecar leaves her with an empty record
            self.episodes = []


def conditions_of(thought: Any, *, kind: str = "") -> Conditions:
    """Read the conditions off a finished :class:`~nyxara.njp.brain.NJPThought`.

    Kept here rather than on the brain so the one definition of "what counts as a condition" lives
    beside the index that uses it. Every field degrades to its empty value: a thought missing an
    organ's output produces a condition that simply carries less, never an exception on the turn
    path.
    """
    try:
        percept = getattr(thought, "percept", None)
        grounding = getattr(percept, "grounding", None)
        act = getattr(thought, "act", None)
        solution = getattr(thought, "solution", None)
        return Conditions(
            kind=str(kind or getattr(solution, "kind", "") or ""),
            act=str(getattr(act, "act", "") or getattr(act, "kind", "") or ""),
            epistemic=str(getattr(thought, "epistemic", "") or ""),
            grounded=bool(getattr(grounding, "grounded", False)),
            novelty=band(getattr(percept, "novelty", None)),
            confidence=band(getattr(thought, "confidence", None)))
    except Exception:  # noqa: BLE001
        return Conditions()
