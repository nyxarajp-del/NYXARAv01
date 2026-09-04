"""NYXARA · njp/blackbox.py — the flight recorder for one act of thinking (🛑, NJP V.12, Phase 6).

**What she could not answer before this.** Every organ keeps its own counters, and between them
they can say how often she was right, which strategy won, what got repaired. None of them can
answer the question the Master's plan asks for:

    *"under which conditions does my strategy historically fail?"*

That is a different shape of question. It is not about a strategy's average — ``metareason``
already has that — it is about the **join**: strategy × situation. A strategy with a 70% record
that fails almost every time the question is causal and she has no world model is not a 70%
strategy; it is a reliable one and a broken one wearing the same name. Averaged, the two halves
hide each other, and the average is the only thing anything held.

**One row per act of thinking**, in the order the plan names::

    INPUT → BELIEF STATE → STRATEGY → PREDICTION → ACTION → RESULT → ERROR → UPDATE

:class:`Episode` is that row. What makes it useful rather than a log is
:attr:`Episode.conditions` — the small set of *situational* facts that were true when she chose,
recorded beside the choice: what kind of problem it was, what speech act, whether the record had
anything to say, whether the substrate recognised the ground. Those are the axes a failure mode
lives on.

**A failure mode is a claim about a join, and it is only reported when it is one.** A condition
seen twice says nothing; a strategy that fails under it once says less. So
:meth:`BlackBox.failure_modes` reports a ``(condition, strategy)`` pair only when it has enough
rows *and* the pair's failure rate stands clearly above that strategy's own baseline — because a
strategy that fails everywhere equally has no failure *mode*, it is simply weak, and
:mod:`nyxara.njp.selfmodel` already says so.

**It records, it does not judge.** Nothing here changes a confidence, retires a strategy or
answers a turn. It is read by :mod:`nyxara.njp.curriculum`, which turns a named weakness into
something to practise, and by the Master, who is entitled to ask why she keeps getting a kind of
question wrong and to get an answer with rows behind it.

Bounded: the recorder keeps the most recent :data:`_CAPACITY` episodes and the aggregate counts
survive them, so a long session degrades to statistics rather than to memory pressure.

Pure standard library. No LLM. Fail-soft: a failed recording costs a row, never a turn.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Episode", "FailureMode", "BlackBox"]

#: Episodes kept in full. Beyond it the oldest are dropped and only the counts remain — which is
#: what `failure_modes` reads, so the analysis does not degrade with the buffer.
_CAPACITY = 512

#: Rows a (condition, strategy) pair needs before its failure rate is reported as a mode rather
#: than as noise. Small enough to be reachable in a session, large enough that one bad turn is
#: not a finding.
_MIN_ROWS = 4

#: How far above the strategy's own baseline failure rate a pair must sit to be a *mode*. A
#: strategy that fails everywhere equally is weak, not conditionally broken, and `selfmodel`
#: already reports weakness.
_LIFT = 0.25


@dataclass
class Episode:
    """One act of thinking, recorded in the order the plan names its stages."""

    turn: int = 0
    #: INPUT
    stimulus: str = ""
    #: BELIEF STATE — what she thought she knew going in, and how sure.
    epistemic: str = ""
    confidence: float = 0.0
    #: STRATEGY — the one actually run, not the one the bandit nominated.
    strategy: str = ""
    problem_kind: str = ""
    #: PREDICTION / ACTION — what she committed to.
    answer: str = ""
    #: RESULT — the gauntlet's verdict where it ran.
    verdict: str = ""
    #: ERROR — whether **this answer** was later contradicted by the Master's own statement.
    #: Deliberately *not* ``LoopReport.correct``: that number is dominated by the manifold's
    #: next-state anticipation, which is a claim about the substrate and not about the strategy
    #: that answered. Charging a fabric miss to the reasoning that produced the sentence read
    #: 38 failures in 44 rows on a session where the answers were mostly right.
    corrected: Optional[bool] = None
    #: UPDATE — what the turn changed about her.
    repaired: int = 0
    #: The situation, as a small set of facts true when she chose. These are the axes a failure
    #: mode lives on; see the module docstring.
    conditions: Tuple[str, ...] = ()
    at: float = field(default_factory=time.time)

    @property
    def failed(self) -> bool:
        """Did this episode go wrong?

        Graded first by reality where reality spoke, and by the gauntlet otherwise. An abstention
        is **not** a failure: declining to answer is the correct outcome for a question she cannot
        settle, and counting it as failure would teach the recorder that honesty is a defect.
        """
        if self.corrected is not None:
            return self.corrected
        return self.verdict == "refuted"

    @property
    def scored(self) -> bool:
        """Was there an outcome at all? An unscored episode is context, never evidence."""
        return self.corrected is not None or self.verdict in ("refuted", "established")

    def to_dict(self) -> Dict[str, Any]:
        return {"turn": self.turn, "stimulus": self.stimulus[:80],
                "epistemic": self.epistemic, "confidence": round(self.confidence, 3),
                "strategy": self.strategy, "problem_kind": self.problem_kind,
                "answer": self.answer[:60], "verdict": self.verdict,
                "corrected": self.corrected, "repaired": self.repaired,
                "conditions": list(self.conditions)}


@dataclass
class FailureMode:
    """A ``(condition, strategy)`` pair that goes wrong far more often than the strategy does."""

    condition: str = ""
    strategy: str = ""
    rows: int = 0
    failures: int = 0
    baseline: float = 0.0

    @property
    def rate(self) -> float:
        return self.failures / self.rows if self.rows else 0.0

    @property
    def lift(self) -> float:
        """How much worse than the strategy's own average. The whole content of the claim."""
        return self.rate - self.baseline

    def why(self) -> str:
        return (f"{self.strategy} fails {self.rate:.0%} of the time when {self.condition}, "
                f"against {self.baseline:.0%} overall ({self.failures}/{self.rows})")

    def to_dict(self) -> Dict[str, Any]:
        return {"condition": self.condition, "strategy": self.strategy,
                "rows": self.rows, "failures": self.failures,
                "rate": round(self.rate, 4), "baseline": round(self.baseline, 4),
                "lift": round(self.lift, 4), "why": self.why()}


class BlackBox:
    """Records episodes and answers where a strategy is conditionally unreliable."""

    def __init__(self, *, capacity: int = _CAPACITY, min_rows: int = _MIN_ROWS,
                 lift: float = _LIFT) -> None:
        self.capacity = max(16, int(capacity))
        self.min_rows = max(2, int(min_rows))
        self.lift = float(lift)
        self.episodes: List[Episode] = []
        self.recorded = 0
        #: (condition, strategy) → [rows, failures], kept beyond the episode buffer so the
        #: analysis does not shrink when memory does.
        self._pairs: Dict[Tuple[str, str], List[int]] = {}
        #: strategy → [rows, failures] — the baseline a mode has to beat.
        self._strategies: Dict[str, List[int]] = {}

    # ---- recording ---------------------------------------------------------- #
    def record(self, thought: Any) -> Optional[Episode]:
        """One row for one turn. Never raises; an unrecordable turn simply is not a row."""
        try:
            episode = self._read(thought)
            if episode is None:
                return None
            self.episodes.append(episode)
            if len(self.episodes) > self.capacity:
                del self.episodes[: len(self.episodes) - self.capacity]
            self.recorded += 1
            # Tallied only where the verdict was available *on this turn* — the gauntlet's. An
            # answer graded by reality is tallied later, by `grade`, on the turn reality speaks.
            if episode.strategy and episode.verdict in ("refuted", "established"):
                self._tally(episode)
            return episode
        except Exception:  # noqa: BLE001 — a lost row is not a lost turn
            return None

    def grade(self, solution: Any, *, correct: bool) -> Optional[Episode]:
        """Reality decided about an answer given on an **earlier** turn. Find it and score it.

        The grade and the choice do not arrive together, and that is the whole difficulty. She
        answers on one turn; the Master states the fact three turns later; the correction is
        counted on *that* turn — which chose no strategy at all, because it is a statement. Read
        naively, every failure is charged to an episode with ``strategy=''`` and the join that
        makes a failure *mode* possible never forms. Measured: six real failures, zero pairs.

        :class:`~nyxara.njp.integrate.LearningLoop` already holds the join — ``_Deferred`` keeps
        the ``Solution`` from the answering turn precisely so the outcome can be routed back to
        it — so this is called from the same place that already tells ``metareason``, and the two
        stay consistent by construction rather than by coincidence.
        """
        try:
            problem = str(getattr(solution, "problem", "") or "")
            strategy = str(getattr(solution, "strategy", "") or "")
            if not problem:
                return None
            for episode in reversed(self.episodes):
                if episode.stimulus != problem:
                    continue
                if strategy and episode.strategy != strategy:
                    continue
                if episode.corrected is not None:
                    return episode          # already graded; reality does not vote twice
                episode.corrected = not bool(correct)
                if episode.strategy:
                    self._tally(episode)
                return episode
        except Exception:  # noqa: BLE001
            return None
        return None

    def _tally(self, episode: Episode) -> None:
        failed = int(episode.failed)
        row = self._strategies.setdefault(episode.strategy, [0, 0])
        row[0] += 1
        row[1] += failed
        for condition in episode.conditions:
            pair = self._pairs.setdefault((condition, episode.strategy), [0, 0])
            pair[0] += 1
            pair[1] += failed

    @staticmethod
    def _read(thought: Any) -> Optional[Episode]:
        """Pull the eight stages off a thought. Only turns that *did* something become rows."""
        solution = getattr(thought, "solution", None)
        strategy = str(getattr(solution, "strategy", "") or "")
        answer = str(getattr(thought, "answer", "") or "")
        judgement = getattr(thought, "judgement", None)
        loop = getattr(thought, "loop", None)
        # A turn that neither chose a strategy nor produced an answer is not an act of thinking
        # she can be held to. Recording it would fill the buffer with greetings.
        if not strategy and not answer:
            return None
        # Left ungraded on purpose. The verdict on an answer arrives on a later turn — see
        # :meth:`grade` — and reading this turn's `corrections` would charge it to whatever
        # episode happened to be current, which is the *statement* that supplied the correction.
        corrected: Optional[bool] = None
        return Episode(
            turn=int(getattr(loop, "turn", 0) or 0),
            stimulus=str(getattr(thought, "stimulus", "") or ""),
            epistemic=str(getattr(thought, "epistemic", "") or ""),
            confidence=float(getattr(thought, "epistemic_confidence", 0.0) or 0.0),
            strategy=strategy,
            problem_kind=str(getattr(solution, "kind", "") or ""),
            answer=answer,
            verdict=str(getattr(judgement, "verdict", "") or ""),
            corrected=corrected,
            repaired=int(getattr(loop, "repaired", 0) or 0),
            conditions=BlackBox._conditions(thought, solution))

    @staticmethod
    def _conditions(thought: Any, solution: Any) -> Tuple[str, ...]:
        """The situation she chose in, as a handful of named facts.

        Deliberately coarse and deliberately few. A condition that is nearly unique to one turn
        can never accumulate the rows a mode needs, so a fine-grained taxonomy here would produce
        a recorder that is precise and silent.
        """
        out: List[str] = []
        try:
            kind = str(getattr(solution, "kind", "") or "")
            if kind:
                out.append(f"problem={kind}")
            act = getattr(thought, "act", None)
            act_kind = str(getattr(act, "kind", "") or "")
            if act_kind:
                out.append(f"act={act_kind}")
            grounding = getattr(getattr(thought, "percept", None), "grounding", None)
            out.append("grounded" if (getattr(grounding, "triples", None) or [])
                       else "ungrounded")
            # Whether the substrate had seen anything like this. The one condition that is about
            # *her* rather than about the question, and the one a strategy cannot compensate for.
            anticipated = getattr(getattr(thought, "percept", None), "anticipated", None)
            if anticipated is not None:
                out.append("familiar" if getattr(anticipated, "trusted", False) else "unfamiliar")
        except Exception:  # noqa: BLE001
            return tuple(out)
        return tuple(out)

    # ---- what it found ------------------------------------------------------ #
    def failure_modes(self, *, limit: int = 8) -> List[FailureMode]:
        """``(condition, strategy)`` pairs that fail far more than that strategy usually does."""
        out: List[FailureMode] = []
        try:
            for (condition, strategy), (rows, failures) in self._pairs.items():
                if rows < self.min_rows:
                    continue
                total, total_failures = self._strategies.get(strategy, [0, 0])
                if total <= 0:
                    continue
                baseline = total_failures / total
                mode = FailureMode(condition=condition, strategy=strategy,
                                   rows=rows, failures=failures, baseline=baseline)
                if mode.lift < self.lift:
                    continue
                out.append(mode)
            out.sort(key=lambda m: (-m.lift, -m.rows))
        except Exception:  # noqa: BLE001
            return out
        return out[:limit]

    def weakest_condition(self) -> Optional[FailureMode]:
        """The single worst join, for a caller that wants one thing to work on."""
        modes = self.failure_modes(limit=1)
        return modes[0] if modes else None

    def stats(self) -> Dict[str, Any]:
        scored = sum(1 for e in self.episodes if e.scored)
        failed = sum(1 for e in self.episodes if e.scored and e.failed)
        modes = self.failure_modes()
        return {
            "recorded": self.recorded,
            "episodes": len(self.episodes),
            "scored": scored,
            "failed": failed,
            "pairs": len(self._pairs),
            "failure_modes": len(modes),
            "worst": modes[0].to_dict() if modes else None,
        }
