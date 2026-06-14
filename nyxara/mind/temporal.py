"""NYXARA · mind/temporal.py — temporal reasoning over remembered time (✦).

Memory stamps every record with a time; until now nothing *reasoned* over those stamps.
This engine gives NYXARA a sense of **when**, and of how events sit in time relative to
one another:

* **Allen's interval algebra** — the thirteen qualitative relations between two intervals
  (``before``, ``meets``, ``overlaps``, ``during``, ``finishes``, ``equals``, …). Point
  events are zero-length intervals, so "X before Y" falls out of the same machinery.

* **Precedence & lag** — does X *consistently* come before Y, and by how long? "the alert
  fires, and ~3 days later the outage follows" is a precedence with a typical lag and a
  consistency score. Precedence is necessary (not sufficient) for causation — the world
  model decides cause; this only reports the temporal shape.

* **Periodicity** — from the gaps between recurrences, detect a stable period and name it
  ("repeats roughly every week"), with a confidence from how regular the rhythm is.

Pure standard library (small lists + ``statistics``). It is a *faculty*: it reports
temporal structure; it never acts.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "SECOND", "MINUTE", "HOUR", "DAY", "WEEK",
    "AllenRelation",
    "Interval",
    "TemporalEvent",
    "Precedence",
    "Periodicity",
    "describe_duration",
    "allen_relation",
    "TemporalReasoner",
]

SECOND = 1.0
MINUTE = 60.0
HOUR = 3600.0
DAY = 86400.0
WEEK = 7 * DAY

_NAMED_UNITS: Tuple[Tuple[str, float], ...] = (
    ("week", WEEK), ("day", DAY), ("hour", HOUR), ("minute", MINUTE), ("second", SECOND),
)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def describe_duration(seconds: float) -> str:
    """A human phrase for a span: 604800s -> "1 week", 259200s -> "3 days"."""
    seconds = abs(seconds)
    for name, size in _NAMED_UNITS:
        if seconds >= size * 0.85:
            mult = seconds / size
            # snap to a whole unit when very close (so ~7.0 days reads as "1 week")
            rounded = round(mult)
            if rounded >= 1 and abs(mult - rounded) <= 0.15 * max(1, rounded):
                return f"{rounded} {name}" + ("s" if rounded != 1 else "")
            return f"{mult:.1f} {name}s"
    return f"{seconds:.0f} seconds"


# --------------------------------------------------------------------------- #
# Allen's interval algebra
# --------------------------------------------------------------------------- #
class AllenRelation(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    MEETS = "meets"
    MET_BY = "met_by"
    OVERLAPS = "overlaps"
    OVERLAPPED_BY = "overlapped_by"
    STARTS = "starts"
    STARTED_BY = "started_by"
    DURING = "during"
    CONTAINS = "contains"
    FINISHES = "finishes"
    FINISHED_BY = "finished_by"
    EQUALS = "equals"

    @property
    def converse(self) -> "AllenRelation":
        return _CONVERSE[self]


_CONVERSE: Dict[AllenRelation, AllenRelation] = {
    AllenRelation.BEFORE: AllenRelation.AFTER,
    AllenRelation.AFTER: AllenRelation.BEFORE,
    AllenRelation.MEETS: AllenRelation.MET_BY,
    AllenRelation.MET_BY: AllenRelation.MEETS,
    AllenRelation.OVERLAPS: AllenRelation.OVERLAPPED_BY,
    AllenRelation.OVERLAPPED_BY: AllenRelation.OVERLAPS,
    AllenRelation.STARTS: AllenRelation.STARTED_BY,
    AllenRelation.STARTED_BY: AllenRelation.STARTS,
    AllenRelation.DURING: AllenRelation.CONTAINS,
    AllenRelation.CONTAINS: AllenRelation.DURING,
    AllenRelation.FINISHES: AllenRelation.FINISHED_BY,
    AllenRelation.FINISHED_BY: AllenRelation.FINISHES,
    AllenRelation.EQUALS: AllenRelation.EQUALS,
}


@dataclass(frozen=True)
class Interval:
    """A closed time span [start, end]; a point event has start == end."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            object.__setattr__(self, "end", self.start)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_point(self) -> bool:
        return self.end == self.start


def allen_relation(a: Interval, b: Interval, *, tol: float = 0.0) -> AllenRelation:
    """The Allen relation of ``a`` *to* ``b`` (read "a <relation> b"), within tolerance."""
    def c(x: float, y: float) -> int:
        if x < y - tol:
            return -1
        if x > y + tol:
            return 1
        return 0

    ss = c(a.start, b.start)      # a.start vs b.start
    ee = c(a.end, b.end)          # a.end vs b.end
    es = c(a.end, b.start)        # a.end vs b.start
    se = c(a.start, b.end)        # a.start vs b.end

    if es < 0:
        return AllenRelation.BEFORE
    if se > 0:
        return AllenRelation.AFTER
    # touching at a single shared endpoint is "meets" only when the spans don't also
    # share the other endpoint (which would make it starts/finishes/equals instead)
    if es == 0 and ss < 0:
        return AllenRelation.MEETS
    if se == 0 and ee > 0:
        return AllenRelation.MET_BY

    if ss == 0 and ee == 0:
        return AllenRelation.EQUALS
    if ss == 0:
        return AllenRelation.STARTS if ee < 0 else AllenRelation.STARTED_BY
    if ee == 0:
        return AllenRelation.FINISHES if ss > 0 else AllenRelation.FINISHED_BY
    if ss < 0 and ee < 0:
        return AllenRelation.OVERLAPS
    if ss > 0 and ee > 0:
        return AllenRelation.OVERLAPPED_BY
    if ss < 0 < ee:
        return AllenRelation.CONTAINS
    return AllenRelation.DURING   # ss > 0 and ee < 0


# --------------------------------------------------------------------------- #
# Events & findings
# --------------------------------------------------------------------------- #
@dataclass
class TemporalEvent:
    label: str
    start: float
    end: Optional[float] = None      # None => instantaneous (a point in time)

    def interval(self) -> Interval:
        return Interval(self.start, self.end if self.end is not None else self.start)


@dataclass
class Precedence:
    """How consistently ``cause`` precedes ``effect``, and by how long."""

    cause: str
    effect: str
    support: int                 # how many cause-events were followed by an effect
    ratio: float                 # fraction of cause-events so followed [0,1]
    median_lag: float            # typical delay, seconds
    consistency: float           # 1 - coefficient-of-variation of the lags [0,1]

    def describe(self) -> str:
        return (f"{self.cause!r} precedes {self.effect!r} by ~{describe_duration(self.median_lag)} "
                f"({self.ratio:.0%} of the time, consistency {self.consistency:.0%})")

    def to_dict(self) -> Dict[str, object]:
        return {"cause": self.cause, "effect": self.effect, "support": self.support,
                "ratio": round(self.ratio, 3), "median_lag": round(self.median_lag, 3),
                "consistency": round(self.consistency, 3)}


@dataclass
class Periodicity:
    """A detected repeating rhythm for one label."""

    label: str
    period: float                # seconds between recurrences
    confidence: float            # how regular the rhythm is [0,1]
    occurrences: int

    def describe(self) -> str:
        return f"{self.label!r} repeats roughly every {describe_duration(self.period)}"

    def to_dict(self) -> Dict[str, object]:
        return {"label": self.label, "period": round(self.period, 3),
                "confidence": round(self.confidence, 3), "occurrences": self.occurrences}


# --------------------------------------------------------------------------- #
# The reasoner
# --------------------------------------------------------------------------- #
class TemporalReasoner:
    """Ingests timestamped events and answers questions about their order in time."""

    def __init__(self, *, simultaneity_tol: float = 1.0, max_events: int = 20_000) -> None:
        self.tol = simultaneity_tol
        self.max_events = max_events
        self._events: List[TemporalEvent] = []

    # ---- ingest ---- #
    def observe(self, label: str, at: Optional[float] = None, *,
                end: Optional[float] = None) -> TemporalEvent:
        ev = TemporalEvent(label=label, start=at if at is not None else time.time(), end=end)
        self._events.append(ev)
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events:]
        return ev

    def ingest_memories(self, records: Sequence[object]) -> int:
        """Bridge: ingest memory records (anything with ``created_at`` + ``tags``/text).
        The first tag is used as the event label, else a truncated text snippet."""
        n = 0
        for rec in records:
            at = getattr(rec, "created_at", None)
            if at is None:
                continue
            tags = list(getattr(rec, "tags", ()) or ())
            label = tags[0] if tags else str(getattr(rec, "text", lambda: "event")())[:24]
            self.observe(label, at)
            n += 1
        return n

    # ---- access ---- #
    def labels(self) -> List[str]:
        return sorted({e.label for e in self._events})

    def occurrences(self, label: str) -> List[float]:
        return sorted(e.start for e in self._events if e.label == label)

    def relation(self, a: TemporalEvent, b: TemporalEvent) -> AllenRelation:
        return allen_relation(a.interval(), b.interval(), tol=self.tol)

    # ---- ordering ---- #
    def strictly_before(self, x_label: str, y_label: str) -> bool:
        """Every X ends before any Y begins — an unambiguous "X happened before Y"."""
        xs = [e.interval().end for e in self._events if e.label == x_label]
        ys = [e.interval().start for e in self._events if e.label == y_label]
        if not xs or not ys:
            return False
        return max(xs) < min(ys) - self.tol

    def order_fraction(self, x_label: str, y_label: str) -> float:
        """Fraction of (X, Y) pairs in which X starts before Y — a soft "X tends to
        precede Y" when events interleave."""
        xs = self.occurrences(x_label)
        ys = self.occurrences(y_label)
        if not xs or not ys:
            return 0.0
        ordered = sum(1 for x in xs for y in ys if x < y - self.tol)
        return ordered / (len(xs) * len(ys))

    # ---- precedence & lag ("X caused Y N later" — temporal shape only) ---- #
    def precedence(self, cause: str, effect: str, *,
                   max_lag: Optional[float] = None) -> Optional[Precedence]:
        causes = self.occurrences(cause)
        effects = self.occurrences(effect)
        if not causes or not effects:
            return None
        lags: List[float] = []
        for ct in causes:
            following = [et - ct for et in effects
                         if et > ct + self.tol and (max_lag is None or et - ct <= max_lag)]
            if following:
                lags.append(min(following))   # the nearest following effect
        if not lags:
            return None
        median_lag = statistics.median(lags)
        consistency = 1.0
        if len(lags) >= 2:
            mean = statistics.fmean(lags)
            if mean > 0:
                consistency = _clamp(1.0 - statistics.pstdev(lags) / mean)
        return Precedence(cause=cause, effect=effect, support=len(lags),
                          ratio=_clamp(len(lags) / len(causes)),
                          median_lag=median_lag, consistency=consistency)

    def causal_candidates(self, *, min_support: int = 2, min_ratio: float = 0.6,
                          max_lag: Optional[float] = None) -> List[Precedence]:
        """All (cause -> effect) label pairs whose temporal precedence is strong enough to
        be worth handing to the world model as a *candidate* causal link."""
        out: List[Precedence] = []
        labels = self.labels()
        for cause in labels:
            for effect in labels:
                if cause == effect:
                    continue
                p = self.precedence(cause, effect, max_lag=max_lag)
                if p is not None and p.support >= min_support and p.ratio >= min_ratio:
                    out.append(p)
        out.sort(key=lambda p: (p.ratio, p.consistency, p.support), reverse=True)
        return out

    # ---- periodicity ("repeats every week") ---- #
    def periodicity(self, label: str, *, min_occurrences: int = 3) -> Optional[Periodicity]:
        times = self.occurrences(label)
        if len(times) < min_occurrences:
            return None
        gaps = [b - a for a, b in zip(times, times[1:]) if b - a > 0]
        if len(gaps) < 2:
            return None
        mean = statistics.fmean(gaps)
        if mean <= 0:
            return None
        cv = statistics.pstdev(gaps) / mean
        consistency = _clamp(1.0 - cv)
        support = _clamp((len(gaps)) / 4.0)          # more cycles => more trust
        confidence = _clamp(consistency * (0.5 + 0.5 * support))
        return Periodicity(label=label, period=mean, confidence=confidence,
                           occurrences=len(times))

    def rhythms(self, *, min_confidence: float = 0.5,
                min_occurrences: int = 3) -> List[Periodicity]:
        """Every label with a confidently-detected repeating period."""
        out: List[Periodicity] = []
        for label in self.labels():
            p = self.periodicity(label, min_occurrences=min_occurrences)
            if p is not None and p.confidence >= min_confidence:
                out.append(p)
        out.sort(key=lambda p: p.confidence, reverse=True)
        return out
