"""NYXARA · njp/predict.py — prediction error as the central learning signal (📉, NJP V.02).

Every organ in NJP already learns from *something*. The fabric potentiates what fired, the
manifold learns its transitions, the world model counts what followed what. What was missing is
the loop that ties them together and, crucially, tells them apart:

    predict → observe → compare → **diagnose** → learn → re-predict

The diagnosis is the part that matters. "She was wrong" is almost useless as a training signal —
it is the same string whether she misheard the question, remembered the wrong thing, reasoned
badly from correct facts, or knew the answer and phrased it terribly. Four different repairs, one
undifferentiated error. So every miss here is classified by **kind**:

``perception`` · ``grounding`` · ``memory`` · ``world_model`` · ``reasoning`` · ``planning`` ·
``language``

and each kind routes its update to the organ actually responsible. A grounding error re-parses; a
memory error adjusts retrieval; a world-model error feeds a transition; a reasoning error opens a
debate. That is what makes this stronger than a scalar "correct/incorrect" fed back into
everything at once, which mostly teaches every organ to be vaguely less confident.

**Attribution is evidence-based, not a guess.** The classifier does not ask a model what went
wrong; it reads what each organ actually produced on the turn. If the grounder extracted nothing
from a sentence that plainly stated a fact, that is a grounding error and the evidence for it is
the empty extraction. If it extracted correctly and the answer still contradicted the record,
grounding is exonerated and something downstream owns it. Where the evidence does not single out
one organ the error is recorded as ``unattributed`` rather than assigned to the most likely
suspect — a wrong attribution actively trains the wrong organ, which is worse than none.

**Calibration.** Alongside accuracy this tracks whether her confidence *means* anything: a mind
that is 90% confident should be right about nine times in ten. :meth:`calibration` reports the gap,
which is the number that says whether a stated confidence can be trusted at all.

Fail-soft: a failed prediction cycle records nothing and never breaks a turn.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["ErrorKind", "Prediction", "Outcome", "Diagnosis", "Calibration", "PredictionEngine"]


class ErrorKind:
    """Where a mistake actually came from. The whole point is that these are different repairs."""

    PERCEPTION = "perception"        # the turn was not read correctly at all
    GROUNDING = "grounding"          # read, but not turned into the right structure
    MEMORY = "memory"                # the structure was right; the wrong thing was retrieved
    WORLD_MODEL = "world_model"      # what she expected to follow did not
    REASONING = "reasoning"          # correct facts, wrong conclusion
    PLANNING = "planning"            # right conclusion, wrong course of action
    LANGUAGE = "language"            # right answer, said wrongly
    UNATTRIBUTED = "unattributed"    # genuinely cannot tell — never guess

    ALL = (PERCEPTION, GROUNDING, MEMORY, WORLD_MODEL,
           REASONING, PLANNING, LANGUAGE, UNATTRIBUTED)


@dataclass
class Prediction:
    """One registered expectation, waiting to meet reality."""

    key: str = ""
    expected: Any = None
    confidence: float = 0.0
    organ: str = ""                  # who predicted it
    at: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "expected": _short(self.expected), "organ": self.organ,
                "confidence": round(self.confidence, 4)}


@dataclass
class Outcome:
    """What actually happened, scored against what was expected."""

    key: str = ""
    expected: Any = None
    actual: Any = None
    error: float = 0.0               # 0.0 exactly right … 1.0 wholly wrong
    confidence: float = 0.0
    organ: str = ""
    diagnosis: Optional["Diagnosis"] = None
    # The evidence the diagnosis was drawn from, carried through to the repair. A repair that
    # cannot see the turn it is fixing can only do something generic, and a generic repair is the
    # thing the whole error taxonomy exists to avoid.
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def correct(self) -> bool:
        return self.error <= _CORRECT_FLOOR

    @property
    def surprise(self) -> float:
        """Error weighted by how sure she was. Being confidently wrong is the costly case."""
        return self.error * (0.5 + 0.5 * self.confidence)

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "expected": _short(self.expected), "actual": _short(self.actual),
                "error": round(self.error, 4), "correct": self.correct,
                "surprise": round(self.surprise, 4), "organ": self.organ,
                "diagnosis": self.diagnosis.to_dict() if self.diagnosis else None}


@dataclass
class Diagnosis:
    """Which organ owns this miss, and the evidence that says so."""

    kind: str = ErrorKind.UNATTRIBUTED
    evidence: str = ""
    confidence: float = 0.0
    repaired: bool = False           # did the routed update actually run?

    @property
    def attributed(self) -> bool:
        return self.kind != ErrorKind.UNATTRIBUTED

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "evidence": self.evidence[:200],
                "confidence": round(self.confidence, 4),
                "attributed": self.attributed, "repaired": self.repaired}


@dataclass
class Calibration:
    """Does a stated confidence mean anything? Measured, in buckets."""

    buckets: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # band -> (right, total)
    gap: float = 0.0                 # mean |stated confidence − observed accuracy|
    samples: int = 0

    @property
    def calibrated(self) -> bool:
        """Within ten points, over enough samples to mean it."""
        return self.samples >= _MIN_CALIBRATION_SAMPLES and self.gap <= 0.10

    def to_dict(self) -> Dict[str, Any]:
        return {"gap": round(self.gap, 4), "samples": self.samples,
                "calibrated": self.calibrated,
                "buckets": {b: {"right": r, "total": t,
                                "accuracy": round(r / t, 4) if t else None}
                            for b, (r, t) in sorted(self.buckets.items())}}


_CORRECT_FLOOR = 0.25
_MIN_CALIBRATION_SAMPLES = 20


def _short(value: Any, limit: int = 80) -> str:
    try:
        return str(value)[:limit]
    except Exception:  # noqa: BLE001
        return "<unrenderable>"


def _similarity(a: Any, b: Any) -> float:
    """How alike two outcomes are, 0…1. Token overlap for text, closeness for numbers."""
    try:
        if a is None or b is None:
            return 0.0
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if isinstance(a, bool) or isinstance(b, bool):
                return 1.0 if bool(a) == bool(b) else 0.0
            scale = max(abs(float(a)), abs(float(b)), 1e-9)
            return max(0.0, 1.0 - abs(float(a) - float(b)) / scale)
        if isinstance(a, (list, tuple, set)) or isinstance(b, (list, tuple, set)):
            sa, sb = set(a or ()), set(b or ())
            return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0
        sa = {t for t in str(a).lower().split() if t}
        sb = {t for t in str(b).lower().split() if t}
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


class PredictionEngine:
    """Register expectations, score them against reality, and route the repair."""

    def __init__(self, brain: Any = None, *, capacity: int = 4096,
                 window: int = 128) -> None:
        self.brain = brain
        self.capacity = max(16, int(capacity))
        self.window = max(8, int(window))

        self._open: Dict[str, Prediction] = {}
        self.outcomes: List[Outcome] = []
        self.by_kind: Dict[str, int] = {k: 0 for k in ErrorKind.ALL}
        self.repaired: Dict[str, int] = {k: 0 for k in ErrorKind.ALL}
        self.predictions = 0
        self.scored = 0
        self.correct = 0
        # Repairs, by error kind. Injected so the engine has no hard dependency on any organ and
        # can be tested without a brain at all.
        self.repairs: Dict[str, Callable[[Outcome], bool]] = {}

    # ---- predict ---------------------------------------------------------- #
    def predict(self, key: str, expected: Any, *, confidence: float = 0.5,
                organ: str = "", **context: Any) -> Prediction:
        """Register an expectation under ``key``. Scored when :meth:`observe` sees that key."""
        got = Prediction(key=str(key), expected=expected,
                         confidence=max(0.0, min(1.0, float(confidence))),
                         organ=str(organ or ""), context=dict(context))
        try:
            self._open[got.key] = got
            self.predictions += 1
            if len(self._open) > self.capacity:
                for stale in list(self._open)[: len(self._open) - self.capacity]:
                    self._open.pop(stale, None)
        except Exception:  # noqa: BLE001
            pass
        return got

    # ---- observe and compare ----------------------------------------------- #
    def observe(self, key: str, actual: Any, *, evidence: Optional[Dict[str, Any]] = None,
                learn: bool = True) -> Optional[Outcome]:
        """Score reality against what was expected. ``None`` when nothing predicted this key.

        An unpredicted observation is not an error — she simply had no expectation, and inventing
        one retroactively so there is something to score would be measuring nothing.
        """
        try:
            pending = self._open.pop(str(key), None)
            if pending is None:
                return None

            evidence = dict(evidence) if evidence else dict(pending.context)
            out = Outcome(key=pending.key, expected=pending.expected, actual=actual,
                          confidence=pending.confidence, organ=pending.organ,
                          context=evidence)
            out.error = max(0.0, min(1.0, 1.0 - _similarity(pending.expected, actual)))
            self.scored += 1
            if out.correct:
                self.correct += 1
            else:
                out.diagnosis = self.diagnose(out, evidence)
                self.by_kind[out.diagnosis.kind] = self.by_kind.get(out.diagnosis.kind, 0) + 1
                if learn:
                    self._repair(out)

            self.outcomes.append(out)
            if len(self.outcomes) > self.capacity:
                del self.outcomes[: len(self.outcomes) - self.capacity]
            return out
        except Exception:  # noqa: BLE001
            return None

    # ---- diagnose ----------------------------------------------------------- #
    def diagnose(self, outcome: Outcome, evidence: Dict[str, Any]) -> Diagnosis:
        """Which organ owns this miss? Read from what each one produced, never guessed.

        The checks run outermost-first — perception before grounding before memory — because an
        organ can only be blamed once the ones feeding it are cleared. Blaming the reasoner for a
        conclusion drawn from facts that were never extracted trains the wrong thing.

        Where nothing singles out one organ this returns ``UNATTRIBUTED``. That is deliberate and
        it is not a gap: a wrong attribution actively trains the wrong organ, which is strictly
        worse than declining to attribute at all.
        """
        try:
            # 1. Did anything reach her? No concepts from a non-empty turn is a perception failure.
            if evidence.get("stimulus") and not evidence.get("concepts"):
                return Diagnosis(kind=ErrorKind.PERCEPTION, confidence=0.8,
                                 evidence="the turn produced no concepts at all")

            # 2. Did the words become structure? A factual sentence that grounded to nothing is a
            # grounding failure regardless of what happened downstream of it.
            if evidence.get("expected_triples") and not evidence.get("triples"):
                return Diagnosis(kind=ErrorKind.GROUNDING, confidence=0.8,
                                 evidence="a statement of fact extracted no relation")

            # 3. Structure was right, but the wrong thing came back. Only claimable when the
            # record actually held the answer — otherwise this is ignorance, not a memory fault.
            if evidence.get("in_memory") and not evidence.get("recalled"):
                return Diagnosis(kind=ErrorKind.MEMORY, confidence=0.75,
                                 evidence="the answer was in the record and was not retrieved")

            # 4. A miss on what-follows-what belongs to the world model by construction.
            if str(outcome.organ) in ("world", "world_model", "manifold"):
                return Diagnosis(kind=ErrorKind.WORLD_MODEL, confidence=0.85,
                                 evidence=f"{outcome.organ} predicted the next state and missed")

            # 5. Facts present and correct, conclusion still wrong.
            if evidence.get("triples") and evidence.get("facts_correct", True):
                return Diagnosis(kind=ErrorKind.REASONING, confidence=0.6,
                                 evidence="the facts were right and the conclusion was not")

            if str(outcome.organ) in ("planner", "planning", "goals"):
                return Diagnosis(kind=ErrorKind.PLANNING, confidence=0.7,
                                 evidence="the plan did not achieve what it predicted")

            # 6. Right content, wrong words. Only when the content genuinely matched — otherwise
            # "she said it badly" is a comfortable way to avoid diagnosing a real error.
            if evidence.get("content_correct") and not outcome.correct:
                return Diagnosis(kind=ErrorKind.LANGUAGE, confidence=0.65,
                                 evidence="the content was right and the phrasing lost it")

            return Diagnosis(kind=ErrorKind.UNATTRIBUTED, confidence=0.0,
                             evidence="no organ is singled out by the evidence on this turn")
        except Exception:  # noqa: BLE001
            return Diagnosis()

    # ---- learn -------------------------------------------------------------- #
    def register_repair(self, kind: str, fn: Callable[[Outcome], bool]) -> None:
        """Attach the update that fixes one kind of error."""
        self.repairs[str(kind)] = fn

    def _repair(self, outcome: Outcome) -> None:
        """Route the error to the organ that owns it. An unattributed error is not routed."""
        try:
            diagnosis = outcome.diagnosis
            if diagnosis is None or not diagnosis.attributed:
                return
            fn = self.repairs.get(diagnosis.kind)
            if fn is None:
                return
            diagnosis.repaired = bool(fn(outcome))
            if diagnosis.repaired:
                self.repaired[diagnosis.kind] = self.repaired.get(diagnosis.kind, 0) + 1
        except Exception:  # noqa: BLE001 — a failed repair is recorded as unrepaired, not fatal
            pass

    # ---- reading ------------------------------------------------------------ #
    @property
    def accuracy(self) -> Optional[float]:
        return (self.correct / self.scored) if self.scored else None

    def recent_error(self, *, window: Optional[int] = None) -> Optional[float]:
        """Mean error over the last ``window`` scored predictions — the trend that matters."""
        rows = self.outcomes[-(window or self.window):]
        return (sum(o.error for o in rows) / len(rows)) if rows else None

    def improving(self) -> Optional[bool]:
        """Is she getting better? Compares the recent half against the one before it."""
        try:
            rows = self.outcomes[-(self.window * 2):]
            if len(rows) < 8:
                return None
            half = len(rows) // 2
            older = sum(o.error for o in rows[:half]) / half
            newer = sum(o.error for o in rows[half:]) / (len(rows) - half)
            return newer < older
        except Exception:  # noqa: BLE001
            return None

    def calibration(self, *, bands: int = 5) -> Calibration:
        """Does a stated confidence predict being right? The number behind every other number."""
        out = Calibration()
        try:
            width = 1.0 / max(1, int(bands))
            for outcome in self.outcomes:
                index = min(bands - 1, int(outcome.confidence / width))
                label = f"{index * width:.1f}-{(index + 1) * width:.1f}"
                right, total = out.buckets.get(label, (0, 0))
                out.buckets[label] = (right + (1 if outcome.correct else 0), total + 1)
            gaps: List[float] = []
            for label, (right, total) in out.buckets.items():
                if not total:
                    continue
                midpoint = (float(label.split("-")[0]) + float(label.split("-")[1])) / 2.0
                gaps.append(abs(midpoint - right / total) * total)
            out.samples = len(self.outcomes)
            out.gap = (sum(gaps) / out.samples) if out.samples else 0.0
            return out
        except Exception:  # noqa: BLE001
            return out

    def worst_kind(self) -> Optional[str]:
        """Which organ is costing her most. What meta-learning should act on first."""
        attributed = {k: n for k, n in self.by_kind.items()
                      if n > 0 and k != ErrorKind.UNATTRIBUTED}
        return max(attributed, key=attributed.get) if attributed else None

    def stats(self) -> Dict[str, Any]:
        return {
            "predictions": self.predictions, "scored": self.scored,
            "open": len(self._open),
            "accuracy": round(self.accuracy, 4) if self.accuracy is not None else None,
            "recent_error": (round(self.recent_error(), 4)
                             if self.recent_error() is not None else None),
            "improving": self.improving(),
            "by_kind": {k: n for k, n in self.by_kind.items() if n},
            "repaired": {k: n for k, n in self.repaired.items() if n},
            "worst_kind": self.worst_kind(),
            "calibration": self.calibration().to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"by_kind": dict(self.by_kind), "repaired": dict(self.repaired),
                "counters": {"predictions": self.predictions, "scored": self.scored,
                             "correct": self.correct},
                "outcomes": [o.to_dict() for o in self.outcomes[-256:]]}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.by_kind.update({str(k): int(v) for k, v in (d.get("by_kind") or {}).items()})
            self.repaired.update({str(k): int(v) for k, v in (d.get("repaired") or {}).items()})
            counters = d.get("counters") or {}
            self.predictions = int(counters.get("predictions", 0))
            self.scored = int(counters.get("scored", 0))
            self.correct = int(counters.get("correct", 0))
        except Exception:  # noqa: BLE001
            pass
