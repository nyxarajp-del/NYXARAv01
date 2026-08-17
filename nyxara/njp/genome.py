"""NYXARA · njp/genome.py — the Reasoning Genome (🧬, NJP V.06).

Every other organ in :mod:`nyxara.njp` compresses what she knows about the *world*.
:mod:`nyxara.njp.concepts` finds the kind behind a set of instances and pays for itself in a
description-length ratio; :mod:`nyxara.njp.core` finds the schema behind a set of relations. This
one compresses what she knows about her own **reasoning**.

**The material was already there and was being thrown away.** Every answer the Core derives comes
back as a :class:`~nyxara.njp.core.Derivation` carrying ``kind``, the chain in ``support``, the
readable ``steps`` and a ``confidence``. :class:`~nyxara.njp.brain.NJPBrain` attaches it to the
thought — and nothing anywhere stored it, mined it, or looked at two of them together. A trace is
the record of *how* an answer was reached, and until now that record survived exactly one turn.

**What a shape is.** Strip the entities out of a chain and what remains is its form::

    aag   —causes→ garmi , garmi —causes→ pasina   ⇒  ('causes', 'causes')
    garmi —causes→ pasina, pasina —causes→ pyaas   ⇒  ('causes', 'causes')

Three different questions, three different subjects, one shape. That is the object worth counting:
a shape that keeps recurring is a piece of reasoning she keeps re-deriving from scratch, and it is
the unit :mod:`nyxara.growth.noesis` can turn into a named primitive.

**What "worth compressing" means here, in numbers.** Not an opinion, and not a threshold anyone
liked the look of. A shape seen ``n`` times, each derivation costing ``k`` steps, costs ``n·k``
steps to keep re-deriving. Named once and applied ``n`` times it costs ``k + n``. So the saving is
``n·k − (k + n)``, positive exactly when a shape both recurs and is not trivially short, which is
the same MDL argument :mod:`nyxara.njp.concepts` already makes about kinds. A shape that recurs
twice and takes two steps saves nothing and is reported as saving nothing.

**Success is tracked separately from recurrence, and neither alone is enough.** A shape that
recurs constantly and is usually *wrong* is a habit worth breaking, not a primitive worth naming,
and promoting it would teach her to make the same mistake faster. :meth:`candidates` requires both,
and :meth:`liabilities` reports the other case so it is visible rather than merely absent.

**Simulated derivations are recorded and never promoted**, for the reason
:mod:`nyxara.njp.truth` gives at length: a reasoning shape learned from her own imagination, then
applied to real questions, is the same closed loop one rung further up. They count toward
``seen`` so the record is honest about what happened, and never toward ``candidates``.

Pure standard library. Every public entry point is fail-soft: on error it degrades to a null
result rather than breaking a turn.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Trace", "Shape", "ReasoningGenome"]

#: A shape has to recur at least this often before it is a pattern rather than a coincidence.
_MIN_SEEN = 3

#: And it has to have been right at least this often. A shape that recurs and fails is a habit.
_MIN_SUCCESS = 0.6

#: Chains shorter than this are not worth naming: a one-hop "derivation" is a lookup, and naming
#: it buys nothing that the fact store does not already give.
_MIN_DEPTH = 2


def shape_of(derivation: Any) -> Tuple[str, ...]:
    """The form of a derivation with its entities removed — the part that can be reused.

    Reads ``support``, which the Core fills with the actual ``(subject, predicate, object)`` edges
    it walked. The predicates in order *are* the shape: two different chains through two different
    subjects that used the same relations in the same order are the same piece of reasoning.
    """
    try:
        return tuple(str(edge[1]) for edge in (getattr(derivation, "support", None) or ())
                     if isinstance(edge, (tuple, list)) and len(edge) >= 2)
    except Exception:  # noqa: BLE001
        return ()


@dataclass
class Trace:
    """One reasoning event: what was asked, how it was answered, and whether that worked."""

    question: str = ""
    kind: str = ""                          # direct | composed | schema | simulated
    shape: Tuple[str, ...] = ()
    depth: int = 0
    confidence: float = 0.0
    answer: str = ""
    simulated: bool = False
    #: None until reality says. Kept as None rather than defaulted to a number, because an
    #: ungraded derivation and a failed one are different things and averaging them lies.
    correct: Optional[bool] = None
    t: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"question": self.question[:200], "kind": self.kind, "shape": list(self.shape),
                "depth": self.depth, "confidence": round(self.confidence, 4),
                "answer": self.answer[:120], "simulated": self.simulated,
                "correct": self.correct, "t": round(self.t, 3)}


@dataclass
class Shape:
    """One reasoning form, and its record across every time it was used."""

    shape: Tuple[str, ...] = ()
    seen: int = 0
    graded: int = 0
    correct: int = 0
    simulated: int = 0
    total_depth: int = 0

    @property
    def depth(self) -> float:
        """Mean chain length — the ``k`` in the saving below."""
        return (self.total_depth / self.seen) if self.seen else 0.0

    @property
    def success(self) -> Optional[float]:
        """Share of graded uses that were right, or None while nothing has been graded."""
        return (self.correct / self.graded) if self.graded else None

    @property
    def real(self) -> int:
        """Uses that came from real reasoning rather than simulation."""
        return max(0, self.seen - self.simulated)

    @property
    def saving(self) -> float:
        """Steps saved by naming this shape once instead of re-deriving it every time.

        ``n·k − (k + n)``: re-deriving costs a chain of ``k`` steps on each of ``n`` uses; naming
        it costs ``k`` once to define plus one application per use. Trivially short or barely
        repeated shapes come out at or below zero and say so.
        """
        n, k = float(self.real), self.depth
        if n <= 0 or k <= 0:
            return 0.0
        return (n * k) - (k + n)

    def to_dict(self) -> Dict[str, Any]:
        return {"shape": list(self.shape), "seen": self.seen, "real": self.real,
                "graded": self.graded, "correct": self.correct, "simulated": self.simulated,
                "depth": round(self.depth, 3),
                "success": (round(self.success, 4) if self.success is not None else None),
                "saving": round(self.saving, 3)}


class ReasoningGenome:
    """The record of how she reasons, and which of those ways is worth naming.

    Holds a bounded window of traces and an unbounded tally per shape — the traces are evidence
    for a person reading the log, the tallies are what the promotion decision is actually made
    from, and only the second needs to survive a long session.
    """

    def __init__(self, *, capacity: int = 512, min_seen: int = _MIN_SEEN,
                 min_success: float = _MIN_SUCCESS, min_depth: int = _MIN_DEPTH) -> None:
        self.capacity = max(16, int(capacity))
        self.min_seen = max(2, int(min_seen))
        self.min_success = float(min_success)
        self.min_depth = max(2, int(min_depth))
        self.traces: List[Trace] = []
        self.shapes: Dict[Tuple[str, ...], Shape] = {}
        self.recorded = 0
        self.graded = 0

    # ---- writing ------------------------------------------------------------ #
    def record(self, derivation: Any, *, question: str = "", simulated: bool = False) -> Optional[Trace]:
        """Keep one derivation. Returns the trace, or None if there was no reasoning to keep.

        A direct answer is deliberately *not* kept as a shape: it is a lookup, its "chain" is one
        edge, and counting it would swamp the tallies with the one form that needs no compressing.
        """
        try:
            shape = shape_of(derivation)
            if len(shape) < self.min_depth:
                return None
            trace = Trace(
                question=str(question or "")[:200],
                kind=str(getattr(derivation, "kind", "") or ""),
                shape=shape, depth=len(shape),
                confidence=float(getattr(derivation, "confidence", 0.0) or 0.0),
                answer=str(getattr(derivation, "answer", "") or "")[:120],
                simulated=bool(simulated),
            )
            self.traces.append(trace)
            if len(self.traces) > self.capacity:
                del self.traces[: len(self.traces) - self.capacity]

            got = self.shapes.get(shape)
            if got is None:
                got = Shape(shape=shape)
                self.shapes[shape] = got
            got.seen += 1
            got.total_depth += trace.depth
            if simulated:
                got.simulated += 1
            self.recorded += 1
            return trace
        except Exception:  # noqa: BLE001
            return None

    def grade(self, trace: Any, *, correct: bool) -> None:
        """Tell the genome how a recorded derivation actually turned out."""
        try:
            if trace is None or getattr(trace, "correct", None) is not None:
                return
            trace.correct = bool(correct)
            got = self.shapes.get(tuple(getattr(trace, "shape", ()) or ()))
            if got is None:
                return
            got.graded += 1
            if correct:
                got.correct += 1
            self.graded += 1
        except Exception:  # noqa: BLE001
            pass

    # ---- reading ------------------------------------------------------------ #
    def candidates(self) -> List[Shape]:
        """Shapes worth compressing into a primitive, best saving first.

        Three conditions, and each rules out a different way of being wrong about this: enough
        real (non-simulated) uses to be a pattern, a success rate that says following it is a good
        idea, and a positive MDL saving so naming it actually costs less than re-deriving it.
        """
        out = [s for s in self.shapes.values()
               if s.real >= self.min_seen
               and s.saving > 0.0
               and (s.success is None or s.success >= self.min_success)]
        return sorted(out, key=lambda s: s.saving, reverse=True)

    def liabilities(self) -> List[Shape]:
        """Shapes she keeps using that keep being wrong.

        The mirror of :meth:`candidates`, and reported rather than merely excluded: a recurring
        reasoning form with a poor record is a habit worth breaking, and the fact that it is
        *frequent* is the reason it matters rather than a reason to ignore it.
        """
        out = [s for s in self.shapes.values()
               if s.graded >= self.min_seen and (s.success or 0.0) < self.min_success]
        return sorted(out, key=lambda s: (s.success or 0.0))

    def stats(self) -> Dict[str, Any]:
        best = self.candidates()
        worst = self.liabilities()
        return {
            "recorded": self.recorded, "graded": self.graded,
            "traces": len(self.traces), "shapes": len(self.shapes),
            "candidates": len(best), "liabilities": len(worst),
            "total_saving": round(sum(s.saving for s in best), 3),
            "best": [s.to_dict() for s in best[:3]],
            "worst": [s.to_dict() for s in worst[:2]],
        }

    # ---- persistence -------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {"shapes": [s.to_dict() for s in self.shapes.values()],
                "recorded": self.recorded, "graded": self.graded}

    def load_dict(self, data: Dict[str, Any]) -> None:
        """Restore the tallies. Traces are a window and are deliberately not persisted."""
        try:
            self.shapes = {}
            for row in (data or {}).get("shapes") or []:
                shape = tuple(str(p) for p in (row.get("shape") or ()))
                if not shape:
                    continue
                got = Shape(shape=shape)
                got.seen = int(row.get("seen", 0) or 0)
                got.graded = int(row.get("graded", 0) or 0)
                got.correct = int(row.get("correct", 0) or 0)
                got.simulated = int(row.get("simulated", 0) or 0)
                got.total_depth = int(round(float(row.get("depth", 0) or 0) * got.seen))
                self.shapes[shape] = got
            self.recorded = int((data or {}).get("recorded", 0) or 0)
            self.graded = int((data or {}).get("graded", 0) or 0)
        except Exception:  # noqa: BLE001
            pass
