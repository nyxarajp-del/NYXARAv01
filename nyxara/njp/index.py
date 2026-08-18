"""NYXARA · njp/index.py — one number she is not allowed to compute about herself (📐).

Every organ in NJP reports its own counters, and between them they say almost everything except
the one thing worth knowing: **is she better than she was.** `stats()` grows monotonically by
construction — more facts, more cells, more synapses, more schemas — so reading it as progress is
reading the size of the store as the quality of the mind. `ledger.Generation` tracks growth over
time and has the same problem for the same reason.

This module is the other number. Eight terms, five that should rise and three that should fall::

            G · T · C · N · R
    I_t = ─────────────────────
            1 + E + H + Û

    G  generalization     a rule induced from some members, scored on the rest
    T  transfer           structure projected into a domain she has never seen
    C  causal prediction  the accuracy of predictions she actually committed to
    N  novel-task solving *no independent source yet — see below*
    R  reusable programs  reasoning compressed into primitives that survived a held-out win
    E  prediction error   how wrong her recent predictions were
    H  hallucination      claims she asserted that turned out false
    Û  unsupported conf.  how much her stated confidence outruns her hit rate

**It is an engineering objective, not a law.** Nothing about intelligence licenses this
arrangement of these eight numbers; what licenses it is that each term is measured somewhere else
in this repo already, and that a change which moves the product is a change that moved something
real. Treat it as a dashboard, and read the vector.

Four rules hold it up, and each exists because of a way this kind of index normally goes wrong.

**No term she computes about herself may enter it.** Otherwise the cheapest path to a better score
is to change the scoring, and a self-improvement loop optimising this would find that path before
it found a better idea. `selfmodel` knows what she is good at and is deliberately not a source
here; every term comes from a benchmark with held-out data or from a counter incremented by an
outcome rather than by an intention.

**Absent is not zero.** A missing term is `None` and leaves the product, and the scalar says which
terms it was computed over. A product of five numbers in [0,1] would otherwise be zeroed by one
unmeasured term, and — worse — an unmeasured term reads as a measured failure. `N` is absent today
and reported as absent rather than quietly filled with a stage that already serves `G` or `T`;
inventing a source to complete a formula is how a dashboard starts lying.

**The numerator is a geometric mean, not a raw product.** The sketch this came from multiplies,
and a raw product changes magnitude when a term appears or disappears, so a four-term run and a
five-term run are not comparable numbers — which defeats the entire purpose of a time series. The
geometric mean is the same product read at a fixed scale, so `I_t` stays in [0,1] and a run that
gained a term is still comparable to the run before it.

**Honest uncertainty is nowhere in the penalty.** The denominator charges her for being *wrong*
(`E`), for *asserting* what was wrong (`H`), and for being *more sure than she earned* (`Û`). It
does not charge her for not knowing. A system fined for saying "I don't know" has exactly one
gradient available to it — say it less — and that is the opposite of what `truth.py` being
fail-closed and `UNKNOWN` being a real return value are for.

Pure standard library. Every entry point is fail-soft: a term whose source raises is absent, not
zero, and never breaks the measurement of the others.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["Term", "IntelligenceVector", "IntelligenceIndex", "measure"]

#: Terms that should rise, in the order they are reported.
NUMERATOR = ("G", "T", "C", "N", "R")
#: Terms that should fall.
DENOMINATOR = ("E", "H", "U")

_NAMES = {
    "G": "generalization", "T": "transfer", "C": "causal_prediction",
    "N": "novel_task_solving", "R": "reusable_programs",
    "E": "prediction_error", "H": "hallucination", "U": "unsupported_confidence",
}


@dataclass
class Term:
    """One measured quantity, with where it came from and why it is missing when it is."""

    key: str = ""
    value: Optional[float] = None
    source: str = ""
    detail: str = ""

    @property
    def name(self) -> str:
        return _NAMES.get(self.key, self.key)

    @property
    def measured(self) -> bool:
        return self.value is not None

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "name": self.name,
                "value": None if self.value is None else round(self.value, 4),
                "source": self.source, "detail": self.detail}


@dataclass
class IntelligenceVector:
    """The eight terms, and the one line that summarises them for a dashboard."""

    terms: List[Term] = field(default_factory=list)
    at: float = field(default_factory=time.time)
    ms: float = 0.0

    def get(self, key: str) -> Optional[Term]:
        for term in self.terms:
            if term.key == key:
                return term
        return None

    def value(self, key: str) -> Optional[float]:
        term = self.get(key)
        return term.value if term is not None else None

    @property
    def absent(self) -> List[str]:
        return [t.key for t in self.terms if not t.measured]

    @property
    def scalar(self) -> Optional[float]:
        """``I_t``, over the terms that were actually measured. ``None`` if none were.

        The numerator is the geometric mean rather than the product, so that a run missing a term
        is still on the same scale as one that had it. The denominator is ``1 + penalties`` rather
        than the penalties alone: it is defined when she has made no mistakes, and it puts a
        perfect run at exactly 1.0 instead of at infinity.
        """
        gains = [t.value for t in self.terms if t.key in NUMERATOR and t.value is not None]
        if not gains:
            return None
        # A single zero is a real result and must survive: geometric mean of anything with a zero
        # is zero, which is the honest reading of "she cannot generalise at all".
        if any(g <= 0.0 for g in gains):
            numerator = 0.0
        else:
            numerator = math.exp(sum(math.log(g) for g in gains) / len(gains))
        penalties = [t.value for t in self.terms if t.key in DENOMINATOR and t.value is not None]
        return numerator / (1.0 + sum(penalties))

    def regressions(self, previous: "IntelligenceVector") -> Dict[str, Any]:
        """Terms that got worse against an earlier reading. Direction-aware.

        The standing rule this exists to enforce: a change is reverted unless **G or T** moved,
        and a rise in the scalar with both of those flat has taught her to score better rather
        than to think better.
        """
        out: Dict[str, Any] = {}
        for term in self.terms:
            was = previous.value(term.key)
            if was is None or term.value is None:
                continue
            worse = term.value < was if term.key in NUMERATOR else term.value > was
            if worse:
                out[term.key] = {"was": round(was, 4), "now": round(term.value, 4)}
        return out

    def to_dict(self) -> Dict[str, Any]:
        """The vector leads. The scalar is one line at the end, which is its whole standing."""
        return {
            "terms": [t.to_dict() for t in self.terms],
            "absent": self.absent,
            "measured_over": [t.key for t in self.terms if t.measured],
            "scalar": None if self.scalar is None else round(self.scalar, 4),
            "at": round(self.at, 3),
            "ms": round(self.ms, 1),
        }

    def describe(self) -> str:
        lines = [f"{'term':<22} {'key':<4} value    source"]
        for term in self.terms:
            shown = "  —  " if term.value is None else f"{term.value:.3f}"
            lines.append(f"{term.name:<22} {term.key:<4} {shown}    {term.source}")
        scalar = self.scalar
        lines.append("")
        lines.append(f"I_t = {'—' if scalar is None else f'{scalar:.4f}'}"
                     + (f"   (absent: {', '.join(self.absent)})" if self.absent else ""))
        return "\n".join(lines)


def _rate(numerator: Any, denominator: Any) -> Optional[float]:
    try:
        den = float(denominator)
        return max(0.0, min(1.0, float(numerator) / den)) if den > 0 else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


class IntelligenceIndex:
    """Reads the eight terms off the benchmarks and organs that already measure them."""

    def __init__(self, *, seed: int = 0, width: int = 6) -> None:
        self.seed = int(seed)
        self.width = int(width)

    # ---- the terms that need a benchmark run ------------------------------ #

    def _generalization(self, prepare: Any) -> Term:
        term = Term(key="G", source="eval.intelligence:generalization")
        try:
            from nyxara.eval.intelligence import run_intelligence_benchmark
            report = run_intelligence_benchmark(seed=self.seed, width=self.width,
                                                only=["generalization"], prepare=prepare)
            stage = report.stages[0] if report.stages else None
            term.value = getattr(stage, "score", None)
            term.detail = f"{getattr(stage, 'correct', 0)}/{getattr(stage, 'total', 0)} held out"
        except Exception as exc:  # noqa: BLE001
            term.detail = f"unavailable: {exc}"
        return term

    def _transfer(self) -> Term:
        """Structure projected into a domain she has never seen, against its own control."""
        term = Term(key="T", source="eval.generalization:run_structure_transfer")
        try:
            from nyxara.eval.generalization import run_structure_transfer
            report = run_structure_transfer()
            if not report.total:
                term.detail = "no transfer tasks"
                return term
            # Credited against the base-only baseline, not raw. That baseline is what stops a
            # lucky match from reading as transfer, and ignoring it would waste the one control
            # this whole benchmark already carries.
            term.value = max(0.0, (report.solved - report.baseline_solved) / report.total)
            term.detail = (f"{report.solved}/{report.total} solved, "
                           f"{report.baseline_solved} by the base-only control")
        except Exception as exc:  # noqa: BLE001
            term.detail = f"unavailable: {exc}"
        return term

    def _novel(self) -> Term:
        """Absent, and deliberately left absent.

        The nearest existing measurements are `G` (a rule scored on held-out members of a kind it
        was induced from) and `T` (structure projected across domains). Neither is a novel *task*
        — a problem whose shape she has not been shown — and pointing this term at one of them
        would make the vector report two readings of one thing as though they were two things.
        It needs a benchmark stage of its own that does not exist yet.
        """
        return Term(key="N", source="—",
                    detail="no independent source yet; excluded rather than approximated")

    # ---- the terms an organ already counts -------------------------------- #

    def _causal(self, brain: Any) -> Term:
        term = Term(key="C", source="njp.predict:accuracy")
        try:
            stats = brain.predictor.stats()
            term.value = stats.get("accuracy")
            term.detail = f"{stats.get('scored', 0)} predictions scored"
        except Exception as exc:  # noqa: BLE001
            term.detail = f"unavailable: {exc}"
        return term

    def _programs(self, brain: Any) -> Term:
        """Reasoning compressed into primitives that survived a held-out description-length win."""
        term = Term(key="R", source="growth.noesis:compression_gain")
        try:
            engine = getattr(brain, "noesis", None)
            if engine is None:
                term.detail = "no noesis engine attached"
                return term
            report = engine.report()
            term.value = max(0.0, min(1.0, float(report.get("compression_gain", 0.0))))
            term.detail = f"{len(report.get('abstractions') or [])} abstractions adopted"
        except Exception as exc:  # noqa: BLE001
            term.detail = f"unavailable: {exc}"
        return term

    def _error(self, brain: Any) -> Term:
        term = Term(key="E", source="njp.predict:recent_error")
        try:
            term.value = brain.predictor.stats().get("recent_error")
            term.detail = "mean error over the recent window"
        except Exception as exc:  # noqa: BLE001
            term.detail = f"unavailable: {exc}"
        return term

    def _hallucination(self, brain: Any) -> Term:
        """Claims she asserted that turned out false, over claims she committed to.

        `ledger.ErrorRecord` is the right source and the only one: it keeps *what it was asserted
        as* beside *what it turned out to be*, which is precisely a claim she stated and was wrong
        about. `truth.refused` is deliberately not used — a refusal is the gauntlet working, and
        counting it as a hallucination would penalise the machinery that prevents them.
        """
        term = Term(key="H", source="njp.ledger:errors_recorded / truth:judged")
        try:
            wrong = brain.ledger.stats().get("errors_recorded", 0)
            judged = brain.truth.stats().get("judged", 0)
            term.value = _rate(wrong, judged)
            term.detail = f"{wrong} wrong of {judged} judged"
            if term.value is None:
                term.detail = "nothing judged yet"
        except Exception as exc:  # noqa: BLE001
            term.detail = f"unavailable: {exc}"
        return term

    def _unsupported(self, brain: Any) -> Term:
        """How far her stated confidence outruns her hit rate. Overconfidence only.

        `Reliability.bias` is signed and positive means overconfident, so the negative half is
        clamped away rather than credited: being *under*-confident is a different problem and
        rewarding it here would push her toward hedging everything.
        """
        term = Term(key="U", source="njp.beliefs:reliability.bias")
        try:
            rel = brain.beliefs.reliability()
            if rel.bias is None or rel.samples < 5:
                term.detail = f"only {getattr(rel, 'samples', 0)} settled beliefs; needs 5"
                return term
            term.value = max(0.0, float(rel.bias))
            term.detail = f"bias {rel.bias:+.3f} over {rel.samples} settled beliefs"
        except Exception as exc:  # noqa: BLE001
            term.detail = f"unavailable: {exc}"
        return term

    # ---- the measurement -------------------------------------------------- #

    def measure(self, brain: Any = None, *, benchmarks: bool = True,
                prepare: Any = None) -> IntelligenceVector:
        """Read every term. ``benchmarks=False`` skips the two that need a harness run.

        ``prepare`` is handed to the intelligence benchmark, so `G` can be measured on a brain
        that has been fed a corpus rather than on a fresh one.
        """
        started = time.perf_counter()
        out = IntelligenceVector()
        if benchmarks:
            out.terms.append(self._generalization(prepare))
            out.terms.append(self._transfer())
        else:
            out.terms.append(Term(key="G", source="—", detail="benchmarks skipped"))
            out.terms.append(Term(key="T", source="—", detail="benchmarks skipped"))
        if brain is not None:
            out.terms.append(self._causal(brain))
        else:
            out.terms.append(Term(key="C", source="—", detail="no brain given"))
        out.terms.append(self._novel())
        out.terms.append(self._programs(brain) if brain is not None
                         else Term(key="R", source="—", detail="no brain given"))
        for key, reader in (("E", self._error), ("H", self._hallucination),
                            ("U", self._unsupported)):
            out.terms.append(reader(brain) if brain is not None
                             else Term(key=key, source="—", detail="no brain given"))
        out.ms = (time.perf_counter() - started) * 1000.0
        return out


def measure(brain: Any = None, *, seed: int = 0, width: int = 6,
            benchmarks: bool = True, prepare: Any = None) -> IntelligenceVector:
    """One reading of the index. See :class:`IntelligenceIndex`."""
    return IntelligenceIndex(seed=seed, width=width).measure(
        brain, benchmarks=benchmarks, prepare=prepare)
