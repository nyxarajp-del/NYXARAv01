"""NYXARA · njp/progress.py — learning as a derivative, not a level (📉, NJP V.02).

Two organs in this package already return an honest compression number.
:meth:`~nyxara.njp.concepts.ConceptGenesis.compression` is a minimum-description-length ratio —
raw cost over modelled cost — and it is deliberately hard to game: *"a concept that over-claims is
charged for the invariants its member lacks, so inventing one enormous concept that swallows
everything makes the number worse, not better."*
:meth:`~nyxara.njp.discover.Discoverer.compression` is observations per surviving abstraction.

Both were read only as **levels**. :mod:`nyxara.njp.field` samples the concept ratio to detect
structural pressure and to refuse a knob trial that made it worse, and that is the whole list of
consumers. Measured on a brain that had just absorbed the unified and experience corpora:
``genesis 2.5938``, ``discoverer 6.0000``, unchanged across three further passes — two numbers
sitting still, read by nobody who could act on them.

**A level is not a reward.** A region she has already compressed well is exactly the region with
nothing left to learn; paying attention to it because the number is high is the opposite of
curiosity. What carries information is the **derivative** — the region where the number is *still
moving*, because that is where more looking is still buying something. This is compression
progress in the sense the literature has meant it for two decades, and the reason it belongs here
rather than in a paper is that both numbers already exist, already move, and already refuse to be
gamed.

**Three rules, and each one closes a way of cheating this.**

*Progress is measured against a high-water mark.* Not against the previous sample. A number that
falls and climbs back has learned nothing, and rewarding the climb would pay her to oscillate —
break the concepts, re-form them, collect twice. Only a **new record** pays.

*The level pays nothing.* The first sample sets the mark, so an organ that arrives compressing
beautifully and then stands still scores exactly zero. Being good at something is not a reason to
keep looking at it.

*Progress is scale-free.* A concept ratio near ``2.6`` and an abstraction ratio near ``6.0`` are
not comparable as differences, so progress is reported as a fraction of the record itself. Without
this the organ with the larger units would always look like the one that was learning.

The trace decays, so a gain two passes ago still counts for something and a gain ten passes ago
does not. That is deliberate: the question this answers is *"is this region still yielding?"*, and
an undecayed sum answers *"did it ever yield?"*, which every region eventually did.

Pure standard library. Fail-soft: an organ that raises is an organ that reported no progress,
which from the outside is what it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = ["Source", "CompressionProgress", "SOURCES"]

#: How fast a gain stops counting. One pass later it is worth 70% of itself, five passes later
#: 17%. Slow enough that a gain survives the pass it was made on, fast enough that a region which
#: has stopped yielding stops being interesting.
_DECAY = 0.7

#: Below this, a change in a compression ratio is floating-point noise in a division, not a new
#: record. Both ratios are sums of small integer counts, so a real gain is never this small.
_FLOOR = 1e-4

#: The most one pass may contribute to the trace, as a fraction of the record it beat.
#:
#: This clamp is not tidiness, it is the difference between "still yielding" and "yielded once".
#: Measured without it, on the unified corpus: every concept and abstraction record lands in one
#: chunk, both ratios jump at once — ``genesis 1.0 → 2.5938``, a rate of 1.59 — and an undecayed
#: contribution that large stays above any sensible threshold for **eight further passes** during
#: which nothing whatsoever was learned. The trace would then be reporting a region as live long
#: after it had gone dead, which is precisely the failure the derivative was chosen to avoid.
#:
#: One pass can buy at most one pass's worth. A burst falls out of the reward in a handful of
#: passes instead of a dozen, and two organs that stopped yielding at different times end up with
#: different traces — which is the only way this number can rank one region above another.
_PER_PASS_CAP = 0.08


@dataclass
class Source:
    """One compressing organ, and what it has recently bought.

    ``best`` is a high-water mark rather than the last sample, which is the whole anti-oscillation
    rule in one field: progress is what a *record* costs to beat.
    """

    name: str = ""
    best: Optional[float] = None       # the record so far; None until the first sample
    last: Optional[float] = None       # the most recent reading, for reporting only
    trace: float = 0.0                 # decayed sum of scale-free gains
    samples: int = 0
    records: int = 0                   # how many times the mark was actually beaten
    total_gain: float = 0.0

    def sample(self, value: Optional[float]) -> float:
        """Fold in one reading. Returns the scale-free progress it bought, ``0.0`` for none."""
        self.trace *= _DECAY
        if value is None:
            return 0.0
        try:
            now = float(value)
        except (TypeError, ValueError):
            return 0.0
        if now != now or now in (float("inf"), float("-inf")):   # NaN / ±inf
            return 0.0
        self.samples += 1
        self.last = now
        if self.best is None:
            # The level pays nothing. Arriving already compressed is not having just learned
            # something, and a first sample that scored would hand every organ one free reward
            # for existing.
            self.best = now
            return 0.0
        gain = now - self.best
        if gain <= _FLOOR:
            return 0.0
        # Scale-free, so a ratio of 6 and a ratio of 2.6 are comparable. Floored at 1.0 in the
        # denominator because a compression ratio below 1 means the model costs more than the raw
        # data, and dividing by that would turn a tiny gain in a failing organ into a huge reward.
        rate = gain / max(1.0, self.best)
        self.best = now
        self.records += 1
        self.total_gain += rate
        self.trace += min(rate, _PER_PASS_CAP)
        return rate

    @property
    def yielding(self) -> bool:
        """Still buying something. The only question this module exists to answer."""
        return self.trace > _FLOOR

    @property
    def full(self) -> bool:
        """Yielding as hard as one pass can. The anchor a consumer scales its reward against."""
        return self.trace >= _PER_PASS_CAP - _FLOOR

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name,
                "level": None if self.last is None else round(self.last, 4),
                "best": None if self.best is None else round(self.best, 4),
                "progress": round(self.trace, 5), "samples": self.samples,
                "records": self.records, "yielding": self.yielding,
                # The raw, unclamped gain this source has ever booked. Kept apart from `progress`
                # so the clamp is visible rather than silent: a source with a large `total_gain`
                # and a zero `progress` learned a great deal and has since stopped, which is a
                # different state from one that never learned anything, and the two must not
                # report identically.
                "total_gain": round(self.total_gain, 5)}


def _genesis(brain: Any) -> Optional[float]:
    organ = getattr(brain, "genesis", None)
    return None if organ is None else float(organ.compression())


def _discoverer(brain: Any) -> Optional[float]:
    organ = getattr(brain, "discoverer", None)
    return None if organ is None else float(organ.compression())


#: The two organs that return a compression number this package trusts. Deliberately short: a
#: source belongs here only if its number is a real description-length ratio that a system cannot
#: raise by doing less. Adding "how many facts do I have" would make the reward a reward for
#: accumulating, which is the behaviour compression progress exists to *not* reward.
SOURCES: Dict[str, Callable[[Any], Optional[float]]] = {
    "genesis": _genesis,
    "discoverer": _discoverer,
}


class CompressionProgress:
    """Samples every compressing organ and reports which ones are still yielding."""

    def __init__(self, brain: Any = None,
                 sources: Optional[Dict[str, Callable[[Any], Optional[float]]]] = None) -> None:
        self.brain = brain
        self._readers = dict(sources if sources is not None else SOURCES)
        self.sources: Dict[str, Source] = {name: Source(name=name) for name in self._readers}
        self.passes = 0

    # ---- sampling ------------------------------------------------------------ #
    def sample(self) -> Dict[str, float]:
        """One reading from every organ. Returns ``{organ: progress bought this pass}``."""
        out: Dict[str, float] = {}
        self.passes += 1
        for name, read in self._readers.items():
            source = self.sources.setdefault(name, Source(name=name))
            try:
                value = read(self.brain)
            except Exception:  # noqa: BLE001 — an organ that raises reported no progress
                value = None
            out[name] = source.sample(value)
        return out

    # ---- reading it back ------------------------------------------------------ #
    def rate(self, name: str) -> float:
        """How much this organ has recently been buying, decayed. ``0.0`` when it has stalled."""
        source = self.sources.get(name)
        return 0.0 if source is None else max(0.0, source.trace)

    def rate_over(self, names: Any) -> float:
        """Mean progress across several organs — what a gap that feeds more than one is worth."""
        rates = [self.rate(n) for n in (names or ()) if n in self.sources]
        return (sum(rates) / len(rates)) if rates else 0.0

    def yielding(self) -> List[str]:
        return [n for n, s in self.sources.items() if s.yielding]

    def stats(self) -> Dict[str, Any]:
        return {"passes": self.passes,
                "yielding": self.yielding(),
                "sources": [s.to_dict() for s in self.sources.values()]}

    def to_dict(self) -> Dict[str, Any]:
        return {"passes": self.passes,
                "sources": {n: {"best": s.best, "last": s.last, "trace": s.trace,
                                "samples": s.samples, "records": s.records,
                                "total_gain": s.total_gain}
                            for n, s in self.sources.items()}}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.passes = int((d or {}).get("passes", 0))
            for name, row in ((d or {}).get("sources") or {}).items():
                source = self.sources.setdefault(name, Source(name=name))
                best, last = row.get("best"), row.get("last")
                source.best = None if best is None else float(best)
                source.last = None if last is None else float(last)
                source.trace = float(row.get("trace", 0.0))
                source.samples = int(row.get("samples", 0))
                source.records = int(row.get("records", 0))
                source.total_gain = float(row.get("total_gain", 0.0))
        except Exception:  # noqa: BLE001
            pass
