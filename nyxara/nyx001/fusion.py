"""NYXARA · nyx001/fusion.py — three brains, one thought (⚖️, verifiable beats probabilistic).

NYX V.001 holds three minds that answer in different currencies:

* **V.01–.03** (:mod:`nyxara.nyx`) — a deliberated answer from the global workspace, with the
  winning specialist's confidence and whether it was *derived* or guessed.
* **NYX-5** (:mod:`nyxara.nyx5`) — a spiking tick with a surprise/entropy appraisal, and sometimes
  a pre-emptive suggestion.
* **Layers 0–17** (:mod:`nyxara.nyx001.layers`) — a cycle with a measured prediction error, a
  concept, and a calibration.

Fusing them is not averaging. Three numbers called "confidence" that were produced by three
unrelated procedures do not live on a common scale, and averaging them produces a number that
means nothing while looking authoritative. So this module does two things instead:

**1. The control law, enforced not weighted.** ``verifiable`` beats ``probabilistic``,
categorically. A checked derivation from V.02's proof core or an induced rule that predicted a
held-out demonstration outranks any unverified answer regardless of how confident that answer
sounds. This is the same law ``nyxara/nyx/__init__.py`` states for V.01, kept here rather than
diluted. A weighted bonus large enough to matter would be indistinguishable from the rule, and one
small enough not to be would not be a rule.

**2. Weights that are earned, not assigned.** Below the control law, each source's vote is scaled
by its own *measured* track record — a reliability EMA over whether that source's answers turned
out right, updated by :meth:`Fusion.resolve` when an outcome becomes known. A source with no track
record contributes at a neutral prior and reports its sample count, so a thin record reads as thin.

:meth:`Fusion.fuse` returns a :class:`Fused` carrying the winner, the runners-up, why the winner
won, and — the field that matters most — ``agreement``: how much the three actually concurred.
Low agreement with high confidence is the signature of a system fooling itself, and it is surfaced
rather than smoothed away.

Honest: reliability here is an exponential moving average of observed outcomes — ordinary online
statistics. It is not a calibration guarantee, and with few resolved outcomes it is mostly prior.
The three sources are also not independent: they share NYXARA's inputs, so their agreement is
weaker evidence than three independent judgements would be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Vote", "Fused", "Fusion"]

_SOURCES = ("v03", "nyx5", "stack")


@dataclass
class Vote:
    """One brain's answer, in its own currency, with what it claims about itself."""

    source: str = ""
    text: str = ""
    confidence: float = 0.0
    verifiable: bool = False
    rationale: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_content(self) -> bool:
        return bool(self.text and self.text.strip())

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "confidence": round(self.confidence, 4),
                "verifiable": self.verifiable, "has_content": self.has_content,
                "rationale": self.rationale[:160]}


@dataclass
class Fused:
    """One thought, assembled from three. Carries why it won and how much they agreed."""

    winner: Optional[Vote] = None
    votes: List[Vote] = field(default_factory=list)
    agreement: float = 0.0
    reason: str = ""
    verifiable: bool = False
    confidence: float = 0.0

    @property
    def answer(self) -> str:
        return self.winner.text if self.winner is not None else ""

    @property
    def contested(self) -> bool:
        """High confidence on low agreement — the signature of a system fooling itself."""
        return bool(self.confidence > 0.6 and self.agreement < 0.34)

    def to_dict(self) -> Dict[str, Any]:
        return {"winner": self.winner.source if self.winner else None,
                "confidence": round(self.confidence, 4), "verifiable": self.verifiable,
                "agreement": round(self.agreement, 4), "contested": self.contested,
                "reason": self.reason, "votes": [v.to_dict() for v in self.votes]}


class Fusion:
    """Combines the three brains' answers under one enforced control law."""

    def __init__(self, *, alpha: float = 0.1, prior: float = 0.5,
                 min_samples: int = 5) -> None:
        self.alpha = float(alpha)
        self.prior = float(prior)
        self.min_samples = int(min_samples)
        self._reliability: Dict[str, float] = {s: prior for s in _SOURCES}
        self._samples: Dict[str, int] = {s: 0 for s in _SOURCES}
        self.fusions = 0
        self.resolutions = 0
        self._last: Optional[Fused] = None

    # ---- reliability ---- #
    def reliability(self, source: str) -> Tuple[float, int]:
        """This source's measured hit rate and how many outcomes it rests on."""
        return float(self._reliability.get(source, self.prior)), int(self._samples.get(source, 0))

    def _weight(self, source: str) -> float:
        """Neutral until there is a real record — a thin record does not get to swing a vote."""
        rel, n = self.reliability(source)
        if n < self.min_samples:
            return self.prior
        return rel

    def resolve(self, source: str, *, correct: float) -> None:
        """Credit a source with how its answer actually turned out. Drives the weights."""
        try:
            k = float(min(1.0, max(0.0, correct)))
            cur = self._reliability.get(source, self.prior)
            self._reliability[source] = float((1.0 - self.alpha) * cur + self.alpha * k)
            self._samples[source] = self._samples.get(source, 0) + 1
            self.resolutions += 1
        except Exception:  # noqa: BLE001
            pass

    # ---- agreement ---- #
    @staticmethod
    def _similar(a: str, b: str) -> float:
        """Token-overlap similarity. Crude on purpose: it is a concurrence signal, not a metric."""
        ta = {w for w in str(a).lower().split() if len(w) > 2}
        tb = {w for w in str(b).lower().split() if len(w) > 2}
        if not ta or not tb:
            return 0.0
        return float(len(ta & tb) / len(ta | tb))

    def _agreement(self, votes: List[Vote]) -> float:
        """Mean pairwise similarity among the votes that actually said something."""
        live = [v for v in votes if v.has_content]
        if len(live) < 2:
            return 0.0
        sims = []
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                sims.append(self._similar(live[i].text, live[j].text))
        return float(sum(sims) / len(sims)) if sims else 0.0

    # ---- the fusion ---- #
    def fuse(self, votes: List[Vote]) -> Fused:
        """Apply the control law, then the earned weights. Never raises; may return nothing."""
        out = Fused(votes=list(votes or []))
        try:
            self.fusions += 1
            live = [v for v in (votes or []) if v is not None and v.has_content]
            if not live:
                out.reason = "no source produced content"
                self._last = out
                return out
            out.agreement = self._agreement(live)

            # THE CONTROL LAW. Verifiable wins categorically — not by a large weight.
            verified = [v for v in live if v.verifiable]
            if verified:
                winner = max(verified, key=lambda v: v.confidence * self._weight(v.source))
                out.winner = winner
                out.verifiable = True
                out.confidence = float(winner.confidence)
                out.reason = (f"{winner.source} was verifiable — a checked derivation outranks "
                              f"any unverified answer, however confident")
                self._last = out
                return out

            # Below the law: earned weight × claimed confidence.
            def score(v: Vote) -> float:
                return float(v.confidence * self._weight(v.source))

            winner = max(live, key=score)
            out.winner = winner
            out.verifiable = False
            rel, n = self.reliability(winner.source)
            # The fused confidence is the winner's, scaled by how much its source has EARNED. A
            # source with no record cannot lend authority it has not demonstrated.
            out.confidence = float(winner.confidence * self._weight(winner.source))
            out.reason = (f"{winner.source} scored highest "
                          f"(confidence {winner.confidence:.2f} × reliability {rel:.2f} "
                          f"over {n} resolved outcomes)")
            self._last = out
            return out
        except Exception:  # noqa: BLE001 — a failed fusion produces no thought, never a crash
            out.reason = "fusion failed"
            self._last = out
            return out

    def explain(self, fused: Optional[Fused] = None) -> str:
        """One honest line. Names contestedness rather than hiding it behind the winner."""
        f = fused or self._last
        if f is None or f.winner is None:
            return "no thought — none of the three brains produced content"
        note = " (CONTESTED: the three disagree)" if f.contested else ""
        return f"{f.reason}{note}"

    def stats(self) -> Dict[str, Any]:
        return {"fusions": self.fusions, "resolutions": self.resolutions,
                "reliability": {s: {"value": round(self._reliability[s], 4),
                                    "samples": self._samples[s],
                                    "thin": self._samples[s] < self.min_samples}
                                for s in _SOURCES},
                "last": self._last.to_dict() if self._last else None}

    def to_dict(self) -> Dict[str, Any]:
        return {"fusions": self.fusions, "resolutions": self.resolutions,
                "reliability": dict(self._reliability), "samples": dict(self._samples)}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.fusions = int(d.get("fusions", 0))
            self.resolutions = int(d.get("resolutions", 0))
            self._reliability.update({str(k): float(v) for k, v in
                                      (d.get("reliability") or {}).items()})
            self._samples.update({str(k): int(v) for k, v in (d.get("samples") or {}).items()})
        except Exception:  # noqa: BLE001
            pass
