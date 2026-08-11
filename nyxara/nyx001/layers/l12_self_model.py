"""NYXARA · nyx001/layers/l12_self_model.py — Layer 12, the self-model (🪞, three questions).

The charter: "Models its own internal state — What do I know? Where am I failing? How confident
am I?"

All three answers here are **measurements of this system's own history**, never introspective
reports. That distinction is the whole design. A system that reports its confidence by consulting
a feeling has an unfalsifiable self-model; a system that reports it as the observed frequency with
which its stated confidence matched its actual correctness has one that can be wrong and shown to
be wrong.

* **What do I know?** — the earned concepts (Layer 4), the surviving hypotheses (Layer 6), and the
  regions curiosity has marked learned (Layer 9). Counted, with their support.
* **Where am I failing?** — the regions with persistent error and no progress, the retired
  hypotheses, and the defects self-critique keeps finding. :meth:`weaknesses` ranks them.
* **How confident am I?** — :meth:`calibration`, the reliability-diagram gap between stated
  confidence and observed correctness over a sliding window. Reported with its sample count, so a
  thin record reads as thin rather than as a confident 1.0.

:meth:`calibration` returning ``None`` before ``min_samples`` is deliberate and is the most
important null in this file. A system with four observations does not know how calibrated it is,
and reporting 1.0 (or 0.0) from four samples would let every layer above it reason on a number
that is noise.

Honest: calibration is measured only on outcomes that were actually resolved. Predictions the
world never answered are excluded rather than counted as either right or wrong, which means the
calibration is over the resolvable subset of what the system does, not over all of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["SelfReport", "SelfModel"]


@dataclass
class SelfReport:
    """What the system can truthfully say about itself, from live numbers only."""

    knows: Dict[str, Any] = field(default_factory=dict)
    failing: List[Dict[str, Any]] = field(default_factory=list)
    calibration: Optional[float] = None
    calibration_samples: int = 0
    confidence_bias: Optional[float] = None
    competence: Optional[float] = None

    @property
    def is_thin(self) -> bool:
        """Too little evidence for this report to mean much. Stated, not hidden."""
        return self.calibration is None or self.calibration_samples < 20

    def to_dict(self) -> Dict[str, Any]:
        return {"knows": dict(self.knows), "failing": list(self.failing[:8]),
                "calibration": None if self.calibration is None else round(self.calibration, 4),
                "calibration_samples": self.calibration_samples,
                "confidence_bias": None if self.confidence_bias is None
                else round(self.confidence_bias, 4),
                "competence": None if self.competence is None else round(self.competence, 4),
                "thin": self.is_thin}

    def summary(self) -> str:
        """One honest sentence. Says "I don't know yet" when that is the true answer."""
        if self.is_thin:
            return (f"I have {self.calibration_samples} resolved outcomes — not enough to know "
                    f"how reliable I am. I hold {self.knows.get('concepts', 0)} concepts and "
                    f"{self.knows.get('hypotheses', 0)} surviving hypotheses.")
        bias = self.confidence_bias or 0.0
        lean = "overconfident" if bias > 0.05 else ("underconfident" if bias < -0.05 else "calibrated")
        return (f"Over {self.calibration_samples} resolved outcomes I am {lean} "
                f"(calibration {self.calibration:.2f}, bias {bias:+.2f}). "
                f"I hold {self.knows.get('concepts', 0)} concepts; "
                f"{len(self.failing)} areas are not improving.")


class SelfModel:
    """Layer 12. Measures what this system knows, where it fails, and how sure it should be."""

    def __init__(self, substrate: Any, *, window: int = 256, min_samples: int = 20,
                 bins: int = 10) -> None:
        self.substrate = substrate
        self.window = int(window)
        self.min_samples = int(min_samples)
        self.bins = int(bins)
        self._outcomes: List[Tuple[float, float]] = []      # (stated_confidence, correct 0/1)

    # ---- how confident am I? ---- #
    def record(self, confidence: float, correct: float) -> None:
        """Log one resolved outcome. Unresolved predictions must not be passed here."""
        try:
            c = float(min(1.0, max(0.0, confidence)))
            k = float(min(1.0, max(0.0, correct)))
            self._outcomes.append((c, k))
            if len(self._outcomes) > self.window:
                self._outcomes = self._outcomes[-self.window:]
        except Exception:  # noqa: BLE001
            pass

    def calibration(self) -> Optional[float]:
        """1 - mean |stated - observed| per bin. ``None`` below ``min_samples``, never a guess."""
        if len(self._outcomes) < self.min_samples:
            return None
        try:
            buckets: Dict[int, List[Tuple[float, float]]] = {}
            for c, k in self._outcomes:
                b = min(self.bins - 1, int(c * self.bins))
                buckets.setdefault(b, []).append((c, k))
            gaps, weights = [], []
            for items in buckets.values():
                stated = sum(c for c, _ in items) / len(items)
                observed = sum(k for _, k in items) / len(items)
                gaps.append(abs(stated - observed))
                weights.append(len(items))
            total = sum(weights)
            weighted = sum(g * w for g, w in zip(gaps, weights)) / max(1, total)
            return float(max(0.0, 1.0 - weighted))
        except Exception:  # noqa: BLE001
            return None

    def confidence_bias(self) -> Optional[float]:
        """Mean(stated) - mean(correct). Positive is overconfident. ``None`` when unknown."""
        if len(self._outcomes) < self.min_samples:
            return None
        stated = sum(c for c, _ in self._outcomes) / len(self._outcomes)
        observed = sum(k for _, k in self._outcomes) / len(self._outcomes)
        return float(stated - observed)

    def competence(self) -> Optional[float]:
        """Plain hit rate — how often it is actually right, independent of how sure it sounded."""
        if len(self._outcomes) < self.min_samples:
            return None
        return float(sum(k for _, k in self._outcomes) / len(self._outcomes))

    def adjusted(self, stated: float) -> float:
        """Correct a stated confidence by the measured bias. Identity when bias is unknown."""
        bias = self.confidence_bias()
        if bias is None:
            return float(min(1.0, max(0.0, stated)))
        return float(min(1.0, max(0.0, stated - bias)))

    # ---- what do I know / where am I failing? ---- #
    def report(self, *, semantic: Any = None, reasoning: Any = None,
               curiosity: Any = None, critique: Any = None) -> SelfReport:
        """Assemble the three answers from whichever layers are present."""
        rep = SelfReport()
        try:
            concepts = semantic.concepts() if semantic is not None else []
            survivors = reasoning.survivors() if reasoning is not None else []
            learned = [r for r in getattr(curiosity, "_regions", {}).values()
                       if getattr(r, "is_learned", False)] if curiosity is not None else []
            rep.knows = {
                "concepts": len(concepts),
                "concept_support": sum(int(getattr(c, "support", 0)) for c in concepts),
                "hypotheses": len(survivors),
                "verifiable": sum(1 for h in survivors if getattr(h, "verifiable", False)),
                "learned_regions": len(learned),
            }
            if curiosity is not None:
                for r in curiosity.stale()[:8]:
                    rep.failing.append({"kind": "no-progress", "where": r.key,
                                        "mean_error": r.mean_error})
            if reasoning is not None:
                retired = int(getattr(reasoning, "retired", 0))
                if retired:
                    rep.failing.append({"kind": "retired-hypotheses", "count": retired})
            if critique is not None:
                cyc = int(getattr(critique, "cycles", 0))
                imp = int(getattr(critique, "improvements", 0))
                if cyc and imp / max(1, cyc) < 0.2:
                    rep.failing.append({"kind": "critique-ineffective",
                                        "improvement_rate": round(imp / cyc, 4)})
            rep.calibration = self.calibration()
            rep.calibration_samples = len(self._outcomes)
            rep.confidence_bias = self.confidence_bias()
            rep.competence = self.competence()
            return rep
        except Exception:  # noqa: BLE001
            return rep

    def weaknesses(self, *, k: int = 5, **layers: Any) -> List[Dict[str, Any]]:
        return self.report(**layers).failing[: max(1, int(k))]

    def stats(self) -> Dict[str, Any]:
        return {"samples": len(self._outcomes), "min_samples": self.min_samples,
                "calibration": self.calibration(), "bias": self.confidence_bias(),
                "competence": self.competence()}

    def to_dict(self) -> Dict[str, Any]:
        return {"outcomes": [[round(c, 4), round(k, 4)] for c, k in self._outcomes[-256:]]}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self._outcomes = [(float(a), float(b)) for a, b in (d.get("outcomes") or [])]
        except Exception:  # noqa: BLE001
            pass
