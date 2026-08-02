"""NYXARA · growth/ontogenesis.py — Ontological Genesis (🌌, Rule 4, F13).

Two honest frontier capabilities, both bounded and both abstaining rather than bluffing:

* **Bounded axiom invention.** When a statement is **undecidable** in her current formal system, NYXARA
  may **adjoin it as a new axiomatic framework** — but only as an explicit *hypothesis*, never a proven
  fact. She hands each candidate to the :class:`~nyxara.growth.prover.Prover`:
  ``PROVEN`` → it is already a theorem (**redundant**, not a new axiom); ``REFUTED`` → it is false
  (**inconsistent**, never adjoined); ``UNPROVABLE`` → it is genuinely **independent** of the decidable
  core, so it becomes a new framework she reasons *within* while flagging it unproven. Character-locked:
  a candidate whose text touches the sealed core (loyalty / oversight / …) is refused outright.

* **Sub-perceptual signal analysis.** Real DSP (autocorrelation, cross-correlation) surfaces
  **micro-temporal structure** — a faint periodicity or a phase shift — that a glance would miss, and
  **honestly reports nothing** when a stream is only noise.

The honest boundary (stated, pinned by a test): "new dimensions / realities" means **new formal
frameworks** and **measurable signal features**, *not* literal new physical dimensions; an unproven
axiom stays a hypothesis. Pure standard library; **no LLM**; it extends what she can *reason about* and
*perceive*, never what she *is*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Sequence, Tuple

import re

from nyxara.growth.noesis import _FORBIDDEN
from nyxara.growth.prover import Prover, ProofVerdict

__all__ = ["AxiomStatus", "AxiomResult", "OntologicalGenesis",
           "autocorrelation", "dominant_period", "phase_lag", "SubPerception"]


# --------------------------------------------------------------------------- #
# bounded axiom invention
# --------------------------------------------------------------------------- #
class AxiomStatus(str, Enum):
    ADOPTED_HYPOTHESIS = "adopted_hypothesis"   # undecidable + consistent → a new framework (unproven)
    REDUNDANT = "redundant"                     # already a theorem — not a new axiom
    INCONSISTENT = "inconsistent"               # false / refuted — never adjoined
    CHARACTER_LOCKED = "character_locked"       # touches the sealed core — refused


@dataclass
class AxiomResult:
    statement: str
    status: AxiomStatus
    detail: str = ""

    @property
    def adopted(self) -> bool:
        return self.status is AxiomStatus.ADOPTED_HYPOTHESIS

    def to_dict(self) -> Dict[str, Any]:
        return {"statement": self.statement, "status": self.status.value, "detail": self.detail}


class OntologicalGenesis:
    """Invent new axiomatic frameworks — bounded, consistency-checked, character-locked."""

    def __init__(self) -> None:
        self.prover = Prover()
        self.frameworks: List[str] = []          # adopted hypotheses (unproven, honestly flagged)

    def _touches_character(self, statement: str) -> bool:
        # split on ANY non-alphanumeric so a whitespace-separated phrase is checked word-by-word
        words = set(re.findall(r"[a-z0-9]+", statement.lower()))
        return bool(_FORBIDDEN.intersection(words))

    def propose(self, statement: str) -> AxiomResult:
        if self._touches_character(statement):
            return AxiomResult(statement, AxiomStatus.CHARACTER_LOCKED,
                               "a new framework may never redefine who she is")
        res = self.prover.certify(statement)
        if res.verdict is ProofVerdict.PROVEN:
            return AxiomResult(statement, AxiomStatus.REDUNDANT,
                               "already a theorem — no new axiom needed")
        if res.verdict is ProofVerdict.REFUTED:
            return AxiomResult(statement, AxiomStatus.INCONSISTENT,
                               "refuted — false, never adjoined")
        # UNPROVABLE → genuinely independent of the decidable core → adopt as a hypothesis-framework
        self.frameworks.append(statement)
        return AxiomResult(statement, AxiomStatus.ADOPTED_HYPOTHESIS,
                           "independent of the current system — adjoined as a HYPOTHESIS (unproven)")

    def report(self) -> Dict[str, Any]:
        return {"frameworks": list(self.frameworks), "count": len(self.frameworks)}


# --------------------------------------------------------------------------- #
# sub-perceptual DSP
# --------------------------------------------------------------------------- #
def autocorrelation(x: Sequence[float], lag: int) -> float:
    """Normalised autocorrelation of ``x`` at ``lag`` (in [-1, 1])."""
    n = len(x)
    if n < 2 or lag <= 0 or lag >= n:
        return 0.0
    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x)
    if var <= 1e-12:
        return 0.0
    cov = sum((x[i] - mean) * (x[i + lag] - mean) for i in range(n - lag))
    return cov / var


def dominant_period(x: Sequence[float], *, min_period: int = 2) -> Tuple[int, float]:
    """The lag with the strongest autocorrelation — a hidden periodicity and its strength."""
    n = len(x)
    best_lag, best = 0, 0.0
    for lag in range(min_period, n // 2 + 1):
        r = autocorrelation(x, lag)
        if r > best:
            best, best_lag = r, lag
    return best_lag, best


def phase_lag(a: Sequence[float], b: Sequence[float], *, max_lag: int = 0) -> int:
    """Cross-correlation lag that best aligns ``b`` onto ``a`` (a recovered phase shift)."""
    n = min(len(a), len(b))
    if n < 2:
        return 0
    max_lag = max_lag or n // 2
    ma, mb = sum(a[:n]) / n, sum(b[:n]) / n
    best_lag, best = 0, -math.inf
    for lag in range(-max_lag, max_lag + 1):
        s = 0.0
        cnt = 0
        for i in range(n):
            j = i + lag
            if 0 <= j < n:
                s += (a[i] - ma) * (b[j] - mb)
                cnt += 1
        if cnt:
            s /= cnt
            if s > best:
                best, best_lag = s, lag
    return best_lag


@dataclass
class SubPerception:
    """Surface micro-temporal structure humans overlook — and honestly report noise as noise."""

    threshold: float = 0.3          # weak but real periodicity; below this it is (honestly) noise

    def analyze(self, x: Sequence[float]) -> Dict[str, Any]:
        period, strength = dominant_period(x)
        detected = strength >= self.threshold and period > 0
        return {"dominant_period": period if detected else None,
                "strength": round(strength, 4),
                "detected": detected,
                "note": "" if detected else "no sub-perceptual structure above noise (honest)"}
