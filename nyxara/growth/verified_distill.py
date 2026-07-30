"""NYXARA · growth/verified_distill.py — the adversarial hallucination trap (Part M).

"Trust the tool, but verify." A cloud model (groq, or GLM-5 via airouter) is a powerful *proposer*, and proposers
hallucinate — so nothing it says is baked into NYXARA's durable knowledge until it survives a check:

* **Decidable claim** (arithmetic / algebra / logic / number theory) → run it through
  :class:`~nyxara.growth.prover.Prover`, which returns a machine-checkable *certificate* (``PROVEN``)
  or a counter-example (``REFUTED``) and **never bluffs** (``UNPROVABLE`` = honest abstention).
* **Open-domain claim** → a graded, honest confidence — *not* a proof. Only high-confidence results are
  marked durable; the rest are tentative (stored with decay) so a hallucination never hardens into a fact.

When a decidable answer is refuted and a model is available, :meth:`self_correct` feeds the counter-example
back to the model (a single-round RLSP-style correction) and re-verifies — so the *corrected, verified*
answer is what gets stored, and the next occurrence needs no cloud call.

Reuses the existing `Prover` (decision procedures + certificates). No network; deterministic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from nyxara.growth.prover import ProofVerdict, Prover

__all__ = ["VerifiedResult", "GroundedVerifier"]

_HEDGES = ("i'm not sure", "not sure", "i think", "i guess", "maybe", "perhaps",
           "i cannot", "i can't", "i don't know", "as an ai", "no idea", "unsure")


@dataclass
class VerifiedResult:
    """The verdict on one (question, answer) pair."""

    verdict: str            # "proven" | "refuted" | "graded"
    confidence: float       # [0, 1]
    method: str
    certificate: str = ""   # a re-checkable witness when proven/refuted
    durable: bool = False   # True only when safe to bake as permanent knowledge

    @property
    def proven(self) -> bool:
        return self.verdict == "proven"

    @property
    def refuted(self) -> bool:
        return self.verdict == "refuted"

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "confidence": round(self.confidence, 4),
                "method": self.method, "certificate": self.certificate, "durable": self.durable}


class GroundedVerifier:
    """Verify a model's answer before it becomes local knowledge (Part M)."""

    def __init__(self, *, prover: Optional[Prover] = None, bake_threshold: float = 0.75) -> None:
        self.prover = prover or Prover()
        self.bake_threshold = float(bake_threshold)

    # ---- the check ---- #
    def verify(self, question: str, answer: str, *, prior_confidence: float = 1.0) -> VerifiedResult:
        statement = self._as_statement(question, answer)
        if statement:
            try:
                res = self.prover.certify(statement)
            except Exception:  # noqa: BLE001 — a checker failure is honest abstention, not a crash
                res = None
            if res is not None and res.verdict is ProofVerdict.PROVEN:
                return VerifiedResult("proven", 1.0, res.method, res.certificate, durable=True)
            if res is not None and res.verdict is ProofVerdict.REFUTED:
                return VerifiedResult("refuted", 0.0, res.method, res.certificate, durable=False)
        # not decidable here — an honest graded confidence, never a proof
        conf = self._grade(answer, prior_confidence)
        return VerifiedResult("graded", conf, "grounded-heuristic",
                              durable=conf >= self.bake_threshold)

    # ---- optional self-correction (needs a model; a no-op without one) ---- #
    def self_correct(self, question: str, answer: str, *,
                     generate: Optional[Callable[[str], str]] = None,
                     prior_confidence: float = 1.0, max_rounds: int = 1) -> tuple[str, VerifiedResult]:
        """Return the best (answer, verdict). If the answer is refuted and a ``generate(prompt)->str``
        model is supplied, feed the counter-example back and re-verify (single-round by default)."""
        best_answer, best = answer, self.verify(question, answer, prior_confidence=prior_confidence)
        rounds = 0
        while best.refuted and generate is not None and rounds < max_rounds:
            rounds += 1
            prompt = (f"Your answer to '{question}' was {best_answer!r}, which is wrong: "
                      f"{best.certificate}. Give ONLY the corrected answer.")
            try:
                corrected = (generate(prompt) or "").strip()
            except Exception:  # noqa: BLE001 — correction is best-effort
                break
            if not corrected or corrected == best_answer:
                break
            cand = self.verify(question, corrected, prior_confidence=prior_confidence)
            best_answer, best = corrected, cand
            if best.proven:
                break
        return best_answer, best

    # ---- helpers ---- #
    @staticmethod
    def _as_statement(question: str, answer: str) -> Optional[str]:
        """Form a decidable statement from (question, answer), or None when open-domain."""
        a = (answer or "").strip()
        q = (question or "").strip()
        if not a:
            return None
        if any(op in a for op in ("=", "≤", "≥", "==")):
            return a                                   # answer is already a checkable (in)equation
        core = re.sub(r"(?i)^\s*(what\s+is|compute|evaluate|calculate|how much is)\b", "", q)
        core = core.strip(" ?.\t")
        if (core and re.search(r"\d.*[-+*/^].*\d", core)
                and re.fullmatch(r"[-+0-9.,/*\s]+", a)):
            return f"({core}) = {a}"                    # e.g. "6*7" + "42" -> "(6*7) = 42"
        return None

    @staticmethod
    def _grade(answer: str, prior_confidence: float) -> float:
        a = (answer or "").strip()
        if not a:
            return 0.0
        conf = max(0.0, min(1.0, float(prior_confidence)))
        low = a.lower()
        if any(h in low for h in _HEDGES):
            conf *= 0.5
        if len(a) < 3:
            conf *= 0.5
        return round(conf, 4)
