"""NYXARA · nyx001/layers/l06_reasoning.py — Layer 6, the reasoning engine (🔍, the cycle).

The charter: "a continuous cycle of self-questioning — Hypothesize → Evaluate → Find
Contradictions → Update Confidence."

The engine holds hypotheses and runs that cycle over them. What makes it reasoning rather than
bookkeeping is that **evaluation is prediction**: a hypothesis here is a claim about what the
world model will do, and it is evaluated by *making that prediction and checking it*. A claim that
cannot be turned into a prediction cannot be evaluated, and this engine says so rather than
scoring it anyway.

Confidence updates by Bayes on the evidence, with a hard rule inherited from the rest of this
repo: **verifiable beats probabilistic**. A hypothesis marked ``verifiable`` — one whose support
came from a checked derivation rather than an accumulation of agreeable evidence — outranks a
higher-confidence unverifiable one whenever the two conflict. Confidence measures how much
evidence has piled up; it does not measure whether the evidence was any good.

Hypotheses are generated from three sources, all of them grounded in something already observed:
Layer 4's succession theories, Layer 3's most surprising episodes, and explicit proposals from
above. Nothing is invented from a template, because a template would be exactly the "hand-coded
reasoning chain" the charter forbids.

Honest: confidence here is a Beta posterior over a hypothesis's hit rate, reported with its
evidence count so a thin record reads as thin. It is not a probability that the claim is true in
the world; it is the observed frequency with which the claim's prediction held. Those differ, and
the difference is the whole reason :attr:`Hypothesis.verifiable` exists as a separate flag.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from nyxara.nyx001.layers.types import Hypothesis

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

__all__ = ["ReasoningEngine"]


class ReasoningEngine:
    """Layer 6. Holds hypotheses and runs hypothesize → evaluate → contradict → update."""

    def __init__(self, substrate: Any, *, max_hypotheses: int = 256, prior_a: float = 1.0,
                 prior_b: float = 1.0, retire_below: float = 0.15,
                 retire_after: int = 8) -> None:
        self.substrate = substrate
        self.max_hypotheses = int(max_hypotheses)
        self.prior_a = float(prior_a)
        self.prior_b = float(prior_b)
        self.retire_below = float(retire_below)
        self.retire_after = int(retire_after)
        self._hyps: Dict[str, Hypothesis] = {}
        self._counts: Dict[str, List[float]] = {}      # claim -> [hits, misses]
        self.cycles = 0
        self.retired = 0

    # ---- hypothesize ---- #
    def propose(self, claim: str, *, verifiable: bool = False,
                support: Optional[List[str]] = None) -> Optional[Hypothesis]:
        """Entertain a claim. Re-proposing an existing one strengthens its support, not its count."""
        try:
            c = str(claim).strip()
            if not c:
                return None
            existing = self._hyps.get(c)
            if existing is not None:
                for s in (support or []):
                    if s not in existing.support:
                        existing.support.append(s)
                existing.verifiable = existing.verifiable or bool(verifiable)
                return existing
            if len(self._hyps) >= self.max_hypotheses:
                self._retire_weakest()
                if len(self._hyps) >= self.max_hypotheses:
                    return None
            h = Hypothesis(claim=c, confidence=0.5, support=list(support or []),
                           verifiable=bool(verifiable))
            self._hyps[c] = h
            self._counts[c] = [self.prior_a, self.prior_b]
            return h
        except Exception:  # noqa: BLE001
            return None

    def from_theories(self, theories: List[Dict[str, Any]]) -> List[Hypothesis]:
        """Turn Layer 4's succession relations into testable claims. Grounded, not templated."""
        out = []
        for t in theories or []:
            a, b = t.get("antecedent"), t.get("consequent")
            if not a or not b:
                continue
            h = self.propose(f"{a} precedes {b}", support=[f"succession:{t.get('strength', 0)}"])
            if h is not None:
                out.append(h)
        return out

    def from_surprises(self, episodes: List[Any]) -> List[Hypothesis]:
        """Where the world model was most wrong, there is something it does not yet model."""
        out = []
        for ep in episodes or []:
            what = getattr(ep, "what", "")
            err = float(getattr(ep, "prediction_error", 0.0))
            if not what or err <= 0.0:
                continue
            h = self.propose(f"unmodelled structure in: {what[:80]}",
                             support=[f"prediction_error:{err:.4f}"])
            if h is not None:
                out.append(h)
        return out

    # ---- evaluate ---- #
    def evaluate(self, claim: str, predictor: Callable[[], Optional[bool]]) -> Optional[Hypothesis]:
        """Test a hypothesis by making its prediction and checking it.

        ``predictor`` returns True (held), False (failed), or None (could not be tested). None is
        not a miss: a claim that could not be turned into a checkable prediction has learned
        nothing about itself, and scoring it either way would be manufacturing evidence.
        """
        h = self._hyps.get(str(claim).strip())
        if h is None:
            return None
        try:
            result = predictor()
        except Exception:  # noqa: BLE001 — a failed test is untested, never a refutation
            result = None
        if result is None:
            return h
        counts = self._counts.setdefault(h.claim, [self.prior_a, self.prior_b])
        if result:
            counts[0] += 1.0
        else:
            counts[1] += 1.0
            h.contradictions.append("prediction failed")
            if len(h.contradictions) > 32:
                h.contradictions = h.contradictions[-32:]
        h.tested += 1
        h.confidence = float(counts[0] / (counts[0] + counts[1]))
        return h

    def contradict(self, claim: str, evidence: str) -> Optional[Hypothesis]:
        """Record evidence against a claim. Layer 10 drives this; it is not optional here."""
        h = self._hyps.get(str(claim).strip())
        if h is None:
            return None
        h.contradictions.append(str(evidence)[:160])
        counts = self._counts.setdefault(h.claim, [self.prior_a, self.prior_b])
        counts[1] += 1.0
        h.confidence = float(counts[0] / (counts[0] + counts[1]))
        return h

    # ---- the control law ---- #
    def best(self, *, k: int = 5) -> List[Hypothesis]:
        """Ranked. Verifiable outranks probabilistic — a checked chain beats a louder guess.

        The sort key puts ``verifiable`` first *categorically*, not as a weighted bonus. A bonus
        large enough to matter is indistinguishable from this rule, and one small enough not to be
        is not a rule at all.
        """
        items = list(self._hyps.values())
        items.sort(key=lambda h: (h.verifiable, h.confidence, h.tested), reverse=True)
        return items[: max(1, int(k))]

    def cycle(self, *, theories: Optional[List[Dict[str, Any]]] = None,
              surprises: Optional[List[Any]] = None) -> Dict[str, Any]:
        """One full pass: hypothesize from evidence, retire what has failed, report."""
        self.cycles += 1
        made = []
        if theories:
            made += self.from_theories(theories)
        if surprises:
            made += self.from_surprises(surprises)
        self._retire_failed()
        return {"cycle": self.cycles, "proposed": len(made), "held": len(self._hyps),
                "retired": self.retired}

    def _retire_failed(self) -> None:
        for claim, h in list(self._hyps.items()):
            if h.tested >= self.retire_after and h.confidence < self.retire_below:
                self._hyps.pop(claim, None)
                self._counts.pop(claim, None)
                self.retired += 1

    def _retire_weakest(self) -> None:
        if not self._hyps:
            return
        victim = min(self._hyps.values(), key=lambda h: (h.verifiable, h.confidence))
        self._hyps.pop(victim.claim, None)
        self._counts.pop(victim.claim, None)
        self.retired += 1

    def get(self, claim: str) -> Optional[Hypothesis]:
        return self._hyps.get(str(claim).strip())

    def survivors(self) -> List[Hypothesis]:
        """Hypotheses that have been tested and not contradicted."""
        return [h for h in self._hyps.values() if h.survived]

    def stats(self) -> Dict[str, Any]:
        items = list(self._hyps.values())
        tested = [h for h in items if h.tested > 0]
        return {"cycles": self.cycles, "held": len(items), "tested": len(tested),
                "survivors": len(self.survivors()), "retired": self.retired,
                "verifiable": sum(1 for h in items if h.verifiable),
                "mean_confidence": round(float(sum(h.confidence for h in tested) / len(tested)), 4)
                if tested else None}

    def to_dict(self) -> Dict[str, Any]:
        return {"cycles": self.cycles, "retired": self.retired,
                "hypotheses": [h.to_dict() for h in self.best(k=128)]}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.cycles = int(d.get("cycles", 0))
            self.retired = int(d.get("retired", 0))
            for item in d.get("hypotheses") or []:
                claim = str(item.get("claim", ""))
                if not claim:
                    continue
                h = Hypothesis(claim=claim, confidence=float(item.get("confidence", 0.5)),
                               tested=int(item.get("tested", 0)),
                               verifiable=bool(item.get("verifiable", False)))
                self._hyps[claim] = h
                conf = max(1e-6, min(1.0 - 1e-6, h.confidence))
                total = max(2.0, float(h.tested) + 2.0)
                self._counts[claim] = [conf * total, (1.0 - conf) * total]
        except Exception:  # noqa: BLE001
            pass
