"""NYXARA · nyx001/substrate/objective.py — the composite objective (🎯, not next-token loss).

A language model optimises one number: how surprised it was by the next token. The NYX V.001
charter rejects that as the whole story — intelligence, it says, is driven by a *multi-dimensional*
objective. This module is that objective, and its design rule is that every term must be
**separately measurable**, because a composite loss whose terms cannot be told apart is one number
wearing five hats.

Five terms, each with a reason to exist:

* **prediction** (``mse`` / ``cross_entropy``) — the world model's error, ``D(s_{t+1}, ŝ_{t+1})``.
  The load-bearing term; the others shape *how* it is reduced.
* **concept** — pull representations of co-occurring things together and push unrelated ones
  apart. Without it a world model can predict well with an internally shapeless latent space, and
  Layer 4 then has nothing to form concepts *from*.
* **compression** — an L1 pressure toward sparse, reusable codes. This is Layer 14's objective
  expressed as a gradient: a representation that says the same thing in fewer active units is the
  one that transfers.
* **calibration** — penalise confidence that outruns correctness. This is what makes the ``U``
  signal in :mod:`~nyxara.nyx001.substrate.optim` mean something: a system whose uncertainty is
  uncorrelated with its error has an uncertainty estimate that is decorative.
* **curiosity** — a *negative*-weighted information-gain term, so the system is rewarded for
  moving toward states it cannot yet predict. This is the only term that pushes error *up* in the
  short run, and it is bounded hard for that reason.

:class:`CompositeObjective` returns a :class:`Loss` carrying both the differentiable total and a
per-term breakdown of scalars. The breakdown is what makes ablation real: a term can be zeroed by
weight and the change in the other four measured, rather than a claim being made about it.

Honest about a real tension: these terms are not independent, and their weights are hyperparameters
chosen by hand, not derived. Curiosity and prediction genuinely pull against each other — that is
the point of curiosity, but it does mean a badly-set ``w_curiosity`` produces a system that seeks
confusion. The default weights keep curiosity an order of magnitude below prediction, and
:meth:`CompositeObjective.contributions` reports what each term actually contributed on the last
call so a bad setting is visible in numbers instead of in behaviour three hours later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from nyxara.nyx001.substrate import autograd as A
from nyxara.nyx001.substrate.autograd import Tensor

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

__all__ = ["Loss", "CompositeObjective"]


@dataclass
class Loss:
    """A differentiable total plus the per-term scalars that produced it."""

    total: Optional[Tensor] = None
    terms: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)

    @property
    def value(self) -> float:
        """The scalar total — 0.0 only when there was genuinely nothing to optimise."""
        if self.total is None:
            return 0.0
        return float(self.total.data)

    def contributions(self) -> Dict[str, float]:
        """Each term's *weighted* share of the total, as a fraction. Empty when the total is 0."""
        weighted = {k: abs(self.weights.get(k, 0.0) * v) for k, v in self.terms.items()}
        denom = sum(weighted.values())
        if denom <= 0.0:
            return {}
        return {k: round(v / denom, 4) for k, v in weighted.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {"total": round(self.value, 6),
                "terms": {k: round(v, 6) for k, v in self.terms.items()},
                "contributions": self.contributions()}


class CompositeObjective:
    """The five-term objective. Any term may be switched off by setting its weight to zero."""

    def __init__(self, *, w_prediction: float = 1.0, w_concept: float = 0.2,
                 w_compression: float = 0.05, w_calibration: float = 0.1,
                 w_curiosity: float = 0.05, curiosity_ceiling: float = 1.0) -> None:
        self.w_prediction = float(w_prediction)
        self.w_concept = float(w_concept)
        self.w_compression = float(w_compression)
        self.w_calibration = float(w_calibration)
        self.w_curiosity = float(w_curiosity)
        self.curiosity_ceiling = float(curiosity_ceiling)
        self.last: Optional[Loss] = None

    # ---- individual terms ---- #
    @staticmethod
    def prediction_term(pred: Tensor, target: "np.ndarray") -> Tensor:
        """``D(s_{t+1}, ŝ_{t+1})`` — the world model's error."""
        return A.mse(pred, target)

    @staticmethod
    def concept_term(anchor: Tensor, positive: Tensor,
                     negatives: Optional[Sequence[Tensor]] = None) -> Tensor:
        """Pull ``anchor`` toward ``positive``, push it from ``negatives``.

        Written as ``(1 - cos(a, p)) + mean(relu(cos(a, n)))`` rather than a softmax contrastive
        loss: with a handful of negatives drawn from episodic memory the softmax normalisation is
        dominated by however many negatives happened to be available, which makes the term's scale
        depend on memory occupancy rather than on representation quality.
        """
        pull = A.mean_all(A.sub(A.constant(np.ones(())), A.cosine(anchor, positive)))
        if not negatives:
            return pull
        pushes = [A.mean_all(A.relu(A.cosine(anchor, n))) for n in negatives]
        push = pushes[0]
        for p in pushes[1:]:
            push = A.add(push, p)
        return A.add(pull, A.scale(push, 1.0 / len(pushes)))

    @staticmethod
    def compression_term(code: Tensor) -> Tensor:
        """L1 pressure toward sparse codes — Layer 14's goal as a gradient.

        L1 and not L2 deliberately: an L2 penalty shrinks every coordinate a little and drives
        none of them to zero, which produces a small dense code. Sparsity — a code where most
        units are *off* and the few that are on are reusable — is what Layer 14 needs, and that
        is what the constant-magnitude subgradient of L1 produces.
        """
        return A.mean_all(A.abs_val(code))

    @staticmethod
    def calibration_term(confidence: Tensor, correct: "np.ndarray") -> Tensor:
        """Penalise the gap between stated confidence and observed correctness (Brier score)."""
        return A.mse(confidence, np.asarray(correct, dtype=np.float64))

    def curiosity_term(self, info_gain: Tensor) -> Tensor:
        """Negative information gain, softly saturated at ``curiosity_ceiling``.

        Returned as ``-c·tanh(gain/c)``: near zero it is just ``-gain``, so ordinary curiosity is
        rewarded linearly; as the gain grows the term asymptotes at ``-c`` and *its gradient goes
        to zero*. That vanishing gradient is the safety property. A hard ``min(gain, c)`` would
        clip the value while leaving a discontinuous subgradient at the boundary; a straight-
        through variant would clip the value while passing the full gradient, which caps the
        number but not the incentive. Only the smooth saturation actually removes the pull toward
        ever-more-confusing states — the failure mode this ceiling exists to prevent.
        """
        c = max(self.curiosity_ceiling, 1e-9)
        return A.neg(A.scale(A.tanh(A.scale(info_gain, 1.0 / c)), c))

    # ---- the composite ---- #
    def __call__(self, *, prediction: Optional[Tensor] = None,
                 concept: Optional[Tensor] = None,
                 compression: Optional[Tensor] = None,
                 calibration: Optional[Tensor] = None,
                 curiosity: Optional[Tensor] = None) -> Loss:
        """Combine whichever terms were supplied. Missing terms are absent, not zero.

        The distinction matters for :meth:`Loss.contributions`: a term that was never computed
        must not dilute the shares of the terms that were.
        """
        pairs = [
            ("prediction", prediction, self.w_prediction),
            ("concept", concept, self.w_concept),
            ("compression", compression, self.w_compression),
            ("calibration", calibration, self.w_calibration),
            ("curiosity", curiosity, self.w_curiosity),
        ]
        total: Optional[Tensor] = None
        terms: Dict[str, float] = {}
        weights: Dict[str, float] = {}
        for name, term, weight in pairs:
            if term is None or weight == 0.0:
                continue
            terms[name] = float(term.data)
            weights[name] = weight
            scaled = A.scale(term, weight)
            total = scaled if total is None else A.add(total, scaled)
        loss = Loss(total=total, terms=terms, weights=weights)
        self.last = loss
        return loss

    def stats(self) -> Dict[str, Any]:
        return {"weights": {"prediction": self.w_prediction, "concept": self.w_concept,
                            "compression": self.w_compression, "calibration": self.w_calibration,
                            "curiosity": self.w_curiosity},
                "last": self.last.to_dict() if self.last is not None else None}
