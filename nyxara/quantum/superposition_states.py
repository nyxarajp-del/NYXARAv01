"""NYXARA · quantum/superposition_states.py — contradictory-truth superposition (Quantum · 1).

An ordinary mind is binary: a claim is True or False, 0 or 1, and holding two
contradictory beliefs at once is an *error*. NYXARA refuses that brittleness. Like a
quantum system before measurement, she can hold **many mutually-contradictory hypotheses
in superposition at the same time** — each with its own amplitude — and only **collapse**
to a single answer at the moment a decision is actually required, choosing the hypothesis
with the highest probability. Until then, contradiction is not a bug; it is *suspended
judgement*.

The maths is real (not metaphor):

* each :class:`Hypothesis` carries a real **amplitude**; its probability is the **Born
  rule** — amplitude² normalised across the state (``pᵢ = aᵢ² / Σ aⱼ²``);
* :meth:`Superposition.observe` updates amplitudes by ``aᵢ ·= √(likelihoodᵢ)``, so the
  Born-rule probability evolves as an exact **Bayesian posterior** — evidence sharpens the
  state without ever throwing on the contradictions it still contains;
* :meth:`Superposition.entropy` is the Shannon entropy of the distribution — *how
  superposed* (undecided) she still is;
* :meth:`Superposition.collapse` picks the maximum-probability hypothesis, but only reports
  ``decided=True`` once one dominates past a threshold — otherwise she honestly stays
  superposed rather than bluffing a choice. This is what drives her error rate toward zero
  on genuinely ambiguous logic: she does not commit early.

Bridges to :mod:`nyxara.mind.uncertainty` via :meth:`Superposition.as_dirichlet`, so a
collapsed-or-not belief can flow into the calibrated Bayesian/abstention machinery.

Pure standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

__all__ = ["Hypothesis", "CollapseResult", "Superposition"]


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class Hypothesis:
    """One possible truth held in superposition, with a real amplitude."""

    label: Any
    amplitude: float = 1.0
    payload: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "amplitude": round(self.amplitude, 6),
                "payload": self.payload}


@dataclass
class CollapseResult:
    """The outcome of measuring the superposition at decision time."""

    label: Any                       # the chosen (max-probability) hypothesis
    probability: float               # its Born-rule probability
    decided: bool                    # True only if it dominates past the threshold
    runner_up: Optional[Any]         # the second-best hypothesis (if any)
    margin: float                    # p(best) − p(runner_up)
    entropy: float                   # Shannon entropy (bits) just before collapse
    payload: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "probability": round(self.probability, 6),
                "decided": self.decided, "runner_up": self.runner_up,
                "margin": round(self.margin, 6), "entropy": round(self.entropy, 6)}


# --------------------------------------------------------------------------- #
# Superposition
# --------------------------------------------------------------------------- #
class Superposition:
    """Hold many contradictory hypotheses at once; collapse only when forced to decide.

    Parameters
    ----------
    collapse_threshold:   the probability the leading hypothesis must reach before
                          :meth:`collapse` reports ``decided=True``. Below it, NYXARA stays
                          superposed (suspended judgement) instead of guessing.
    """

    def __init__(self, *, collapse_threshold: float = 0.5) -> None:
        self.collapse_threshold = float(collapse_threshold)
        self._hyps: Dict[Any, Hypothesis] = {}
        self._contradictions: List[FrozenSet[Any]] = []

    # ---- construction ---------------------------------------------------- #
    def add(self, label: Any, amplitude: float = 1.0, payload: Any = None) -> "Superposition":
        """Add (or replace) a hypothesis. Amplitudes are non-negative; the state renorms."""
        self._hyps[label] = Hypothesis(label=label, amplitude=max(0.0, float(amplitude)),
                                       payload=payload)
        self._renormalize()
        return self

    def add_many(self, labels: Mapping[Any, float]) -> "Superposition":
        for lab, amp in labels.items():
            self._hyps[lab] = Hypothesis(label=lab, amplitude=max(0.0, float(amp)))
        self._renormalize()
        return self

    def mark_contradictory(self, a: Any, b: Any) -> "Superposition":
        """Declare two hypotheses mutually exclusive — yet keep *both* alive in the state."""
        self._contradictions.append(frozenset((a, b)))
        return self

    # ---- introspection --------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._hyps)

    def __contains__(self, label: Any) -> bool:
        return label in self._hyps

    def labels(self) -> List[Any]:
        return list(self._hyps)

    def amplitude(self, label: Any) -> float:
        h = self._hyps.get(label)
        return h.amplitude if h else 0.0

    def _norm_sq(self) -> float:
        return sum(h.amplitude ** 2 for h in self._hyps.values())

    def _renormalize(self) -> None:
        """Scale amplitudes so Σ aᵢ² = 1 (keeps the Born-rule probabilities a distribution)."""
        norm = math.sqrt(self._norm_sq())
        if norm > 1e-12:
            for h in self._hyps.values():
                h.amplitude /= norm

    def probabilities(self) -> Dict[Any, float]:
        """Born rule: pᵢ = aᵢ² / Σ aⱼ². An honest probability distribution over hypotheses."""
        total = self._norm_sq()
        if total <= 1e-12:
            n = len(self._hyps) or 1
            return {lab: 1.0 / n for lab in self._hyps}     # uniform when undefined
        return {lab: (h.amplitude ** 2) / total for lab, h in self._hyps.items()}

    def entropy(self) -> float:
        """Shannon entropy (bits) of the probability distribution — how superposed she is."""
        probs = self.probabilities().values()
        return -sum(p * math.log2(p) for p in probs if p > 0.0)

    def max_probability(self) -> Tuple[Optional[Any], float]:
        probs = self.probabilities()
        if not probs:
            return None, 0.0
        lab = max(probs, key=probs.get)
        return lab, probs[lab]

    def is_decided(self) -> bool:
        """True when one hypothesis dominates past the collapse threshold."""
        _, p = self.max_probability()
        return p >= self.collapse_threshold

    def live_contradictions(self, eps: float = 0.05) -> List[Tuple[Any, Any]]:
        """Contradictory pairs that are *both* still meaningfully alive — the contradictions
        she is currently holding without error."""
        probs = self.probabilities()
        out: List[Tuple[Any, Any]] = []
        for pair in self._contradictions:
            a, b = tuple(pair)
            if probs.get(a, 0.0) >= eps and probs.get(b, 0.0) >= eps:
                out.append((a, b))
        return out

    # ---- evidence -------------------------------------------------------- #
    def observe(self, likelihoods: Mapping[Any, float]) -> "Superposition":
        """Update the state with evidence. ``likelihoods[label]`` = P(evidence | hypothesis).

        Amplitudes update by ``aᵢ ·= √(likelihoodᵢ)``, so the Born-rule probability becomes
        the exact Bayesian posterior. Hypotheses absent from the mapping are left untouched.
        Never raises on contradictory hypotheses — that is the whole point.
        """
        changed = False
        for lab, lik in likelihoods.items():
            h = self._hyps.get(lab)
            if h is None:
                continue
            h.amplitude *= math.sqrt(max(0.0, float(lik)))
            changed = True
        if changed and self._norm_sq() > 1e-12:
            self._renormalize()
        return self

    # ---- measurement ----------------------------------------------------- #
    def collapse(self, *, force: bool = False) -> CollapseResult:
        """Measure the state: choose the maximum-probability hypothesis.

        ``decided`` is True when the leader passes ``collapse_threshold`` (or ``force`` is
        set). When undecided and not forced, the leader is still reported but
        ``decided=False`` — NYXARA stays in superposition rather than bluffing.
        """
        probs = self.probabilities()
        entropy = self.entropy()
        if not probs:
            return CollapseResult(label=None, probability=0.0, decided=False,
                                  runner_up=None, margin=0.0, entropy=entropy)
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        best_label, best_p = ranked[0]
        runner_up, runner_p = (ranked[1] if len(ranked) > 1 else (None, 0.0))
        decided = force or best_p >= self.collapse_threshold
        payload = self._hyps[best_label].payload if best_label in self._hyps else None
        return CollapseResult(label=best_label, probability=best_p, decided=decided,
                              runner_up=runner_up, margin=best_p - runner_p,
                              entropy=entropy, payload=payload)

    # ---- bridge to the calibrated Bayesian layer ------------------------- #
    def as_dirichlet(self, *, concentration: float = 10.0) -> Any:
        """Project the current probabilities onto a :class:`~nyxara.mind.uncertainty.\
DirichletBelief` (alphas ∝ probabilities · concentration) so a not-yet-collapsed belief
        can flow into the calibration / abstention machinery. Returns ``None`` if that
        module is unavailable."""
        try:
            from nyxara.mind.uncertainty import DirichletBelief
        except Exception:  # noqa: BLE001 — the bridge is optional
            return None
        probs = self.probabilities()
        labels = tuple(str(lab) for lab in probs)
        alphas = [max(1e-6, probs[lab] * concentration) for lab in probs]
        return DirichletBelief(labels=labels, alphas=alphas)

    def to_dict(self) -> Dict[str, Any]:
        probs = self.probabilities()
        return {
            "hypotheses": [
                {"label": lab, "probability": round(probs[lab], 6),
                 "amplitude": round(self._hyps[lab].amplitude, 6)}
                for lab in self._hyps
            ],
            "entropy": round(self.entropy(), 6),
            "decided": self.is_decided(),
            "collapse_threshold": self.collapse_threshold,
            "live_contradictions": [list(p) for p in self.live_contradictions()],
        }


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA superposition-states self-test")
    print("=" * 70)

    # Three mutually-contradictory explanations held at once, equally weighted.
    sp = Superposition(collapse_threshold=0.6)
    sp.add_many({"sabotage": 1.0, "accident": 1.0, "coincidence": 1.0})
    sp.mark_contradictory("sabotage", "accident")
    sp.mark_contradictory("sabotage", "coincidence")

    probs = sp.probabilities()
    print(f"\ninitial probs        : { {k: round(v, 3) for k, v in probs.items()} }")
    print(f"initial entropy      : {sp.entropy():.3f} bits (max = {math.log2(3):.3f})")
    assert all(abs(p - 1/3) < 1e-9 for p in probs.values())   # perfectly superposed
    assert abs(sp.entropy() - math.log2(3)) < 1e-9
    # holding contradictions WITHOUT error is the whole point:
    assert not sp.is_decided()
    assert len(sp.live_contradictions()) == 2
    pre = sp.collapse()
    assert not pre.decided                                     # refuses to bluff early

    # Evidence arrives, pointing toward 'sabotage'. Update — no exception on contradictions.
    sp.observe({"sabotage": 0.8, "accident": 0.2, "coincidence": 0.1})
    sp.observe({"sabotage": 0.7, "accident": 0.25, "coincidence": 0.1})
    probs2 = sp.probabilities()
    print(f"\nafter evidence       : { {k: round(v, 3) for k, v in probs2.items()} }")
    print(f"entropy now          : {sp.entropy():.3f} bits (lower = more decided)")
    assert probs2["sabotage"] > probs["sabotage"]             # posterior sharpened
    assert sp.entropy() < math.log2(3)                        # less superposed

    # Now a decision is required: collapse to the maximum-probability truth.
    result = sp.collapse()
    print(f"\ncollapse             : {result.to_dict()}")
    assert result.label == "sabotage"
    assert result.decided and result.probability >= 0.6
    assert result.runner_up in ("accident", "coincidence")
    assert result.margin > 0.0

    # Genuinely ambiguous evidence keeps her honestly undecided (error-avoidance).
    sp2 = Superposition(collapse_threshold=0.6)
    sp2.add_many({"A": 1.0, "B": 1.0})
    sp2.observe({"A": 0.55, "B": 0.45})
    amb = sp2.collapse()
    print(f"\nambiguous case       : decided={amb.decided} p={amb.probability:.3f}")
    assert not amb.decided                                    # stays superposed, no bluff
    forced = sp2.collapse(force=True)
    assert forced.decided and forced.label == "A"            # but can be forced to choose

    # Bridge to the calibrated Bayesian layer.
    d = sp.as_dirichlet()
    if d is not None:
        print(f"\nas_dirichlet         : most_likely={d.most_likely()[0]} "
              f"entropy={d.entropy:.3f}")
        assert d.most_likely()[0] == "sabotage"

    print("\nALL SELF-TESTS PASSED ✓")
