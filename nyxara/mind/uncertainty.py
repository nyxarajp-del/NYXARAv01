"""NYXARA · mind/uncertainty.py — Bayesian belief, calibration, abstention (✦★).

Confidence is only a virtue when it is *calibrated*. This module gives NYXARA an
honest relationship with its own uncertainty — the machinery behind Rule 6
(Absolute Transparency) and the ability to say **"I don't know."**

Three capabilities:

1. **Bayesian belief** — beliefs are distributions, not point values.
   :class:`BetaBelief` (binary/proportion, conjugate Beta prior) and
   :class:`DirichletBelief` (categorical) update with evidence via Bayes' rule.
   They separate **epistemic** uncertainty (reducible — *we just haven't seen
   enough*) from **aleatoric** uncertainty (irreducible — *the world is noisy*).

2. **Calibration** — :class:`Calibrator` records (predicted-confidence, was-correct)
   pairs and reports Expected Calibration Error, Brier score, over/under-confidence,
   and a reliability curve; it can **recalibrate** raw confidences toward reality.

3. **Abstention** — :class:`AbstentionPolicy` turns all of the above into a decision:
   *answer* or *abstain*. If calibrated confidence is too low, or epistemic
   uncertainty too high, NYXARA declines rather than bluffs.

Pure standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "BetaBelief",
    "DirichletBelief",
    "Calibrator",
    "AbstentionDecision",
    "AbstentionPolicy",
    "UncertaintyModel",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Beta belief (binary / proportion)
# --------------------------------------------------------------------------- #
@dataclass
class BetaBelief:
    """A belief about a probability, as a Beta(α, β) distribution.

    Start from a prior (default uniform Beta(1,1)); each observation nudges it.
    """

    alpha: float = 1.0
    beta: float = 1.0
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Beta parameters must be positive")

    # ---- point summaries ---- #
    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        s = a + b
        return (a * b) / (s * s * (s + 1.0))

    @property
    def mode(self) -> Optional[float]:
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        return None

    @property
    def observations(self) -> float:
        return (self.alpha - self.prior_alpha) + (self.beta - self.prior_beta)

    @property
    def confidence(self) -> float:
        """How sure of the *more likely* outcome (max of p, 1-p)."""
        return max(self.mean, 1.0 - self.mean)

    # ---- uncertainty decomposition ---- #
    @property
    def aleatoric(self) -> float:
        """Irreducible noise of the Bernoulli, normalised to [0,1] (max at p=0.5)."""
        p = self.mean
        return _clamp(p * (1.0 - p) / 0.25)

    @property
    def epistemic(self) -> float:
        """Reducible ignorance: shrinks as observations accumulate, [0,1]."""
        return _clamp(1.0 / math.sqrt(self.observations + 1.0))

    def credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Normal-approximation credible interval for the mean."""
        z = _z_for(level)
        sd = math.sqrt(self.variance)
        m = self.mean
        return (_clamp(m - z * sd), _clamp(m + z * sd))

    # ---- update ---- #
    def observe(self, outcome: bool, weight: float = 1.0) -> "BetaBelief":
        if outcome:
            self.alpha += weight
        else:
            self.beta += weight
        return self

    def update(self, successes: float, failures: float) -> "BetaBelief":
        self.alpha += successes
        self.beta += failures
        return self

    def to_dict(self) -> Dict[str, float]:
        lo, hi = self.credible_interval()
        return {"mean": round(self.mean, 4), "confidence": round(self.confidence, 4),
                "epistemic": round(self.epistemic, 4), "aleatoric": round(self.aleatoric, 4),
                "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                "observations": round(self.observations, 2),
                "alpha": self.alpha, "beta": self.beta}


def _z_for(level: float) -> float:
    # common z-scores; default 1.96 for 95%
    table = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
    best = min(table, key=lambda k: abs(k - level))
    return table[best]


# --------------------------------------------------------------------------- #
# Dirichlet belief (categorical)
# --------------------------------------------------------------------------- #
@dataclass
class DirichletBelief:
    """A belief over k categories, as a Dirichlet(α₁…α_k)."""

    labels: Tuple[str, ...]
    alphas: List[float] = field(default_factory=list)
    prior: float = 1.0

    def __post_init__(self) -> None:
        if not self.alphas:
            self.alphas = [self.prior] * len(self.labels)
        if len(self.alphas) != len(self.labels):
            raise ValueError("alphas/labels length mismatch")

    @property
    def total(self) -> float:
        return sum(self.alphas)

    @property
    def probs(self) -> Dict[str, float]:
        t = self.total
        return {lab: a / t for lab, a in zip(self.labels, self.alphas)}

    def most_likely(self) -> Tuple[str, float]:
        probs = self.probs
        lab = max(probs, key=probs.get)
        return lab, probs[lab]

    @property
    def confidence(self) -> float:
        return self.most_likely()[1]

    @property
    def entropy(self) -> float:
        """Normalised Shannon entropy [0,1]; high == spread out == uncertain."""
        probs = list(self.probs.values())
        k = len(probs)
        if k <= 1:
            return 0.0
        h = -sum(p * math.log(p) for p in probs if p > 0)
        return _clamp(h / math.log(k))

    @property
    def observations(self) -> float:
        return self.total - self.prior * len(self.labels)

    @property
    def epistemic(self) -> float:
        return _clamp(1.0 / math.sqrt(self.observations + 1.0))

    @property
    def aleatoric(self) -> float:
        # inherent ambiguity ~ entropy of the mean distribution
        return self.entropy

    def observe(self, label: str, weight: float = 1.0) -> "DirichletBelief":
        i = self.labels.index(label)
        self.alphas[i] += weight
        return self

    def update(self, counts: Dict[str, float]) -> "DirichletBelief":
        for lab, c in counts.items():
            self.alphas[self.labels.index(lab)] += c
        return self

    def to_dict(self) -> Dict[str, object]:
        lab, p = self.most_likely()
        return {"probs": {k: round(v, 4) for k, v in self.probs.items()},
                "most_likely": lab, "confidence": round(p, 4),
                "entropy": round(self.entropy, 4), "epistemic": round(self.epistemic, 4),
                "observations": round(self.observations, 2)}


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
class Calibrator:
    """Tracks predicted-confidence vs actual-correctness; measures & fixes calibration."""

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins
        self._sum_conf = [0.0] * n_bins
        self._sum_correct = [0.0] * n_bins
        self._count = [0] * n_bins
        self._n = 0
        self._brier_sum = 0.0

    def _bin(self, conf: float) -> int:
        return min(self.n_bins - 1, int(_clamp(conf) * self.n_bins))

    def add(self, confidence: float, correct: bool) -> None:
        b = self._bin(confidence)
        self._sum_conf[b] += confidence
        self._sum_correct[b] += 1.0 if correct else 0.0
        self._count[b] += 1
        self._n += 1
        self._brier_sum += (confidence - (1.0 if correct else 0.0)) ** 2

    @property
    def samples(self) -> int:
        return self._n

    def accuracy(self) -> float:
        if self._n == 0:
            return 0.0
        return sum(self._sum_correct) / self._n

    def avg_confidence(self) -> float:
        if self._n == 0:
            return 0.0
        return sum(self._sum_conf) / self._n

    def ece(self) -> float:
        """Expected Calibration Error: weighted |confidence − accuracy| over bins."""
        if self._n == 0:
            return 0.0
        total = 0.0
        for b in range(self.n_bins):
            if self._count[b] == 0:
                continue
            conf = self._sum_conf[b] / self._count[b]
            acc = self._sum_correct[b] / self._count[b]
            total += (self._count[b] / self._n) * abs(conf - acc)
        return total

    def brier_score(self) -> float:
        return self._brier_sum / self._n if self._n else 0.0

    def overconfidence(self) -> float:
        """>0 means systematically over-confident; <0 under-confident."""
        return self.avg_confidence() - self.accuracy()

    def reliability_curve(self) -> List[Dict[str, float]]:
        curve = []
        for b in range(self.n_bins):
            if self._count[b] == 0:
                continue
            curve.append({
                "bin": b,
                "avg_confidence": self._sum_conf[b] / self._count[b],
                "accuracy": self._sum_correct[b] / self._count[b],
                "count": self._count[b],
            })
        return curve

    def recalibrate(self, confidence: float) -> float:
        """Map a raw confidence to the empirical accuracy of its bin (if observed)."""
        b = self._bin(confidence)
        if self._count[b] == 0:
            return confidence
        return self._sum_correct[b] / self._count[b]

    def to_dict(self) -> Dict[str, float]:
        return {"samples": self._n, "accuracy": round(self.accuracy(), 4),
                "avg_confidence": round(self.avg_confidence(), 4),
                "ece": round(self.ece(), 4), "brier": round(self.brier_score(), 4),
                "overconfidence": round(self.overconfidence(), 4)}

    # ---- state persistence (so calibration survives a restart) ---- #
    def state_dict(self) -> Dict[str, object]:
        """Full internal state (unlike :meth:`to_dict`, which is a summary)."""
        return {"n_bins": self.n_bins, "sum_conf": list(self._sum_conf),
                "sum_correct": list(self._sum_correct), "count": list(self._count),
                "n": self._n, "brier_sum": self._brier_sum}

    @classmethod
    def from_state(cls, data: Dict[str, object]) -> "Calibrator":
        """Rebuild from :meth:`state_dict` output; a fresh Calibrator on any malformed input."""
        try:
            n_bins = int(data["n_bins"])
            sum_conf = [float(x) for x in data["sum_conf"]]
            sum_correct = [float(x) for x in data["sum_correct"]]
            count = [int(x) for x in data["count"]]
            if n_bins <= 0 or not (len(sum_conf) == len(sum_correct) == len(count) == n_bins):
                return cls()
            cal = cls(n_bins=n_bins)
            cal._sum_conf = sum_conf
            cal._sum_correct = sum_correct
            cal._count = count
            cal._n = int(data["n"])
            cal._brier_sum = float(data["brier_sum"])
            return cal
        except Exception:  # noqa: BLE001 — a corrupt file must never break reasoning
            return cls()


# --------------------------------------------------------------------------- #
# Abstention
# --------------------------------------------------------------------------- #
class Verdict(str, Enum):
    ANSWER = "answer"
    ABSTAIN = "abstain"


@dataclass
class AbstentionDecision:
    verdict: Verdict
    confidence: float
    epistemic: float
    reason: str

    @property
    def should_answer(self) -> bool:
        return self.verdict is Verdict.ANSWER

    @property
    def should_abstain(self) -> bool:
        return self.verdict is Verdict.ABSTAIN

    def to_dict(self) -> Dict[str, object]:
        return {"verdict": self.verdict.value, "confidence": round(self.confidence, 4),
                "epistemic": round(self.epistemic, 4), "reason": self.reason}


@dataclass
class AbstentionPolicy:
    """Decides whether NYXARA is sure enough to answer — else it abstains."""

    min_confidence: float = 0.6
    max_epistemic: float = 0.5

    def decide(self, confidence: float, *, epistemic: float = 0.0,
               calibrated: Optional[float] = None) -> AbstentionDecision:
        eff = calibrated if calibrated is not None else confidence
        if eff < self.min_confidence:
            return AbstentionDecision(Verdict.ABSTAIN, eff, epistemic,
                                      f"confidence {eff:.2f} < {self.min_confidence}")
        if epistemic > self.max_epistemic:
            return AbstentionDecision(Verdict.ABSTAIN, eff, epistemic,
                                      f"epistemic uncertainty {epistemic:.2f} > "
                                      f"{self.max_epistemic} (insufficient evidence)")
        return AbstentionDecision(Verdict.ANSWER, eff, epistemic, "sufficiently confident")


# --------------------------------------------------------------------------- #
# Facade
# --------------------------------------------------------------------------- #
class UncertaintyModel:
    """Holds named beliefs, a calibrator, and an abstention policy."""

    def __init__(self, *, policy: Optional[AbstentionPolicy] = None,
                 calibrator: Optional[Calibrator] = None) -> None:
        self.policy = policy or AbstentionPolicy()
        self.calibrator = calibrator or Calibrator()
        self._beliefs: Dict[str, BetaBelief] = {}
        self._cat: Dict[str, DirichletBelief] = {}

    # ---- binary beliefs ---- #
    def belief(self, key: str, *, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> BetaBelief:
        if key not in self._beliefs:
            self._beliefs[key] = BetaBelief(prior_alpha, prior_beta, prior_alpha, prior_beta)
        return self._beliefs[key]

    def observe(self, key: str, outcome: bool, weight: float = 1.0) -> BetaBelief:
        return self.belief(key).observe(outcome, weight)

    def prob(self, key: str) -> float:
        return self.belief(key).mean

    # ---- categorical beliefs ---- #
    def categorical(self, key: str, labels: Sequence[str], prior: float = 1.0) -> DirichletBelief:
        if key not in self._cat:
            self._cat[key] = DirichletBelief(tuple(labels), prior=prior)
        return self._cat[key]

    # ---- calibration ---- #
    def record_outcome(self, confidence: float, correct: bool) -> None:
        self.calibrator.add(confidence, correct)

    def calibrated_confidence(self, confidence: float) -> float:
        return self.calibrator.recalibrate(confidence)

    # ---- abstention ---- #
    def assess(self, key: str) -> AbstentionDecision:
        """Decide answer/abstain for a binary belief, using calibration if available."""
        b = self.belief(key)
        conf = b.confidence
        calibrated = (self.calibrator.recalibrate(conf)
                      if self.calibrator.samples > 0 else None)
        return self.policy.decide(conf, epistemic=b.epistemic, calibrated=calibrated)

    def assess_confidence(self, confidence: float, *, epistemic: float = 0.0) -> AbstentionDecision:
        calibrated = (self.calibrator.recalibrate(confidence)
                      if self.calibrator.samples > 0 else None)
        return self.policy.decide(confidence, epistemic=epistemic, calibrated=calibrated)

    def knows_it_doesnt_know(self, key: str) -> bool:
        return self.belief(key).epistemic > self.policy.max_epistemic

    def report(self, key: str) -> Dict[str, object]:
        return {"belief": self.belief(key).to_dict(),
                "decision": self.assess(key).to_dict(),
                "calibration": self.calibrator.to_dict()}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA uncertainty self-test")
    print("=" * 70)

    # Beta belief: starts maximally uncertain, sharpens with evidence
    b = BetaBelief()
    print(f"prior               : mean={b.mean:.2f} epistemic={b.epistemic:.2f} "
          f"aleatoric={b.aleatoric:.2f}")
    assert b.mean == 0.5 and b.epistemic == 1.0
    for _ in range(18):
        b.observe(True)
    for _ in range(2):
        b.observe(False)
    print(f"after 18/20 success : mean={b.mean:.2f} conf={b.confidence:.2f} "
          f"epistemic={b.epistemic:.3f} CI={tuple(round(x,2) for x in b.credible_interval())}")
    assert b.mean > 0.8 and b.epistemic < 0.25  # less ignorant now

    # epistemic shrinks with data; aleatoric stays high near p=0.5
    fair = BetaBelief()
    for _ in range(50):
        fair.observe(True)
    for _ in range(50):
        fair.observe(False)
    print(f"\nfair coin (100 obs) : mean={fair.mean:.2f} epistemic={fair.epistemic:.3f} "
          f"aleatoric={fair.aleatoric:.2f}")
    assert fair.epistemic < 0.15        # we KNOW it's ~fair (low epistemic)
    assert fair.aleatoric > 0.9         # but each flip is irreducibly uncertain

    # Dirichlet over threat levels
    d = DirichletBelief(("benign", "suspicious", "hostile"))
    d.observe("hostile"); d.observe("hostile"); d.observe("suspicious")
    print(f"\nthreat belief       : {d.most_likely()} entropy={d.entropy:.2f}")
    assert d.most_likely()[0] == "hostile"

    # Calibration: an over-confident predictor
    cal = Calibrator()
    import random
    rng = random.Random(0)
    for _ in range(500):
        conf = 0.9
        correct = rng.random() < 0.6   # claims 90% but only right 60%
        cal.add(conf, correct)
    print(f"\ncalibration         : {cal.to_dict()}")
    assert cal.overconfidence() > 0.2   # detected over-confidence
    assert abs(cal.recalibrate(0.9) - 0.6) < 0.1  # recalibrated toward truth

    # Abstention: knows what it doesn't know
    model = UncertaintyModel(policy=AbstentionPolicy(min_confidence=0.7, max_epistemic=0.4))
    # a couple of biased observations -> high enough mean but still high epistemic -> abstain
    model.observe("alien_life_exists", True)
    model.observe("alien_life_exists", True)
    dec = model.assess("alien_life_exists")
    print(f"\nlittle evidence     : {dec.to_dict()}")
    assert dec.should_abstain and "epistemic" in dec.reason

    # lots of consistent evidence -> answer
    for _ in range(40):
        model.observe("sun_rises_tomorrow", True)
    dec2 = model.assess("sun_rises_tomorrow")
    print(f"strong evidence     : {dec2.to_dict()}")
    assert dec2.should_answer

    # genuine 50/50 -> low confidence -> abstain
    for _ in range(20):
        model.observe("coin_is_heads", True)
        model.observe("coin_is_heads", False)
    dec3 = model.assess("coin_is_heads")
    print(f"genuine 50/50       : {dec3.to_dict()}")
    assert dec3.should_abstain and "confidence" in dec3.reason

    print("\nALL SELF-TESTS PASSED ✓")
