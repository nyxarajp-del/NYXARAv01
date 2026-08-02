"""NYXARA · mind/synesthesia.py — Cross-Domain Biomimetic Synesthesia (🎨⚛, Rule 4).

The critique this module answers, in the Master's words: *"Har organ apni specific domain mein kaam
karta hai. Naya: ek synesthetic translation matrix — music ke patterns dekh kar physics solve kare, DNA
ke behaviour ko market par apply kare — poori kainaat ke saare systems ek single unified pattern ki tarah
dikhein."*

Two real substrates already exist: :mod:`nyxara.cognition.hyper_dimensional_vectors` (a shared
10,000-D space where any pattern becomes comparable geometry) and
:mod:`nyxara.mind.category_transfer` (structure-preserving functorial transfer between her DSL
categories, adopted only when it *verifiably* works). This module joins them into a **universal pattern
transposition** over *arbitrary numeric domains*:

1. It extracts a scale-invariant **shape signature** of a pattern — trend, curvature, oscillation,
   monotonicity, exponential/power linearity, autocorrelation — and lifts it into the shared HDC space
   via a signed random projection. Two patterns from *different* domains that share *shape* (a
   population curve and a compound-interest curve; a heartbeat and a market oscillation) land **near
   each other** in that space, regardless of their units.
2. It finds the nearest **cross-domain** analogue of a target pattern (a different domain, same shape).
3. It **transposes** the analogue's known functional law onto the target and — like category_transfer
   — adopts it **only if it verifiably fits** the target (R² past a threshold on the target's own
   data); otherwise it **abstains**. An unverified analogy is a hypothesis, not knowledge.

So "music → physics" is made concrete and honest: the *same shape* is recognised across domains and a
law is carried over **only when it demonstrably predicts** the new domain. Pure numeric maths;
**no LLM**; composes the HDC faculty and reuses least-squares fitting. Degrades to shape-features-only
when NumPy is absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.cognition.hyper_dimensional_vectors import (
    Hypervector,
    HyperSpace,
    RandomProjector,
)

__all__ = ["Pattern", "TranspositionResult", "SynestheticMatrix"]

# the functional-law families she can recognise and transpose
_FORMS = ("linear", "exponential", "power", "logarithmic", "sine")
_FEAT_DIM = 12


def _log_shift(y: Sequence[float]) -> float:
    """The additive shift that makes ``y`` positive for a log — zero when it already is, so a genuine
    exponential/power is not distorted by an unnecessary offset."""
    lo = min(y)
    return 0.0 if lo > 0.0 else (1e-6 - lo)


# --------------------------------------------------------------------------- #
# shape features — a scale-invariant fingerprint of a pattern's *form*
# --------------------------------------------------------------------------- #
def _r2(pred: Sequence[float], truth: Sequence[float]) -> float:
    n = len(truth)
    if n == 0:
        return 0.0
    mean = sum(truth) / n
    ss_tot = sum((t - mean) ** 2 for t in truth)
    ss_res = sum((p - t) ** 2 for p, t in zip(pred, truth))
    if ss_tot <= 1e-12:
        return 1.0 if ss_res <= 1e-12 else 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def _linfit(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float, float]:
    """Least-squares slope, intercept, R² for y = a·x + b."""
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0), 0.0
    sx = sum(xs); sy = sum(ys); sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0, sy / n, 0.0
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    pred = [a * x + b for x in xs]
    return a, b, _r2(pred, ys)


def _shape_features(series: Sequence[float]) -> List[float]:
    """A scale-invariant feature vector capturing a pattern's *form* (not its units)."""
    y = [float(v) for v in series if _finite(v)]
    n = len(y)
    if n < 3:
        return [0.0] * _FEAT_DIM
    xs = [float(i) for i in range(n)]
    # normalise y to zero-mean/unit-std so only the SHAPE, not the scale, is encoded
    mean = sum(y) / n
    var = sum((v - mean) ** 2 for v in y) / n
    std = math.sqrt(var) or 1.0
    yn = [(v - mean) / std for v in y]

    slope, _b, lin_r2 = _linfit(xs, yn)
    diffs = [yn[i + 1] - yn[i] for i in range(n - 1)]
    second = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
    curvature = (sum(second) / len(second)) if second else 0.0
    sign_changes = sum(1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i + 1] < 0)
    oscillation = sign_changes / max(1, len(diffs) - 1)
    monotonic = max(sum(1 for d in diffs if d >= 0), sum(1 for d in diffs if d <= 0)) / max(1, len(diffs))
    autocorr1 = (sum(yn[i] * yn[i + 1] for i in range(n - 1)) / (n - 1))
    # exponential linearity: fit a line to log(y) (shifted positive only if needed) — high R² ⇒ exp
    shift = _log_shift(y)
    logy = [math.log(v + shift) for v in y]
    _s2, _i2, exp_r2 = _linfit(xs, logy)
    # power linearity: line in log-log (positive x only)
    xs_pos = [math.log(i + 1.0) for i in range(n)]
    _s3, _i3, pow_r2 = _linfit(xs_pos, logy)
    # dominant-oscillation strength via lag autocorrelation trough/peak spread
    ac = [sum(yn[i] * yn[i + k] for i in range(n - k)) / (n - k) for k in range(1, min(n, 6))]
    osc_strength = (max(ac) - min(ac)) if ac else 0.0
    rng = (max(y) - min(y))
    tail_ratio = (y[-1] - y[0]) / (rng or 1.0)

    return [slope, curvature, oscillation, monotonic, autocorr1, lin_r2, exp_r2, pow_r2,
            osc_strength, tail_ratio, float(min(1.0, std / (abs(mean) + 1.0))),
            float(n) / (n + 10.0)]


def _finite(x: Any) -> bool:
    try:
        f = float(x)
        return not (math.isnan(f) or math.isinf(f))
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# functional-law fitters (for verify-or-abstain transposition)
# --------------------------------------------------------------------------- #
def _fit_form(form: str, series: Sequence[float]) -> float:
    """Fit ``form`` to ``series`` and return its R² (goodness of fit) in [0,1]."""
    y = [float(v) for v in series if _finite(v)]
    n = len(y)
    if n < 3:
        return 0.0
    xs = [float(i) for i in range(n)]
    try:
        if form == "linear":
            _a, _b, r2 = _linfit(xs, y)
            return r2
        if form == "exponential":
            shift = _log_shift(y)
            logy = [math.log(v + shift) for v in y]
            a, b, _r = _linfit(xs, logy)
            pred = [math.exp(a * x + b) - shift for x in xs]
            return _r2(pred, y)
        if form == "power":
            shift = _log_shift(y)
            logy = [math.log(v + shift) for v in y]
            lx = [math.log(i + 1.0) for i in range(n)]
            a, b, _r = _linfit(lx, logy)
            pred = [math.exp(a * math.log(i + 1.0) + b) - shift for i in range(n)]
            return _r2(pred, y)
        if form == "logarithmic":
            lx = [math.log(i + 1.0) for i in range(n)]
            a, b, r2 = _linfit(lx, y)
            return r2
        if form == "sine":
            return _fit_sine(y)
    except Exception:  # noqa: BLE001 — a fit that blows up simply does not fit
        return 0.0
    return 0.0


def _fit_sine(y: Sequence[float]) -> float:
    """Best R² over a small bank of periods for y ≈ A·sin(ωx) + B·cos(ωx) + C (linear in A,B,C)."""
    n = len(y)
    if n < 4:
        return 0.0
    best = 0.0
    for period in range(2, n + 1):
        w = 2.0 * math.pi / period
        s = [math.sin(w * i) for i in range(n)]
        c = [math.cos(w * i) for i in range(n)]
        # solve least squares for [A,B,C] via normal equations (3x3)
        cols = [s, c, [1.0] * n]
        g = [[sum(cols[a][i] * cols[b][i] for i in range(n)) for b in range(3)] for a in range(3)]
        rhs = [sum(cols[a][i] * y[i] for i in range(n)) for a in range(3)]
        coef = _solve3(g, rhs)
        if coef is None:
            continue
        pred = [coef[0] * s[i] + coef[1] * c[i] + coef[2] for i in range(n)]
        best = max(best, _r2(pred, y))
    return best


def _solve3(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """Solve a 3×3 system by Gaussian elimination (returns None if singular)."""
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        m[col] = [v / pv for v in m[col]]
        for r in range(3):
            if r != col:
                f = m[r][col]
                m[r] = [m[r][k] - f * m[col][k] for k in range(4)]
    return [m[0][3], m[1][3], m[2][3]]


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
@dataclass
class Pattern:
    label: str
    domain: str
    signature: Any                    # the HDC signature (or the raw feature vector without numpy)
    features: List[float]
    form: Optional[str] = None        # the known functional law of this pattern, if any


@dataclass
class TranspositionResult:
    """The outcome of transposing a cross-domain law onto a target pattern."""

    target_domain: str
    source_label: Optional[str]
    source_domain: Optional[str]
    form: Optional[str]
    similarity: float                 # cross-domain HDC similarity of the shapes
    fit_r2: float                     # how well the transposed law actually fits the target
    adopted: bool                     # True only if it verifiably fits (else she abstains)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"target_domain": self.target_domain, "source_label": self.source_label,
                "source_domain": self.source_domain, "form": self.form,
                "similarity": round(self.similarity, 4), "fit_r2": round(self.fit_r2, 4),
                "adopted": self.adopted, "reason": self.reason}


# --------------------------------------------------------------------------- #
# the matrix
# --------------------------------------------------------------------------- #
class SynestheticMatrix:
    """Encode patterns from any domain into one space; find cross-domain isomorphisms; transpose laws."""

    def __init__(self, *, dim: int = 10000, seed: int = 7, adopt_r2: float = 0.9,
                 min_similarity: float = 0.15) -> None:
        self.space = HyperSpace(dim, seed=seed)
        self.projector = RandomProjector(_FEAT_DIM, dim, seed=seed)
        self.adopt_r2 = float(adopt_r2)
        self.min_similarity = float(min_similarity)
        self._patterns: List[Pattern] = []

    # ---- encode + learn -------------------------------------------------- #
    def encode(self, series: Sequence[float]) -> Tuple[Any, List[float]]:
        feats = _shape_features(series)
        sig = self.projector.project(feats)
        return sig, feats

    def learn(self, label: str, domain: str, series: Sequence[float],
              *, form: Optional[str] = None) -> Pattern:
        """Register a domain's pattern (optionally with its known functional law) into the space."""
        sig, feats = self.encode(series)
        if form is None:
            form = self.best_form(series)
        p = Pattern(label=str(label), domain=str(domain), signature=sig, features=feats, form=form)
        self._patterns.append(p)
        return p

    @staticmethod
    def best_form(series: Sequence[float]) -> Optional[str]:
        """The functional family that best fits ``series`` (the pattern's own law), or None."""
        scored = [(f, _fit_form(f, series)) for f in _FORMS]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[0][0] if scored and scored[0][1] >= 0.75 else None

    # ---- cross-domain isomorphism --------------------------------------- #
    def find_isomorphisms(self, series: Sequence[float], *, target_domain: str = "",
                          cross_domain_only: bool = True) -> List[Tuple[Pattern, float]]:
        """Rank stored patterns by shape-similarity to ``series`` (optionally only OTHER domains)."""
        sig, _feats = self.encode(series)
        out: List[Tuple[Pattern, float]] = []
        for p in self._patterns:
            if cross_domain_only and target_domain and p.domain == target_domain:
                continue
            out.append((p, _similarity(sig, p.signature)))
        out.sort(key=lambda t: t[1], reverse=True)
        return out

    # ---- transpose (verify or abstain) ---------------------------------- #
    def transpose(self, target_series: Sequence[float], target_domain: str) -> TranspositionResult:
        """Carry the nearest cross-domain law onto the target — adopting it only if it verifiably fits."""
        matches = self.find_isomorphisms(target_series, target_domain=target_domain,
                                         cross_domain_only=True)
        best_form_match = next(((p, s) for p, s in matches if p.form is not None), None)
        if best_form_match is None:
            return TranspositionResult(target_domain, None, None, None, 0.0, 0.0, False,
                                       "no cross-domain analogue with a known law")
        src, sim = best_form_match
        if sim < self.min_similarity:
            return TranspositionResult(target_domain, src.label, src.domain, src.form, sim, 0.0,
                                       False, "nearest analogue is not structurally similar enough")
        fit = _fit_form(src.form, target_series)
        adopted = fit >= self.adopt_r2
        reason = (f"transposed '{src.form}' law from {src.domain} — verified (R²={fit:.3f})" if adopted
                  else f"analogue found but the law did not fit the target (R²={fit:.3f}); abstaining")
        return TranspositionResult(target_domain, src.label, src.domain, src.form, sim, fit,
                                   adopted, reason)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _similarity(a: Any, b: Any) -> float:
    if isinstance(a, Hypervector) and isinstance(b, Hypervector):
        return a.cosine(b)
    # pure-python fallback: cosine over the raw feature vectors
    import math as _m
    va, vb = list(a), list(b)
    dot = sum(x * y for x, y in zip(va, vb))
    na = _m.sqrt(sum(x * x for x in va)) or 1.0
    nb = _m.sqrt(sum(y * y for y in vb)) or 1.0
    return dot / (na * nb)


# --------------------------------------------------------------------------- #
# Self-test / demo — real cross-domain transposition (no LLM, no network)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA cross-domain synesthesia self-test — one shape across many domains")
    print("=" * 70)

    m = SynestheticMatrix(adopt_r2=0.9)

    # DOMAIN A (biology): a population that grows exponentially — she learns its law.
    bio = [2.0 * math.exp(0.30 * t) for t in range(16)]
    pa = m.learn("bacterial_growth", "biology", bio)
    print(f"learned biology     : bacterial_growth, law={pa.form}")
    assert pa.form == "exponential"

    # DOMAIN B (finance): compound interest — SAME shape, different units, unknown law.
    fin = [100.0 * (1.05 ** t) for t in range(16)]
    res = m.transpose(fin, "finance")
    print("\n" + res.to_dict().__str__())
    assert res.source_domain == "biology" and res.form == "exponential"
    assert res.adopted and res.fit_r2 >= 0.9, res.to_dict()
    print("music→physics style : exponential law transposed biology→finance, VERIFIED ✓")

    # DOMAIN C (oscillation): a heartbeat-like sine — she should NOT transpose the exp law to it.
    osc = [math.sin(0.6 * t) for t in range(20)]
    res2 = m.transpose(osc, "cardiology")
    print("\n" + res2.to_dict().__str__())
    assert not res2.adopted, "an exponential law must not be forced onto an oscillation"
    print("verify-or-abstain   : refused to transpose a law that did not fit (abstained) ✓")

    # cross-domain nearest: the oscillation's best analogue is another oscillation, not the exp curves
    m.learn("market_cycle", "economics", [math.sin(0.5 * t + 0.3) for t in range(20)], form="sine")
    iso = m.find_isomorphisms(osc, target_domain="cardiology")
    print(f"\nnearest analogue    : {iso[0][0].label} ({iso[0][0].domain}) sim={iso[0][1]:.3f}")
    assert iso[0][0].domain == "economics", "the oscillation should match the sine analogue"
    print("universal pattern   : oscillation matched oscillation across domains ✓")

    print("\nALL SELF-TESTS PASSED ✓")
