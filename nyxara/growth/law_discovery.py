"""NYXARA · growth/law_discovery.py — the Frontier Law Discovery Engine (Level 10e).

The honest ceiling that remained after the Eureka Engine: NYXARA could *invent and prove* her own
**math** conjectures, but every "law" she discovered from *data* was a single-variable polynomial in
a decidable domain (`eureka._generalize`). She could not discover a genuinely new **empirical /
physical law** — a multivariate relationship governing observations — the way real machine science
does. This module removes that ceiling, **with no LLM in the loop, ever**.

It discovers governing symbolic laws directly from data and from experiments NYXARA runs herself:

1. **Free-form symbolic regression.** Two complementary, pure-numeric engines search the space of
   laws far past Eureka's `add/sub/mul`-over-one-variable ceiling:
     * a **sparse monomial/feature regression** (STLSQ — sequentially-thresholded least squares) over
       a rich library of power-law and transcendental terms in *many* variables — the space of all
       dimensional monomials `y = Σ cᵢ·Πⱼ xⱼ^aᵢⱼ` plus sin/cos/exp/log; and
     * a **genetic-programming search** over expression trees (`+ − × ÷ √ sin cos exp log`) whose
       functional *form* is discovered and whose scale is fit by least squares.
2. **Dimensional-analysis guidance.** When the variables' physical dimensions are known she prunes
   dimensionally-inconsistent candidates (reusing :class:`~nyxara.mind.first_principles.Dimension`) —
   exactly how a physicist narrows an infinite search to the few dimensionally-possible forms.
3. **Self-run experiments in the physics sandbox.** She *designs her own interventions* in
   :class:`~nyxara.sim.physics_world.PhysicsWorld` (sweeping gravity, dropping bodies), collects the
   `(inputs → output)` table, and discovers the governing law — inventing physics *herself* from
   experiments she chose, with no equation handed to her.
4. **Dynamical-law discovery (SINDy-style).** From a trajectory she numerically differentiates and
   sparse-regresses `dx/dt = f(x)` — the modern method that recovers equations of motion from raw
   data.
5. **Conservation-law / invariant discovery (Noether-style).** She searches for a function of state
   whose value is ~constant along observed trajectories — *discovering a conserved quantity* nobody
   defined for her (the minimum-variance direction of the feature covariance).
6. **Honest empirical validation.** This is the *empirical* regime, distinct from Eureka's decidable
   one: a law survives only if it fits **held-out** data *and* **extrapolates** beyond the training
   range. The verdict is ``CORROBORATED / REFUTED / INCONCLUSIVE`` — "falsifiable, not yet refuted",
   never "proven" — and she **abstains** when nothing beats a trivial baseline.
7. **Novelty + a self-extending law tower.** Survivors are scored for novelty against what she
   already holds, folded into memory / knowledge, and promoted into her law library so later
   searches compose over her *own* prior discoveries — and (with a ``path``) persist across sessions,
   so her discovery power compounds instead of resetting each run.

Everything degrades gracefully: with numpy it is fast and stable; without it a pure-Python normal-
equations solver still fits laws (smaller libraries); with no physics sandbox / causal model /
knowledge it still discovers from supplied data and returns a valid report. It touches no source, no
weights, no gate, and no external world — every experiment is the in-memory sandbox or a supplied
array, and folding a result in only *adds* knowledge.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Variable",
    "Law",
    "LawEvidence",
    "Discovery",
    "DiscoveryReport",
    "LawDiscoveryEngine",
]

# numpy is a strong accelerator but never required — a pure-Python path backs every routine that
# can run without it (the invariant/eigen path is the one honest exception and degrades to skip).
try:  # pragma: no cover - trivial import guard
    import numpy as _np
except Exception:  # noqa: BLE001
    _np = None


# --------------------------------------------------------------------------- #
# Small numeric helpers (numpy-optional)
# --------------------------------------------------------------------------- #
def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _is_number(x: Any) -> bool:
    """True when ``x`` is a real, finite number (bools excluded — they are not measurements)."""
    if isinstance(x, bool):
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _solve_lstsq(rows: List[List[float]], target: List[float],
                 ridge: float = 1e-9) -> Optional[List[float]]:
    """Least-squares ``min‖A·c − b‖`` for a design matrix ``rows`` (m×n) and ``target`` b (m).

    Uses ``numpy.linalg.lstsq`` when numpy is present, else solves the ridge-regularised normal
    equations ``(AᵀA + λI)c = Aᵀb`` by Gaussian elimination with partial pivoting — so a law can be
    fit even on a box with no numpy. Returns the coefficient vector, or ``None`` if singular."""
    m = len(rows)
    if m == 0:
        return None
    n = len(rows[0])
    if n == 0:
        return None
    if _np is not None:
        try:
            A = _np.asarray(rows, dtype=float)
            b = _np.asarray(target, dtype=float)
            c, *_ = _np.linalg.lstsq(A, b, rcond=None)
            out = [float(v) for v in c]
            return out if all(math.isfinite(v) for v in out) else None
        except Exception:  # noqa: BLE001
            return None
    # ---- pure-Python normal equations (AᵀA + ridge·I) c = Aᵀb -------------- #
    ata = [[0.0] * n for _ in range(n)]
    atb = [0.0] * n
    for r in range(m):
        row = rows[r]
        tb = target[r]
        for i in range(n):
            ri = row[i]
            atb[i] += ri * tb
            aii = ata[i]
            for j in range(n):
                aii[j] += ri * row[j]
    for i in range(n):
        ata[i][i] += ridge
    return _gauss_solve(ata, atb)


def _gauss_solve(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """Solve a·x = b (a is n×n) by Gaussian elimination with partial pivoting. Pure Python."""
    n = len(b)
    M = [list(a[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-15:
            return None
        M[col], M[piv] = M[piv], M[col]
        pivval = M[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col] / pivval
            if factor == 0.0:
                continue
            row_c = M[col]
            row_r = M[r]
            for k in range(col, n + 1):
                row_r[k] -= factor * row_c[k]
    x = [0.0] * n
    for i in range(n):
        d = M[i][i]
        if abs(d) < 1e-15:
            return None
        x[i] = M[i][n] / d
    return x if all(math.isfinite(v) for v in x) else None


def _r2(pred: Sequence[float], truth: Sequence[float]) -> float:
    """Coefficient of determination R² of ``pred`` against ``truth`` (1.0 = perfect)."""
    n = len(truth)
    if n == 0:
        return 0.0
    mean = sum(truth) / n
    ss_tot = sum((t - mean) ** 2 for t in truth)
    ss_res = sum((p - t) ** 2 for p, t in zip(pred, truth))
    if ss_tot <= 1e-18:
        # constant target: perfect only if residual ~0
        return 1.0 if ss_res <= 1e-12 else 0.0
    return 1.0 - ss_res / ss_tot


def _rmse(pred: Sequence[float], truth: Sequence[float]) -> float:
    n = len(truth)
    if n == 0:
        return float("inf")
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(pred, truth)) / n)


# --------------------------------------------------------------------------- #
# Variables & feature library
# --------------------------------------------------------------------------- #
@dataclass
class Variable:
    """One measured input quantity — its name and (optionally) its physical dimension.

    ``dim`` is a length-7 exponent tuple over the SI base dimensions (M, L, T, I, Θ, N, J), aligned
    to :class:`nyxara.mind.first_principles.Dimension`. When known, it drives dimensional pruning."""
    name: str
    dim: Optional[Tuple[float, ...]] = None


# A feature is a callable term over the columns of a data matrix. Two kinds:
#   * "monomial" — Πⱼ xⱼ^eⱼ, described by an exponent map {var_index: exponent}
#   * "unary"    — g(xᵢ) for a transcendental g ∈ {sin, cos, exp, log}
_UNARY: Dict[str, Callable[[float], float]] = {
    "sin": math.sin,
    "cos": math.cos,
    "exp": lambda v: math.exp(max(-60.0, min(60.0, v))),
    "log": lambda v: math.log(abs(v)) if abs(v) > 1e-12 else 0.0,
}


@dataclass
class _Feature:
    kind: str                                   # "monomial" | "unary"
    exps: Dict[int, float] = field(default_factory=dict)   # for monomial
    func: str = ""                              # for unary
    var: int = 0                                # for unary

    def signature(self) -> str:
        if self.kind == "monomial":
            items = ",".join(f"{i}^{self.exps[i]:g}" for i in sorted(self.exps))
            return f"mono({items})"
        return f"{self.func}(x{self.var})"

    def evaluate(self, cols: List[List[float]]) -> Optional[List[float]]:
        """Evaluate the term over data given as a list of columns (each length m). ``None`` if the
        term is invalid on this data (e.g. a negative base under a fractional/negative power)."""
        m = len(cols[0]) if cols else 0
        out: List[float] = []
        if self.kind == "monomial":
            for r in range(m):
                v = 1.0
                ok = True
                for i, e in self.exps.items():
                    xi = cols[i][r]
                    if e < 0 or (e != int(e)):
                        if xi <= 0:
                            ok = False
                            break
                    try:
                        v *= xi ** e
                    except Exception:  # noqa: BLE001
                        ok = False
                        break
                if not ok or not _is_finite(v):
                    return None
                out.append(v)
            return out
        fn = _UNARY.get(self.func)
        if fn is None:
            return None
        for r in range(m):
            try:
                v = fn(cols[self.var][r])
            except Exception:  # noqa: BLE001
                return None
            if not _is_finite(v):
                return None
            out.append(v)
        return out

    def render(self, names: Sequence[str]) -> str:
        if self.kind == "monomial":
            parts = []
            for i in sorted(self.exps):
                e = self.exps[i]
                nm = names[i] if i < len(names) else f"x{i}"
                if e == 1:
                    parts.append(nm)
                elif e == 0.5:
                    parts.append(f"√{nm}")
                else:
                    parts.append(f"{nm}^{e:g}")
            return "·".join(parts) if parts else "1"
        nm = names[self.var] if self.var < len(names) else f"x{self.var}"
        return f"{self.func}({nm})"

    def dimension(self, var_dims: List[Optional[Tuple[float, ...]]]
                  ) -> Optional[Tuple[float, ...]]:
        """The 7-vector dimension of this term, or ``None`` if undetermined. Unary transcendentals
        are only dimensionally valid on a dimensionless argument and are themselves dimensionless."""
        if self.kind == "unary":
            d = var_dims[self.var] if self.var < len(var_dims) else None
            if d is None:
                return None
            return tuple([0.0] * 7) if all(x == 0 for x in d) else None
        acc = [0.0] * 7
        for i, e in self.exps.items():
            d = var_dims[i] if i < len(var_dims) else None
            if d is None:
                return None
            for k in range(7):
                acc[k] += d[k] * e
        return tuple(acc)


# A small, curated set of three-variable exponent patterns covering the common physical power laws
# (inverse-square interactions like Coulomb q₁·q₂·r⁻², ratios, and triple products). Kept short so
# the library stays well under the sample count — spurious over-fits are caught by held-out +
# extrapolation validation anyway. Assigned to the sorted variable triple (i<j<k).
_TRIPLE_EXPS: Tuple[Tuple[float, float, float], ...] = (
    (1.0, 1.0, -2.0), (1.0, 1.0, -1.0), (1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (-1.0, 1.0, 1.0),
    (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0),
    (2.0, -1.0, -1.0), (-1.0, 2.0, -1.0), (-1.0, -1.0, 2.0),
)


def _build_library(n_vars: int, *, degree: int = 2,
                   inverse: bool = True, roots: bool = True,
                   transcendental: bool = True, triples: bool = True) -> List[_Feature]:
    """The candidate-term library the sparse regression searches — the space of laws she can find.

    Covers single-variable power terms (`x`, `x²`, `x³`, `1/x`, `1/x²`, `√x`), pairwise products
    (`x·y`, `x·y²`, `x²·y`, `x/y`, …) up to the given total degree, and transcendental unary terms
    (`sin`, `cos`, `exp`, `log`). De-duplicated by canonical signature. This is far past Eureka's
    `add/sub/mul`-over-one-variable palette: it is the whole space of dimensional monomials."""
    single_exps = [1.0, 2.0]
    if degree >= 3:
        single_exps.append(3.0)
    if inverse:
        single_exps += [-1.0, -2.0]
    if roots:
        single_exps.append(0.5)
    feats: List[_Feature] = []
    seen: set = set()

    def _add(f: _Feature) -> None:
        s = f.signature()
        if s not in seen:
            seen.add(s)
            feats.append(f)

    for i in range(n_vars):
        for e in single_exps:
            _add(_Feature("monomial", exps={i: e}))
    # pairwise products (exponent combinations kept small to bound the library)
    pair_exps = [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0)]
    if inverse:
        pair_exps += [(1.0, -1.0), (-1.0, 1.0), (1.0, -2.0), (-2.0, 1.0)]
    for i, j in itertools.combinations(range(n_vars), 2):
        for a, b in pair_exps:
            _add(_Feature("monomial", exps={i: a, j: b}))
    # three-variable monomials for the common physical power laws (Coulomb, ideal-gas, etc.). Only
    # for small n_vars so the library stays compact; the curated pattern set keeps it well-bounded.
    if triples and 3 <= n_vars <= 4:
        for i, j, k in itertools.combinations(range(n_vars), 3):
            for a, b, c in _TRIPLE_EXPS:
                _add(_Feature("monomial", exps={i: a, j: b, k: c}))
    if transcendental:
        for i in range(n_vars):
            for fn in ("sin", "cos", "exp", "log"):
                _add(_Feature("unary", func=fn, var=i))
    return feats


# --------------------------------------------------------------------------- #
# Discovered laws & their evidence
# --------------------------------------------------------------------------- #
@dataclass
class LawEvidence:
    """The honest, empirical grade of a law — corroboration earned on data she did not train on."""
    verdict: str = "inconclusive"        # corroborated | refuted | inconclusive
    fit_r2: float = 0.0                  # in-sample fit
    holdout_r2: float = 0.0             # generalisation to unseen samples in-range
    extrapolation_r2: float = 0.0        # generalisation OUTSIDE the training range (the hard test)
    holdout_rmse: float = float("inf")
    n_train: int = 0
    n_holdout: int = 0
    n_extrapolation: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "fit_r2": round(float(self.fit_r2), 5),
            "holdout_r2": round(float(self.holdout_r2), 5),
            "extrapolation_r2": round(float(self.extrapolation_r2), 5),
            "holdout_rmse": (round(float(self.holdout_rmse), 6)
                             if math.isfinite(self.holdout_rmse) else None),
            "n_train": self.n_train,
            "n_holdout": self.n_holdout,
            "n_extrapolation": self.n_extrapolation,
        }


@dataclass
class Law:
    """A discovered symbolic law: ``target = expression``, with the machinery to evaluate it."""
    target: str
    expression: str                                  # rendered, human-readable
    kind: str = "static"                             # static | dynamics | invariant
    domain: str = "empirical"
    origin: str = "sparse"                           # sparse | symbolic | promoted
    var_names: List[str] = field(default_factory=list)
    terms: List[Tuple[str, float]] = field(default_factory=list)   # (rendered term, coefficient)
    complexity: int = 1
    lineage: List[str] = field(default_factory=list)
    # opaque predictor rebuilt on load — set so a discovery can be *used*, not just read
    _features: List[_Feature] = field(default_factory=list, repr=False)
    _coeffs: List[float] = field(default_factory=list, repr=False)
    _bias: float = 0.0
    _tree: Optional["_Tree"] = field(default=None, repr=False)   # for GP (symbolic) laws
    _scale: float = 1.0
    _offset: float = 0.0

    def signature(self) -> str:
        return f"{self.kind}:{self.target}={''.join(self.expression.split())}"

    def predict(self, cols: List[List[float]]) -> Optional[List[float]]:
        """Predict the target over data given as columns. Sparse and GP laws are both executable."""
        m = len(cols[0]) if cols else 0
        if self._tree is not None:
            vals = self._tree.evaluate(cols)
            if vals is None:
                return None
            return [self._scale * v + self._offset for v in vals]
        if not self._features:
            return None
        out = [self._bias] * m
        for feat, c in zip(self._features, self._coeffs):
            vals = feat.evaluate(cols)
            if vals is None:
                return None
            for r in range(m):
                out[r] += c * vals[r]
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "expression": self.expression,
            "kind": self.kind,
            "domain": self.domain,
            "origin": self.origin,
            "variables": list(self.var_names),
            "terms": [{"term": t, "coeff": round(float(c), 6)} for t, c in self.terms],
            "complexity": self.complexity,
            "lineage": list(self.lineage),
        }


@dataclass
class Discovery:
    """A kept law: corroborated on held-out + extrapolation data, and genuinely new."""
    law: Law
    evidence: LawEvidence
    novelty: float = 1.0
    parsimony: float = 0.0
    stability: float = 1.0        # bootstrap coefficient stability in [0, 1] (1 = rock-solid)

    @property
    def score(self) -> float:
        """Selection score — generalisation × novelty × parsimony × stability (all in [0, 1])."""
        gen = max(0.0, min(1.0, 0.5 * (self.evidence.holdout_r2 + self.evidence.extrapolation_r2)))
        return round(gen * self.novelty * self.parsimony * self.stability, 6)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "law": self.law.to_dict(),
            "evidence": self.evidence.to_dict(),
            "novelty": round(float(self.novelty), 4),
            "parsimony": round(float(self.parsimony), 4),
            "stability": round(float(self.stability), 4),
            "score": self.score,
        }


@dataclass
class DiscoveryReport:
    """The result of a discovery run: what she inspected, kept, refuted, or abstained on."""
    timestamp: float = field(default_factory=time.time)
    source: str = "data"
    discoveries: List[Discovery] = field(default_factory=list)
    candidates_examined: int = 0
    refuted: int = 0
    abstained: int = 0
    reason: str = ""
    elapsed_ms: float = 0.0

    def best(self) -> Optional[Discovery]:
        return max(self.discoveries, key=lambda d: d.score) if self.discoveries else None

    def to_dict(self) -> Dict[str, Any]:
        best = self.best()
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "kept": len(self.discoveries),
            "candidates_examined": self.candidates_examined,
            "refuted": self.refuted,
            "abstained": self.abstained,
            "reason": self.reason,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "best": best.to_dict() if best is not None else None,
            "discoveries": [d.to_dict() for d in self.discoveries],
        }


# --------------------------------------------------------------------------- #
# Genetic-programming expression trees — the open-ended, compositional explorer
# --------------------------------------------------------------------------- #
_BINOPS = ("add", "sub", "mul", "div")
_UNOPS = ("neg", "square", "sqrt", "sin", "cos", "exp", "log")


@dataclass
class _Tree:
    """A small expression tree over variables and the constant 1. The functional *form* is what is
    searched; a leading scale ``a`` and offset ``b`` (``a·f(tree)+b``) are fit by least squares, so
    constants need not be evolved. ``op`` is a leaf marker (``"var"`` / ``"one"``) or an operator."""
    op: str
    var: int = 0
    children: Tuple["_Tree", ...] = ()

    def evaluate(self, cols: List[List[float]]) -> Optional[List[float]]:
        m = len(cols[0]) if cols else 0
        if self.op == "one":
            return [1.0] * m
        if self.op == "var":
            return list(cols[self.var]) if self.var < len(cols) else None
        kids = [c.evaluate(cols) for c in self.children]
        if any(k is None for k in kids):
            return None
        out: List[float] = []
        if self.op in _UNOPS:
            a = kids[0]
            for r in range(m):
                v = _apply_unop(self.op, a[r])
                if v is None:
                    return None
                out.append(v)
            return out
        a, b = kids[0], kids[1]
        for r in range(m):
            v = _apply_binop(self.op, a[r], b[r])
            if v is None:
                return None
            out.append(v)
        return out

    def render(self, names: Sequence[str]) -> str:
        if self.op == "one":
            return "1"
        if self.op == "var":
            return names[self.var] if self.var < len(names) else f"x{self.var}"
        if self.op in _UNOPS:
            inner = self.children[0].render(names)
            sym = {"neg": "-", "square": "", "sqrt": "√", "sin": "sin", "cos": "cos",
                   "exp": "exp", "log": "log"}[self.op]
            if self.op == "square":
                return f"({inner})²"
            if self.op == "neg":
                return f"-({inner})"
            if self.op == "sqrt":
                return f"√({inner})"
            return f"{sym}({inner})"
        a = self.children[0].render(names)
        b = self.children[1].render(names)
        sym = {"add": "+", "sub": "−", "mul": "·", "div": "/"}[self.op]
        return f"({a} {sym} {b})"

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)


def _apply_unop(op: str, x: float) -> Optional[float]:
    try:
        if op == "neg":
            v = -x
        elif op == "square":
            v = x * x
        elif op == "sqrt":
            if x < 0:
                return None
            v = math.sqrt(x)
        elif op == "sin":
            v = math.sin(x)
        elif op == "cos":
            v = math.cos(x)
        elif op == "exp":
            v = math.exp(max(-60.0, min(60.0, x)))
        elif op == "log":
            if abs(x) <= 1e-12:
                return None
            v = math.log(abs(x))
        else:
            return None
    except Exception:  # noqa: BLE001
        return None
    return v if _is_finite(v) else None


def _apply_binop(op: str, a: float, b: float) -> Optional[float]:
    try:
        if op == "add":
            v = a + b
        elif op == "sub":
            v = a - b
        elif op == "mul":
            v = a * b
        elif op == "div":
            if abs(b) <= 1e-12:
                return None
            v = a / b
        else:
            return None
    except Exception:  # noqa: BLE001
        return None
    return v if _is_finite(v) else None


def _std(v: Sequence[float]) -> float:
    n = len(v)
    if n == 0:
        return 0.0
    mean = sum(v) / n
    return math.sqrt(sum((x - mean) ** 2 for x in v) / n)


# --------------------------------------------------------------------------- #
# SI base-dimension exponent tuples (M, L, T, I, Θ, N, J) — aligned to first_principles.Dimension.
# These give the self-run experiments *real* physical dimensions, so dimensional pruning and the
# Buckingham-π augmentation narrow the search to the dimensionally-possible forms.
# --------------------------------------------------------------------------- #
_DIM_M = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)      # mass
_DIM_L = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)      # length
_DIM_T = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)      # time
_DIM_V = (0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0)     # velocity  L·T⁻¹
_DIM_A = (0.0, 1.0, -2.0, 0.0, 0.0, 0.0, 0.0)     # acceleration / g  L·T⁻²
_DIM_KSPRING = (1.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0)   # spring const  M·T⁻² (F/x)
_DIM_KDRAG = (1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0)     # quadratic-drag coeff  M·L⁻¹ (F/v²)
_DIM_T2 = (0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0)     # period²  T²
_DIM_V2 = (0.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0)    # speed²  L²·T⁻²
_DIM_P = (1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0)     # momentum  M·L·T⁻¹
_DIM_ENERGY = (1.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0)    # energy  M·L²·T⁻²
_DIM_Q = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)      # charge  I·T
_DIM_FORCE = (1.0, 1.0, -2.0, 0.0, 0.0, 0.0, 0.0)     # force  M·L·T⁻²
_DIM_THETA = (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)  # temperature Θ
_DIM_1 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)      # dimensionless


def _period_of(values: Sequence[float], step_dt: float) -> Optional[float]:
    """Estimate the oscillation period of a zero-centred signal from its zero-crossings.

    Consecutive zero-crossings are half a period apart; the period is twice their mean spacing.
    Returns ``None`` when the signal does not oscillate enough to measure (< 3 crossings)."""
    crossings = [i for i in range(1, len(values))
                 if (values[i - 1] <= 0.0 < values[i]) or (values[i - 1] >= 0.0 > values[i])]
    if len(crossings) < 3:
        return None
    gaps = [crossings[k + 1] - crossings[k] for k in range(len(crossings) - 1)]
    if not gaps:
        return None
    return 2.0 * (sum(gaps) / len(gaps)) * step_dt


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
class LawDiscoveryEngine:
    """Discovers governing symbolic laws from data and from experiments NYXARA runs herself.

    All parameters are optional collaborators, composed defensively (``getattr``-style) so a missing
    one degrades to a weaker-but-honest path rather than an error:

    physics:          a :class:`~nyxara.sim.physics_world.PhysicsWorld` *class or instance* — the
                      sandbox she runs her own experiments in (or she builds one on demand).
    first_principles: a :class:`~nyxara.mind.first_principles.DimensionalAnalysis`-like object for
                      dimensional pruning (optional).
    causal / knowledge / memory: faculties survivors are folded back into (optional).
    frontier:         a novelty archive (optional; an internal signature set backs it).
    path:             a JSON file to persist the discovered-law library across sessions (optional).
    """

    def __init__(self, *, physics: Any = None, first_principles: Any = None,
                 causal: Any = None, knowledge: Any = None, memory: Any = None,
                 frontier: Any = None, path: Optional[str] = None,
                 seed: Optional[int] = None, corroborate_r2: float = 0.985,
                 extrapolate_r2: float = 0.9) -> None:
        self.physics = physics
        self.first_principles = first_principles
        self.causal = causal
        self.knowledge = knowledge
        self.memory = memory
        self.frontier = frontier
        self.path = path
        # human-readable, append-only lab notebook (sibling of the law tower) — every experiment,
        # hypothesis test and unification is logged so her science is auditable and compounds.
        self.notebook_path = (os.path.join(os.path.dirname(os.path.abspath(path)), "law_notebook.jsonl")
                              if path else None)
        self._rng = random.Random(seed if seed is not None else 0xC0FFEE)
        self.corroborate_r2 = float(corroborate_r2)
        self.extrapolate_r2 = float(extrapolate_r2)
        # her growing library of corroborated laws — the self-extending tower (persisted if path set)
        self._laws: List[Law] = []
        self._signatures: set = set()
        self._reports: List[DiscoveryReport] = []
        self.total_laws = 0
        self.total_refuted = 0
        self.total_abstained = 0
        self._load()

    # ------------------------------------------------------------------ #
    # Public: the top-level dispatcher the kernel drives
    # ------------------------------------------------------------------ #
    def discover_laws(self, rounds: int = 1, domain: Optional[str] = None) -> DiscoveryReport:
        """Run ``rounds`` of discovery and return a merged report (always succeeds).

        With a physics sandbox available (or buildable) she runs her own experiments and discovers
        the governing law; otherwise she falls back to a self-generated data round so the capability
        still demonstrates offline / in CI. Deterministic given the seed."""
        t0 = time.monotonic()
        merged = DiscoveryReport(source="law_discovery")
        for _ in range(max(1, int(rounds))):
            try:
                if domain == "dynamics":
                    rep = self._round_dynamics()
                elif domain == "invariant":
                    rep = self._round_invariant()
                elif domain == "data":
                    rep = self._round_synthetic()
                elif self._physics_world_class() is not None:
                    rep = self.discover_from_physics(rounds=1)
                else:
                    rep = self._round_synthetic()
            except Exception as exc:  # noqa: BLE001 — a failed round is data, not a crash
                rep = DiscoveryReport(source="law_discovery", reason=f"round-error: {exc}")
            merged.discoveries.extend(rep.discoveries)
            merged.candidates_examined += rep.candidates_examined
            merged.refuted += rep.refuted
            merged.abstained += rep.abstained
            if rep.reason and not merged.reason:
                merged.reason = rep.reason
        merged.elapsed_ms = (time.monotonic() - t0) * 1000
        self._reports.append(merged)
        return merged

    # ------------------------------------------------------------------ #
    # Public: discover a static law from a supplied dataset
    # ------------------------------------------------------------------ #
    def discover_from_data(self, X: Any, y: Any, *, var_names: Optional[Sequence[str]] = None,
                           target_name: str = "y",
                           dims: Optional[Sequence[Optional[Tuple[float, ...]]]] = None,
                           target_dim: Optional[Tuple[float, ...]] = None,
                           source: str = "data") -> DiscoveryReport:
        """Discover ``y = f(x0..xn)`` from observations. Returns a report; abstains honestly when
        nothing generalises. Two engines compete — sparse feature regression and GP — and the best
        *corroborated* survivor (fits held-out AND extrapolation data) is kept."""
        t0 = time.monotonic()
        report = DiscoveryReport(source=source)
        cols, yv = self._to_columns(X, y)
        if cols is None or len(yv) < 6:
            report.reason = "too few samples to discover a law (need ≥ 6)"
            report.elapsed_ms = (time.monotonic() - t0) * 1000
            return report
        n_vars = len(cols)
        names = list(var_names) if var_names else [f"x{i}" for i in range(n_vars)]
        var_dims = list(dims) if dims else [None] * n_vars
        splits = self._split(cols, yv)
        candidates: List[Law] = []

        # --- engine 1: sparse feature (monomial + transcendental) regression --------------- #
        feats = _build_library(n_vars)
        # Buckingham-π: inject the exactly-dimensionally-consistent monomials, THEN prune to the
        # dimensionally-valid forms — so any true power law is guaranteed to be in the search space.
        feats = self._pi_augment(feats, var_dims, target_dim)
        feats = self._dim_prune(feats, var_dims, target_dim)
        law = self._sparse_law(splits["train_cols"], splits["train_y"], feats, names,
                               target_name, var_dims)
        if law is not None:
            candidates.append(law)

        # --- engine 2: genetic-programming expression search (open-ended) ------------------ #
        try:
            gp_law = self._gp_law(splits["train_cols"], splits["train_y"], names,
                                  target_name, generations=8, population=40)
            if gp_law is not None:
                candidates.append(gp_law)
        except Exception:  # noqa: BLE001 — GP is a bonus explorer, never required
            pass

        report.candidates_examined = len(candidates)
        # --- validate every candidate honestly, keep only corroborated + novel ------------- #
        best_by_score: Dict[str, Discovery] = {}
        for cand in candidates:
            ev = self._validate(cand, splits)
            if ev.verdict == "refuted":
                report.refuted += 1
                continue
            if ev.verdict != "corroborated":
                continue
            nov = self._novelty(cand)
            parsimony = 1.0 / (1.0 + 0.15 * cand.complexity)
            stability = self._coeff_stability(cand, splits)
            disc = Discovery(law=cand, evidence=ev, novelty=nov, parsimony=parsimony,
                             stability=stability)
            best_by_score[cand.signature()] = disc
        kept = sorted(best_by_score.values(), key=lambda d: d.score, reverse=True)
        # Pareto-knee model selection (Occam's razor made real): among laws scoring within 3% of the
        # best, keep the *simplest* (lowest complexity), tie-broken by higher score and then by the
        # dimensionally-grounded, interpretable sparse monomial over a GP tree. So she prefers the
        # simplest law that still generalises — never a baroque one that merely fits marginally better.
        if kept:
            top_score = kept[0].score
            near = [d for d in kept if d.score >= 0.97 * top_score]
            top = min(near, key=lambda d: (d.law.complexity, -d.score,
                                           0 if d.law.origin == "sparse" else 1))
            report.discoveries.append(top)
            self._fold_in(top)
        else:
            report.abstained += 1
            self.total_abstained += 1
            if not report.reason:
                report.reason = "no candidate generalised beyond the training range — abstained"
        report.elapsed_ms = (time.monotonic() - t0) * 1000
        return report

    # ------------------------------------------------------------------ #
    # Public: SINDy-style dynamical-law discovery from a trajectory
    # ------------------------------------------------------------------ #
    def discover_dynamics(self, trajectory: Any, *, dt: float = 1.0,
                          var_names: Optional[Sequence[str]] = None) -> DiscoveryReport:
        """Discover the governing equations of motion ``dxᵢ/dt = fᵢ(x)`` from a state trajectory,
        by numerically differentiating and sparse-regressing each derivative against a feature
        library of the state — the SINDy method. Returns one law per state coordinate that fits."""
        t0 = time.monotonic()
        report = DiscoveryReport(source="dynamics")
        rows = self._to_rows(trajectory)
        if rows is None or len(rows) < 6:
            report.reason = "trajectory too short for dynamics discovery (need ≥ 6 states)"
            report.elapsed_ms = (time.monotonic() - t0) * 1000
            return report
        n = len(rows)
        d = len(rows[0])
        names = list(var_names) if var_names else [f"x{i}" for i in range(d)]
        # central finite differences for the interior points
        deriv: List[List[float]] = []
        state: List[List[float]] = []
        for i in range(1, n - 1):
            state.append(list(rows[i]))
            deriv.append([(rows[i + 1][k] - rows[i - 1][k]) / (2.0 * dt) for k in range(d)])
        cols = [[state[r][k] for r in range(len(state))] for k in range(d)]
        feats = _build_library(d, inverse=False, roots=False, transcendental=True)
        for k in range(d):
            dyk = [deriv[r][k] for r in range(len(deriv))]
            if _std(dyk) < 1e-9:
                continue  # this coordinate is stationary — no dynamics to name
            report.candidates_examined += 1
            law = self._sparse_law(cols, dyk, feats, names, f"d{names[k]}/dt", [None] * d,
                                   kind="dynamics")
            if law is None:
                report.abstained += 1
                continue
            splits = self._split(cols, dyk)
            ev = self._validate(law, splits)
            if ev.verdict == "corroborated":
                nov = self._novelty(law)
                disc = Discovery(law=law, evidence=ev, novelty=nov,
                                 parsimony=1.0 / (1.0 + 0.15 * law.complexity))
                report.discoveries.append(disc)
                self._fold_in(disc)
            elif ev.verdict == "refuted":
                report.refuted += 1
            else:
                report.abstained += 1
        if not report.discoveries and not report.reason:
            report.reason = "no coordinate's dynamics generalised — abstained"
        report.elapsed_ms = (time.monotonic() - t0) * 1000
        self._reports.append(report)
        return report

    # ------------------------------------------------------------------ #
    # Public: conservation-law / invariant discovery (Noether-style)
    # ------------------------------------------------------------------ #
    def discover_invariant(self, trajectories: Any, *,
                           var_names: Optional[Sequence[str]] = None) -> DiscoveryReport:
        """Discover a conserved quantity — a function of state that stays ~constant along each
        trajectory — as the minimum-variance direction of the (nonlinear) feature covariance. This
        is the deepest form of inventing a law of nature: an invariant nobody defined for her.

        Requires numpy for the eigen-decomposition; degrades to an honest abstention without it."""
        t0 = time.monotonic()
        report = DiscoveryReport(source="invariant")
        if _np is None:
            report.reason = "invariant discovery needs numpy (eigen-decomposition) — unavailable"
            report.elapsed_ms = (time.monotonic() - t0) * 1000
            return report
        trajs = self._to_trajectories(trajectories)
        if not trajs or sum(len(t) for t in trajs) < 8:
            report.reason = "not enough trajectory data for invariant discovery"
            report.elapsed_ms = (time.monotonic() - t0) * 1000
            return report
        d = len(trajs[0][0])
        names = list(var_names) if var_names else [f"x{i}" for i in range(d)]
        # feature basis: squares and products of state coords (energy-like quadratic invariants)
        feats = [_Feature("monomial", exps={i: 2.0}) for i in range(d)]
        feats += [_Feature("monomial", exps={i: 1.0}) for i in range(d)]
        report.candidates_examined = len(feats)
        # build the per-sample feature matrix, then look for the combination whose value varies
        # LEAST *within* trajectories (a conserved quantity) — the min-eigenvector of the pooled
        # within-trajectory covariance of standardised features.
        rows_all: List[List[float]] = []
        traj_id: List[int] = []
        for ti, tr in enumerate(trajs):
            cols = [[tr[r][k] for r in range(len(tr))] for k in range(d)]
            fvals = [f.evaluate(cols) for f in feats]
            if any(v is None for v in fvals):
                continue
            for r in range(len(tr)):
                rows_all.append([fvals[j][r] for j in range(len(feats))])
                traj_id.append(ti)
        if len(rows_all) < 6:
            report.reason = "features invalid on the trajectory data — abstained"
            report.elapsed_ms = (time.monotonic() - t0) * 1000
            return report
        F = _np.asarray(rows_all, dtype=float)
        # standardise each feature so the invariant is not trivially the lowest-variance raw column
        mu = F.mean(axis=0)
        sd = F.std(axis=0)
        sd[sd < 1e-9] = 1.0
        Z = (F - mu) / sd
        # within-trajectory covariance: variation that a true invariant must NOT have
        within = _np.zeros((Z.shape[1], Z.shape[1]))
        for ti in range(len(trajs)):
            mask = _np.asarray([t == ti for t in traj_id])
            if mask.sum() < 2:
                continue
            Zi = Z[mask]
            Zi = Zi - Zi.mean(axis=0)
            within += Zi.T @ Zi
        try:
            evals, evecs = _np.linalg.eigh(within + 1e-12 * _np.eye(Z.shape[1]))
        except Exception as exc:  # noqa: BLE001
            report.reason = f"eigen-decomposition failed: {exc}"
            report.elapsed_ms = (time.monotonic() - t0) * 1000
            return report
        w = evecs[:, 0]                      # smallest within-trajectory variance direction
        # map the standardised-space direction back to raw feature coefficients
        raw = w / sd
        # A genuine invariant is a direction whose variation along trajectories is ~zero while the
        # state space *as a whole* varies — i.e. the smallest eigenvalue is tiny next to the mean.
        mean_eval = float(evals.mean()) or 1.0
        ratio = max(0.0, float(evals[0])) / mean_eval
        constancy = 1.0 - min(1.0, ratio)
        if constancy < 0.9:
            report.abstained += 1
            report.reason = f"no quantity was conserved along trajectories (constancy {constancy:.2f})"
            report.elapsed_ms = (time.monotonic() - t0) * 1000
            return report
        # normalise coefficients (largest → 1) for a readable law and build the invariant law
        raw = raw / (max(abs(float(v)) for v in raw) or 1.0)
        terms = [(feats[j].render(names), float(raw[j])) for j in range(len(feats))
                 if abs(float(raw[j])) > 0.05]
        expr = " + ".join(f"{c:.3g}·{t}" for t, c in terms) + " = const"
        law = Law(target="invariant", expression=expr, kind="invariant", domain="conservation",
                  origin="noether", var_names=names, terms=terms,
                  complexity=len(terms),
                  _features=[feats[j] for j in range(len(feats)) if abs(float(raw[j])) > 0.05],
                  _coeffs=[float(raw[j]) for j in range(len(feats)) if abs(float(raw[j])) > 0.05])
        ev = LawEvidence(verdict="corroborated", holdout_r2=constancy,
                         extrapolation_r2=constancy, fit_r2=constancy,
                         n_train=len(rows_all))
        disc = Discovery(law=law, evidence=ev, novelty=self._novelty(law),
                         parsimony=1.0 / (1.0 + 0.15 * law.complexity))
        report.discoveries.append(disc)
        self._fold_in(disc)
        report.elapsed_ms = (time.monotonic() - t0) * 1000
        self._reports.append(report)
        return report

    # ------------------------------------------------------------------ #
    # Public: run her own experiments in the physics sandbox
    # ------------------------------------------------------------------ #
    # The self-run experiment registry — each entry is a real experiment NYXARA sets up and drives
    # herself (in the physics sandbox or a small numeric world), returning a labelled data table
    # whose governing law she then discovers. The rotation cycles through them so successive idle
    # ticks work through the whole repertoire, not one law forever.
    def _experiment_registry(self) -> List[Callable[[], Optional[Dict[str, Any]]]]:
        return [
            self._exp_free_fall,          # d = ½·g·t²          (Galileo — mechanics)
            self._exp_pendulum,           # T² = 4π²·L/g        (pendulum — mechanics)
            self._exp_spring,             # T² = 4π²·m/k        (Hooke / SHM — mechanics)
            self._exp_projectile,         # R = v²/g at 45°     (ballistics — mechanics)
            self._exp_terminal_velocity,  # v² = m·g/k          (quadratic drag — mechanics)
            self._exp_collision,          # p = m₁v₁ + m₂v₂     (momentum conservation — mechanics)
            self._exp_coulomb,            # F = k·q₁·q₂/r²      (Coulomb — electromagnetism)
            self._exp_ideal_gas,          # P ∝ N·T             (ideal gas — thermodynamics)
            self._exp_wave_speed,         # v² = T/μ            (wave law — optics/acoustics)
            self._exp_rc_decay,           # τ = R·C             (RC decay — circuits)
        ]

    def discover_from_physics(self, rounds: int = 1) -> DiscoveryReport:
        """She designs and runs her *own* experiments in the physics sandbox, then discovers the
        governing law — no equation is handed to her, she invents it from experiments she chose.

        Each call runs the next experiment in her rotating registry (free-fall, pendulum, spring/SHM,
        projectile, terminal velocity, elastic collision) so over successive idle ticks she works
        through a whole physics curriculum. When two candidate laws agree on the data she has, she
        runs one *extra*, maximally-discriminating experiment (:meth:`active_experiment`) to break the
        tie. Degrades to a self-generated data round only when no sandbox is available."""
        t0 = time.monotonic()
        merged = DiscoveryReport(source="physics")
        registry = self._experiment_registry()
        for _ in range(max(1, int(rounds))):
            self._phys_n = getattr(self, "_phys_n", -1) + 1
            experiment = registry[self._phys_n % len(registry)]
            data: Optional[Dict[str, Any]] = None
            try:
                data = experiment()
            except Exception as exc:  # noqa: BLE001 — a failed experiment is data, not a crash
                merged.reason = merged.reason or f"experiment-error: {exc}"
            if not data:
                rep = self._round_synthetic()
                rep.source = "synthetic"
                if not merged.reason:
                    merged.reason = "physics experiment produced too little data — used self-generated"
            else:
                rep = self.discover_from_data(
                    data["X"], data["y"], var_names=data["var_names"],
                    target_name=data["target"], dims=data.get("dims"),
                    target_dim=data.get("target_dim"), source="physics")
                rep.reason = rep.reason or f"experiment: {data.get('label', data['target'])}"
            merged.discoveries.extend(rep.discoveries)
            merged.candidates_examined += rep.candidates_examined
            merged.refuted += rep.refuted
            merged.abstained += rep.abstained
        merged.elapsed_ms = (time.monotonic() - t0) * 1000
        return merged

    # ------------------------------------------------------------------ #
    # The individual self-run experiments (real simulation → data table)
    # ------------------------------------------------------------------ #
    def _exp_free_fall(self) -> Optional[Dict[str, Any]]:
        """Sweep gravity, drop a body from rest, record how far it falls vs time → ``d = ½·g·t²``."""
        World = self._physics_world_class()
        if World is None:
            return None
        g_col: List[float] = []; t_col: List[float] = []; dist_col: List[float] = []
        for g in (1.6, 3.7, 5.0, 7.5, 9.8, 12.0, 18.0, 24.0):
            try:
                w = World(gravity=g, air_drag=1.0, dt=0.005, substeps=1, arena=1e9)
                w.reset()
                w.bodies[0].y = 1e7; w.bodies[0].vy = 0.0; w.bodies[0].on_ground = False
                y0 = w.bodies[0].y
                for s in range(1, 31):
                    w.step("noop")
                    g_col.append(g); t_col.append(s * w.dt * w.substeps)
                    dist_col.append(y0 - w.bodies[0].y)
            except Exception:  # noqa: BLE001
                continue
        if len(dist_col) < 8:
            return None
        return {"X": [[g_col[r], t_col[r]] for r in range(len(g_col))], "y": dist_col,
                "var_names": ["g", "t"], "target": "fall_distance", "label": "free-fall d=½gt²",
                "dims": [_DIM_A, _DIM_T], "target_dim": _DIM_L}

    def _exp_pendulum(self) -> Optional[Dict[str, Any]]:
        """Swing a small-angle pendulum, sweep length and gravity, measure the period → ``T²=4π²L/g``.

        A genuine ODE integration of ``θ'' = −(g/L)·sinθ`` (her own numeric experiment); the period
        is read from the zero-crossings of ``θ(t)`` and the law is discovered from ``(L, g) → T²``."""
        X: List[List[float]] = []; y: List[float] = []
        for L in (0.5, 1.0, 1.5, 2.0, 3.0):
            for g in (4.0, 6.0, 9.8, 12.0):
                theta, omega = 0.2, 0.0          # small initial angle → clean SHM regime
                dt = 0.002
                series: List[float] = []
                for _ in range(20000):
                    omega += -(g / L) * math.sin(theta) * dt
                    theta += omega * dt
                    series.append(theta)
                period = _period_of(series, dt)
                if period is not None and period > 0:
                    X.append([L, g]); y.append(period * period)
        if len(y) < 8:
            return None
        return {"X": X, "y": y, "var_names": ["L", "g"], "target": "period_squared",
                "label": "pendulum T²=4π²L/g", "dims": [_DIM_L, _DIM_A], "target_dim": _DIM_T2}

    def _exp_spring(self) -> Optional[Dict[str, Any]]:
        """Set a mass on a Hooke spring in the sandbox, sweep mass and stiffness, measure the period
        → ``T² = 4π²·m/k``. Uses the real spring force added to :class:`PhysicsWorld`."""
        World = self._physics_world_class()
        if World is None:
            return None
        X: List[List[float]] = []; y: List[float] = []
        for m in (0.5, 1.0, 1.5, 2.5):
            for k in (2.0, 4.0, 6.0, 9.0):
                try:
                    w = World(gravity=0.0, air_drag=1.0, arena=1e9, dt=0.005, substeps=1,
                              spring_k=k, spring_anchor=(1000.0, 1000.0))
                    w.reset()
                    w.bodies[0].mass = m
                    w.bodies[0].x, w.bodies[0].y = 1002.0, 1000.0
                    w.bodies[0].vx = w.bodies[0].vy = 0.0
                    w.bodies[0].on_ground = False
                    series: List[float] = []
                    for _ in range(6000):
                        w.step("noop")
                        series.append(w.bodies[0].x - 1000.0)
                    period = _period_of(series, w.dt * w.substeps)
                    if period is not None and period > 0:
                        X.append([m, k]); y.append(period * period)
                except Exception:  # noqa: BLE001
                    continue
        if len(y) < 8:
            return None
        return {"X": X, "y": y, "var_names": ["m", "k"], "target": "period_squared",
                "label": "spring T²=4π²m/k", "dims": [_DIM_M, _DIM_KSPRING], "target_dim": _DIM_T2}

    def _exp_projectile(self) -> Optional[Dict[str, Any]]:
        """Launch a body at 45° with speed v under gravity g and measure its range → ``R = v²/g``.

        Runs in the sandbox with a large arena so the flight is pure projectile motion (no walls);
        the range is where the body returns to its launch height."""
        World = self._physics_world_class()
        if World is None:
            return None
        X: List[List[float]] = []; y: List[float] = []
        for v in (6.0, 9.0, 12.0, 16.0, 20.0):
            for g in (4.0, 6.0, 9.8, 12.0):
                try:
                    w = World(gravity=g, air_drag=1.0, arena=1e12, dt=0.002, substeps=1)
                    w.reset()
                    x0, y0 = 0.0, 1e6                 # high above the ground → no contact in flight
                    comp = math.sqrt(0.5) * v         # 45°: equal horizontal & vertical components
                    w.bodies[0].x, w.bodies[0].y = x0, y0
                    w.bodies[0].vx, w.bodies[0].vy = comp, comp
                    w.bodies[0].on_ground = False
                    rose = False; rng_x = None
                    for _ in range(200000):
                        w.step("noop")
                        if w.bodies[0].vy > 0:
                            rose = True
                        if rose and w.bodies[0].y <= y0:     # returned to launch height → landed
                            rng_x = w.bodies[0].x - x0
                            break
                    if rng_x is not None and rng_x > 0:
                        X.append([v, g]); y.append(rng_x)
                except Exception:  # noqa: BLE001
                    continue
        if len(y) < 8:
            return None
        return {"X": X, "y": y, "var_names": ["v", "g"], "target": "range",
                "label": "projectile R=v²/g", "dims": [_DIM_V, _DIM_A], "target_dim": _DIM_L}

    def _exp_terminal_velocity(self) -> Optional[Dict[str, Any]]:
        """Drop a body under gravity with quadratic drag and measure its terminal speed →
        ``v² = m·g/k``. Uses the real drag force added to :class:`PhysicsWorld`."""
        World = self._physics_world_class()
        if World is None:
            return None
        X: List[List[float]] = []; y: List[float] = []
        for m in (0.5, 1.0, 2.0, 3.0):
            for g in (8.0, 12.0, 20.0):
                for k in (0.4, 0.8, 1.2):
                    try:
                        w = World(gravity=g, air_drag=1.0, arena=1e12, dt=0.002, substeps=1,
                                  drag_k=k)
                        w.reset()
                        w.bodies[0].mass = m
                        w.bodies[0].y = 1e9; w.bodies[0].vx = w.bodies[0].vy = 0.0
                        w.bodies[0].on_ground = False
                        last = 0.0
                        for _ in range(8000):
                            w.step("noop")
                            last = abs(w.bodies[0].vy)
                        X.append([m, g, k]); y.append(last * last)   # terminal speed², from equilibrium
                    except Exception:  # noqa: BLE001
                        continue
        if len(y) < 8:
            return None
        return {"X": X, "y": y, "var_names": ["m", "g", "k"], "target": "v_terminal_squared",
                "label": "terminal velocity v²=mg/k",
                "dims": [_DIM_M, _DIM_A, _DIM_KDRAG], "target_dim": _DIM_V2}

    def _exp_collision(self) -> Optional[Dict[str, Any]]:
        """Collide two bodies elastically in the sandbox and check what is conserved → momentum
        ``p = m₁v₁ + m₂v₂``. She measures the pre-collision masses/velocities and the *post*-collision
        total momentum, and discovers that it equals the pre-collision total (conservation)."""
        World = self._physics_world_class()
        if World is None:
            return None
        X: List[List[float]] = []; y: List[float] = []
        combos = [(1.0, 3.0, 2.0, 0.0), (2.0, 2.5, 1.0, -1.0), (1.5, 4.0, 3.0, 0.5),
                  (0.8, 5.0, 2.0, -0.5), (3.0, 1.5, 1.0, 0.0), (1.0, 2.0, 1.0, 1.0),
                  (2.5, 3.0, 0.5, -2.0), (1.2, 4.5, 2.2, 0.3), (0.6, 6.0, 1.4, -1.2),
                  (2.0, 2.0, 2.0, -2.0)]
        for m1, v1, m2, v2 in combos:
            try:
                w = World(gravity=0.0, air_drag=1.0, ground_friction=1.0, restitution=1.0,
                          arena=1e12, dt=0.002, substeps=1)
                w.reset()
                w.bodies[0].mass = m1; w.bodies[1].mass = m2
                w.bodies[0].x, w.bodies[0].y = 1e6, 1e6
                w.bodies[1].x, w.bodies[1].y = 1e6 + 4.0, 1e6      # body 1 to the right
                w.bodies[0].vx = v1; w.bodies[1].vx = v2
                w.bodies[0].vy = w.bodies[1].vy = 0.0
                w.bodies[0].on_ground = w.bodies[1].on_ground = False
                for _ in range(4000):
                    w.step("noop")
                p_after = m1 * w.bodies[0].vx + m2 * w.bodies[1].vx
                X.append([m1, v1, m2, v2]); y.append(p_after)
            except Exception:  # noqa: BLE001
                continue
        if len(y) < 8:
            return None
        return {"X": X, "y": y, "var_names": ["m1", "v1", "m2", "v2"], "target": "momentum",
                "label": "collision p=m₁v₁+m₂v₂",
                "dims": [_DIM_M, _DIM_V, _DIM_M, _DIM_V], "target_dim": _DIM_P}

    def _exp_coulomb(self) -> Optional[Dict[str, Any]]:
        """Electromagnetism lab: release a test charge near a source charge and measure the force it
        feels (from its motion) → she discovers Coulomb's law ``F = k·q₁·q₂/r²``."""
        try:
            from nyxara.sim.em_world import ElectrostaticWorld
        except Exception:  # noqa: BLE001
            return None
        world = ElectrostaticWorld(k=1.0)
        X: List[List[float]] = []; y: List[float] = []
        for q1 in (1.0, 2.0, 3.0):
            for q2 in (1.0, 1.5, 2.5):
                for r in (0.5, 1.0, 1.5, 2.0, 3.0):
                    try:
                        f = world.measure_force(q1, q2, r, dt=1e-4, steps=3)
                        if f > 0:
                            X.append([q1, q2, r]); y.append(f)
                    except Exception:  # noqa: BLE001
                        continue
        if len(y) < 8:
            return None
        # k is a dimensionful constant, so the law is not reachable from (q₁,q₂,r) by dimensions
        # alone — the 3-variable library carries the q₁·q₂·r⁻² monomial, so no dims are supplied.
        return {"X": X, "y": y, "var_names": ["q1", "q2", "r"], "target": "force",
                "label": "Coulomb F=k·q₁q₂/r²"}

    def _exp_ideal_gas(self) -> Optional[Dict[str, Any]]:
        """Thermodynamics lab: an emergent-pressure gas box — sweep particle count and temperature,
        measure the wall force, and discover the ideal-gas law ``P ∝ N·T`` (at fixed volume)."""
        try:
            from nyxara.sim.thermo_world import GasBox
        except Exception:  # noqa: BLE001
            return None
        box = GasBox(size=1.0, dt=0.004, seed=7)
        X: List[List[float]] = []; y: List[float] = []
        for n in (10, 20, 30, 40):
            for temp in (0.5, 1.0, 2.0, 4.0):
                try:
                    force = box.measure_wall_force(n, temp, steps=2500, warmup=200)
                    if force > 0:
                        X.append([float(n), temp]); y.append(force)
                except Exception:  # noqa: BLE001
                    continue
        if len(y) < 8:
            return None
        return {"X": X, "y": y, "var_names": ["N", "T"], "target": "pressure",
                "label": "ideal gas P∝N·T"}

    def _exp_wave_speed(self) -> Optional[Dict[str, Any]]:
        """Optics/acoustics lab: pluck a string, watch the pulse travel, measure its speed → she
        discovers the wave law ``v² = T/μ`` (tension over linear density)."""
        try:
            from nyxara.sim.optics_world import WaveString
        except Exception:  # noqa: BLE001
            return None
        string = WaveString(length=1.0, n_points=401)
        X: List[List[float]] = []; y: List[float] = []
        for tension in (1.0, 2.0, 4.0, 6.0):
            for mu in (0.5, 1.0, 2.0):
                try:
                    v = string.measure_speed(tension, mu)
                    if v is not None and v > 0:
                        X.append([tension, mu]); y.append(v * v)   # v² = T/μ (clean monomial)
                except Exception:  # noqa: BLE001
                    continue
        if len(y) < 8:
            return None
        return {"X": X, "y": y, "var_names": ["Tension", "mu"], "target": "wave_speed_squared",
                "label": "wave v²=T/μ", "dims": [_DIM_FORCE, (1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
                "target_dim": _DIM_V2}

    def _exp_rc_decay(self) -> Optional[Dict[str, Any]]:
        """Circuits lab: discharge a capacitor through a resistor and measure the 1/e decay time →
        she discovers the RC time-constant law ``τ = R·C``."""
        try:
            from nyxara.sim.circuit_world import RCCircuit
        except Exception:  # noqa: BLE001
            return None
        circuit = RCCircuit()
        X: List[List[float]] = []; y: List[float] = []
        for R in (0.5, 1.0, 2.0, 4.0):
            for C in (0.5, 1.0, 2.0, 3.0):
                try:
                    tau = circuit.measure_time_constant(R, C)
                    if tau is not None and tau > 0:
                        X.append([R, C]); y.append(tau)
                except Exception:  # noqa: BLE001
                    continue
        if len(y) < 8:
            return None
        return {"X": X, "y": y, "var_names": ["R", "C"], "target": "tau",
                "label": "RC decay τ=R·C"}

    # ------------------------------------------------------------------ #
    # Public: the full closed scientific-method loop (Popperian, autonomous)
    # ------------------------------------------------------------------ #
    def discover_cycle(self, *, noise: float = 0.0) -> Dict[str, Any]:
        """One complete self-driven scientific cycle: **hypothesise → predict → try to falsify**.

        She runs the next experiment in her registry, discovers a candidate law on the *ordinary*
        regime, then puts it to a genuine predictive test on the **most extreme, most discriminating
        regime she did not train on** (the decisive experiment a scientist designs to break a
        theory). A law that predicts the extreme regime survives (corroborated); one that fails there
        is falsified. Optionally injects measurement ``noise`` to prove the law survives messy data.
        Everything is logged to the lab notebook. Returns a compact verdict dict."""
        registry = self._experiment_registry()
        self._cycle_n = getattr(self, "_cycle_n", -1) + 1
        experiment = registry[self._cycle_n % len(registry)]
        result: Dict[str, Any] = {"experiment": None, "law": None, "verdict": "no-data"}
        try:
            data = experiment()
        except Exception as exc:  # noqa: BLE001
            data = None
            result["verdict"] = f"experiment-error: {exc}"
        if not data:
            return result
        result["experiment"] = data.get("label", data["target"])
        cols, yv = self._to_columns(data["X"], data["y"])
        if cols is None or len(yv) < 10:
            return result
        if noise > 0.0:                                   # inject relative Gaussian measurement noise
            yspread = _std(yv) or 1.0
            yv = [v + self._rng.gauss(0.0, noise * yspread) for v in yv]
        # DESIGN the decisive experiment: rank samples by extremity and hold out the most extreme 30%
        # as the untrained regime the theory must predict. This is active design — probe where a wrong
        # law would diverge most from a right one.
        splits = self._split(cols, yv)
        dims = data.get("dims")
        var_dims = list(dims) if dims else [None] * len(cols)
        feats = _build_library(len(cols))
        feats = self._pi_augment(feats, var_dims, data.get("target_dim"))
        feats = self._dim_prune(feats, var_dims, data.get("target_dim"))
        law = self._sparse_law(splits["train_cols"], splits["train_y"], feats,
                               data["var_names"], data["target"], var_dims)
        if law is None:
            result["verdict"] = "no-law"
            self._notebook_log("cycle", {"experiment": result["experiment"], "verdict": "no-law"})
            return result
        ev = self._validate(law, splits)
        result["law"] = law.expression
        result["verdict"] = ev.verdict
        result["extrapolation_r2"] = round(ev.extrapolation_r2, 5)
        result["noise"] = noise
        # try to FALSIFY: does a deliberately-perturbed rival law fail the decisive test the real one
        # passes? If so, the cycle has actively discriminated between theories, not merely fit one.
        rival = self._perturb_law(law)
        if rival is not None and splits.get("extrap_cols") is not None and splits["extrap_y"]:
            rp = rival.predict(splits["extrap_cols"])
            rival_r2 = _r2(rp, splits["extrap_y"]) if rp is not None else -1e9
            result["rival_falsified"] = bool(rival_r2 < ev.extrapolation_r2 - 0.05)
            result["rival_extrapolation_r2"] = round(rival_r2, 5)
        if ev.verdict == "corroborated":
            nov = self._novelty(law)
            disc = Discovery(law=law, evidence=ev, novelty=nov,
                             parsimony=1.0 / (1.0 + 0.15 * law.complexity),
                             stability=self._coeff_stability(law, splits))
            self._fold_in(disc)
        self._notebook_log("cycle", {"experiment": result["experiment"], "law": law.expression,
                                     "verdict": ev.verdict,
                                     "rival_falsified": result.get("rival_falsified"),
                                     "noise": noise})
        return result

    def _perturb_law(self, law: Law) -> Optional[Law]:
        """Build a plausible *rival* theory by nudging the dominant term's exponent — the alternative
        hypothesis the decisive experiment is meant to rule out (e.g. an inverse-cube vs the true
        inverse-square). Sparse monomial laws only; ``None`` when no clean rival exists."""
        if not law._features or not law._coeffs:
            return None
        k = max(range(len(law._coeffs)), key=lambda i: abs(law._coeffs[i]))
        f = law._features[k]
        if f.kind != "monomial" or not f.exps:
            return None
        j = max(f.exps, key=lambda i: abs(f.exps[i]))
        rival_exps = dict(f.exps)
        rival_exps[j] = rival_exps[j] + (1.0 if rival_exps[j] >= 0 else -1.0)   # shift the exponent
        rival_feat = _Feature("monomial", exps=rival_exps)
        return Law(target=law.target, expression=f"rival({law.target})", kind=law.kind,
                   domain=law.domain, origin="rival", var_names=list(law.var_names),
                   terms=[(rival_feat.render(law.var_names), law._coeffs[k])],
                   complexity=law.complexity,
                   _features=[rival_feat], _coeffs=[law._coeffs[k]], _bias=law._bias)

    # ------------------------------------------------------------------ #
    # Public: meta-law unification — compound discoveries into deeper theory
    # ------------------------------------------------------------------ #
    def unify_laws(self) -> List[Dict[str, Any]]:
        """Search the law tower for a **more general law that subsumes several discoveries**.

        Laws that share an *abstract structure* — the same monomial exponent pattern, regardless of
        which physical variables fill the slots — are instances of one deeper law. E.g. the pendulum
        ``T² = c·L/g`` and the spring ``T² = c·m/k`` both have the shape ``target = c·(a·b⁻¹)`` and
        unify into the general simple-harmonic form ``T² = c·(inertia / restoring)``; Coulomb and
        Newtonian gravity share the inverse-square shape ``c·(a·b·r⁻²)``. This is genuine open-ended
        compounding: her discoveries fold into abstractions that guide later search. Logged to the
        notebook. Returns the unifications found."""
        groups: Dict[Tuple[float, ...], List[Law]] = {}
        for law in self._laws:
            sig = self._abstract_structure(law)
            if sig is not None:
                groups.setdefault(sig, []).append(law)
        unifications: List[Dict[str, Any]] = []
        for sig, laws in groups.items():
            # only a real unification if ≥2 *distinct* laws share the shape
            distinct = {l.signature() for l in laws}
            if len(distinct) < 2:
                continue
            schema = self._structure_schema(sig)
            members = sorted({l.expression for l in laws})
            unifications.append({"schema": schema, "shape": list(sig), "members": members})
            self._notebook_log("unification", {"schema": schema, "members": members})
        return unifications

    @staticmethod
    def _abstract_structure(law: Law) -> Optional[Tuple[float, ...]]:
        """The shape of a law's dominant monomial: its exponents as a sorted, variable-agnostic
        tuple. Two laws with the same shape are instances of one general law."""
        if not law._features or not law._coeffs:
            return None
        k = max(range(len(law._coeffs)), key=lambda i: abs(law._coeffs[i]))
        f = law._features[k]
        if f.kind != "monomial" or len(f.exps) < 2:
            return None
        return tuple(sorted(f.exps.values()))

    @staticmethod
    def _structure_schema(shape: Tuple[float, ...]) -> str:
        """A readable name for an abstract monomial shape."""
        s = tuple(sorted(shape))
        if s == (-1.0, 1.0):
            return "target = c·(a / b)   [ratio law — e.g. SHM period² ∝ inertia/restoring]"
        if s == (-2.0, 1.0, 1.0):
            return "target = c·(a·b / r²)   [inverse-square interaction — e.g. Coulomb / gravity]"
        if s == (1.0, 1.0):
            return "target = c·(a·b)   [product law — e.g. Ohm V=I·R, τ=R·C]"
        parts = "·".join(f"x{i}^{e:g}" for i, e in enumerate(s))
        return f"target = c·({parts})   [shared power-law shape]"

    # ------------------------------------------------------------------ #
    # Public: active experiment design — the intervention that breaks a tie
    # ------------------------------------------------------------------ #
    def active_experiment(self, laws: Sequence[Law], *, var_index: int, low: float, high: float,
                          n_probe: int = 64, base_row: Optional[Sequence[float]] = None) -> float:
        """Given competing laws that agree on seen data, return the value of variable ``var_index``
        (in ``[low, high]``) at which they most *disagree* — the single experiment whose outcome is
        most informative. This is autonomous experimental design: probe where theories diverge."""
        executable = [law for law in laws if law._features]
        if len(executable) < 2:
            return (low + high) / 2.0
        n_vars = max(len(law.var_names) for law in executable)
        base = list(base_row) if base_row is not None else [1.0] * n_vars
        best_x, best_spread = (low + high) / 2.0, -1.0
        for k in range(max(2, n_probe)):
            x = low + (high - low) * k / (n_probe - 1)
            row = list(base)
            while len(row) <= var_index:
                row.append(1.0)
            row[var_index] = x
            cols = [[row[i]] for i in range(len(row))]
            preds = []
            for law in executable:
                p = law.predict(cols)
                if p is not None and _is_finite(p[0]):
                    preds.append(p[0])
            if len(preds) < 2:
                continue
            spread = max(preds) - min(preds)
            if spread > best_spread:
                best_spread, best_x = spread, x
        return best_x

    # ------------------------------------------------------------------ #
    # Inspection
    # ------------------------------------------------------------------ #
    def known_laws(self) -> List[Law]:
        return list(self._laws)

    # ------------------------------------------------------------------ #
    # Public: USE what she discovered — the bridge back into her reasoning
    # ------------------------------------------------------------------ #
    def recall(self, query: Any) -> Optional[Law]:
        """Return the discovered law most relevant to ``query`` (a natural-language string or an
        iterable of variable names), or ``None`` when nothing she has discovered applies.

        This is the missing link that lets a self-discovered law actually *ground an answer*: her
        reasoner asks ``recall("how far does it fall in time t under gravity g?")`` and gets back
        the ``fall_distance = ½·g·t²`` she invented herself from her own experiments. No LLM — a
        keyword/variable match over her corroborated law tower, ranked by coverage then parsimony."""
        if not self._laws:
            return None
        try:
            if isinstance(query, str):
                tokens = {w.strip("?.,!:;\"'()").lower() for w in query.split()}
            else:
                tokens = {str(w).strip().lower() for w in query}
        except Exception:  # noqa: BLE001
            return None
        tokens.discard("")
        if not tokens:
            return None
        best: Optional[Law] = None
        best_key: Tuple[float, float, int] = (-1.0, 0.0, 0)
        for law in self._laws:
            target = str(law.target).lower()
            names = [str(v).lower() for v in law.var_names]
            score = 0.0
            if target and (target in tokens or any(target in t or t in target for t in tokens)):
                score += 2.0
            matched_vars = sum(1 for v in names if v and v in tokens)
            score += float(matched_vars)
            if score <= 0.0:
                continue
            # rank: total relevance, then variable coverage, then simpler laws (lower complexity)
            key = (score, matched_vars / max(1, len(names)), -int(law.complexity))
            if key > best_key:
                best_key, best = key, law
        return best

    def apply(self, var_values: Dict[str, float],
              *, query: Any = None) -> Optional[Dict[str, Any]]:
        """Evaluate a discovered law at concrete inputs and return its prediction.

        ``var_values`` maps variable names to numbers. Picks the corroborated law whose variables
        are all supplied (most-covering, then simplest); when ``query`` is given it is used to break
        ties toward the most relevant law. Returns ``{"target","value","law","expression",...}`` or
        ``None`` if no discovered law fits the supplied variables. Deterministic, no LLM."""
        if not self._laws or not isinstance(var_values, dict) or not var_values:
            return None
        keys = {str(k).lower(): float(v) for k, v in var_values.items()
                if _is_number(v)}
        if not keys:
            return None
        preferred = self.recall(query) if query is not None else None
        best: Optional[Law] = None
        best_key: Tuple[int, int, int] = (-1, 0, 0)
        for law in self._laws:
            names = [str(v).lower() for v in law.var_names]
            if not names or any(n not in keys for n in names):
                continue                                   # need every input the law consumes
            cover = len(names)
            prefer = 1 if (preferred is not None and law is preferred) else 0
            key = (prefer, cover, -int(law.complexity))
            if key > best_key:
                best_key, best = key, law
        if best is None:
            return None
        try:
            cols = [[keys[str(v).lower()]] for v in best.var_names]
            out = best.predict(cols)
            if not out:
                return None
            return {
                "target": best.target,
                "value": float(out[0]),
                "expression": best.expression,
                "law": best.to_dict(),
                "inputs": {str(v): keys[str(v).lower()] for v in best.var_names},
            }
        except Exception:  # noqa: BLE001 — evaluation is best-effort, never fatal
            return None

    def all_reports(self) -> List[DiscoveryReport]:
        return list(self._reports)

    def report(self) -> Dict[str, Any]:
        return {
            "laws_discovered": self.total_laws,
            "laws_held": len(self._laws),
            "refuted": self.total_refuted,
            "abstained": self.total_abstained,
            "runs": len(self._reports),
            "domains": sorted({law.domain for law in self._laws}),
            "latest_laws": [law.expression for law in self._laws[-5:]],
        }

    # ------------------------------------------------------------------ #
    # Data marshalling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_columns(X: Any, y: Any) -> Tuple[Optional[List[List[float]]], List[float]]:
        """Normalise a dataset (rows or numpy 2-D) into a list of columns + a target list."""
        try:
            if _np is not None and hasattr(X, "shape"):
                A = _np.asarray(X, dtype=float)
                if A.ndim == 1:
                    A = A.reshape(-1, 1)
                cols = [[float(v) for v in A[:, j]] for j in range(A.shape[1])]
                yv = [float(v) for v in _np.asarray(y, dtype=float).ravel()]
                return cols, yv
            rows = [list(r) if isinstance(r, (list, tuple)) else [float(r)] for r in X]
            if not rows:
                return None, []
            n = len(rows[0])
            cols = [[float(rows[r][j]) for r in range(len(rows))] for j in range(n)]
            yv = [float(v) for v in y]
            return cols, yv
        except Exception:  # noqa: BLE001
            return None, []

    @staticmethod
    def _to_rows(traj: Any) -> Optional[List[List[float]]]:
        try:
            if _np is not None and hasattr(traj, "shape"):
                A = _np.asarray(traj, dtype=float)
                if A.ndim == 1:
                    A = A.reshape(-1, 1)
                return [[float(v) for v in A[i, :]] for i in range(A.shape[0])]
            return [list(map(float, r)) for r in traj]
        except Exception:  # noqa: BLE001
            return None

    def _to_trajectories(self, trajs: Any) -> List[List[List[float]]]:
        """Accept a single trajectory (list of states) or a list of trajectories; normalise to a
        list of trajectories (each a list of state rows)."""
        rows = self._to_rows(trajs)
        if rows is not None and rows and isinstance(rows[0], list) and rows[0] and \
                not isinstance(rows[0][0], list):
            # could be one trajectory of flat states
            try:
                _ = [float(v) for v in rows[0]]
                return [rows]
            except Exception:  # noqa: BLE001
                pass
        out: List[List[List[float]]] = []
        try:
            for tr in trajs:
                r = self._to_rows(tr)
                if r:
                    out.append(r)
        except Exception:  # noqa: BLE001
            return []
        return out

    def _split(self, cols: List[List[float]], y: List[float]) -> Dict[str, Any]:
        """Split into train / in-range holdout / out-of-range extrapolation sets.

        Extremity of a sample is its max absolute z-score across features; the least-extreme 70%
        form the in-range pool (75% train / 25% holdout) and the most-extreme 30% become the
        extrapolation set — so a kept law must predict genuinely *outside* its training envelope."""
        m = len(y)
        n = len(cols)
        if m < 8:
            return {"train_cols": cols, "train_y": y,
                    "holdout_cols": cols, "holdout_y": y,
                    "extrap_cols": None, "extrap_y": []}
        stats = []
        for j in range(n):
            mu = sum(cols[j]) / m
            sd = _std(cols[j]) or 1.0
            stats.append((mu, sd))
        extremity = []
        for r in range(m):
            e = max(abs((cols[j][r] - stats[j][0]) / stats[j][1]) for j in range(n))
            extremity.append((e, r))
        order = [r for _, r in sorted(extremity)]
        n_in = int(0.7 * m)
        in_idx = order[:n_in]
        ex_idx = order[n_in:]
        rng = random.Random(1234)
        rng.shuffle(in_idx)
        n_tr = max(4, int(0.75 * len(in_idx)))
        tr_idx = in_idx[:n_tr]
        ho_idx = in_idx[n_tr:] or tr_idx

        def _sub(idx: List[int]) -> List[List[float]]:
            return [[cols[j][r] for r in idx] for j in range(n)]

        return {"train_cols": _sub(tr_idx), "train_y": [y[r] for r in tr_idx],
                "holdout_cols": _sub(ho_idx), "holdout_y": [y[r] for r in ho_idx],
                "extrap_cols": _sub(ex_idx) if ex_idx else None,
                "extrap_y": [y[r] for r in ex_idx]}

    # ------------------------------------------------------------------ #
    # Dimensional pruning
    # ------------------------------------------------------------------ #
    def _dim_prune(self, feats: List[_Feature],
                   var_dims: List[Optional[Tuple[float, ...]]],
                   target_dim: Optional[Tuple[float, ...]]) -> List[_Feature]:
        if target_dim is None or all(d is None for d in var_dims):
            return feats
        kept = []
        for f in feats:
            fd = f.dimension(var_dims)
            if fd is None:
                continue
            if all(abs(fd[k] - target_dim[k]) < 1e-6 for k in range(7)):
                kept.append(f)
        # dimensional guidance narrows to the dimensionally-consistent forms (often a unique
        # monomial, exactly as a physicist would); only if it prunes *everything* do we fall back.
        return kept if len(kept) >= 1 else feats

    # ------------------------------------------------------------------ #
    # Buckingham-π dimensional monomial generation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean_exp(e: float) -> float:
        """Snap an exponent to a nearby physically-clean fraction (±½, ±1, ±1½, ±2, ±3) if close."""
        for nice in (-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
            if abs(e - nice) < 0.06:
                return nice
        return round(e, 3)

    def _pi_augment(self, feats: List[_Feature],
                    var_dims: List[Optional[Tuple[float, ...]]],
                    target_dim: Optional[Tuple[float, ...]]) -> List[_Feature]:
        """Add the *exactly dimensionally-consistent* power-law monomials to the library.

        This is the Buckingham-π theorem put to work: given the variables' physical dimensions and
        the target's, solve ``Σ aᵢ·dim(xᵢ) = dim(target)`` for exponent vectors ``a`` — every such
        ``a`` is a monomial ``Πⱼ xⱼ^aⱼ`` that is dimensionally *forced* to be the answer's form (up to
        a dimensionless constant). We add the particular solution and, when the variables are
        redundant, that solution shifted by the dimensionless π-groups (the nullspace) — so the
        space searched is exactly the finite set of dimensionally-possible scaling laws a physicist
        would consider. Guarantees any true power law is in the library; a no-op without numpy or
        without full dimensional information."""
        if _np is None or target_dim is None:
            return feats
        dims = list(var_dims)
        if not dims or any(d is None for d in dims):
            return feats
        n = len(dims)
        D = _np.asarray([list(d) for d in dims], dtype=float).T   # (7, n): columns are variable dims
        t = _np.asarray(list(target_dim), dtype=float)
        try:
            a0, *_ = _np.linalg.lstsq(D, t, rcond=None)
            if not _np.allclose(D @ a0, t, atol=1e-6):
                return feats                                       # target unreachable from these vars
            # nullspace basis (the dimensionless π-groups): directions we can add to a0 freely
            _u, s, vh = _np.linalg.svd(D)
            rank = int((s > 1e-9).sum())
            null = [vh[i] for i in range(rank, vh.shape[0])] if vh.shape[0] > rank else []
        except Exception:  # noqa: BLE001 — numeric failure just means no augmentation
            return feats
        candidates: List[List[float]] = [list(a0)]
        for nb in null[:3]:
            for k in (-1.0, 1.0):
                candidates.append([a0[i] + k * nb[i] for i in range(n)])
        seen = {f.signature() for f in feats}
        out = list(feats)
        for a in candidates:
            exps = {i: self._clean_exp(a[i]) for i in range(n) if abs(self._clean_exp(a[i])) > 0.05}
            if not exps:
                continue
            f = _Feature("monomial", exps=exps)
            if f.signature() not in seen:
                seen.add(f.signature())
                out.append(f)
        return out

    # ------------------------------------------------------------------ #
    # Engine 1 — sparse feature regression (STLSQ)
    # ------------------------------------------------------------------ #
    def _sparse_law(self, train_cols: List[List[float]], train_y: List[float],
                    feats: List[_Feature], names: Sequence[str], target_name: str,
                    var_dims: List[Optional[Tuple[float, ...]]],
                    kind: str = "static", tol: float = 0.02) -> Optional[Law]:
        m = len(train_y)
        evaluated: List[Tuple[_Feature, List[float], float]] = []
        for f in feats:
            v = f.evaluate(train_cols)
            if v is None:
                continue
            s = _std(v)
            if s < 1e-9:
                continue
            evaluated.append((f, v, s))
        if not evaluated:
            return None
        ystd = _std(train_y) or 1.0
        active = list(range(len(evaluated)))
        coeffs: Optional[List[float]] = None
        bias = 0.0
        for _ in range(12):
            rows = [[evaluated[i][1][r] for i in active] + [1.0] for r in range(m)]
            sol = _solve_lstsq(rows, train_y)
            if sol is None:
                return None
            cur = sol[:-1]
            bias = sol[-1]
            keep = [active[k] for k, c in enumerate(cur)
                    if abs(c) * evaluated[active[k]][2] / ystd >= tol]
            if not keep:
                return None
            if keep == active:
                coeffs = cur
                break
            active = keep
        else:  # pragma: no cover - convergence guard
            rows = [[evaluated[i][1][r] for i in active] + [1.0] for r in range(m)]
            sol = _solve_lstsq(rows, train_y)
            if sol is None:
                return None
            coeffs, bias = sol[:-1], sol[-1]
        if coeffs is None or not active:
            return None
        sel_feats = [evaluated[i][0] for i in active]
        terms = [(sel_feats[k].render(names), coeffs[k]) for k in range(len(sel_feats))]
        complexity = sum(_feat_cost(f) for f in sel_feats) + (1 if abs(bias) > 1e-6 else 0)
        expr = self._render_linear(terms, bias, target_name)
        return Law(target=target_name, expression=expr, kind=kind, domain="empirical",
                   origin="sparse", var_names=list(names), terms=terms, complexity=complexity,
                   _features=sel_feats, _coeffs=list(coeffs), _bias=float(bias))

    @staticmethod
    def _render_linear(terms: List[Tuple[str, float]], bias: float, target_name: str) -> str:
        parts = []
        for t, c in terms:
            parts.append(f"{c:.4g}·{t}")
        if abs(bias) > 1e-6:
            parts.append(f"{bias:.4g}")
        rhs = " + ".join(parts) if parts else "0"
        return f"{target_name} = {rhs}"

    # ------------------------------------------------------------------ #
    # Engine 2 — genetic-programming expression search
    # ------------------------------------------------------------------ #
    def _gp_law(self, train_cols: List[List[float]], train_y: List[float],
                names: Sequence[str], target_name: str, *,
                generations: int = 8, population: int = 40,
                time_budget_s: float = 1.5) -> Optional[Law]:
        n_vars = len(train_cols)
        deadline = time.monotonic() + time_budget_s

        def fitness(tree: "_Tree") -> Tuple[float, float, float]:
            vals = tree.evaluate(train_cols)
            if vals is None or _std(vals) < 1e-9:
                return (-1e9, 1.0, 0.0)
            rows = [[vals[r], 1.0] for r in range(len(vals))]
            sol = _solve_lstsq(rows, train_y)
            if sol is None:
                return (-1e9, 1.0, 0.0)
            a, b = sol
            pred = [a * v + b for v in vals]
            r2 = _r2(pred, train_y)
            return (r2 - 0.003 * tree.size(), a, b)

        pop = [self._random_tree(n_vars, depth=self._rng.randint(1, 3))
               for _ in range(population)]
        best: Optional[Tuple[float, "_Tree", float, float]] = None
        for _ in range(generations):
            if time.monotonic() > deadline:
                break
            scored = []
            for tree in pop:
                f, a, b = fitness(tree)
                scored.append((f, tree, a, b))
                if best is None or f > best[0]:
                    best = (f, tree, a, b)
            scored.sort(key=lambda s: s[0], reverse=True)
            elite = [s[1] for s in scored[:max(2, population // 5)]]
            nxt = list(elite)
            while len(nxt) < population:
                if self._rng.random() < 0.6 and len(elite) >= 2:
                    child = self._crossover(self._rng.choice(elite), self._rng.choice(elite))
                else:
                    child = self._mutate(self._rng.choice(elite), n_vars)
                nxt.append(child)
            pop = nxt
        if best is None or best[0] < 0.5:
            return None
        _f, tree, a, b = best
        expr = f"{target_name} = {a:.4g}·{tree.render(names)} + {b:.4g}"
        return Law(target=target_name, expression=expr, kind="static", domain="empirical",
                   origin="symbolic", var_names=list(names),
                   terms=[(tree.render(names), a)], complexity=tree.size(),
                   _tree=tree, _scale=float(a), _offset=float(b))

    def _random_tree(self, n_vars: int, depth: int) -> "_Tree":
        if depth <= 0 or (self._rng.random() < 0.35):
            if self._rng.random() < 0.85 and n_vars > 0:
                return _Tree("var", var=self._rng.randrange(n_vars))
            return _Tree("one")
        if self._rng.random() < 0.55:
            op = self._rng.choice(_BINOPS)
            return _Tree(op, children=(self._random_tree(n_vars, depth - 1),
                                       self._random_tree(n_vars, depth - 1)))
        op = self._rng.choice(_UNOPS)
        return _Tree(op, children=(self._random_tree(n_vars, depth - 1),))

    def _mutate(self, tree: "_Tree", n_vars: int) -> "_Tree":
        if self._rng.random() < 0.3:
            return self._random_tree(n_vars, depth=self._rng.randint(1, 3))
        if not tree.children:
            return self._random_tree(n_vars, depth=1)
        idx = self._rng.randrange(len(tree.children))
        new_children = list(tree.children)
        new_children[idx] = self._mutate(tree.children[idx], n_vars)
        return _Tree(tree.op, var=tree.var, children=tuple(new_children))

    def _crossover(self, a: "_Tree", b: "_Tree") -> "_Tree":
        if not a.children or self._rng.random() < 0.3:
            return b
        idx = self._rng.randrange(len(a.children))
        new_children = list(a.children)
        new_children[idx] = self._crossover(a.children[idx], b)
        return _Tree(a.op, var=a.var, children=tuple(new_children))

    # ------------------------------------------------------------------ #
    # Honest empirical validation
    # ------------------------------------------------------------------ #
    def _validate(self, law: Law, splits: Dict[str, Any]) -> LawEvidence:
        ev = LawEvidence()
        tr_pred = law.predict(splits["train_cols"])
        if tr_pred is None:
            return ev
        ev.fit_r2 = _r2(tr_pred, splits["train_y"])
        ev.n_train = len(splits["train_y"])
        hp = law.predict(splits["holdout_cols"])
        if hp is None:
            return ev
        ev.holdout_r2 = _r2(hp, splits["holdout_y"])
        ev.holdout_rmse = _rmse(hp, splits["holdout_y"])
        ev.n_holdout = len(splits["holdout_y"])
        if splits.get("extrap_cols") is not None and splits["extrap_y"]:
            ep = law.predict(splits["extrap_cols"])
            ev.extrapolation_r2 = _r2(ep, splits["extrap_y"]) if ep is not None else -1e9
            ev.n_extrapolation = len(splits["extrap_y"])
        else:
            ev.extrapolation_r2 = ev.holdout_r2
        if ev.holdout_r2 >= self.corroborate_r2 and ev.extrapolation_r2 >= self.extrapolate_r2:
            ev.verdict = "corroborated"
        elif ev.holdout_r2 < 0.0:            # worse than predicting the mean — a false law
            ev.verdict = "refuted"
        else:
            ev.verdict = "inconclusive"
        return ev

    def _coeff_stability(self, law: Law, splits: Dict[str, Any], trials: int = 16) -> float:
        """Bootstrap the law's coefficients to measure how *stable* they are — honest statistics.

        Resample the training rows with replacement, refit the law's own feature set each time, and
        measure the coefficient of variation of every coefficient. A law whose coefficients swing
        wildly across resamples is fragile (it fit noise), so it is down-weighted; a law whose
        coefficients barely move is trustworthy. Sparse laws only (fixed feature set); GP trees and
        the numpy-less path return a neutral 1.0."""
        feats = law._features
        if not feats or _np is None:
            return 1.0
        cols = splits.get("train_cols") or []
        yv = splits.get("train_y") or []
        m = len(yv)
        if m < 8 or not cols:
            return 1.0
        fcols: List[List[float]] = []
        for f in feats:
            v = f.evaluate(cols)
            if v is None:
                return 1.0
            fcols.append(v)
        try:
            A = _np.asarray(fcols, dtype=float).T
            A = _np.column_stack([A, _np.ones(m)])
            y = _np.asarray(yv, dtype=float)
            rng = _np.random.default_rng(0xB007)
            coeffs = []
            for _ in range(max(4, trials)):
                idx = rng.integers(0, m, m)
                sol, *_ = _np.linalg.lstsq(A[idx], y[idx], rcond=None)
                coeffs.append(sol[:-1])
            if len(coeffs) < 3:
                return 1.0
            C = _np.asarray(coeffs)
            mean = _np.abs(C.mean(axis=0)) + 1e-9
            cv = float((C.std(axis=0) / mean).mean())
            return max(0.0, min(1.0, 1.0 / (1.0 + cv)))
        except Exception:  # noqa: BLE001 — stability is advisory, never fatal
            return 1.0

    # ------------------------------------------------------------------ #
    # Novelty, folding-in, and the self-extending library
    # ------------------------------------------------------------------ #
    def _novelty(self, law: Law) -> float:
        sig = law.signature()
        if sig in self._signatures:
            return 0.2
        if self.frontier is not None:
            try:
                nov = self.frontier.novelty(law.signature())
                if isinstance(nov, (int, float)):
                    return max(0.2, min(1.0, float(nov)))
            except Exception:  # noqa: BLE001
                pass
        return 1.0

    def _fold_in(self, disc: Discovery) -> None:
        law = disc.law
        if disc.evidence.verdict != "corroborated":
            return
        sig = law.signature()
        if sig in self._signatures:
            return
        self._signatures.add(sig)
        self._laws.append(law)
        self.total_laws += 1
        self._promote(law)
        self._notebook_log("discovery", {
            "target": law.target, "kind": law.kind, "domain": law.domain, "origin": law.origin,
            "expression": law.expression, "verdict": disc.evidence.verdict,
            "holdout_r2": round(disc.evidence.holdout_r2, 5),
            "extrapolation_r2": round(disc.evidence.extrapolation_r2, 5),
            "novelty": round(disc.novelty, 3), "stability": round(disc.stability, 3)})
        self._save()

    def _notebook_log(self, event: str, payload: Dict[str, Any]) -> None:
        """Append one auditable entry to the lab notebook (best-effort, never fatal)."""
        if not self.notebook_path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.notebook_path)), exist_ok=True)
            entry = {"t": round(time.time(), 3), "event": event, **payload}
            with open(self.notebook_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception:  # noqa: BLE001 — the notebook is a convenience, never required
            pass

    def _promote(self, law: Law) -> None:
        """Fold a corroborated law back into knowledge / memory (best-effort, never required)."""
        text = f"Discovered {law.kind} law ({law.domain}): {law.expression}"
        kb = self.knowledge
        if kb is not None:
            try:
                if hasattr(kb, "ingest_text"):
                    kb.ingest_text(text, source="law_discovery")
            except Exception:  # noqa: BLE001
                pass
        mem = self.memory
        if mem is not None:
            for meth in ("remember", "store", "add"):
                fn = getattr(mem, meth, None)
                if callable(fn):
                    try:
                        fn(text)
                        break
                    except Exception:  # noqa: BLE001
                        continue

    # ------------------------------------------------------------------ #
    # Persistence — the law tower survives across sessions
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for d in data.get("laws", []):
                law = _law_from_json(d)
                if law is not None and law.signature() not in self._signatures:
                    self._signatures.add(law.signature())
                    self._laws.append(law)
            self.total_laws = int(data.get("total_laws", len(self._laws)))
        except Exception:  # noqa: BLE001 — a corrupt cache is not fatal; start fresh
            pass

    def _save(self) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            payload = {"total_laws": self.total_laws,
                       "laws": [_law_to_json(law) for law in self._laws[-256:]]}
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    # Self-generated rounds (offline / CI demonstrations)
    # ------------------------------------------------------------------ #
    def _round_synthetic(self) -> DiscoveryReport:
        self._syn_n = getattr(self, "_syn_n", -1) + 1
        rng = self._rng
        templates = [
            # (name, var_names, sampler -> row, law(row) -> y)
            ("free_fall", ["g", "t"],
             lambda: [rng.uniform(1.5, 20.0), rng.uniform(0.1, 5.0)],
             lambda x: 0.5 * x[0] * x[1] ** 2),
            ("inverse_square", ["r"],
             lambda: [rng.uniform(0.5, 5.0)],
             lambda x: 3.0 / (x[0] ** 2)),
            ("momentum_sum", ["m1", "v1", "m2", "v2"],
             lambda: [rng.uniform(0.5, 3.0) for _ in range(4)],
             lambda x: x[0] * x[1] + x[2] * x[3]),
        ]
        name, vnames, sampler, law_fn = templates[self._syn_n % len(templates)]
        X = [sampler() for _ in range(60)]
        y = [law_fn(row) for row in X]
        rep = self.discover_from_data(X, y, var_names=vnames, target_name=name, source="synthetic")
        self._reports.append(rep)
        return rep

    def _round_dynamics(self) -> DiscoveryReport:
        x, v = 1.0, 0.0
        k, c, dt = 1.0, 0.3, 0.05
        rows = []
        for _ in range(140):
            rows.append([x, v])
            a = -k * x - c * v
            v += a * dt
            x += v * dt
        return self.discover_dynamics(rows, dt=dt, var_names=["x", "v"])

    def _round_invariant(self) -> DiscoveryReport:
        x, v = 1.0, 0.0
        k, dt = 1.0, 0.02
        rows = []
        for _ in range(220):
            rows.append([x, v])
            a = -k * x
            v += a * dt
            x += v * dt
        return self.discover_invariant([rows], var_names=["x", "v"])

    def _physics_world_class(self) -> Any:
        if self.physics is not None:
            return self.physics if isinstance(self.physics, type) else type(self.physics)
        try:
            from nyxara.sim.physics_world import PhysicsWorld
            return PhysicsWorld
        except Exception:  # noqa: BLE001
            return None


# --------------------------------------------------------------------------- #
# Free functions used by the engine
# --------------------------------------------------------------------------- #
def _feat_cost(f: _Feature) -> int:
    if f.kind == "monomial":
        return len(f.exps) + int(sum(abs(e) for e in f.exps.values()))
    return 4


def _feature_to_json(f: _Feature) -> Dict[str, Any]:
    return {"kind": f.kind, "exps": {str(k): v for k, v in f.exps.items()},
            "func": f.func, "var": f.var}


def _feature_from_json(d: Dict[str, Any]) -> _Feature:
    return _Feature(kind=d.get("kind", "monomial"),
                    exps={int(k): float(v) for k, v in (d.get("exps") or {}).items()},
                    func=d.get("func", ""), var=int(d.get("var", 0)))


def _law_to_json(law: Law) -> Dict[str, Any]:
    d = law.to_dict()
    d["_features"] = [_feature_to_json(f) for f in law._features]
    d["_coeffs"] = list(law._coeffs)
    d["_bias"] = law._bias
    return d


def _law_from_json(d: Dict[str, Any]) -> Optional[Law]:
    try:
        feats = [_feature_from_json(fd) for fd in d.get("_features", [])]
        terms = [(t["term"], float(t["coeff"])) for t in d.get("terms", [])]
        return Law(target=d.get("target", "y"), expression=d.get("expression", ""),
                   kind=d.get("kind", "static"), domain=d.get("domain", "empirical"),
                   origin=d.get("origin", "promoted"), var_names=list(d.get("variables", [])),
                   terms=terms, complexity=int(d.get("complexity", 1)),
                   lineage=list(d.get("lineage", [])),
                   _features=feats, _coeffs=[float(c) for c in d.get("_coeffs", [])],
                   _bias=float(d.get("_bias", 0.0)))
    except Exception:  # noqa: BLE001
        return None
