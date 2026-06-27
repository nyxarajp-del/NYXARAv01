"""NYXARA · growth/open_world.py — open-world generalization from first principles (🛸).

The hardest test of a mind is not answering a question it was trained on — it is meeting a
thing it has **never seen before** and *figuring it out anyway*. Drop NYXARA in front of an
**alien machine**: a black box she can only *poke* (feed it an input, watch what comes out),
whose rules she has zero prior knowledge of. A trained model abstains. A *sovereign* mind does
what a scientist does — from first principles:

    OBSERVE   →   HYPOTHESIZE   →   TEST   →   MODEL
    (probe it)   (induce laws)   (experiment)  (predict + generalize)
       ↑                                            │
       └─────────────── refine ─────────────────────┘

This is the one thing the existing discovery faculties do **not** do. The
:class:`~nyxara.growth.autonomous_scientist.AutonomousScientist` studies her *own* self-generated
math/logic propositions; :class:`~nyxara.growth.active_curiosity.ActiveCuriosity` wonders about
events she has *already lived*. Neither can be pointed at an arbitrary unknown interactive system.
:class:`OpenWorldGeneralizer` can — and it **composes** the existing faculties rather than
duplicating them (it reuses :class:`~nyxara.growth.autonomous_scientist.BeliefModel` for honest
confidence and folds findings into the world / causal models when present).

How it works, concretely and *for real* (pure stdlib, deterministic, numpy-free → runs in CI):

1. **Observe** — it probes the box at well-chosen inputs: the origin, one axis at a time
   (so each input's effect is isolated), the domain boundaries, then a deterministic spread.
2. **Hypothesize** — from the ``(input → output)`` pairs it *fits candidate laws from first
   principles*: constant, affine (least squares), low-degree polynomial, multiplicative,
   modular/periodic, threshold/piecewise, and — for discrete boxes — boolean operators and
   categorical lookups. Each candidate is ranked by **MDL** (fit + simplicity), so the *simplest
   law that actually fits* wins — never an over-fit table.
3. **Test** — it then runs the experiment a real scientist would: it picks the next probe where
   the surviving candidates **disagree most** (maximum information gain), queries the box there,
   and prunes whatever mispredicts. A handful of decisive experiments, not brute force.
4. **Model** — the winning law becomes a callable predictor with a human-readable description
   (``"out = 2*x + 1"``). It is **validated on fresh, never-probed inputs** to prove it
   *generalizes*, with a calibrated confidence. When *nothing* fits, it says so honestly —
   :attr:`Verdict.UNMODELLED`, low confidence, the best partial model kept — and never bluffs.

Nothing here touches the world or side-steps the control law: every probe is a call into the
black box the Master handed her, exactly as sandboxed as the rest of :mod:`nyxara.growth`.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.growth.autonomous_scientist import BeliefModel

__all__ = [
    "OpenWorldGeneralizer",
    "LawFamily",
    "Verdict",
    "DomainSpec",
    "Probe",
    "CandidateLaw",
    "UnderstandingReport",
    "build_alien_machine",
]


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
class LawFamily(str, Enum):
    """The shape of a candidate law fitted to an unknown system's behaviour."""
    CONSTANT = "constant"
    AFFINE = "affine"
    POLYNOMIAL = "polynomial"
    MULTIPLICATIVE = "multiplicative"
    MODULAR = "modular"
    THRESHOLD = "threshold"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    UNMODELLED = "unmodelled"


class Verdict(str, Enum):
    """How well the unknown system was understood — honest about failure."""
    MODELLED = "modelled"          # a law that generalizes to unseen inputs was found
    PARTIAL = "partial"            # a law that fits most, but not cleanly, of the behaviour
    INCONCLUSIVE = "inconclusive"  # not enough usable evidence to decide
    UNMODELLED = "unmodelled"      # nothing fit — she honestly could not crack it


# --------------------------------------------------------------------------- #
# The black-box system NYXARA confronts
# --------------------------------------------------------------------------- #
class _Interactor:
    """Wraps an unknown system into a single ``run(action) -> observation`` call.

    Accepts a bare callable (``system(action)``) or any object exposing ``.interact(action)``.
    A raised exception is captured as ``None`` rather than crashing the whole inquiry — a failed
    poke is itself data ("that input is rejected"), exactly as in a real experiment.
    """

    def __init__(self, system: Any) -> None:
        if callable(system):
            self._fn: Callable[[Any], Any] = system
        elif hasattr(system, "interact") and callable(getattr(system, "interact")):
            self._fn = getattr(system, "interact")
        else:
            raise TypeError("system must be callable or expose an .interact(action) method")
        self.errors = 0
        self.calls = 0

    def run(self, action: Any) -> Any:
        self.calls += 1
        try:
            return self._fn(action)
        except Exception:  # noqa: BLE001 — a rejected probe is data, never fatal
            self.errors += 1
            return None


@dataclass
class DomainSpec:
    """A minimal, prior-free description of an unknown system's input space.

    ``dims`` is how many numbers an action carries (1 for a scalar). ``kind`` is per-dim:
    ``"real"`` (continuous), ``"int"`` (small integers), or ``"bool"`` (0/1). ``low``/``high``
    bound the search; ``scalar`` records whether actions are passed as a bare number (dims==1)
    versus a tuple/list. With nothing known she defaults to a single real input on ``[-6, 6]``.
    """
    dims: int = 1
    kind: str = "real"
    low: float = -6.0
    high: float = 6.0
    scalar: bool = True

    @classmethod
    def infer(cls, example_action: Any) -> "DomainSpec":
        """Guess the shape of the input space from one example action — no rules assumed."""
        if isinstance(example_action, bool):
            return cls(dims=1, kind="bool", low=0.0, high=1.0, scalar=True)
        if isinstance(example_action, (int, float)):
            kind = "int" if isinstance(example_action, int) else "real"
            return cls(dims=1, kind=kind, low=-6.0, high=6.0, scalar=True)
        if isinstance(example_action, (tuple, list)) and example_action:
            allbool = all(isinstance(v, bool) for v in example_action)
            allint = all(isinstance(v, int) and not isinstance(v, bool) for v in example_action)
            kind = "bool" if allbool else ("int" if allint else "real")
            return cls(dims=len(example_action), kind=kind, low=-6.0, high=6.0, scalar=False)
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        return {"dims": self.dims, "kind": self.kind, "low": self.low, "high": self.high,
                "scalar": self.scalar}


# --------------------------------------------------------------------------- #
# Records of the inquiry
# --------------------------------------------------------------------------- #
@dataclass
class Probe:
    """One experiment: an action fed to the box and the observation that came back."""
    action: Any
    observation: Any
    vec: Tuple[float, ...] = ()        # the action normalised to a float vector (internal use)

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "observation": self.observation}


@dataclass
class CandidateLaw:
    """A hypothesised law: a callable predictor plus a human-readable description and scores.

    ``predict`` maps a *normalised float-vector* action to a predicted observation; it is never
    serialised. ``mdl`` is the description length (lower = a simpler law that fits better);
    ``train_error`` is the mean error on the probes seen so far. ``params`` is the model's
    complexity — the Occam term that keeps a memorising lookup from beating a real rule.
    """
    family: LawFamily
    description: str
    params: int
    predict: Callable[[Tuple[float, ...]], Any] = field(repr=False, default=lambda _x: 0.0)
    numeric: bool = True
    train_error: float = math.inf
    mdl: float = math.inf

    def to_dict(self) -> Dict[str, Any]:
        return {"family": self.family.value, "description": self.description,
                "params": self.params, "train_error": _round(self.train_error),
                "mdl": _round(self.mdl)}


@dataclass
class UnderstandingReport:
    """The result of confronting an unknown system: what she figured out, and how sure she is."""
    system: str = "unknown-system"
    domain: Dict[str, Any] = field(default_factory=dict)
    verdict: Verdict = Verdict.INCONCLUSIVE
    confidence: float = 0.0
    law: Optional[str] = None
    law_family: Optional[str] = None
    holdout_accuracy: float = 0.0
    residual: float = math.inf
    probes_used: int = 0
    candidates: List[CandidateLaw] = field(default_factory=list)
    winner: Optional[CandidateLaw] = field(default=None, repr=False)
    samples: List[Tuple[Tuple[float, ...], Any]] = field(default_factory=list, repr=False)
    elapsed_ms: float = 0.0

    def predict(self, action: Any) -> Any:
        """Use the learned model to predict the box's output for a *new* action."""
        if self.winner is None:
            return None
        spec = DomainSpec(**self.domain) if self.domain else DomainSpec()
        return self.winner.predict(_to_vec(action, spec))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system": self.system,
            "domain": self.domain,
            "verdict": self.verdict.value,
            "confidence": _round(self.confidence),
            "law": self.law,
            "law_family": self.law_family,
            "holdout_accuracy": _round(self.holdout_accuracy),
            "residual": _round(self.residual),
            "probes_used": self.probes_used,
            "candidates": [c.to_dict() for c in self.candidates[:6]],
            "elapsed_ms": _round(self.elapsed_ms, 1),
        }


# --------------------------------------------------------------------------- #
# Small numeric helpers (pure stdlib — no numpy)
# --------------------------------------------------------------------------- #
def _round(x: float, n: int = 4) -> Optional[float]:
    try:
        if x is None or math.isinf(x) or math.isnan(x):
            return None
        return round(float(x), n)
    except (TypeError, ValueError):
        return None


def _to_vec(action: Any, spec: DomainSpec) -> Tuple[float, ...]:
    """Normalise an arbitrary action to a tuple of floats (booleans → 0/1)."""
    if isinstance(action, bool):
        return (1.0 if action else 0.0,)
    if isinstance(action, (int, float)):
        return (float(action),)
    if isinstance(action, (tuple, list)):
        out: List[float] = []
        for v in action:
            out.append(1.0 if v is True else 0.0 if v is False else float(v))
        return tuple(out)
    return (0.0,) * max(1, spec.dims)


def _from_vec(vec: Tuple[float, ...], spec: DomainSpec) -> Any:
    """Turn a normalised float-vector back into an action the system expects."""
    if spec.kind == "bool":
        vals = [bool(round(v)) for v in vec]
    elif spec.kind == "int":
        vals = [int(round(v)) for v in vec]
    else:
        vals = [float(v) for v in vec]
    if spec.scalar or spec.dims == 1:
        return vals[0]
    return tuple(vals)


def _solve_linear(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """Solve ``a x = b`` by Gaussian elimination with partial pivoting. ``None`` if singular."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        pivval = m[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col] / pivval
            if factor:
                for c in range(col, n + 1):
                    m[r][c] -= factor * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def _least_squares(phi: List[List[float]], y: List[float]) -> Optional[List[float]]:
    """Ordinary least squares for ``y ≈ phi · w`` via the normal equations ``φᵀφ w = φᵀy``."""
    if not phi or not phi[0]:
        return None
    k = len(phi[0])
    ata = [[0.0] * k for _ in range(k)]
    aty = [0.0] * k
    for row, target in zip(phi, y):
        for i in range(k):
            aty[i] += row[i] * target
            for j in range(k):
                ata[i][j] += row[i] * row[j]
    # tiny ridge term keeps near-collinear designs invertible without distorting clean fits
    for i in range(k):
        ata[i][i] += 1e-9
    return _solve_linear(ata, aty)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# --------------------------------------------------------------------------- #
# MDL scoring — the Occam razor that ranks the candidate laws
# --------------------------------------------------------------------------- #
_PARAM_BITS = 8.0          # bits charged per free parameter (the simplicity penalty)
_SIGMA_FLOOR = 1e-9        # keeps an exact numeric fit's data-cost finite (but very cheap)


def _numeric_errors(predict: Callable[[Tuple[float, ...]], Any],
                    xs: List[Tuple[float, ...]], ys: List[float]) -> List[float]:
    out: List[float] = []
    for x, y in zip(xs, ys):
        try:
            p = float(predict(x))
            out.append(abs(p - y))
        except Exception:  # noqa: BLE001
            out.append(math.inf)
    return out


def _mdl_numeric(errors: List[float], params: int) -> Tuple[float, float]:
    """Return (mdl, mean_abs_error) for a numeric law from its residuals."""
    n = len(errors)
    if n == 0 or any(math.isinf(e) for e in errors):
        return math.inf, math.inf
    sse = sum(e * e for e in errors)
    sigma2 = max(sse / n, _SIGMA_FLOOR)
    data_bits = 0.5 * n * math.log2(2.0 * math.pi * math.e * sigma2)
    return data_bits + params * _PARAM_BITS, sum(errors) / n


def _mdl_discrete(errors: int, n: int, classes: int, params: int) -> float:
    """Description length for a classifier: misclassification cost + simplicity penalty."""
    if n == 0:
        return math.inf
    bits_per_miss = math.log2(max(classes, 2))
    return errors * bits_per_miss + params * _PARAM_BITS


# --------------------------------------------------------------------------- #
# The generalizer
# --------------------------------------------------------------------------- #
class OpenWorldGeneralizer:
    """Confront an unknown black-box system and model it from first principles.

    Parameters
    ----------
    world_model / causal_model :
        Optional :mod:`nyxara.mind` faculties to fold the discovered ``input → output``
        behaviour into (as transitions and as ``do``-interventions). Both optional.
    belief_model :
        Optional :class:`~nyxara.growth.autonomous_scientist.BeliefModel` to record, with honest
        Beta-Bernoulli confidence, "I have a working model of system X". One is created on demand.
    novelty :
        Optional :class:`~nyxara.growth.frontier.NoveltyArchive` to bias probes toward unexplored
        regions of the input space (active, open-ended exploration).
    memory / knowledge :
        Optional stores a one-line summary of each cracked system is written to. Best-effort.
    seed :
        RNG seed — the whole inquiry is deterministic, so results reproduce (and CI is stable).
    """

    def __init__(self, *, world_model: Any = None, causal_model: Any = None,
                 belief_model: Any = None, novelty: Any = None, memory: Any = None,
                 knowledge: Any = None, seed: int = 0) -> None:
        self.world_model = world_model
        self.causal_model = causal_model
        self.belief_model = belief_model if belief_model is not None else BeliefModel()
        self.novelty = novelty
        self.memory = memory
        self.knowledge = knowledge
        self.seed = int(seed)

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #
    def understand(self, system: Any, *, domain: Optional[DomainSpec] = None,
                   example_action: Any = None, budget: int = 48,
                   label: str = "unknown-system",
                   fold: bool = True) -> UnderstandingReport:
        """Crack ``system`` — an unknown black box — from first principles. Always returns a report.

        ``system`` is a callable ``system(action) -> observation`` or an object with
        ``.interact(action)``. ``budget`` caps the total number of probes. With no ``domain`` or
        ``example_action`` she assumes a single real input and discovers the rest by probing.
        """
        t0 = time.monotonic()
        rng = random.Random(self.seed)
        spec = domain or (DomainSpec.infer(example_action) if example_action is not None
                          else DomainSpec())
        box = _Interactor(system)

        # 1. OBSERVE — seed the input space and poke the box.
        seeds = self._seed_actions(spec, rng, budget)
        probes: List[Probe] = []
        for vec in seeds:
            if len(probes) >= max(8, budget // 2):
                break
            probes.append(self._probe(box, vec, spec))

        # If the box rejects almost everything, there is nothing to model — say so honestly.
        usable = [p for p in probes if p.observation is not None]
        if len(usable) < 4:
            return self._finish(label, spec, [], None, Verdict.INCONCLUSIVE, 0.0, 0.0,
                                math.inf, box, t0, fold, usable)

        # 2/3. HYPOTHESIZE + TEST — fit laws, then run discriminating experiments.
        candidates = self._fit_candidates(usable, spec)
        candidates = self._actively_discriminate(box, spec, probes, candidates, rng, budget)

        # 4. MODEL — choose the simplest law that fits, then prove it generalizes.
        usable = [p for p in probes if p.observation is not None]
        candidates = self._fit_candidates(usable, spec)        # final fit on all evidence
        winner = candidates[0] if candidates else None
        acc, residual = self._validate(box, spec, winner, rng)
        verdict, confidence = self._judge(candidates, acc)
        return self._finish(label, spec, candidates, winner, verdict, confidence, acc,
                            residual, box, t0, fold, usable)

    # ---------------------------------------------------------------------- #
    # OBSERVE — choosing what to poke
    # ---------------------------------------------------------------------- #
    def _seed_actions(self, spec: DomainSpec, rng: random.Random,
                      budget: int) -> List[Tuple[float, ...]]:
        """A prior-free opening: the origin, one axis at a time, the boundaries, then a spread."""
        d = max(1, spec.dims)
        seeds: List[Tuple[float, ...]] = []

        if spec.kind in ("bool", "int") and d <= 6:
            # small discrete space → enumerate it (exhaustive truth is better than sampling)
            if spec.kind == "bool":
                levels = [0.0, 1.0]
            else:
                lo, hi = int(spec.low), int(spec.high)
                levels = [float(v) for v in range(max(lo, -4), min(hi, 4) + 1)]
            grid = [()]
            for _ in range(d):
                grid = [g + (v,) for g in grid for v in levels]
                if len(grid) > 256:
                    break
            seeds.extend(grid)
            return _dedup(seeds)

        bound = max(1.0, spec.high)
        # origin
        seeds.append((0.0,) * d)
        # vary one axis at a time around the origin — isolates each input's effect
        for axis in range(d):
            for val in (-bound, -2.0, -1.0, 1.0, 2.0, bound):
                v = [0.0] * d
                v[axis] = val
                seeds.append(tuple(v))
        # a deterministic joint spread so interactions (products) become visible
        for _ in range(max(8, budget // 3)):
            seeds.append(tuple(rng.uniform(spec.low, spec.high) for _ in range(d)))
        for combo in ((1.0,) * d, (2.0,) * d, (-1.0,) * d, (bound,) * d):
            seeds.append(combo)
        return _dedup(seeds)

    def _probe(self, box: _Interactor, vec: Tuple[float, ...], spec: DomainSpec) -> Probe:
        action = _from_vec(vec, spec)
        obs = box.run(action)
        # store the *realized* action as the normalised vector — so an int/bool domain keeps
        # integer/0-1 coordinates (a float pool sample fed as int 7 is recorded as 7.0, not 7.3).
        return Probe(action=action, observation=obs, vec=_to_vec(action, spec))

    # ---------------------------------------------------------------------- #
    # HYPOTHESIZE — fit candidate laws and rank by MDL
    # ---------------------------------------------------------------------- #
    def _fit_candidates(self, probes: List[Probe], spec: DomainSpec) -> List[CandidateLaw]:
        xs = [p.vec for p in probes]
        obs = [p.observation for p in probes]
        kind = _output_kind(obs)
        cands: List[CandidateLaw] = []
        if kind == "numeric":
            ys = [float(o) for o in obs]
            cands.extend(self._fit_numeric(xs, ys))
            cands.extend(self._fit_modular(xs, ys))
            cands.extend(self._fit_threshold(xs, ys))
            for c in cands:
                errs = _numeric_errors(c.predict, xs, ys)
                c.mdl, c.train_error = _mdl_numeric(errs, c.params)
        elif kind == "boolean":
            ys = [1 if (o is True or o == 1) else 0 for o in obs]
            cands.extend(self._fit_boolean(xs, ys))
            self._score_discrete(cands, xs, ys, classes=2)
        else:  # categorical
            cands.extend(self._fit_categorical(xs, obs))
            self._score_discrete(cands, xs, obs, classes=len({str(o) for o in obs}))
        cands.sort(key=lambda c: c.mdl)
        return cands

    def _fit_numeric(self, xs: List[Tuple[float, ...]], ys: List[float]) -> List[CandidateLaw]:
        out: List[CandidateLaw] = []
        d = len(xs[0])
        # CONSTANT
        c = _mean(ys)
        out.append(CandidateLaw(LawFamily.CONSTANT, f"out = {c:.4g}", 1,
                                predict=(lambda _x, c=c: c)))
        # AFFINE: out = b + Σ wᵢ·xᵢ
        phi = [[1.0] + list(x) for x in xs]
        w = _least_squares(phi, ys)
        if w is not None:
            out.append(CandidateLaw(LawFamily.AFFINE, _affine_str(w), d + 1,
                                    predict=_affine_pred(w)))
        # POLYNOMIAL (degree 2: squares + pairwise products), only for small d
        if d <= 3:
            labels, feats = _poly_features(d)
            phi2 = [[f(x) for f in feats] for x in xs]
            w2 = _least_squares(phi2, ys)
            if w2 is not None:
                out.append(CandidateLaw(LawFamily.POLYNOMIAL, _poly_describe(w2, labels),
                                        len(feats), predict=_poly_pred(w2, feats)))
        # MULTIPLICATIVE: out = c · Πxᵢ
        prod = [math.prod(x) for x in xs]
        denom = sum(p * p for p in prod)
        if denom > 1e-12:
            cm = sum(p * y for p, y in zip(prod, ys)) / denom
            out.append(CandidateLaw(LawFamily.MULTIPLICATIVE, f"out = {cm:.4g}·∏xᵢ", 1,
                                    predict=(lambda x, cm=cm: cm * math.prod(x))))
        return out

    def _fit_modular(self, xs: List[Tuple[float, ...]], ys: List[float]) -> List[CandidateLaw]:
        """out = (a·x_j + b) mod m — only when inputs and outputs are integer-valued."""
        if not _all_int(ys) or not all(_all_int(x) for x in xs):
            return []
        out: List[CandidateLaw] = []
        d = len(xs[0])
        for j in range(d):
            col = [int(round(x[j])) for x in xs]
            if len({*col}) < 2:
                continue
            for m in range(2, 17):
                for a in (1, 2, -1):
                    for b in range(m):
                        if all((a * cj + b) % m == int(round(y)) % m
                               for cj, y in zip(col, ys)):
                            desc = f"out = ({_term(a, j)} + {b}) mod {m}" if b else \
                                   f"out = {_term(a, j)} mod {m}"
                            out.append(CandidateLaw(
                                LawFamily.MODULAR, desc.replace("+ -", "- "), 2,
                                predict=(lambda x, a=a, b=b, m=m, j=j:
                                         (a * int(round(x[j])) + b) % m)))
                            return out  # one exact modular law is enough
        return out

    def _fit_threshold(self, xs: List[Tuple[float, ...]], ys: List[float]) -> List[CandidateLaw]:
        """out = lo if x_j < t else hi — a single-split piecewise-constant law."""
        out: List[CandidateLaw] = []
        d = len(xs[0])
        for j in range(d):
            vals = sorted({x[j] for x in xs})
            if len(vals) < 3:
                continue
            best: Optional[Tuple[float, float, float, float]] = None  # (sse, t, lo, hi)
            for t in [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]:
                lo = [y for x, y in zip(xs, ys) if x[j] < t]
                hi = [y for x, y in zip(xs, ys) if x[j] >= t]
                if not lo or not hi:
                    continue
                lm, hm = _mean(lo), _mean(hi)
                sse = sum((y - lm) ** 2 for y in lo) + sum((y - hm) ** 2 for y in hi)
                if best is None or sse < best[0]:
                    best = (sse, t, lm, hm)
            if best is not None:
                _, t, lm, hm = best
                out.append(CandidateLaw(
                    LawFamily.THRESHOLD,
                    f"out = {lm:.4g} if x[{j}] < {t:.4g} else {hm:.4g}", 3,
                    predict=(lambda x, t=t, lm=lm, hm=hm, j=j: lm if x[j] < t else hm)))
        return out

    def _fit_boolean(self, xs: List[Tuple[float, ...]], ys: List[int]) -> List[CandidateLaw]:
        """Match the observed truth table against the standard boolean operators, then lookup."""
        out: List[CandidateLaw] = []
        d = len(xs[0])
        bits = [tuple(int(round(v)) for v in x) for x in xs]
        for name, fn, params in _boolean_ops(d):
            if all(fn(b) == y for b, y in zip(bits, ys)):
                out.append(CandidateLaw(LawFamily.BOOLEAN, f"out = {name}", params,
                                        predict=_bool_pred(fn), numeric=False))
        # always offer the exact lookup as a fallback (penalised by its size)
        table = {b: y for b, y in zip(bits, ys)}
        maj = round(_mean(ys)) if ys else 0
        out.append(CandidateLaw(
            LawFamily.CATEGORICAL, f"out = truth-table lookup ({len(table)} rows)", len(table),
            predict=_lookup_pred(table, maj), numeric=False))
        return out

    def _fit_categorical(self, xs: List[Tuple[float, ...]], obs: List[Any]) -> List[CandidateLaw]:
        bits = [tuple(round(v, 6) for v in x) for x in xs]
        table: Dict[Tuple[float, ...], Any] = {}
        for b, o in zip(bits, obs):
            table.setdefault(b, o)
        maj = _majority(obs)
        return [CandidateLaw(
            LawFamily.CATEGORICAL, f"out = lookup over {len(table)} cases", len(table),
            predict=_lookup_pred({tuple(round(v, 6) for v in k): v for k, v in table.items()},
                                 maj), numeric=False)]

    def _score_discrete(self, cands: List[CandidateLaw], xs: List[Tuple[float, ...]],
                        ys: List[Any], classes: int) -> None:
        for c in cands:
            errs = 0
            for x, y in zip(xs, ys):
                try:
                    if c.predict(x) != y:
                        errs += 1
                except Exception:  # noqa: BLE001
                    errs += 1
            c.mdl = _mdl_discrete(errs, len(ys), classes, c.params)
            c.train_error = errs / max(1, len(ys))

    # ---------------------------------------------------------------------- #
    # TEST — pick the probe that best tells the surviving hypotheses apart
    # ---------------------------------------------------------------------- #
    def _actively_discriminate(self, box: _Interactor, spec: DomainSpec, probes: List[Probe],
                               candidates: List[CandidateLaw], rng: random.Random,
                               budget: int) -> List[CandidateLaw]:
        seen = {p.vec for p in probes}
        rounds = max(0, budget - len(probes))
        for _ in range(rounds):
            if box.calls >= budget:
                break
            # only laws that actually FIT the evidence so far are worth telling apart; a relative
            # tolerance scales with the output magnitude. When the fitting laws already agree
            # everywhere, no experiment can inform us — stop (this is what keeps probing bounded).
            obs = [p.observation for p in probes
                   if isinstance(p.observation, (int, float)) and not isinstance(p.observation,
                                                                                 bool)]
            scale = max((abs(o) for o in obs), default=1.0)
            tol = 1e-6 + 1e-3 * scale
            survivors = [c for c in candidates if c.train_error <= tol][:4]
            if len(survivors) < 2:
                break
            pool = [tuple(rng.uniform(spec.low, spec.high) for _ in range(max(1, spec.dims)))
                    for _ in range(24)]
            best_vec, best_disagree = None, -1.0
            for vec in pool:
                if vec in seen:
                    continue
                preds = []
                for c in survivors:
                    try:
                        preds.append(c.predict(vec))
                    except Exception:  # noqa: BLE001
                        preds.append(None)
                disagree = _disagreement(preds)
                if self.novelty is not None:
                    disagree += 1e-6 * self._novelty_bonus(vec)
                if disagree > best_disagree:
                    best_vec, best_disagree = vec, disagree
            if best_vec is None or best_disagree <= 1e-9:
                break  # the survivors already agree everywhere — no experiment would inform us
            probes.append(self._probe(box, best_vec, spec))
            seen.add(best_vec)
            usable = [p for p in probes if p.observation is not None]
            candidates = self._fit_candidates(usable, spec)
        return candidates

    def _novelty_bonus(self, vec: Tuple[float, ...]) -> float:
        try:
            return float(self.novelty.novelty(list(vec)))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return 0.0

    # ---------------------------------------------------------------------- #
    # MODEL — prove generalization on fresh inputs, judge honestly
    # ---------------------------------------------------------------------- #
    def _validate(self, box: _Interactor, spec: DomainSpec,
                  winner: Optional[CandidateLaw], rng: random.Random) -> Tuple[float, float]:
        if winner is None:
            return 0.0, math.inf
        d = max(1, spec.dims)
        fresh: List[Tuple[float, ...]] = []
        vrng = random.Random(self.seed + 9973)
        for _ in range(16):
            if spec.kind == "bool":
                fresh.append(tuple(float(vrng.randint(0, 1)) for _ in range(d)))
            elif spec.kind == "int":
                fresh.append(tuple(float(vrng.randint(int(spec.low), int(spec.high)))
                                   for _ in range(d)))
            else:
                fresh.append(tuple(vrng.uniform(spec.low, spec.high) for _ in range(d)))
        hits, errs = 0, []
        for vec in _dedup(fresh):
            actual = box.run(_from_vec(vec, spec))
            if actual is None:
                continue
            try:
                pred = winner.predict(vec)
            except Exception:  # noqa: BLE001
                continue
            if winner.numeric:
                try:
                    e = abs(float(pred) - float(actual))
                except (TypeError, ValueError):
                    continue
                errs.append(e)
                if e <= 1e-4 + 1e-3 * abs(float(actual)):
                    hits += 1
            else:
                errs.append(0.0 if pred == actual else 1.0)
                if pred == actual:
                    hits += 1
        n = len(errs)
        if n == 0:
            return 0.0, math.inf
        return hits / n, sum(errs) / n

    def _judge(self, candidates: List[CandidateLaw], acc: float) -> Tuple[Verdict, float]:
        """Decide the verdict from how well the winning law generalizes to *fresh* inputs.

        Confidence is driven by held-out accuracy (the real generalization signal) and stays
        honestly *never-certain* — capped at 0.95 even on a perfect fit, and held low when the
        law only partly explains the box or fails it outright (it never claims to understand
        what it does not)."""
        if not candidates:
            return Verdict.UNMODELLED, 0.0
        if acc >= 0.9:
            return Verdict.MODELLED, min(0.95, 0.5 + 0.45 * acc)
        if acc >= 0.6:
            return Verdict.PARTIAL, min(0.6, 0.25 + 0.4 * acc)
        return Verdict.UNMODELLED, min(0.25, 0.3 * acc)

    # ---------------------------------------------------------------------- #
    # finish — fold findings into the wider mind, assemble the report
    # ---------------------------------------------------------------------- #
    def _finish(self, label: str, spec: DomainSpec, candidates: List[CandidateLaw],
                winner: Optional[CandidateLaw], verdict: Verdict, confidence: float,
                acc: float, residual: float, box: _Interactor, t0: float,
                fold: bool, usable: List[Probe]) -> UnderstandingReport:
        report = UnderstandingReport(
            system=label, domain=spec.to_dict(), verdict=verdict, confidence=confidence,
            law=winner.description if winner else None,
            law_family=winner.family.value if winner else LawFamily.UNMODELLED.value,
            holdout_accuracy=acc, residual=residual, probes_used=box.calls,
            candidates=candidates, winner=winner,
            samples=[(p.vec, p.observation) for p in usable],
            elapsed_ms=(time.monotonic() - t0) * 1000.0)
        if fold:
            self._fold(report, box, spec)
        return report

    def _fold(self, report: UnderstandingReport, box: _Interactor, spec: DomainSpec) -> None:
        """Best-effort: record the finding in the belief / world / causal models and memory."""
        # belief: "I have a working model of <system>" with honest Beta-Bernoulli confidence.
        # The *evidential weight* is not the display confidence: a failed exhaustive law-search is
        # itself real evidence against modelability, so UNMODELLED folds a weighted REFUTED.
        try:
            verdict_map = {Verdict.MODELLED: ("supported", report.confidence),
                           Verdict.PARTIAL: ("inconclusive", 0.4),
                           Verdict.INCONCLUSIVE: ("inconclusive", 0.0),
                           Verdict.UNMODELLED: ("refuted", 0.5)}
            verdict, weight = verdict_map[report.verdict]
            shim = _ShimReport(
                question=f"Can I model {report.system}?",
                variable=f"open_world:{report.system}",
                statement=report.law or f"behaviour of {report.system}",
                verdict=verdict, confidence=weight,
                reasoning=f"{report.law_family}; holdout_acc={_round(report.holdout_accuracy)}")
            self.belief_model.update(shim)
        except Exception:  # noqa: BLE001
            pass
        # nothing else to fold if she never got clean observations
        if report.winner is None:
            return
        # world model: every probe is a real (state, action, next_state) transition
        if self.world_model is not None:
            try:
                for vec, obs in report.samples:
                    nxt = (float(obs),) if isinstance(obs, (int, float, bool)) else (0.0,)
                    self.world_model.observe(vec, "probe", nxt)
            except Exception:  # noqa: BLE001
                pass
        # causal model: each probe is a do-intervention with an observed effect
        if self.causal_model is not None:
            try:
                self.causal_model.observe(f"probe:{report.system}", intervention=True)
                self.causal_model.observe(f"response:{report.law_family}")
            except Exception:  # noqa: BLE001
                pass
        # memory: a one-line, honest record of what she figured out
        if self.memory is not None and report.verdict in (Verdict.MODELLED, Verdict.PARTIAL):
            try:
                self.memory.remember(
                    f"Open-world: modelled {report.system} → {report.law} "
                    f"(conf {_round(report.confidence)})",
                    tags=["open_world", "generalization", report.law_family])
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# A shim that looks like a Scientist InvestigationReport (so BeliefModel.update folds it)
# --------------------------------------------------------------------------- #
@dataclass
class _ShimHyp:
    variable: str
    statement: str


@dataclass
class _ShimVerdict:
    value: str


@dataclass
class _ShimConclusion:
    verdict: _ShimVerdict
    confidence: float
    reasoning: str


class _ShimReport:
    def __init__(self, *, question: str, variable: str, statement: str, verdict: str,
                 confidence: float, reasoning: str) -> None:
        self.question = question
        self.hypotheses = [_ShimHyp(variable=variable, statement=statement)]
        self.conclusion = _ShimConclusion(_ShimVerdict(verdict), float(confidence), reasoning)


# --------------------------------------------------------------------------- #
# Feature builders, predictors, boolean ops, small utilities
# --------------------------------------------------------------------------- #
def _affine_str(w: Sequence[float]) -> str:
    terms = [f"{w[0]:.4g}"]
    for i, wi in enumerate(w[1:]):
        terms.append(f"{wi:+.4g}·x[{i}]")
    return "out = " + " ".join(terms)


def _affine_pred(w: Sequence[float]) -> Callable[[Tuple[float, ...]], float]:
    def pred(x: Tuple[float, ...], w=tuple(w)) -> float:
        return w[0] + sum(wi * xi for wi, xi in zip(w[1:], x))
    return pred


def _poly_features(d: int) -> Tuple[List[str], List[Callable[[Tuple[float, ...]], float]]]:
    labels: List[str] = ["1"]
    feats: List[Callable[[Tuple[float, ...]], float]] = [lambda _x: 1.0]
    for i in range(d):
        labels.append(f"x[{i}]")
        feats.append(lambda x, i=i: x[i])
    for i in range(d):
        labels.append(f"x[{i}]^2")
        feats.append(lambda x, i=i: x[i] * x[i])
    for i in range(d):
        for j in range(i + 1, d):
            labels.append(f"x[{i}]·x[{j}]")
            feats.append(lambda x, i=i, j=j: x[i] * x[j])
    return labels, feats


def _poly_describe(w: Sequence[float], labels: Sequence[str]) -> str:
    terms = []
    for wi, lab in zip(w, labels):
        if abs(wi) < 1e-9:
            continue
        terms.append(f"{wi:.4g}" if lab == "1" else f"{wi:+.4g}·{lab}")
    return "out = " + (" ".join(terms) if terms else "0")


def _poly_pred(w: Sequence[float],
               feats: List[Callable[[Tuple[float, ...]], float]]
               ) -> Callable[[Tuple[float, ...]], float]:
    def pred(x: Tuple[float, ...], w=tuple(w), feats=feats) -> float:
        return sum(wi * f(x) for wi, f in zip(w, feats))
    return pred


def _term(a: int, j: int) -> str:
    if a == 1:
        return f"x[{j}]"
    if a == -1:
        return f"-x[{j}]"
    return f"{a}·x[{j}]"


def _boolean_ops(d: int) -> List[Tuple[str, Callable[[Tuple[int, ...]], int], int]]:
    ops: List[Tuple[str, Callable[[Tuple[int, ...]], int], int]] = [
        ("FALSE", lambda b: 0, 1),
        ("TRUE", lambda b: 1, 1),
    ]
    for i in range(d):
        ops.append((f"x[{i}]", lambda b, i=i: b[i], 1))
        ops.append((f"NOT x[{i}]", lambda b, i=i: 1 - b[i], 1))
    if d >= 2:
        ops += [
            ("AND", lambda b: int(all(b)), 2),
            ("OR", lambda b: int(any(b)), 2),
            ("XOR", lambda b: sum(b) % 2, 2),
            ("NAND", lambda b: 1 - int(all(b)), 2),
            ("NOR", lambda b: 1 - int(any(b)), 2),
            ("XNOR", lambda b: 1 - (sum(b) % 2), 2),
        ]
    return ops


def _bool_pred(fn: Callable[[Tuple[int, ...]], int]) -> Callable[[Tuple[float, ...]], int]:
    def pred(x: Tuple[float, ...], fn=fn) -> int:
        return int(fn(tuple(int(round(v)) for v in x)))
    return pred


def _lookup_pred(table: Dict[Tuple[float, ...], Any],
                 default: Any) -> Callable[[Tuple[float, ...]], Any]:
    def pred(x: Tuple[float, ...], table=table, default=default) -> Any:
        return table.get(tuple(round(v, 6) if isinstance(v, float) else v for v in x),
                         table.get(tuple(int(round(v)) for v in x), default))
    return pred


def _output_kind(obs: List[Any]) -> str:
    vals = [o for o in obs if o is not None]
    if vals and all(isinstance(o, bool) for o in vals):
        return "boolean"
    if vals and all(isinstance(o, (int, float)) and not isinstance(o, bool) for o in vals):
        return "numeric"
    return "categorical"


def _all_int(xs: Sequence[float]) -> bool:
    return all(abs(x - round(x)) < 1e-9 for x in xs)


def _disagreement(preds: List[Any]) -> float:
    clean = [p for p in preds if p is not None]
    if len(clean) < 2:
        return 0.0
    if all(isinstance(p, (int, float)) and not isinstance(p, bool) for p in clean):
        m = _mean([float(p) for p in clean])
        return sum((float(p) - m) ** 2 for p in clean) / len(clean)
    return float(len({str(p) for p in clean}) - 1)


def _majority(obs: List[Any]) -> Any:
    counts: Dict[str, int] = {}
    rep: Dict[str, Any] = {}
    for o in obs:
        k = str(o)
        counts[k] = counts.get(k, 0) + 1
        rep[k] = o
    if not counts:
        return None
    return rep[max(counts, key=lambda k: counts[k])]


def _dedup(vecs: List[Tuple[float, ...]]) -> List[Tuple[float, ...]]:
    seen: set = set()
    out: List[Tuple[float, ...]] = []
    for v in vecs:
        key = tuple(round(x, 6) for x in v)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


# --------------------------------------------------------------------------- #
# A hidden "alien machine" for demonstrations (the engine never sees its rule)
# --------------------------------------------------------------------------- #
def build_alien_machine(seed: int = 0) -> Tuple[Callable[[Any], Any], DomainSpec, str]:
    """Return ``(system, domain, secret)`` — a randomly-parameterised black box and its hidden law.

    The :class:`OpenWorldGeneralizer` is given only ``system`` (and optionally ``domain``); it has
    **never seen** the rule and must recover it by probing. Used by the ``/generalize`` demo and the
    HTTP endpoint to show her cracking a system she has no prior knowledge of.
    """
    rng = random.Random(seed)
    choice = rng.choice(["affine", "quadratic", "modular", "product", "threshold", "xor"])

    if choice == "affine":
        a, b = rng.randint(-5, 5) or 2, rng.randint(-9, 9)
        return (lambda x, a=a, b=b: a * x + b),  DomainSpec(1, "real"), f"out = {a}*x + {b}"
    if choice == "quadratic":
        a, b, c = rng.randint(1, 4), rng.randint(-4, 4), rng.randint(-6, 6)
        return ((lambda x, a=a, b=b, c=c: a * x * x + b * x + c),
                DomainSpec(1, "real"), f"out = {a}*x^2 + {b}*x + {c}")
    if choice == "modular":
        m = rng.randint(3, 9)
        return ((lambda x, m=m: x % m), DomainSpec(1, "int", -10, 10), f"out = x mod {m}")
    if choice == "product":
        k = rng.randint(2, 4)
        return ((lambda xy, k=k: k * xy[0] * xy[1]),
                DomainSpec(2, "real", scalar=False), f"out = {k}*x0*x1")
    if choice == "threshold":
        t, lo, hi = rng.randint(-3, 3), rng.randint(0, 3), rng.randint(5, 9)
        return ((lambda x, t=t, lo=lo, hi=hi: lo if x < t else hi),
                DomainSpec(1, "real"), f"out = {lo} if x < {t} else {hi}")
    # xor
    return ((lambda ab: bool(int(bool(ab[0])) ^ int(bool(ab[1])))),
            DomainSpec(2, "bool", scalar=False), "out = x0 XOR x1")
