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
   modular/periodic, threshold/piecewise; the transcendental/periodic shapes a polynomial cannot
   reach — power, exponential, logarithmic, rational, and true real-valued sinusoids; structural
   laws — absolute value, min/max extrema, and integer gcd; a **linear recurrence** for stateful /
   sequential systems whose next output depends on their own recent outputs (Fibonacci-like,
   feedback); and — for discrete boxes — boolean operators and categorical lookups. Each candidate
   is ranked by **MDL** (fit + simplicity), so the *simplest law that actually fits* wins — never an
   over-fit table.
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
    "build_system",
    "rebuild_predict",
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
    # --- max-level additions: wider first-principles reach into new environment classes --- #
    POWER = "power"              # out = a·x^b
    EXPONENTIAL = "exponential"  # out = a·e^(k·x)
    LOGARITHMIC = "logarithmic"  # out = a·ln|x| + b
    RATIONAL = "rational"        # out = a/(x+b) + c
    SINUSOIDAL = "sinusoidal"    # out = a·sin(ωx) + b·cos(ωx) + c (real periodicity)
    RECURRENCE = "recurrence"    # out_n = Σ cᵢ·out_(n-i) + b (stateful / sequential systems)
    ABSOLUTE = "absolute"        # out = a·|x_j| + b
    EXTREMUM = "extremum"        # out = min/max over the inputs (with scale + offset)
    GCD = "gcd"                  # out = gcd(x_i, x_j)   (integer structure)
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
    # JSON-safe parameters of the law, so a cracked system can be *persisted* and its predictor
    # rebuilt later (:func:`rebuild_predict`) without re-probing. Empty ⇒ not serialisable
    # (e.g. a raw lookup that only memorises what it saw).
    coeffs: Dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {"family": self.family.value, "description": self.description,
                "params": self.params, "train_error": _round(self.train_error),
                "mdl": _round(self.mdl), "coeffs": dict(self.coeffs)}


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
    recognized: bool = False   # True ⇒ this system was recognised from the registry, not re-cracked

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
            "recognized": self.recognized,
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
                 knowledge: Any = None, registry: Any = None, seed: int = 0) -> None:
        self.world_model = world_model
        self.causal_model = causal_model
        self.belief_model = belief_model if belief_model is not None else BeliefModel()
        self.novelty = novelty
        self.memory = memory
        self.knowledge = knowledge
        # Optional persistent memory of cracked environments (nyxara.growth.env_registry). When
        # wired, a re-encountered system is RECOGNISED instantly instead of being re-probed.
        self.registry = registry
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

        # 0. RECOGNISE — if she has cracked this environment before, reuse the saved law instantly
        # (a few cheap confirmation probes) instead of re-deriving it from scratch. Adaptation is
        # remembering, not amnesia.
        recognized = self._recognize(system, spec, t0)
        if recognized is not None:
            return recognized

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
    # RECOGNITION — reuse a previously-cracked environment instead of re-probing it
    # ---------------------------------------------------------------------- #
    def _recognize(self, system: Any, spec: DomainSpec,
                   t0: float) -> Optional[UnderstandingReport]:
        """If the registry holds this exact environment (verified by re-poking), return a
        MODELLED report built straight from the saved law — no full inquiry. Else ``None``."""
        if self.registry is None:
            return None
        try:
            prof = self.registry.match(system, spec)      # cheap: replays the fingerprint only
        except Exception:  # noqa: BLE001 — recognition is a shortcut, never required
            return None
        if prof is None:
            return None
        predict = prof.rebuild()
        if predict is None:
            return None
        try:
            family = LawFamily(prof.law_family)
        except ValueError:
            family = LawFamily.UNMODELLED
        numeric = family not in (LawFamily.BOOLEAN, LawFamily.CATEGORICAL)
        winner = CandidateLaw(family, prof.law_description, params=1, predict=predict,
                              numeric=numeric, coeffs=dict(prof.coeffs), train_error=0.0, mdl=0.0)
        # count the confirmation probes the match spent, so the report is honest about the cost
        probes = min(len(prof.fingerprint), getattr(self.registry, "min_checks", 3))
        return UnderstandingReport(
            system=prof.label, domain=dict(prof.domain), verdict=Verdict.MODELLED,
            confidence=float(prof.confidence), law=prof.law_description,
            law_family=prof.law_family, holdout_accuracy=1.0, residual=0.0,
            probes_used=probes, candidates=[winner], winner=winner, recognized=True,
            elapsed_ms=(time.monotonic() - t0) * 1000.0)

    def _remember(self, report: UnderstandingReport) -> None:
        """Best-effort: persist a freshly-modelled environment so it is recognised next time."""
        if self.registry is None or report.recognized:
            return
        if report.verdict not in (Verdict.MODELLED, Verdict.PARTIAL):
            return
        try:
            prof = self.registry.build_profile(report)
            if prof is not None:
                self.registry.save(prof)
        except Exception:  # noqa: BLE001 — persistence is a capability, never required
            pass

    # ---------------------------------------------------------------------- #
    # STATIC DATASET — crack a law stated as an (input, output) table (no live box)
    # ---------------------------------------------------------------------- #
    def model_dataset(self, pairs: Sequence[Tuple[Any, Any]], *, query: Any = None,
                      holdout: float = 0.25, label: str = "dataset",
                      fold: bool = False) -> UnderstandingReport:
        """Fit the simplest law that GENERALIZES to a held-out row of a given ``(x, y)`` table.

        Unlike :meth:`understand`, there is no live box to probe — the evidence is exactly the
        rows provided. She fits candidate laws on a training split, validates on the held-out
        rows (the honest generalization signal), then refits the winner on all rows for the best
        final model. Single numeric input only; returns an honest ``UNMODELLED`` verdict when
        nothing generalizes. Reuses the same law family / MDL / judging machinery as ``understand``.
        """
        t0 = time.monotonic()
        clean: List[Tuple[float, float]] = []
        for x, y in pairs:
            try:
                clean.append((float(x), float(y)))
            except (TypeError, ValueError):
                continue
        # dedup by input, order preserved
        seen: set = set()
        rows: List[Tuple[float, float]] = []
        for x, y in clean:
            if x not in seen:
                seen.add(x)
                rows.append((x, y))
        spec = DomainSpec(dims=1, kind=("int" if all(v == int(v) for v, _ in rows) else "real"),
                          scalar=True)
        if len(rows) < 3:
            return UnderstandingReport(system=label, domain=spec.to_dict(),
                                       verdict=Verdict.INCONCLUSIVE, confidence=0.0,
                                       elapsed_ms=(time.monotonic() - t0) * 1000.0)
        probes = [Probe(action=x, observation=y, vec=_to_vec(x, spec)) for x, y in rows]

        # deterministic holdout split: every k-th row is held out (at least one)
        n_hold = max(1, int(round(holdout * len(probes))))
        step = max(2, len(probes) // n_hold)
        held = [p for i, p in enumerate(probes) if (i + 1) % step == 0]
        train = [p for p in probes if p not in held]
        if len(train) < 2:
            train, held = probes, probes[-1:]

        train_cands = self._fit_candidates(train, spec)
        winner = train_cands[0] if train_cands else None
        acc, residual = self._validate_static(winner, held)
        verdict, confidence = self._judge(train_cands, acc)

        # refit the winner on ALL rows so the reported model uses every datum
        all_cands = self._fit_candidates(probes, spec)
        if all_cands:
            winner = all_cands[0]
        report = UnderstandingReport(
            system=label, domain=spec.to_dict(), verdict=verdict, confidence=confidence,
            law=winner.description if winner else None,
            law_family=winner.family.value if winner else LawFamily.UNMODELLED.value,
            holdout_accuracy=acc, residual=residual, probes_used=len(probes),
            candidates=all_cands or train_cands, winner=winner,
            samples=[(p.vec, p.observation) for p in probes],
            elapsed_ms=(time.monotonic() - t0) * 1000.0)
        return report

    @staticmethod
    def _validate_static(winner: Optional[CandidateLaw],
                         held: Sequence[Probe]) -> Tuple[float, float]:
        """Held-out accuracy / mean error of ``winner`` on rows it was not fitted on."""
        if winner is None or not held:
            return 0.0, math.inf
        hits, errs = 0, []
        for p in held:
            try:
                pred = winner.predict(p.vec)
                actual = float(p.observation)
                err = abs(float(pred) - actual)
            except (TypeError, ValueError):
                continue
            errs.append(err)
            if err <= 1e-4 + 1e-3 * abs(actual):
                hits += 1
        n = len(errs)
        return (hits / n, sum(errs) / n) if n else (0.0, math.inf)

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
            elif d == 1:
                # A single integer axis: enumerate a LONG contiguous run (bounded), so a sequential
                # law with memory (a linear recurrence) has enough consecutive terms to be induced,
                # while staying within the stated domain.
                lo, hi = int(spec.low), int(spec.high)
                start = max(lo, -8)
                end = min(hi, start + 23)
                levels = [float(v) for v in range(start, end + 1)]
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

        # Continuous domain: probe *within the stated bounds* only (a law is only defined where the
        # Master said the system lives — probing outside it invents undefined/complex outputs).
        lo, hi = float(spec.low), float(spec.high)
        span = hi - lo if hi > lo else max(1.0, abs(hi))
        baseline = 0.0 if lo <= 0.0 <= hi else lo   # the "hold other axes here" reference point
        axis_vals = sorted({round(lo + span * f, 6)
                            for f in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)}
                           | ({0.0} if lo <= 0.0 <= hi else set()))
        # vary one axis at a time (others at the baseline) — isolates each input's effect
        seeds.append((baseline,) * d)
        for axis in range(d):
            for val in axis_vals:
                v = [baseline] * d
                v[axis] = val
                seeds.append(tuple(v))
        # a deterministic joint spread so interactions (products) become visible
        for _ in range(max(8, budget // 3)):
            seeds.append(tuple(rng.uniform(lo, hi) for _ in range(d)))
        for frac in (0.25, 0.5, 0.75, 1.0):
            seeds.append((round(lo + span * frac, 6),) * d)
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
            cands.extend(self._fit_transcendental(xs, ys))
            cands.extend(self._fit_structural(xs, ys))
            cands.extend(self._fit_recurrence(xs, ys))
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
                                predict=(lambda _x, c=c: c), coeffs={"c": c}))
        # AFFINE: out = b + Σ wᵢ·xᵢ
        phi = [[1.0] + list(x) for x in xs]
        w = _least_squares(phi, ys)
        if w is not None:
            out.append(CandidateLaw(LawFamily.AFFINE, _affine_str(w), d + 1,
                                    predict=_affine_pred(w), coeffs={"w": [float(v) for v in w]}))
        # POLYNOMIAL (degree 2: squares + pairwise products), only for small d
        if d <= 3:
            labels, feats = _poly_features(d)
            phi2 = [[f(x) for f in feats] for x in xs]
            w2 = _least_squares(phi2, ys)
            if w2 is not None:
                out.append(CandidateLaw(LawFamily.POLYNOMIAL, _poly_describe(w2, labels),
                                        len(feats), predict=_poly_pred(w2, feats),
                                        coeffs={"w": [float(v) for v in w2], "d": d}))
        # MULTIPLICATIVE: out = c · Πxᵢ
        prod = [math.prod(x) for x in xs]
        denom = sum(p * p for p in prod)
        if denom > 1e-12:
            cm = sum(p * y for p, y in zip(prod, ys)) / denom
            out.append(CandidateLaw(LawFamily.MULTIPLICATIVE, f"out = {cm:.4g}·∏xᵢ", 1,
                                    predict=(lambda x, cm=cm: cm * math.prod(x)),
                                    coeffs={"c": float(cm)}))
        return out

    # ---- max-level families: transcendental / periodic / rational (single-input) ---- #
    def _fit_transcendental(self, xs: List[Tuple[float, ...]],
                            ys: List[float]) -> List[CandidateLaw]:
        """POWER / EXPONENTIAL / LOG / RATIONAL / SINUSOIDAL — the shapes a polynomial can't reach.

        Single-input only (``d == 1``): these laws are about how one quantity bends, saturates or
        oscillates, and multi-input versions would overfit. Each is fitted by a *linearising*
        transform + least squares (so the fit is exact when the law holds), guarded against invalid
        domains (log/^ of non-positive values), and left to the shared MDL/holdout machinery to
        accept or reject — a shape that does not truly hold simply loses to a simpler law or fails
        the fresh-input validation."""
        if len(xs[0]) != 1:
            return []
        out: List[CandidateLaw] = []
        x1 = [x[0] for x in xs]
        _MIN_VALID = 5   # enough valid points that a transform fit is real, not two-point luck

        # POWER: y = a·x^b  →  ln y = ln a + b·ln x   (fit on the x>0, y>0 subset)
        pw = [(v, y) for v, y in zip(x1, ys) if v > 1e-9 and y > 1e-9]
        if len(pw) >= _MIN_VALID:
            lw = _least_squares([[1.0, math.log(v)] for v, _ in pw], [math.log(y) for _, y in pw])
            if lw is not None:
                a, b = math.exp(lw[0]), lw[1]
                out.append(CandidateLaw(LawFamily.POWER, f"out = {a:.4g}·x^{b:.4g}", 2,
                                        predict=(lambda x, a=a, b=b:
                                                 a * (x[0] ** b) if x[0] > 0 else 0.0),
                                        coeffs={"a": float(a), "b": float(b)}))

        # EXPONENTIAL: y = a·e^(k·x)  →  ln|y| = ln|a| + k·x   (fit on the single-signed subset)
        pos = [(v, y) for v, y in zip(x1, ys) if y > 1e-9]
        neg = [(v, y) for v, y in zip(x1, ys) if y < -1e-9]
        ex = pos if len(pos) >= len(neg) else neg
        if len(ex) >= _MIN_VALID:
            sign = 1.0 if ex is pos else -1.0
            lw = _least_squares([[1.0, v] for v, _ in ex], [math.log(abs(y)) for _, y in ex])
            if lw is not None:
                a, k = sign * math.exp(lw[0]), lw[1]
                out.append(CandidateLaw(LawFamily.EXPONENTIAL, f"out = {a:.4g}·e^({k:.4g}·x)", 2,
                                        predict=(lambda x, a=a, k=k: a * math.exp(k * x[0])),
                                        coeffs={"a": float(a), "k": float(k)}))

        # LOGARITHMIC: y = a·ln|x| + b   (fit on the x≠0 subset)
        lg = [(v, y) for v, y in zip(x1, ys) if abs(v) > 1e-9]
        if len(lg) >= _MIN_VALID:
            lw = _least_squares([[1.0, math.log(abs(v))] for v, _ in lg], [y for _, y in lg])
            if lw is not None:
                b, a = lw[0], lw[1]
                out.append(CandidateLaw(LawFamily.LOGARITHMIC, f"out = {a:.4g}·ln|x| + {b:.4g}", 2,
                                        predict=(lambda x, a=a, b=b:
                                                 a * math.log(abs(x[0])) + b if x[0] else b),
                                        coeffs={"a": float(a), "b": float(b)}))

        # RATIONAL: y = a/(x+p) + c — grid the pole offset p, least-squares a,c for each, keep best.
        best_rat = self._fit_rational(x1, ys)
        if best_rat is not None:
            out.append(best_rat)

        # SINUSOIDAL: y = a·sin(ωx) + b·cos(ωx) + c — grid ω, least-squares [sin,cos,1].
        best_sin = self._fit_sinusoid(x1, ys)
        if best_sin is not None:
            out.append(best_sin)
        return out

    @staticmethod
    def _fit_rational(x1: List[float], ys: List[float]) -> Optional[CandidateLaw]:
        """out = a/(x+p) + c. Solved EXACTLY, not by grid: the identity y·x = (a+c·p) + c·x − p·y
        is linear in (a+c·p, c, −p), so ordinary least squares over features ``[1, x, y]`` recovers
        the pole and both coefficients in one shot when the law holds."""
        if len(x1) < 4:
            return None
        w = _least_squares([[1.0, v, y] for v, y in zip(x1, ys)], [v * y for v, y in zip(x1, ys)])
        if w is None:
            return None
        c = w[1]
        p = -w[2]
        a = w[0] - c * p                      # w[0] = a + c·p
        if abs(a) < 1e-9 or any(abs(v + p) < 1e-6 for v in x1):
            return None                       # degenerate (constant) or a pole sitting on the data
        sign = "+" if p >= 0 else "-"
        return CandidateLaw(LawFamily.RATIONAL,
                            f"out = {a:.4g}/(x {sign} {abs(p):.4g}) + {c:.4g}", 3,
                            predict=(lambda x, a=a, p=p, c=c:
                                     a / (x[0] + p) + c if abs(x[0] + p) > 1e-9 else c),
                            coeffs={"a": float(a), "p": float(p), "c": float(c)})

    @staticmethod
    def _fit_sinusoid(x1: List[float], ys: List[float]) -> Optional[CandidateLaw]:
        best: Optional[Tuple[float, float, float, float, float]] = None  # (sse, a, b, c, w)
        for j in range(1, 61):
            omega = 0.1 * j
            phi = [[math.sin(omega * v), math.cos(omega * v), 1.0] for v in x1]
            wv = _least_squares(phi, ys)
            if wv is None:
                continue
            a, b, c = wv
            sse = sum((a * math.sin(omega * v) + b * math.cos(omega * v) + c - y) ** 2
                      for v, y in zip(x1, ys))
            if best is None or sse < best[0]:
                best = (sse, a, b, c, omega)
        if best is None:
            return None
        _sse, a, b, c, omega = best
        return CandidateLaw(LawFamily.SINUSOIDAL,
                            f"out = {a:.4g}·sin({omega:.4g}x) + {b:.4g}·cos({omega:.4g}x) + {c:.4g}",
                            4,
                            predict=(lambda x, a=a, b=b, c=c, w=omega:
                                     a * math.sin(w * x[0]) + b * math.cos(w * x[0]) + c),
                            coeffs={"a": float(a), "b": float(b), "c": float(c),
                                    "omega": float(omega)})

    # ---- structural families: absolute value, extremum, integer gcd ---- #
    def _fit_structural(self, xs: List[Tuple[float, ...]], ys: List[float]) -> List[CandidateLaw]:
        out: List[CandidateLaw] = []
        d = len(xs[0])
        # ABSOLUTE: out = a·|x_j| + b (per axis)
        for j in range(d):
            phi = [[1.0, abs(x[j])] for x in xs]
            w = _least_squares(phi, ys)
            if w is not None:
                b, a = w
                out.append(CandidateLaw(LawFamily.ABSOLUTE, f"out = {a:.4g}·|x[{j}]| + {b:.4g}", 2,
                                        predict=(lambda x, a=a, b=b, j=j: a * abs(x[j]) + b),
                                        coeffs={"a": float(a), "b": float(b), "j": j}))
        if d >= 2:
            # EXTREMUM: out = s·min(x) + t  and  out = s·max(x) + t
            for which, fn in (("min", min), ("max", max)):
                phi = [[1.0, fn(x)] for x in xs]
                w = _least_squares(phi, ys)
                if w is not None:
                    t, s = w
                    out.append(CandidateLaw(
                        LawFamily.EXTREMUM, f"out = {s:.4g}·{which}(x) + {t:.4g}", 2,
                        predict=(lambda x, s=s, t=t, fn=fn: s * fn(x) + t),
                        coeffs={"s": float(s), "t": float(t), "which": which}))
            # GCD: out = gcd(x_i, x_j) — only when everything is integer-valued
            if _all_int(ys) and all(_all_int(x) for x in xs):
                for i in range(d):
                    for k in range(i + 1, d):
                        pred = (lambda x, i=i, k=k:
                                float(math.gcd(int(round(x[i])), int(round(x[k])))))
                        if all(abs(pred(x) - y) < 1e-9 for x, y in zip(xs, ys)):
                            out.append(CandidateLaw(
                                LawFamily.GCD, f"out = gcd(x[{i}], x[{k}])", 1,
                                predict=pred, coeffs={"i": i, "k": k}))
                            break
        return out

    # ---- recurrence: the first family that models a system WITH MEMORY ---- #
    def _fit_recurrence(self, xs: List[Tuple[float, ...]], ys: List[float]) -> List[CandidateLaw]:
        """out_n = Σ_{i=1..order} cᵢ·out_(n-i) + b — fitted over the sequence ordered by a 1-D
        integer index. This is the shape a stateless polynomial cannot reach: a system whose next
        output depends on its own recent outputs (Fibonacci-like, feedback, decay). Requires a
        contiguous/arithmetic 1-D integer index so 'the next value' is well defined."""
        if len(xs[0]) != 1:
            return []
        pairs = sorted(zip((x[0] for x in xs), ys), key=lambda p: p[0])
        idx = [p[0] for p in pairs]
        seq = [p[1] for p in pairs]
        if len(seq) < 6 or not _all_int(idx):
            return []
        # index must be a clean arithmetic progression (so out_(n-1) is the true predecessor)
        steps = {round(idx[i + 1] - idx[i], 6) for i in range(len(idx) - 1)}
        if len(steps) != 1 or abs(next(iter(steps))) < 1e-9:
            return []
        out: List[CandidateLaw] = []
        for order in (1, 2, 3):
            if len(seq) <= order + 2:
                break
            phi = [[*seq[n - order:n][::-1], 1.0] for n in range(order, len(seq))]
            tgt = seq[order:]
            w = _least_squares(phi, tgt)
            if w is None:
                continue
            coeffs = [float(v) for v in w[:order]]
            b = float(w[order])
            err = sum(abs(sum(c * s for c, s in zip(coeffs, phi[n][:order])) + b - tgt[n])
                      for n in range(len(tgt))) / max(1, len(tgt))
            # STRICT: accept only a recurrence that holds (near-)exactly, so a merely
            # near-geometric sequence (e.g. Fibonacci at order 1) is rejected in favour of the
            # true lowest exact order — never an approximate memory law that then diverges.
            if err > 1e-6 * (max(abs(v) for v in seq) + 1.0):
                continue
            terms = " + ".join(f"{c:.4g}·out[n-{i + 1}]" for i, c in enumerate(coeffs))
            desc = f"out[n] = {terms}" + (f" + {b:.4g}" if abs(b) > 1e-9 else "")
            # a recurrence predicts the NEXT value from the last `order` outputs; as a point law
            # over the index it extrapolates the sequence forward deterministically.
            pred = _recurrence_pred(coeffs, b, idx[0], idx[1] - idx[0], seq[:order])
            out.append(CandidateLaw(LawFamily.RECURRENCE, desc, order + 1, predict=pred,
                                    coeffs={"c": coeffs, "b": b, "x0": float(idx[0]),
                                            "step": float(idx[1] - idx[0]),
                                            "seed": [float(v) for v in seq[:order]]}))
            break  # the lowest order that fits is the honest one
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
                                         (a * int(round(x[j])) + b) % m),
                                coeffs={"a": a, "b": b, "m": m, "j": j}))
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
                    predict=(lambda x, t=t, lm=lm, hm=hm, j=j: lm if x[j] < t else hm),
                    coeffs={"t": float(t), "lo": float(lm), "hi": float(hm), "j": j}))
        return out

    def _fit_boolean(self, xs: List[Tuple[float, ...]], ys: List[int]) -> List[CandidateLaw]:
        """Match the observed truth table against the standard boolean operators, then lookup."""
        out: List[CandidateLaw] = []
        d = len(xs[0])
        bits = [tuple(int(round(v)) for v in x) for x in xs]
        for name, fn, params in _boolean_ops(d):
            if all(fn(b) == y for b, y in zip(bits, ys)):
                out.append(CandidateLaw(LawFamily.BOOLEAN, f"out = {name}", params,
                                        predict=_bool_pred(fn), numeric=False,
                                        coeffs={"op": name, "d": d}))
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
        self._remember(report)
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


def _recurrence_pred(coeffs: Sequence[float], b: float, x0: float, step: float,
                     seed: Sequence[float]) -> Callable[[Tuple[float, ...]], float]:
    """Roll a linear recurrence forward from its seed to predict the value at any index.

    ``out[n] = Σ coeffsᵢ·out[n-1-i] + b`` with ``out[0..order-1] = seed``. The index of an action
    is ``n = round((x - x0)/step)``; values below the seed clamp to the seed, values above are
    computed deterministically by iterating the recurrence — so the law *extrapolates the sequence*
    rather than memorising it."""
    order = len(coeffs)

    def pred(x: Tuple[float, ...], coeffs=tuple(coeffs), b=float(b), x0=float(x0),
             step=float(step), seed=tuple(float(s) for s in seed), order=order) -> float:
        if not step or order == 0:
            return seed[0] if seed else 0.0
        n = int(round((x[0] - x0) / step))
        if n < 0:
            return seed[0] if seed else 0.0
        cur = list(seed)
        while len(cur) <= n:
            m = len(cur)
            cur.append(sum(c * cur[m - 1 - i] for i, c in enumerate(coeffs)) + b)
        return cur[n]
    return pred


def rebuild_predict(family: Any, coeffs: Dict[str, Any],
                    dims: int = 1) -> Optional[Callable[[Tuple[float, ...]], Any]]:
    """Reconstruct a law's predictor from its serialised ``coeffs`` — the inverse of fitting.

    This is what lets a cracked system be *persisted* and its behaviour reused later without
    re-probing (:class:`~nyxara.growth.env_registry.EnvironmentRegistry`). Returns ``None`` for a
    law with no serialisable parameters (a raw lookup that only memorised what it saw), so the
    caller can fall back to re-modelling honestly."""
    try:
        fam = family.value if hasattr(family, "value") else str(family)
        c = coeffs or {}
        if fam == LawFamily.CONSTANT.value:
            k = float(c["c"]); return lambda _x, k=k: k
        if fam == LawFamily.AFFINE.value:
            return _affine_pred([float(v) for v in c["w"]])
        if fam == LawFamily.POLYNOMIAL.value:
            _labels, feats = _poly_features(int(c.get("d", dims)))
            return _poly_pred([float(v) for v in c["w"]], feats)
        if fam == LawFamily.MULTIPLICATIVE.value:
            cm = float(c["c"]); return lambda x, cm=cm: cm * math.prod(x)
        if fam == LawFamily.MODULAR.value:
            a, b, m, j = int(c["a"]), int(c["b"]), int(c["m"]), int(c["j"])
            return lambda x, a=a, b=b, m=m, j=j: (a * int(round(x[j])) + b) % m
        if fam == LawFamily.THRESHOLD.value:
            t, lo, hi, j = float(c["t"]), float(c["lo"]), float(c["hi"]), int(c["j"])
            return lambda x, t=t, lo=lo, hi=hi, j=j: lo if x[j] < t else hi
        if fam == LawFamily.POWER.value:
            a, b = float(c["a"]), float(c["b"])
            return lambda x, a=a, b=b: a * (x[0] ** b) if x[0] > 0 else 0.0
        if fam == LawFamily.EXPONENTIAL.value:
            a, k = float(c["a"]), float(c["k"])
            return lambda x, a=a, k=k: a * math.exp(k * x[0])
        if fam == LawFamily.LOGARITHMIC.value:
            a, b = float(c["a"]), float(c["b"])
            return lambda x, a=a, b=b: a * math.log(abs(x[0])) + b if x[0] else b
        if fam == LawFamily.RATIONAL.value:
            a, p, cc = float(c["a"]), float(c["p"]), float(c["c"])
            return lambda x, a=a, p=p, cc=cc: a / (x[0] + p) + cc if abs(x[0] + p) > 1e-9 else cc
        if fam == LawFamily.SINUSOIDAL.value:
            a, b, cc, w = float(c["a"]), float(c["b"]), float(c["c"]), float(c["omega"])
            return lambda x, a=a, b=b, cc=cc, w=w: a * math.sin(w * x[0]) + b * math.cos(w * x[0]) + cc
        if fam == LawFamily.RECURRENCE.value:
            return _recurrence_pred([float(v) for v in c["c"]], float(c["b"]),
                                    float(c["x0"]), float(c["step"]),
                                    [float(v) for v in c["seed"]])
        if fam == LawFamily.ABSOLUTE.value:
            a, b, j = float(c["a"]), float(c["b"]), int(c["j"])
            return lambda x, a=a, b=b, j=j: a * abs(x[j]) + b
        if fam == LawFamily.EXTREMUM.value:
            s, t, fn = float(c["s"]), float(c["t"]), (min if c.get("which") == "min" else max)
            return lambda x, s=s, t=t, fn=fn: s * fn(x) + t
        if fam == LawFamily.GCD.value:
            i, k = int(c["i"]), int(c["k"])
            return lambda x, i=i, k=k: float(math.gcd(int(round(x[i])), int(round(x[k]))))
        if fam == LawFamily.BOOLEAN.value:
            name = c["op"]
            for nm, fn, _p in _boolean_ops(int(c.get("d", dims))):
                if nm == name:
                    return _bool_pred(fn)
        return None
    except Exception:  # noqa: BLE001 — a malformed profile is simply not rebuildable
        return None


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
# Build a black box from a *declarative* law spec (so a system can cross an API boundary)
# --------------------------------------------------------------------------- #
def build_system(family: Any, params: Dict[str, Any], *, dims: int = 1, kind: str = "real",
                 low: float = -6.0, high: float = 6.0,
                 scalar: Optional[bool] = None) -> Tuple[Optional[Callable[[Any], Any]], DomainSpec]:
    """Turn a named law family + parameters into a callable black box and its :class:`DomainSpec`.

    A live callable cannot cross an HTTP/RPC boundary, but a *declaration* of one can — a family
    name (``"affine"``, ``"sinusoidal"`` …) plus its parameters. This rebuilds the predictor with
    :func:`rebuild_predict` and wraps it to accept ordinary actions, so the Master can hand NYXARA a
    system to model over the wire. Returns ``(system, spec)``; ``system`` is ``None`` when the
    family/params are not rebuildable."""
    spec = DomainSpec(dims=int(dims), kind=str(kind), low=float(low), high=float(high),
                      scalar=(int(dims) == 1 if scalar is None else bool(scalar)))
    predict = rebuild_predict(family, params or {}, dims=int(dims))
    if predict is None:
        return None, spec

    def system(action: Any, predict=predict, spec=spec) -> Any:
        return predict(_to_vec(action, spec))
    return system, spec


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
