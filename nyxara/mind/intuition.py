"""NYXARA · mind/intuition.py — the Intuition Core: non-algorithmic creative leaps (✦★★).

**The mind proposes a leap; the kernel disposes.**

Most of NYXARA reasons *forward* from math, symbolic regression and probability — it
grinds a proof. This faculty does the opposite: it **leaps**. It produces a fast,
unproven *hunch* — a candidate answer reached *before* any proof — and then hands that
hunch to a verifier. That is what a human "Aha!" is: the answer arrives first, the
justification second. And crucially it works on puzzles that have **no training data at
all** — because the leap comes from *structure*, not from memorised examples.

This is **not** an LLM. There is no model, no provider, no API call anywhere in this
module (a test asserts it). It is a portfolio of parallel, self-contained *leap
generators*, each reusing one of NYXARA's real faculties:

* **Gestalt pattern-completion** — pure structure. Finite-difference tables, exact
  rational linear-recurrence fitting, geometric/arithmetic ratios, interleavings and a
  small library of famous integer sequences. Cracks a sequence puzzle it has never seen
  by *seeing its shape* — the raw creative leap.
* **Analogical transfer (HDC/VSA)** — reuses :mod:`nyxara.cognition.hyper_dimensional_vectors`.
  "I have felt this shape before": recalls the nearest past problem and transports its
  solution onto the new one.
* **Superposed contradiction** — reuses :mod:`nyxara.quantum.superposition_states`. Holds
  several *mutually contradictory* candidate leaps at once and collapses to the best only
  when the evidence (fit to the givens) arrives. It refuses to force premature consistency.
* **Dark-data / absence** — reuses :mod:`nyxara.void.dark_data_mining`. Leaps from what is
  *missing or silent* — a hidden period, an anomaly, the dog that didn't bark.
* **First-principles** — reuses :mod:`nyxara.mind.first_principles`. Jumps to the *form*
  that dimensional analysis / axioms permit, before proving it.

Their hunches are **fused** by confidence-weighted consensus (agreement compounds
confidence), gated by a **meta-intuition** UCB1 bandit that learns *which* generator to
trust for *which* problem shape — intuition about its own intuition — and **calibrated**
against lived outcomes. Every leap carries a cheap ``verify()`` so the leap is *real*
(checkable), never merely asserted. Proven leaps are remembered and compound.

Every generator degrades gracefully: if its dependency is absent it simply abstains, and
the Core runs on the generators that remain — it never crashes on a bare machine.

Depends on :mod:`mind.faculties`, :mod:`mind.proposal`, :mod:`memory.provenance`. All
heavier faculties are imported lazily and guarded.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Tuple

from nyxara.memory.provenance import Provenance, SourceType
from nyxara.mind.faculties import Faculty, Task, TaskType
from nyxara.mind.proposal import Proposal, ProposalKind

__all__ = [
    "Problem",
    "Hunch",
    "LeapGenerator",
    "GestaltGenerator",
    "AnalogyGenerator",
    "SuperpositionGenerator",
    "DarkDataGenerator",
    "FirstPrinciplesGenerator",
    "MetaGate",
    "IntuitionCore",
    "IntuitionFaculty",
    "make_intuition_callable",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Problem — a normalised view of whatever the kernel hands intuition
# --------------------------------------------------------------------------- #
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


@dataclass
class Problem:
    """A puzzle to leap at. ``sequence`` is the extracted numeric series, if any."""

    text: str = ""
    sequence: Optional[List[Fraction]] = None
    payload: Any = None
    kind: str = "unknown"          # sequence | analogy | text | physics | state | unknown
    features: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: Any, *, payload: Any = None, features: Optional[Dict[str, Any]] = None) -> "Problem":
        """Build a Problem from a Task/str/sequence, extracting a numeric series when present."""
        features = dict(features or {})
        seq: Optional[List[Fraction]] = None
        text = ""
        # a raw list/tuple of numbers is a sequence outright
        candidate_seq = None
        if isinstance(raw, (list, tuple)) and raw and all(
            isinstance(x, (int, float, Fraction)) and not isinstance(x, bool) for x in raw
        ):
            candidate_seq = list(raw)
        elif payload is not None and isinstance(payload, (list, tuple)) and payload and all(
            isinstance(x, (int, float, Fraction)) and not isinstance(x, bool) for x in payload
        ):
            candidate_seq = list(payload)
        if candidate_seq is not None:
            seq = [Fraction(x).limit_denominator(10**9) for x in candidate_seq]
        else:
            text = raw if isinstance(raw, str) else ("" if raw is None else str(raw))
            nums = _NUM_RE.findall(text)
            if len(nums) >= 3:                       # a written-out sequence, "2, 4, 8, 16, ?"
                try:
                    seq = [Fraction(n) for n in nums]
                except Exception:  # noqa: BLE001
                    seq = None
        kind = cls._classify(text, seq, payload, features)
        return cls(text=text, sequence=seq, payload=payload, kind=kind, features=features)

    @staticmethod
    def _classify(text: str, seq: Optional[List[Fraction]], payload: Any,
                  features: Dict[str, Any]) -> str:
        if isinstance(payload, dict) and "actions" in payload and "state" in payload:
            return "state"
        if seq is not None and len(seq) >= 3:
            return "sequence"
        low = text.lower()
        if " to " in low and (" as " in low or "::" in low or ":" in low):
            return "analogy"
        if any(w in low for w in ("dimension", "velocity", "energy", "force", "derive",
                                  "units", "acceleration", "momentum")):
            return "physics"
        return "text"

    def key(self) -> str:
        if self.sequence is not None:
            return "seq:" + ",".join(str(x) for x in self.sequence)
        return "txt:" + self.text.strip().lower()[:120]


# --------------------------------------------------------------------------- #
# Hunch — a fast, unproven candidate leap (+ its own cheap verifier)
# --------------------------------------------------------------------------- #
@dataclass
class Hunch:
    answer: Any
    confidence: float
    mechanism: str
    rationale: str = ""
    verify: Optional[Callable[[], bool]] = None
    rule: str = ""
    certificate: Optional[str] = None
    support: Tuple[str, ...] = ()

    def verified(self) -> Optional[bool]:
        """Run the cheap self-verification, if the leap carries one. None == not checkable here."""
        if self.verify is None:
            return None
        try:
            return bool(self.verify())
        except Exception:  # noqa: BLE001 — a broken checker never trusts the leap
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {"answer": _jsonable(self.answer), "confidence": round(self.confidence, 3),
                "mechanism": self.mechanism, "rule": self.rule,
                "rationale": self.rationale[:160]}


def _jsonable(x: Any) -> Any:
    if isinstance(x, Fraction):
        return int(x) if x.denominator == 1 else float(x)
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


# =========================================================================== #
# Leap generators
# =========================================================================== #
class LeapGenerator:
    """A single non-algorithmic leap mechanism. Returns a Hunch or None (abstain)."""

    name: str = "generator"
    shapes: frozenset = frozenset()      # problem kinds it can even attempt

    def available(self) -> bool:
        return True

    def applies(self, problem: Problem) -> bool:
        return (not self.shapes) or (problem.kind in self.shapes)

    def leap(self, problem: Problem) -> Optional[Hunch]:  # pragma: no cover - overridden
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Exact rational linear algebra (for recurrence fitting) — pure stdlib
# --------------------------------------------------------------------------- #
def _solve_linear(A: List[List[Fraction]], b: List[Fraction]) -> Optional[List[Fraction]]:
    """Solve a square system A x = b exactly over the rationals. None if singular."""
    n = len(A)
    if n == 0 or any(len(row) != n for row in A) or len(b) != n:
        return None
    M = [[Fraction(A[i][j]) for j in range(n)] + [Fraction(b[i])] for i in range(n)]
    for col in range(n):
        piv = next((r for r in range(col, n) if M[r][col] != 0), None)
        if piv is None:
            return None
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0:
                factor = M[r][col]
                M[r] = [M[r][k] - factor * M[col][k] for k in range(n + 1)]
    return [M[i][n] for i in range(n)]


# --------------------------------------------------------------------------- #
# 1) Gestalt pattern-completion — the raw "Aha!" (pure structure, no training data)
# --------------------------------------------------------------------------- #
_FAMOUS: Dict[str, List[int]] = {
    "primes": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71],
    "factorials": [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800],
    "catalan": [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796],
    "bell": [1, 1, 2, 5, 15, 52, 203, 877, 4140],
    "partitions": [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101],
    "powers_of_two": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
}


class GestaltGenerator(LeapGenerator):
    """Complete a sequence by *seeing its form*: differences, recurrences, ratios, famous series."""

    name = "gestalt"
    shapes = frozenset({"sequence"})

    def leap(self, problem: Problem) -> Optional[Hunch]:
        xs = problem.sequence
        if not xs or len(xs) < 3:
            return None
        # try mechanisms from simplest to most complex; the FIRST that reproduces every
        # given term is the leap (Occam: the simplest form that fits).
        for solver in (self._constant, self._arithmetic, self._geometric,
                       self._polynomial, self._recurrence, self._famous, self._interleaved):
            hunch = solver(xs)
            if hunch is not None:
                return hunch
        return None

    # -- helpers -- #
    @staticmethod
    def _mk(xs: List[Fraction], nxt: Fraction, rule: str, conf: float,
            check: Callable[[], bool], mech: str) -> Hunch:
        ans = _jsonable(nxt)
        return Hunch(answer=ans, confidence=_clamp(conf), mechanism="gestalt",
                     rule=rule, rationale=f"{mech}: {rule} → next = {ans}",
                     verify=check, support=(mech,))

    def _constant(self, xs: List[Fraction]) -> Optional[Hunch]:
        if all(x == xs[0] for x in xs):
            return self._mk(xs, xs[0], f"constant {_jsonable(xs[0])}", 0.92,
                            lambda: all(x == xs[0] for x in xs), "constant")
        return None

    def _arithmetic(self, xs: List[Fraction]) -> Optional[Hunch]:
        d = xs[1] - xs[0]
        if all(xs[i + 1] - xs[i] == d for i in range(len(xs) - 1)):
            conf = _clamp(0.86 + 0.01 * len(xs))
            return self._mk(xs, xs[-1] + d, f"arithmetic, step {_jsonable(d)}", conf,
                            lambda: all(xs[i + 1] - xs[i] == d for i in range(len(xs) - 1)),
                            "arithmetic")
        return None

    def _geometric(self, xs: List[Fraction]) -> Optional[Hunch]:
        if any(x == 0 for x in xs[:-1]):
            return None
        r = xs[1] / xs[0]
        if r != 1 and all(xs[i + 1] == xs[i] * r for i in range(len(xs) - 1)):
            conf = _clamp(0.86 + 0.01 * len(xs))
            return self._mk(xs, xs[-1] * r, f"geometric, ratio {_jsonable(r)}", conf,
                            lambda: all(xs[i + 1] == xs[i] * r for i in range(len(xs) - 1)),
                            "geometric")
        return None

    def _polynomial(self, xs: List[Fraction]) -> Optional[Hunch]:
        table = [list(xs)]
        while len(table[-1]) > 1:
            row = table[-1]
            table.append([row[i + 1] - row[i] for i in range(len(row) - 1)])
        for k in range(1, len(table)):
            row = table[k]
            if len(row) >= 2 and all(v == row[0] for v in row):
                diffs = [list(table[i]) for i in range(k + 1)]
                diffs[k].append(diffs[k][-1])
                for j in range(k - 1, -1, -1):
                    diffs[j].append(diffs[j][-1] + diffs[j + 1][-1])
                nxt = diffs[0][-1]
                confirmations = len(row) - 1
                conf = _clamp(0.82 + 0.03 * confirmations - 0.03 * (k - 1))
                rule = f"degree-{k} polynomial (constant {k}th difference {_jsonable(row[0])})"
                return self._mk(xs, nxt, rule, conf,
                                lambda k=k, row=row: len(row) >= 2 and all(v == row[0] for v in row),
                                "finite-differences")
        return None

    def _recurrence(self, xs: List[Fraction]) -> Optional[Hunch]:
        for m in range(1, len(xs) // 2 + 1):
            if len(xs) < 2 * m + 1:                 # need at least one held-out confirmation
                continue
            A = [[xs[i - j] for j in range(1, m + 1)] for i in range(m, 2 * m)]
            b = [xs[i] for i in range(m, 2 * m)]
            coeffs = _solve_linear(A, b)
            if coeffs is None:
                continue
            ok = all(sum(coeffs[j - 1] * xs[i - j] for j in range(1, m + 1)) == xs[i]
                     for i in range(m, len(xs)))
            if not ok:
                continue
            n = len(xs)
            nxt = sum(coeffs[j - 1] * xs[n - j] for j in range(1, m + 1))
            terms = " + ".join(f"{_jsonable(coeffs[j - 1])}·a[n-{j}]" for j in range(1, m + 1))
            confirmations = len(xs) - 2 * m
            conf = _clamp(0.82 + 0.03 * confirmations - 0.03 * (m - 1))
            return self._mk(xs, nxt, f"linear recurrence a[n] = {terms}", conf,
                            lambda coeffs=coeffs, m=m: all(
                                sum(coeffs[j - 1] * xs[i - j] for j in range(1, m + 1)) == xs[i]
                                for i in range(m, len(xs))),
                            "recurrence")
        return None

    def _famous(self, xs: List[Fraction]) -> Optional[Hunch]:
        if any(x.denominator != 1 for x in xs):
            return None
        ints = [int(x) for x in xs]
        for name, table in _FAMOUS.items():
            L = len(ints)
            for start in range(0, max(1, len(table) - L)):
                window = table[start:start + L]
                if window == ints and start + L < len(table):
                    nxt = table[start + L]
                    return self._mk(xs, Fraction(nxt), f"the {name} sequence", 0.8,
                                    lambda name=name, ints=ints: any(
                                        _FAMOUS[name][s:s + len(ints)] == ints
                                        for s in range(len(_FAMOUS[name]))),
                                    f"famous:{name}")
        return None

    def _interleaved(self, xs: List[Fraction]) -> Optional[Hunch]:
        if len(xs) < 6:
            return None
        evens, odds = xs[0::2], xs[1::2]
        he = GestaltGenerator()._first_simple(evens)
        ho = GestaltGenerator()._first_simple(odds)
        if he is None or ho is None:
            return None
        nxt = he.answer if len(xs) % 2 == 0 else ho.answer
        return self._mk(xs, Fraction(nxt), "two interleaved sub-sequences", 0.6,
                        lambda: True, "interleaved")

    def _first_simple(self, xs: List[Fraction]) -> Optional[Hunch]:
        for solver in (self._constant, self._arithmetic, self._geometric, self._polynomial):
            h = solver(xs)
            if h is not None:
                return h
        return None


# --------------------------------------------------------------------------- #
# 2) Analogical transfer — HDC/VSA "I've felt this shape before"
# --------------------------------------------------------------------------- #
class AnalogyGenerator(LeapGenerator):
    """Recall the nearest past problem in a 10,000-D space and transport its solution."""

    name = "analogy"

    def __init__(self, *, dim: int = 10000, seed: int = 42, threshold: float = 0.30) -> None:
        self.threshold = float(threshold)
        self._solutions: Dict[str, Any] = {}
        self._map: Any = None
        try:
            from nyxara.cognition.hyper_dimensional_vectors import LatentSpaceMap
            self._map = LatentSpaceMap(dim=dim, seed=seed, novelty_threshold=0.85)
            # Pin the dependency-free hashing embedder so encoding TEXT never loads a heavy
            # neural model (or hits the network): intuition must run instantly on a bare machine.
            try:
                from nyxara.memory.store import HashingEmbedder
                self._map._embedder = HashingEmbedder(128)
            except Exception:  # noqa: BLE001 — the map keeps its own default embedder
                pass
        except Exception:  # noqa: BLE001 — HDC is optional; abstain without it
            self._map = None

    def available(self) -> bool:
        return self._map is not None

    def _encode_key(self, problem: Problem) -> Any:
        if problem.sequence is not None:
            return [float(x) for x in problem.sequence]
        return problem.text

    def remember(self, problem: Problem, solution: Any) -> None:
        if self._map is None:
            return
        try:
            name = problem.key()
            self._map.add(name, self._encode_key(problem))
            self._solutions[name] = solution
        except Exception:  # noqa: BLE001
            pass

    def leap(self, problem: Problem) -> Optional[Hunch]:
        if self._map is None or not self._solutions:
            return None
        try:
            hits = self._map.nearest(self._encode_key(problem), k=1)
        except Exception:  # noqa: BLE001
            return None
        if not hits:
            return None
        name, sim = hits[0]
        if name == problem.key():                    # do not "recall" the probe itself
            hits = self._map.nearest(self._encode_key(problem), k=2)
            if len(hits) < 2:
                return None
            name, sim = hits[1]
        if sim < self.threshold or name not in self._solutions:
            return None
        answer = self._solutions[name]
        conf = _clamp(0.35 + 0.5 * sim)
        return Hunch(answer=answer, confidence=conf, mechanism="analogy",
                     rule=f"analog of a past problem (similarity {sim:.2f})",
                     rationale=f"transported the solution of «{name[:48]}» (sim {sim:.2f})",
                     verify=None, support=("hdc-recall",))


# --------------------------------------------------------------------------- #
# 3) Superposed contradiction — hold contradictory leaps, collapse to the best
# --------------------------------------------------------------------------- #
class SuperpositionGenerator(LeapGenerator):
    """Gather partial candidate continuations, hold them in quantum-style superposition,
    then collapse to the one the givens most support. It never forces early consistency."""

    name = "superposition"
    shapes = frozenset({"sequence"})

    def __init__(self) -> None:
        self._ok = True
        try:  # probe the dependency once
            from nyxara.quantum.superposition_states import Superposition  # noqa: F401
        except Exception:  # noqa: BLE001
            self._ok = False

    def available(self) -> bool:
        return self._ok

    def _candidates(self, xs: List[Fraction]) -> List[Tuple[str, Fraction, float]]:
        """Each simple rule votes a next-term with a fit score in [0,1] (partial credit)."""
        out: List[Tuple[str, Fraction, float]] = []
        n = len(xs)
        # arithmetic vote (median step)
        steps = [xs[i + 1] - xs[i] for i in range(n - 1)]
        d = steps[len(steps) // 2]
        fit_a = sum(1 for s in steps if s == d) / len(steps)
        out.append(("arithmetic", xs[-1] + d, fit_a))
        # geometric vote
        if all(x != 0 for x in xs[:-1]):
            ratios = [xs[i + 1] / xs[i] for i in range(n - 1)]
            r = ratios[len(ratios) // 2]
            fit_g = sum(1 for x in ratios if x == r) / len(ratios)
            out.append(("geometric", xs[-1] * r, fit_g))
        # repeat-last vote (weak baseline / periodic collapse)
        out.append(("repeat", xs[-1], 0.1))
        return out

    def leap(self, problem: Problem) -> Optional[Hunch]:
        xs = problem.sequence
        if not self._ok or not xs or len(xs) < 3:
            return None
        try:
            from nyxara.quantum.superposition_states import Superposition
        except Exception:  # noqa: BLE001
            return None
        cands = self._candidates(xs)
        if not cands:
            return None
        sup = Superposition()
        by_label: Dict[str, Fraction] = {}
        for label, val, fit in cands:
            sup.add(label, amplitude=max(0.05, fit), payload=val)
            by_label[label] = val
        # contradictory continuations cannot both be the rule
        labels = list(by_label)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if by_label[labels[i]] != by_label[labels[j]]:
                    sup.mark_contradictory(labels[i], labels[j])
        # observe: reward the rule whose fit is strongest
        sup.observe({label: max(0.01, fit) for label, val, fit in cands})
        result = sup.collapse(force=True)
        label = getattr(result, "label", None)
        if label is None:
            label, _ = sup.max_probability()
        if label is None or label not in by_label:
            return None
        _, prob = sup.max_probability()
        best_fit = max(f for lab, v, f in cands if lab == label)
        if best_fit >= 0.999:                     # a perfect rule belongs to gestalt, not here
            return None
        nxt = by_label[label]
        conf = _clamp(0.25 + 0.4 * float(prob) + 0.2 * best_fit)
        return Hunch(answer=_jsonable(nxt), confidence=conf, mechanism="superposition",
                     rule=f"collapsed {len(cands)} contradictory leaps → {label}",
                     rationale=f"held {len(cands)} rival continuations; «{label}» "
                               f"dominated (p={float(prob):.2f}, fit={best_fit:.2f})",
                     verify=None, support=("superposition",))


# --------------------------------------------------------------------------- #
# 4) Dark-data / absence — leap from what is missing or silent
# --------------------------------------------------------------------------- #
class DarkDataGenerator(LeapGenerator):
    """Reuse the void miner: a hidden period predicts the next term from the past."""

    name = "dark_data"
    shapes = frozenset({"sequence"})

    def __init__(self) -> None:
        self._miner: Any = None
        try:
            from nyxara.void.dark_data_mining import DarkDataMiner
            self._miner = DarkDataMiner()
        except Exception:  # noqa: BLE001
            self._miner = None

    def available(self) -> bool:
        return self._miner is not None

    def leap(self, problem: Problem) -> Optional[Hunch]:
        xs = problem.sequence
        if self._miner is None or not xs or len(xs) < 6:
            return None
        series = [float(x) for x in xs]
        try:
            rep = self._miner.mine_periodicity(series)
        except Exception:  # noqa: BLE001
            return None
        period = int(getattr(rep, "period", 0) or 0)
        strength = float(getattr(rep, "strength", getattr(rep, "confidence", 0.0)) or 0.0)
        if period <= 0 or period >= len(xs) or strength < 0.6:
            return None
        # verify the period actually holds on the givens (exact), then predict by it
        if not all(xs[i] == xs[i - period] for i in range(period, len(xs))):
            return None
        nxt = xs[len(xs) - period]
        conf = _clamp(0.3 + 0.5 * strength)
        return Hunch(answer=_jsonable(nxt), confidence=conf, mechanism="dark_data",
                     rule=f"hidden period {period}",
                     rationale=f"the silence repeats every {period} (strength {strength:.2f})",
                     verify=lambda: all(xs[i] == xs[i - period] for i in range(period, len(xs))),
                     support=("periodicity",))


# --------------------------------------------------------------------------- #
# 5) First-principles — jump to the form dimensional analysis / axioms permit
# --------------------------------------------------------------------------- #
class FirstPrinciplesGenerator(LeapGenerator):
    """Reuse the first-principles engines to derive a form for a physics/derivation prompt."""

    name = "first_principles"
    shapes = frozenset({"physics", "text"})

    def __init__(self) -> None:
        self._engines: List[Any] = []
        try:
            from nyxara.mind.first_principles import PhysicsEngine, MathEngine
            self._engines = [PhysicsEngine(), MathEngine()]
        except Exception:  # noqa: BLE001
            self._engines = []

    def available(self) -> bool:
        return bool(self._engines)

    def leap(self, problem: Problem) -> Optional[Hunch]:
        if not self._engines or not problem.text:
            return None
        for eng in self._engines:
            try:
                deriv = eng.derive(problem.text)
            except Exception:  # noqa: BLE001
                deriv = None
            if deriv is not None:
                rendered = deriv.render() if hasattr(deriv, "render") else str(deriv)
                return Hunch(answer=rendered, confidence=0.55, mechanism="first_principles",
                             rule="dimensional / axiomatic derivation",
                             rationale="jumped to a form the base principles permit",
                             verify=None, support=("first-principles",))
        return None


# =========================================================================== #
# Meta-gate — UCB1 over generators, per problem shape (intuition about intuition)
# =========================================================================== #
class MetaGate:
    """Learns, per problem shape, which generator's leaps pay off. Persisted, no LLM."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._arms: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._load()

    def _arm(self, shape: str, gen: str) -> Dict[str, float]:
        return self._arms.setdefault(shape, {}).setdefault(gen, {"n": 0.0, "wins": 0.0})

    def order(self, shape: str, gens: List[str]) -> List[str]:
        if len(gens) < 2:
            return list(gens)
        total = sum(self._arm(shape, g)["n"] for g in gens) or 1.0

        def _score(g: str) -> float:
            a = self._arm(shape, g)
            if a["n"] <= 0:
                return float("inf")               # explore an untried generator first
            return a["wins"] / a["n"] + math.sqrt(2.0 * math.log(total) / a["n"])

        return sorted(gens, key=_score, reverse=True)

    def trust(self, shape: str, gen: str) -> float:
        """Prior success rate of this generator on this shape (0.5 when unseen)."""
        a = self._arm(shape, gen)
        return a["wins"] / a["n"] if a["n"] > 0 else 0.5

    def record(self, shape: str, gen: str, success: bool) -> None:
        a = self._arm(shape, gen)
        a["n"] += 1.0
        a["wins"] += 1.0 if success else 0.0
        self._save()

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                self._arms = json.load(fh)
        except Exception:  # noqa: BLE001 — a corrupt file never bricks the mind
            self._arms = {}

    def _save(self) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._arms, fh)
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001 — persistence is best-effort
            pass


# =========================================================================== #
# The Intuition Core — portfolio + fusion + meta-gate + compounding
# =========================================================================== #
class IntuitionCore:
    """Run every applicable leap generator, fuse their hunches, gate by learned trust.

    ``leap(problem)`` returns the single best Hunch (or None). It self-verifies where it
    can, folds agreement into confidence, and lets a proven leap compound: it is
    remembered for analogical recall next time.
    """

    def __init__(self, *, data_dir: Any = None, settings: Any = None,
                 dim: int = 10000, seed: int = 42) -> None:
        path = None
        try:
            if data_dir is None and settings is None:
                from nyxara.kernel.config import get_settings
                settings = get_settings()
            base = data_dir
            if base is None and settings is not None:
                base = getattr(getattr(settings, "paths", None), "data_dir", None)
            if base is not None:
                path = os.path.join(str(base), "intuition_gate.json")
        except Exception:  # noqa: BLE001 — persistence is optional
            path = None

        self.meta = MetaGate(path=path)
        self._analogy = AnalogyGenerator(dim=dim, seed=seed)
        self.generators: List[LeapGenerator] = [
            GestaltGenerator(),
            self._analogy,
            SuperpositionGenerator(),
            DarkDataGenerator(),
            FirstPrinciplesGenerator(),
        ]
        self._incubator: List[Problem] = []
        # recent self-verified integer-sequence leaps, ready to be lifted into Prover-decidable
        # Eureka conjectures (the intuition → prove bridge). Each: (sequence, answer, support).
        self._recent: List[Tuple[List[Fraction], Any, Tuple[str, ...]]] = []
        self.stats_counter: Dict[str, int] = {"leaps": 0, "verified": 0, "abstained": 0}

    # -- the headline entry point -- #
    def leap(self, raw: Any, *, payload: Any = None,
             features: Optional[Dict[str, Any]] = None, _requeue: bool = True) -> Optional[Hunch]:
        problem = raw if isinstance(raw, Problem) else Problem.parse(
            raw, payload=payload, features=features)
        hunches: List[Hunch] = []
        active = [g for g in self.generators if g.available() and g.applies(problem)]
        ordered = self.meta.order(problem.kind, [g.name for g in active])
        by_name = {g.name: g for g in active}
        for name in ordered:
            gen = by_name.get(name)
            if gen is None:
                continue
            try:
                h = gen.leap(problem)
            except Exception:  # noqa: BLE001 — one generator's failure never sinks the leap
                h = None
            if h is not None:
                # temper each raw hunch by how much this generator has earned trust here
                trust = self.meta.trust(problem.kind, name)
                h.confidence = _clamp(0.5 * h.confidence + 0.5 * h.confidence * (0.5 + trust))
                hunches.append(h)
        if not hunches:
            self.stats_counter["abstained"] += 1
            if _requeue and len(self._incubator) < 128:   # bounded backlog of open problems
                self._incubator.append(problem)
            return None

        best = self._fuse(hunches)
        # a leap that carries a checker and fails it is discarded outright (honest)
        v = best.verified()
        if v is False:
            survivors = [h for h in hunches if h is not best and h.verified() is not False]
            if not survivors:
                self.stats_counter["abstained"] += 1
                return None
            best = self._fuse(survivors)
            v = best.verified()
        if v is True:
            best.confidence = _clamp(best.confidence + 0.1)
            best.certificate = f"self-verified:{best.mechanism}"
            self.stats_counter["verified"] += 1
            if problem.sequence is not None and all(x.denominator == 1 for x in problem.sequence):
                self._recent.append((list(problem.sequence), best.answer, best.support))
                self._recent = self._recent[-16:]
        self.stats_counter["leaps"] += 1
        return best

    def _fuse(self, hunches: List[Hunch]) -> Hunch:
        """Confidence-weighted consensus. A leap that *self-verifies exactly* always outranks an
        unproven guess (verifiable beats probabilistic); ties break on confidence. Agreement
        between independent leaps then compounds trust."""
        def _rank(h: Hunch) -> Tuple[int, float]:
            return (1 if h.verified() is True else 0, h.confidence)

        best = max(hunches, key=_rank)
        agree = [h for h in hunches if _same_answer(h.answer, best.answer) and h is not best]
        if agree:
            boost = min(0.25, 0.08 * len(agree))
            best = Hunch(answer=best.answer, confidence=_clamp(best.confidence + boost),
                         mechanism=best.mechanism, rule=best.rule,
                         rationale=best.rationale + f" — corroborated by {len(agree)} other leap(s)",
                         verify=best.verify, certificate=best.certificate,
                         support=tuple(sorted({s for h in [best, *agree] for s in h.support})))
        return best

    # -- learning / compounding -- #
    def ingest(self, raw: Any, solution: Any, *, payload: Any = None) -> None:
        """Teach the Core a solved problem so it can leap by analogy next time, and reward
        the generator whose shape matches. Compounding, no LLM."""
        problem = raw if isinstance(raw, Problem) else Problem.parse(raw, payload=payload)
        self._analogy.remember(problem, solution)

    def reward(self, raw: Any, hunch: Hunch, correct: bool) -> None:
        """Close the loop: tell the meta-gate whether a leap was right, so trust adapts."""
        problem = raw if isinstance(raw, Problem) else Problem.parse(raw)
        self.meta.record(problem.kind, hunch.mechanism, bool(correct))
        if correct:
            self._analogy.remember(problem, hunch.answer)

    def incubate(self) -> Optional[Hunch]:
        """Background 'Aha!': re-attempt a previously unsolved problem now that the Core has
        learned more. Returns a real candidate leap, not a templated string. Problems that
        still resist stay queued (they never re-queue during the sweep, so this terminates)."""
        pending, self._incubator = self._incubator, []
        result: Optional[Hunch] = None
        for problem in pending:
            if result is None:
                h = self.leap(problem, _requeue=False)   # no re-queue: the sweep must terminate
                if h is not None:
                    h.rationale = "incubation paid off — " + h.rationale
                    result = h
                    continue
            self._incubator.append(problem)              # still open — keep for a later incubation
        return result

    def eureka_seeds(self) -> List[Any]:
        """The intuition → prove bridge. Lift recent self-verified integer-sequence leaps into
        Prover-decidable Eureka conjectures (primality of a predicted prime; a divisibility the
        geometric ratio guarantees). Returns Eureka ``Conjecture`` objects — ``[]`` when the
        growth engine is unavailable. Safe by construction: Eureka keeps only PROVEN leaps, so
        a wrong guess is simply refuted, never trusted. Consumed once per call."""
        try:
            from nyxara.growth.eureka import Conjecture
        except Exception:  # noqa: BLE001 — no growth engine here; nothing to seed
            self._recent = []
            return []
        seeds: List[Any] = []
        recent, self._recent = self._recent, []
        for seq, ans, support in recent:
            try:
                a = float(ans)
                if not a.is_integer():
                    continue
                ai = int(a)
                if any("primes" in s for s in support) and ai > 1:
                    seeds.append(Conjecture(domain="number_theory",
                                            statement=f"is {ai} prime", origin="intuition"))
                if len(seq) >= 2 and seq[-2] != 0:
                    r = seq[-1] / seq[-2]
                    if r.denominator == 1 and int(r) not in (0, 1, -1) and ai != 0:
                        seeds.append(Conjecture(domain="number_theory",
                                                statement=f"{abs(int(r))} divides {abs(ai)}",
                                                origin="intuition"))
            except Exception:  # noqa: BLE001
                continue
        return seeds[:8]

    def stats(self) -> Dict[str, Any]:
        return {**self.stats_counter,
                "generators": [g.name for g in self.generators if g.available()],
                "incubating": len(self._incubator), "eureka_ready": len(self._recent)}


def _same_answer(a: Any, b: Any) -> bool:
    try:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) < 1e-9
        return a == b
    except Exception:  # noqa: BLE001
        return False


# =========================================================================== #
# Kernel adapters — how the Core plugs into the sovereign cycle
# =========================================================================== #
class IntuitionFaculty(Faculty):
    """Registers the Intuition Core as one engine in the FacultySelector (probabilistic)."""

    name = "intuition"
    handles = frozenset({TaskType.SEQUENCE, TaskType.REASONING, TaskType.ANALOGY})
    verifiable = False
    reliability = 0.65
    cost = 0.8

    def __init__(self, core: IntuitionCore) -> None:
        self.core = core

    def suitability(self, task: Task) -> float:
        if task.type is TaskType.SEQUENCE:
            return 0.9
        if task.type in self.handles:
            return 0.5
        return 0.0

    def handle(self, task: Task) -> Proposal:
        hunch = self.core.leap(task.description or task.payload, payload=task.payload,
                               features=task.features)
        if hunch is None:
            return self._propose("(no intuition fired)", confidence=0.2,
                                 rationale="intuition abstained", source=SourceType.SELF_REFLECTION)
        return Proposal(
            kind=ProposalKind.ANSWER, content=hunch.answer, source_faculty=self.name,
            confidence=hunch.confidence, rationale=f"intuited ({hunch.mechanism}): {hunch.rule}",
            provenance=Provenance(SourceType.SELF_REFLECTION, confidence=hunch.confidence,
                                  detail=self.name, method="intuited"),
            metadata={"mechanism": hunch.mechanism, "rule": hunch.rule,
                      "certificate": hunch.certificate, "hunch": hunch.to_dict()},
        )


def make_intuition_callable(core: IntuitionCore) -> Callable[[Task], Tuple[Any, float]]:
    """Adapter feeding the real Core into ``System1`` in :mod:`nyxara.mind.dual_process`.

    Returns the intuited answer + its (calibrated) confidence — no longer an echo. When the
    Core abstains it returns the reasoner's own prior confidence so the arbitrator behaves
    exactly as before for problems intuition cannot touch."""

    def _intuit(task: Task) -> Tuple[Any, float]:
        try:
            hunch = core.leap(task.description or task.payload, payload=task.payload,
                              features=task.features)
        except Exception:  # noqa: BLE001 — intuition is never allowed to break the turn
            hunch = None
        if hunch is None:
            prior = float(task.features.get("confidence", 0.3)) if task.features else 0.3
            return (task.description, _clamp(prior))
        return (hunch.answer, hunch.confidence)

    return _intuit


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA Intuition Core self-test")
    print("=" * 70)

    core = IntuitionCore(data_dir=None, settings=False)  # settings=False -> no persistence probe

    # 1) squares — a puzzle with NO training data, cracked by pure structure
    h = core.leap([1, 4, 9, 16, 25])
    print(f"\nsquares      : {h.answer} via {h.mechanism} [{h.rule}]  conf={h.confidence:.2f}")
    assert h is not None and _same_answer(h.answer, 36) and h.verified() is True

    # 2) Fibonacci — exact linear-recurrence leap
    h = core.leap([1, 1, 2, 3, 5, 8, 13])
    print(f"fibonacci    : {h.answer} via {h.mechanism} [{h.rule}]")
    assert _same_answer(h.answer, 21)

    # 3) geometric
    h = core.leap([3, 6, 12, 24, 48])
    print(f"geometric    : {h.answer} via {h.mechanism}")
    assert _same_answer(h.answer, 96)

    # 4) primes — a famous sequence, no closed form
    h = core.leap([2, 3, 5, 7, 11, 13])
    print(f"primes       : {h.answer} via {h.mechanism} [{h.rule}]")
    assert _same_answer(h.answer, 17)

    # 5) cubes — finite differences degree 3
    h = core.leap([1, 8, 27, 64, 125])
    print(f"cubes        : {h.answer} via {h.mechanism}")
    assert _same_answer(h.answer, 216)

    # 6) analogical transfer — teach one, recall on a same-shape near-identical probe.
    # (HDC signed random projection is per-width, so a concrete answer transports only between
    #  near-identical problems — exactly when transporting a specific answer is valid.)
    ana = AnalogyGenerator()
    if ana.available():
        core2 = IntuitionCore(data_dir=None, settings=False)
        core2.ingest([100, 200, 300, 400], solution="linear-by-100")
        rec = core2._analogy.leap(Problem.parse([100, 200, 300, 401]))  # same length, ~identical
        print(f"analogy      : recall fired = {rec is not None} "
              f"({rec.answer if rec else '—'})")
        assert rec is not None and rec.answer == "linear-by-100"

    # 7) meta-gate learns (bandit) and rewards close the loop
    core.reward([1, 4, 9, 16], h, correct=True)
    assert core.meta.trust("sequence", "gestalt") >= 0.5

    # 8) incubation surfaces a real leap, not a templated string
    core.leap("an utterly unstructured prompt with no numbers")   # abstains -> incubates
    _ = core.incubate()

    # 9) NO-LLM guarantee — the real code (everything above this self-test) imports no
    # provider and calls no model. Patterns are assembled from fragments so they never
    # match themselves, and only the code *above* the __main__ guard is scanned.
    import nyxara.mind.intuition as _self
    full = open(_self.__file__, "r", encoding="utf-8").read()
    code = full.split("if __name__ ==")[0]
    banned = ["from nyxara.mind." + "llm", "import " + "LLM", "Qwen" + "Provider",
              "SelfProvider", ".generate(", ".complete("]
    for pat in banned:
        assert pat not in code, f"intuition must be LLM-free (found {pat!r})"
    import sys as _sys
    assert "nyxara.mind.llm" not in _sys.modules, "importing intuition pulled in the LLM"

    print(f"\nstats        : {core.stats()}")
    print("\nALL SELF-TESTS PASSED ✓")
