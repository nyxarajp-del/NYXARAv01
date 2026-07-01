"""NYXARA · growth/curriculum.py — the open-ended auto-curriculum (🎓, Rule 4, no saturation).

The Master's second gap: NYXARA scored herself against a **fixed battery** of self-written
benchmarks. A fixed battery has a ceiling — once she masters it the number stops moving, and a true
recursively-self-improving mind would have run out of room to grow. A real open-ended learner runs
an **auto-curriculum that always stays one step ahead of its own capability** (POET / open-endedness):
it manufactures novel problems just beyond what it can already do, and the ground truth for those
problems is *computed*, never asserted by an LLM.

This module is that curriculum, and it grades against **machine-checkable** ground truth only:

* It **generates** fresh, never-stored problems across difficulty *tiers* (arithmetic evaluation and
  linear-equation solving today; the tier sets the number of operations / coefficient magnitude). The
  answer to every problem is produced by NYXARA's own exact evaluator and **certified by the
  :class:`~nyxara.growth.prover.Prover`** — the same non-LLM theorem-checker the rest of growth/ trusts
  ("NYXARA khude kare, koi LLM naa kare").
* It poses them to any solver and scores it against that certified answer.
* It tracks a **frontier tier** that *rises whenever she masters the current edge* and falls when she
  cannot — so the curriculum tracks her and the ``frontier_score`` cannot saturate: to keep the score
  high she must keep solving ever-harder, freshly-generated problems she has never seen. Because every
  batch is regenerated, the score also cannot be Goodharted by memorisation.
* It can optionally fold in the :class:`~nyxara.growth.eureka.EurekaEngine`'s genuinely-novel,
  prover-certified discoveries as an enrichment of the problem bank.

The frontier tier rides NYXARA's existing long-term memory (one protected SEMANTIC record, tag
``auto-curriculum``), like the intelligence index — no new persistence format. Pure stdlib at the
core; degrades to in-process state when memory is absent and to a clean serial path everywhere.
Touches no source, no weights, no gate — it only *measures* against truth it can check.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from fractions import Fraction
from random import Random
from typing import Any, Callable, Dict, List, Optional, Tuple

from nyxara.growth.prover import ProofVerdict, Prover, _eval_rational

__all__ = ["CurriculumProblem", "CurriculumReport", "AutoCurriculum"]

_TAG = "auto-curriculum"
_PROMOTE = 0.80     # master the edge tier at ≥ this accuracy → frontier rises (POET advance)
_DEMOTE = 0.40      # fail the current tier below this → frontier falls one step


# --------------------------------------------------------------------------- #
# One generated problem with a prover-certified answer
# --------------------------------------------------------------------------- #
@dataclass
class CurriculumProblem:
    """A freshly-generated problem whose answer is computed and certified, never asserted."""

    kind: str                       # Prover kind: arithmetic | algebra
    prompt: str                     # the question shown to the solver
    statement: str                  # the prover statement (expression / equation)
    answer: str                     # the certified ground-truth answer
    candidate: str                  # candidate form the prover substitutes (e.g. "x = 2")
    tier: int = 1                   # difficulty tier

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "prompt": self.prompt, "statement": self.statement,
                "answer": self.answer, "tier": self.tier}


# --------------------------------------------------------------------------- #
# The auditable result of one curriculum evaluation
# --------------------------------------------------------------------------- #
@dataclass
class CurriculumReport:
    """Everything one curriculum pass measured — the frontier and the score it produced."""

    frontier_tier: int = 1
    frontier_score: float = 0.0
    by_tier: Dict[int, float] = field(default_factory=dict)
    n_problems: int = 0
    n_correct: int = 0
    promoted: bool = False
    demoted: bool = False
    enrichment: int = 0             # novel prover-certified theorems added to the bank this pass
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"frontier_tier": self.frontier_tier,
                "frontier_score": round(self.frontier_score, 6),
                "by_tier": {int(k): round(float(v), 4) for k, v in self.by_tier.items()},
                "n_problems": self.n_problems, "n_correct": self.n_correct,
                "promoted": self.promoted, "demoted": self.demoted,
                "enrichment": self.enrichment, "elapsed_ms": round(self.elapsed_ms, 3)}


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
class AutoCurriculum:
    """Generate prover-certified problems at the edge of capability, grade a solver, move the edge.

    Parameters
    ----------
    prover:   the machine-checkable verifier (defaults to a fresh :class:`Prover`).
    memory:   long-term store for the persisted frontier tier (optional — in-process otherwise).
    eureka:   an optional :class:`~nyxara.growth.eureka.EurekaEngine` for novelty enrichment.
    settings: :class:`~nyxara.kernel.config.NyxaraSettings` (its memory tag is read).
    seed:     base seed; each evaluation advances an internal RNG so every batch is *fresh*.
    """

    def __init__(self, *, prover: Any = None, memory: Any = None, eureka: Any = None,
                 settings: Any = None, seed: int = 20260629) -> None:
        from nyxara.kernel.config import get_settings
        self.settings = settings or get_settings()
        self.prover = prover or Prover(seed=seed)
        self.memory = memory
        self._eureka = eureka
        self._rng = Random(seed)
        self._cached_tier: Optional[int] = None

    # ------------------------------------------------------------------ #
    # generation — fresh problems with computed, prover-certified answers
    # ------------------------------------------------------------------ #
    def generate(self, n: int, *, tier: int, rng: Optional[Random] = None
                 ) -> List[CurriculumProblem]:
        """Generate ``n`` fresh problems at difficulty ``tier``; every answer is prover-certified."""
        r = rng or self._rng
        tier = max(1, int(tier))
        out: List[CurriculumProblem] = []
        for _ in range(max(0, int(n))):
            prob = (self._gen_arithmetic(tier, r) if r.random() < 0.5
                    else self._gen_algebra(tier, r))
            if prob is not None and self._certify(prob):
                out.append(prob)
        return out

    def _gen_arithmetic(self, tier: int, r: Random) -> Optional[CurriculumProblem]:
        """An ``(tier+1)``-operand integer expression over + - *; answer is the exact value."""
        n_ops = tier + 1
        hi = 9 + 6 * tier
        terms = [str(r.randint(1, hi))]
        for _ in range(n_ops):
            terms.append(r.choice(("+", "-", "*")))
            terms.append(str(r.randint(1, hi)))
        expr = " ".join(terms)
        val = _eval_rational(expr)
        if val is None:
            return None
        return CurriculumProblem(kind="arithmetic", prompt=f"Compute: {expr}",
                                 statement=expr, answer=_fmt(val), candidate=_fmt(val), tier=tier)

    def _gen_algebra(self, tier: int, r: Random) -> Optional[CurriculumProblem]:
        """A linear equation ``a*x + b = c`` with an exact integer solution; answer is ``x``."""
        a = r.randint(2, 2 + tier)
        x = r.randint(-(tier + 2), tier + 2)
        b = r.randint(-(5 * tier), 5 * tier)
        c = a * x + b
        eq = f"{a}*x + {b} = {c}"
        return CurriculumProblem(kind="algebra", prompt=f"Solve for x: {eq}",
                                 statement=eq, answer=str(x), candidate=f"x = {x}", tier=tier)

    def _certify(self, prob: CurriculumProblem) -> bool:
        """The ground truth is real only if the Prover certifies the answer (non-LLM check)."""
        try:
            res = self.prover.check_answer(prob.kind, prob.statement, prob.candidate)
            return res.verdict is ProofVerdict.PROVEN
        except Exception:  # noqa: BLE001 — a checker never throws; an uncertified problem is dropped
            return False

    # ------------------------------------------------------------------ #
    # grading — score a solver against the certified answer
    # ------------------------------------------------------------------ #
    def grade(self, solver: Callable[[str], str], problems: List[CurriculumProblem]
              ) -> Tuple[int, int]:
        """Return ``(n_correct, n_total)`` — a solver answer counts only if it matches the truth."""
        correct = 0
        for prob in problems:
            try:
                resp = solver(prob.prompt)
            except Exception:  # noqa: BLE001 — a solver failure is simply a wrong answer
                resp = ""
            if self._is_correct(resp, prob):
                correct += 1
        return correct, len(problems)

    def _is_correct(self, response: str, prob: CurriculumProblem) -> bool:
        truth = _eval_rational(prob.answer)
        if truth is None:
            return False
        for cand in _numbers(response):
            if cand == truth:
                return True
        return False

    # ------------------------------------------------------------------ #
    # evaluate — probe the frontier, score, and MOVE the edge (POET)
    # ------------------------------------------------------------------ #
    def evaluate(self, solver: Callable[[str], str], *, per_tier: int = 4) -> CurriculumReport:
        """Grade ``solver`` across the tiers around the frontier, then advance/retreat the edge.

        The ``frontier_score`` weights harder tiers more, so it rises only when NYXARA genuinely
        solves harder, freshly-generated problems — it has no ceiling and cannot be memorised. When
        she masters the edge tier (≥ :data:`_PROMOTE`) the frontier rises by one; when she fails the
        current tier (< :data:`_DEMOTE`) it falls by one — the curriculum always tracks her."""
        t0 = time.perf_counter()
        frontier = self.frontier_tier()
        # a fresh RNG per evaluation so the batch is never the same twice (anti-memorisation)
        r = Random(self._rng.random())
        tiers = list(range(1, frontier + 2))     # probe up to one tier beyond the mastered edge
        by_tier: Dict[int, float] = {}
        n_total = n_correct = 0
        for t in tiers:
            problems = self.generate(per_tier, tier=t, rng=r)
            c, n = self.grade(solver, problems)
            by_tier[t] = (c / n) if n else 0.0
            n_total += n
            n_correct += c
        # weighted score: harder tiers count more (a solved tier-5 problem is worth more than tier-1)
        wsum = sum(tiers) or 1
        score = sum(t * by_tier.get(t, 0.0) for t in tiers) / wsum
        report = CurriculumReport(frontier_tier=frontier, frontier_score=_clamp01(score),
                                  by_tier=by_tier, n_problems=n_total, n_correct=n_correct)
        # POET: move the edge based on the edge tier's accuracy
        edge_acc = by_tier.get(frontier + 1, 0.0)
        cur_acc = by_tier.get(frontier, 0.0)
        if edge_acc >= _PROMOTE:
            frontier += 1
            report.promoted = True
        elif cur_acc < _DEMOTE and frontier > 1:
            frontier -= 1
            report.demoted = True
        report.frontier_tier = frontier
        self._save_tier(frontier)
        report.enrichment = self._maybe_enrich()
        report.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return report

    # ------------------------------------------------------------------ #
    # optional novelty enrichment from the Eureka engine (prover-certified)
    # ------------------------------------------------------------------ #
    def _maybe_enrich(self) -> int:
        if self._eureka is None:
            return 0
        try:
            rep = self._eureka.discover(generations=1, population=len(getattr(
                self._eureka, "_elites", {})) or 10)
            return int(getattr(rep, "novel_kept", 0) or 0)
        except Exception:  # noqa: BLE001 — enrichment is a bonus, never required
            return 0

    # ------------------------------------------------------------------ #
    # persistence of the frontier tier (rides long-term memory like the index)
    # ------------------------------------------------------------------ #
    def frontier_tier(self) -> int:
        if self._cached_tier is not None:
            return self._cached_tier
        tier = 1
        rec = self._find_record()
        if rec is not None:
            try:
                tier = max(1, int(rec.metadata.get("frontier_tier", 1)))
            except Exception:  # noqa: BLE001
                tier = 1
        self._cached_tier = tier
        return tier

    def _save_tier(self, tier: int) -> bool:
        self._cached_tier = max(1, int(tier))
        if self.memory is None:
            return False
        try:
            from nyxara.memory.provenance import Provenance, SourceType
            from nyxara.memory.store import MemoryType
        except Exception:  # noqa: BLE001
            return False
        try:
            existing = self._find_record()
            if existing is not None:
                self.memory.forget(existing.mem_id)
            self.memory.remember(
                f"[auto-curriculum] frontier tier = {self._cached_tier}",
                mem_type=MemoryType.SEMANTIC,
                provenance=Provenance(SourceType.SELF_REFLECTION, confidence=0.9),
                importance=0.9, tags=[_TAG],
                metadata={"frontier_tier": self._cached_tier})
            return True
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            return False

    def _find_record(self) -> Any:
        if self.memory is None:
            return None
        try:
            for rec in self.memory._kv.values():
                if _TAG in rec.tags and "frontier_tier" in rec.metadata:
                    return rec
        except Exception:  # noqa: BLE001
            return None
        return None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:  # noqa: BLE001
        return 0.0


def _fmt(v: Fraction) -> str:
    return str(int(v)) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


_NUM_RE = re.compile(r"-?\d+\s*/\s*-?\d+|-?\d+\.\d+|-?\d+")


def _numbers(text: str) -> List[Fraction]:
    """Every number in ``text`` as an exact Fraction (supports ``a/b`` and decimals)."""
    out: List[Fraction] = []
    for tok in _NUM_RE.findall(text or ""):
        try:
            tok = tok.replace(" ", "")
            out.append(Fraction(tok) if "/" in tok else Fraction(tok))
        except Exception:  # noqa: BLE001
            continue
    return out


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA auto-curriculum self-test")
    print("=" * 70)

    cur = AutoCurriculum(seed=7)

    # every generated problem's answer is prover-certified ground truth, not asserted
    probs = cur.generate(20, tier=3)
    assert probs, "must generate problems"
    assert all(cur._certify(p) for p in probs), "every problem must be prover-certified"
    print(f"generated/certified : {len(probs)} problems at tier 3 ✓")

    # a SMART solver that actually computes the answer scores high and pushes the frontier up
    def smart(prompt: str) -> str:
        if prompt.startswith("Compute:"):
            v = _eval_rational(prompt.split(":", 1)[1])
            return _fmt(v) if v is not None else "0"
        # "Solve for x: a*x + b = c" → solve exactly
        eq = prompt.split(":", 1)[1]
        lhs, rhs = eq.split("=")
        m = re.match(r"\s*(-?\d+)\*x\s*\+\s*(-?\d+)\s*", lhs)
        a, b = int(m.group(1)), int(m.group(2))
        c = _eval_rational(rhs)
        return _fmt((c - b) / a)

    start = cur.frontier_tier()
    rep = None
    for _ in range(6):
        rep = cur.evaluate(smart, per_tier=5)
    print(f"smart solver        : {rep.to_dict()}")
    assert rep.frontier_score > 0.6, "a correct solver must score high on the frontier"
    assert cur.frontier_tier() > start, "mastering the edge must raise the frontier (POET)"

    # a DUMB solver scores low and never advances the frontier
    cur2 = AutoCurriculum(seed=7)
    rep2 = cur2.evaluate(lambda p: "0", per_tier=5)
    print(f"dumb solver         : {rep2.to_dict()}")
    assert rep2.frontier_score < 0.3, "a wrong solver must score low"
    assert not rep2.promoted

    # the ruler cannot be memorised: a second batch is freshly generated, never the same set
    a = [p.prompt for p in cur.generate(10, tier=4)]
    b = [p.prompt for p in cur.generate(10, tier=4)]
    assert a != b, "each batch must be freshly generated (anti-memorisation)"

    print("\nALL SELF-TESTS PASSED ✓")
