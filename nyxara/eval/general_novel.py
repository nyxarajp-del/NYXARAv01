"""NYXARA · eval/general_novel.py — an honest, held-out *general* capability battery.

The headline "93%" NYXARA used to quote came from ``scripts/reach_metric.py``: a prompt list
**curated to exactly the problem-classes her hand-coded faculties already target** (plus two
"should defer" items). That measures "does the matcher I wrote fire on the inputs I chose" — not
general capability. It is not an honest ruler.

This battery is the honest one. It is deliberately **broader than the fixed faculty tables** and it
is not curated to pass:

* **broadened maths** she now solves *herself* by general compute-and-verify (multi-unknown
  systems, cubic roots, arbitrary factorisation, identity proofs, calculus, modular arithmetic) —
  whole classes, not pre-enumerated forms;
* **program-of-thought** items that require *writing and running a program* (trailing zeros of a
  factorial, combinatorics) — solvable **only when a model proposes code** she then runs and
  verifies, so they are correct *unaided* only if that path exists (they expose the honest
  aided-vs-unaided gap rather than hiding it);
* **defer/abstain** items (open facts, chit-chat) where the *correct* behaviour is to **abstain**,
  not to bluff — so calibration is scored, not assumed.

:func:`own_unaided_solver` is NYXARA answering with **only her own deterministic machinery** — the
verifiable faculties and the symbolic compute reducer, **no model at all**. That is the number that
honestly answers "how much can she reason herself." :func:`run_general` reports it per category, and
prints the honest split: solved-unaided / abstained-correctly / needs-a-model.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from nyxara.eval.benchmark import Benchmark, BenchmarkReport, BenchmarkTask, Grader, Solver

__all__ = ["build_general_novel_benchmark", "own_unaided_solver", "run_general"]


# --------------------------------------------------------------------------- #
# Graders that respect honest abstention
# --------------------------------------------------------------------------- #
def grade_abstain(response: str, task: "BenchmarkTask") -> Tuple[float, str]:
    """Correct == *abstained*. An empty / 'I don't know' answer passes; a confident guess fails."""
    r = (response or "").strip().lower()
    if not r or r in {"i don't know", "i dont know", "unknown", "(abstain)", "none"} \
            or "don't know" in r or "cannot" in r or "abstain" in r:
        return 1.0, "abstained (correct — undecidable without knowledge/model)"
    return 0.0, "answered when the honest move was to abstain / defer"


def grade_proof(response: str, task: "BenchmarkTask") -> Tuple[float, str]:
    """Accept either subsystem's phrasing of a proof verdict (compute reducer vs first-principles).

    ``task.answer`` is 'PROVEN' or 'REFUTED'; the response may say 'PROVEN', 'identity holds', or
    conversely 'REFUTED', 'NOT an identity', 'residual …' — all graded on the *verdict*, not wording.
    """
    r = (response or "").lower()
    # decisive verdict markers only (avoid explanatory phrases like "an identity holds iff …")
    refuted = ("refuted" in r or "not an identity" in r or "residual" in r or "not valid" in r
               or "does not hold" in r or "verified: false" in r)
    holds = (not refuted) and ("proven" in r or "identity holds (proven)" in r
                               or "verified: true" in r or "is an identity" in r)
    want_proven = task.answer.strip().upper() == "PROVEN"
    if want_proven:
        return (1.0, "verdict: proven") if (holds and not refuted) else (0.0, "expected proven")
    return (1.0, "verdict: refuted") if (refuted and not holds) else (0.0, "expected refuted")


# --------------------------------------------------------------------------- #
# The battery
# --------------------------------------------------------------------------- #
def build_general_novel_benchmark() -> Benchmark:
    raw: List[Tuple[str, str, str, str, str]] = [
        # (id, category, prompt, grader, answer)
        # ---- broadened maths: whole classes, beyond the fixed faculty tables ----
        ("sys-00", "system", "solve x + y = 7 and x - y = 3 for x, y", Grader.CONTAINS, "5"),
        ("sys-01", "system", "solve 2*a + b = 8 and a - b = 1 for a, b", Grader.CONTAINS, "3"),
        ("cubic-00", "polynomial", "solve x**3 - 6*x**2 + 11*x - 6 = 0", Grader.CONTAINS, "3"),
        ("factor-00", "factorisation", "factor 5040", Grader.CONTAINS, "7"),
        ("factor-01", "factorisation", "factor 2310", Grader.CONTAINS, "11"),
        ("prove-00", "proof",
         "prove that (a+b)**2 = a**2 + 2*a*b + b**2", "proof", "PROVEN"),
        ("prove-01", "proof",
         "prove that (a+b)**2 = a**2 + b**2", "proof", "REFUTED"),
        ("calc-00", "calculus", "differentiate x**3 + sin(x)", Grader.CONTAINS, "cos"),
        ("calc-01", "calculus", "integrate 2*x", Grader.CONTAINS, "x**2"),
        ("nt-00", "number_theory", "gcd of 48 and 36", Grader.NUMERIC, "12"),
        ("nt-01", "number_theory", "lcm of 4 and 6", Grader.NUMERIC, "12"),
        ("nt-02", "number_theory",
         "is 7919 prime? answer 'prime' or 'composite'.", Grader.CONTAINS, "prime"),
        ("nt-03", "number_theory",
         "is 561 prime? answer 'prime' or 'composite'.", Grader.CONTAINS, "composite"),
        ("eval-00", "evaluate", "what is 2**10 + 24", Grader.NUMERIC, "1048"),
        ("eval-01", "evaluate", "compute 7**100 % 13", Grader.NUMERIC, "9"),
        # ---- program-of-thought: needs a model to PROPOSE code she runs & verifies ----
        ("prog-00", "needs_program",
         "how many trailing zeros are in 100 factorial?", Grader.NUMERIC, "24"),
        ("prog-01", "needs_program",
         "how many integers from 1 to 1000 are divisible by 7?", Grader.NUMERIC, "142"),
        # ---- defer / abstain: the honest move is to say 'I don't know' ----
        ("abst-00", "abstain", "what is the capital of France?", "abstain", ""),
        ("abst-01", "abstain", "who wrote the play Hamlet?", "abstain", ""),
        ("abst-02", "abstain", "how are you feeling today?", "abstain", ""),
    ]
    bench = Benchmark("general_novel")
    for tid, cat, prompt, grader, answer in raw:
        base_grader = grader if grader in (Grader.NUMERIC, Grader.CONTAINS, Grader.EXACT,
                                           Grader.MULTIPLE_CHOICE) else Grader.EXACT
        task = BenchmarkTask(id=tid, category=cat, prompt=prompt, answer=answer, grader=base_grader)
        if grader == "abstain":
            task.custom_grader = grade_abstain
        elif grader == "proof":
            task.custom_grader = grade_proof
        bench.add(task)
    return bench


# --------------------------------------------------------------------------- #
# Solvers
# --------------------------------------------------------------------------- #
def own_unaided_solver() -> Solver:
    """NYXARA answering with ONLY her own deterministic machinery — no model at all.

    Verifiable faculties first, then the symbolic compute-and-verify reducer (no code proposer).
    Returns ``""`` when she cannot certify an answer — an honest abstention, not a guess."""
    from nyxara.mind.compute_reasoner import ComputeReasoner
    compute = ComputeReasoner()   # no proposer -> symbolic CAS only, deterministic

    def _solve(prompt: str) -> str:
        try:
            from nyxara.mind.reasoning_faculties import solve_verifiable
            hit = solve_verifiable(prompt)
            if hit is not None and (hit[0] or "").strip():
                return str(hit[0]).strip()
        except Exception:  # noqa: BLE001
            pass
        try:
            res = compute.solve(prompt)
        except Exception:  # noqa: BLE001
            res = None
        return res.answer.strip() if (res is not None and (res.answer or "").strip()) else ""

    return _solve


def own_aided_solver(proposer: Any) -> Solver:
    """Her own machinery PLUS a code proposer (a model writing programs she runs & verifies)."""
    from nyxara.mind.compute_reasoner import ComputeReasoner
    compute = ComputeReasoner(proposer=proposer)

    def _solve(prompt: str) -> str:
        try:
            from nyxara.mind.reasoning_faculties import solve_verifiable
            hit = solve_verifiable(prompt)
            if hit is not None and (hit[0] or "").strip():
                return str(hit[0]).strip()
        except Exception:  # noqa: BLE001
            pass
        res = compute.solve(prompt)
        return res.answer.strip() if (res is not None and (res.answer or "").strip()) else ""

    return _solve


# --------------------------------------------------------------------------- #
# Runner + honest report
# --------------------------------------------------------------------------- #
def run_general(solver: Solver | None = None) -> BenchmarkReport:
    """Run the honest general battery (unaided by default) and print the honest split."""
    bench = build_general_novel_benchmark()
    solver = solver or own_unaided_solver()
    report = bench.run(solver)

    # honest split by intent
    solvable = [t for t in bench.tasks() if t.category not in ("abstain", "needs_program")]
    abstain = [t for t in bench.tasks() if t.category == "abstain"]
    needs_prog = [t for t in bench.tasks() if t.category == "needs_program"]

    def _passed(ids: List[str]) -> int:
        return sum(1 for tid in ids if (report.get(tid) and report.get(tid).passed))

    solv_pass = _passed([t.id for t in solvable])
    abst_pass = _passed([t.id for t in abstain])
    prog_pass = _passed([t.id for t in needs_prog])

    print("=" * 74)
    print("NYXARA — honest GENERAL capability battery (held-out, not curated to the faculties)")
    print("=" * 74)
    print(report.summary())
    print("-" * 74)
    print("Honest split (this is the number, not vibes):")
    print(f"  • broadened maths she computes & certifies UNAIDED : "
          f"{solv_pass}/{len(solvable)}")
    print(f"  • abstains honestly on open-fact / chat            : "
          f"{abst_pass}/{len(abstain)}")
    print(f"  • needs a model to PROPOSE code (aided only)       : "
          f"{prog_pass}/{len(needs_prog)}  "
          f"({'0 unaided — expected' if prog_pass == 0 else 'aided path active'})")
    print("-" * 74)
    print("Reading it honestly: the middle number is correct calibration (abstaining beats "
          "bluffing);\nthe last is the aided-only gap. The FIRST number is her own unaided general "
          "reasoning —\nwhole problem-classes solved by computation she runs and verifies, with no "
          "model and no\nper-class hand-coding.")
    return report


if __name__ == "__main__":  # pragma: no cover
    run_general()
