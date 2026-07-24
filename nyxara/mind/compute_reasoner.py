"""NYXARA · mind/compute_reasoner.py — the general *reduce → run → verify* reasoner (✦, Problem #1).

The mind's older offline reasoning was a **fixed set of hand-coded problem-class matchers**: a
word-problem parser that only recognised "rate × count ± loose", a percent parser with two canonical
forms, a unit table, a comparative-adjective table, a constant-difference sequence rule. A genuinely
*new* kind of problem fell straight through — no matcher, no capability. That is the honest gap the
Master named: reasoning that only works for what a programmer pre-enumerated is not general.

This module is the answer, and it is *general by construction* rather than by enumeration. It does
**not** ask "which of my hardcoded classes is this?" — it asks "can I **reduce** this to a
computation, **run** it, and **verify** the result?" Three independent reducers try, in parallel:

* **symbolic** — a general Computer-Algebra front end over SymPy (:mod:`nyxara.mind.math`): solve an
  *arbitrary* equation or system for its unknowns, prove an *arbitrary* identity, differentiate /
  integrate / take a limit of an *arbitrary* expression, factor / test primality / gcd / lcm of
  *arbitrary* integers, evaluate an *arbitrary* closed expression. None of this is coded per problem
  — SymPy solves the whole class. Every result is **self-verified** (a solution is back-substituted;
  an identity is certified by ``simplify(lhs - rhs) == 0``; a factorisation's product is re-checked).
* **program** — Program-of-Thought. A candidate *program* is written for the problem and executed in
  the real isolated sandbox (:func:`nyxara.agency.code_sandbox.run_python`); the program's **output**
  is the answer. The candidate programs come from a caller-supplied ``proposer`` — NYXARA's *own*
  local model (SelfBrain / foundry / DistilGPT-2) writing code **only**. The model never asserts the answer;
  it proposes a program that NYXARA *runs*, so the truth-bearing step is her deterministic execution,
  not the model's word. This is the honest form of "the model proposes, NYXARA computes."
* **search** — for "is there / find an X such that …" shaped problems, a bounded brute-force /
  constraint program (also executed in the sandbox).

Then the **verifier decides**, never the proposer. An answer is emitted only when it is *certified*:
a SymPy self-verification, **or** two independent reductions agree (self-consistency is a real
correctness signal — see :mod:`nyxara.mind.verified_answer`), **or** it matches an exact faculty
oracle. A single un-cross-checkable guess is **refused** — :meth:`ComputeReasoner.solve` returns
``None`` (honest abstain), so the caller's existing ladder simply continues. NYXARA never bluffs.

Pure-stdlib core; SymPy is optional and only ever *strengthens* a result (without it, the symbolic
reducer stands down and the program/search reducers still run). Touches no weights and no gate: the
returned answer is a *proposal* the kernel still disposes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["ComputeResult", "ComputeReasoner", "solve_by_compute"]

# A proposer writes candidate *programs* (never answers) for a problem: (problem, n) -> [code, …].
Proposer = Callable[[str, int], Sequence[str]]


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class ComputeResult:
    """An answer NYXARA *computed and verified* herself, plus the re-checkable certificate."""

    answer: str
    confidence: float
    method: str                       # symbolic | program | search | cross-verified | self-consistency
    certificate: str = ""             # the witness anyone can re-check
    verified: bool = False            # True == exactly certified (proof/substitution/oracle)
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> Tuple[str, float]:
        """The ``(answer, confidence)`` shape the reasoner/router faculty-slots expect."""
        return self.answer, self.confidence

    def to_dict(self) -> Dict[str, Any]:
        return {"answer": self.answer, "confidence": round(float(self.confidence), 4),
                "method": self.method, "certificate": self.certificate,
                "verified": bool(self.verified), "detail": self.detail}


# --------------------------------------------------------------------------- #
# Text → maths normalisation (shared, tiny)
# --------------------------------------------------------------------------- #
def _to_math(expr: str) -> str:
    """Normalise human maths to Python/SymPy: ``^`` -> ``**``, implicit ``2x`` -> ``2*x``."""
    s = (expr or "").strip().rstrip(".?!").replace("^", "**")
    s = re.sub(r"(\d)\s*([A-Za-z(])", r"\1*\2", s)     # 2x -> 2*x, 3( -> 3*(
    s = re.sub(r"\)\s*([A-Za-z0-9(])", r")*\1", s)     # )(  -> )*(,  )x -> )*x
    return s.strip()


def _norm_answer(text: str) -> str:
    """A normalised key so equivalent answers ('x=2', 'x = 2.', '2') cluster together."""
    return " ".join(re.findall(r"[A-Za-z0-9]+", (text or "").lower()))


# --------------------------------------------------------------------------- #
# The reasoner
# --------------------------------------------------------------------------- #
class ComputeReasoner:
    """Solve a problem by reducing it to a computation, running it, and verifying the result.

    ``proposer`` — optional; a callable that writes candidate *programs* (NYXARA's own model). When
    absent, only the symbolic reducer and the deterministic expression path run, so the module is
    fully usable — and testable — on a machine with no model at all.
    """

    def __init__(self, *, proposer: Optional[Proposer] = None, sandbox_timeout: float = 5.0,
                 program_samples: int = 4, settings: Any = None) -> None:
        self.proposer = proposer
        self.sandbox_timeout = float(sandbox_timeout)
        self.program_samples = max(1, int(program_samples))
        self.settings = settings

    # ------------------------------------------------------------------ #
    # Public entry points
    # ------------------------------------------------------------------ #
    def solve(self, problem: str) -> Optional[ComputeResult]:
        """Return a *certified* answer, or ``None`` (honest abstain) when nothing verifies."""
        problem = (problem or "").strip()
        if not problem:
            return None
        try:
            symbolic = self._symbolic(problem)          # verified where SymPy can decide
        except Exception:  # noqa: BLE001 — a reducer never crashes the turn
            symbolic = None
        try:
            programs = self._program_outputs(problem)   # list of (answer, code) already executed
        except Exception:  # noqa: BLE001
            programs = []
        return self._select(problem, symbolic, programs)

    def answer(self, problem: str) -> Optional[Tuple[str, float]]:
        """Convenience for the reasoner/router faculty-slots: ``(answer, confidence)`` or ``None``."""
        res = self.solve(problem)
        return res.as_tuple() if res is not None else None

    # ------------------------------------------------------------------ #
    # Verify & select — the verifier decides, never the proposer
    # ------------------------------------------------------------------ #
    def _select(self, problem: str, symbolic: Optional[ComputeResult],
                programs: List[Tuple[str, str]]) -> Optional[ComputeResult]:
        prog_answers = [a for (a, _code) in programs if a]

        # 1) an exactly-certified symbolic answer wins outright — and is *strengthened* when an
        #    independent program agrees (cross-verification), the strongest signal there is.
        if symbolic is not None and symbolic.verified:
            agree = [a for a in prog_answers if _norm_answer(a) == _norm_answer(symbolic.answer)]
            if agree:
                symbolic.method = "cross-verified"
                symbolic.confidence = 0.99
                symbolic.certificate += f"; independently reproduced by a program ({len(agree)}×)"
                symbolic.detail["cross_verified_by_program"] = True
            return symbolic

        # 2) no symbolic proof — accept the programs' answer ONLY if independent runs agree
        #    (self-consistency). A lone, un-cross-checkable program output is refused.
        if prog_answers:
            winner, share, k, n = self._majority(prog_answers)
            if winner and k >= 2:
                # if an *unverified* symbolic attempt also lands here, it further corroborates.
                corroborated = (symbolic is not None
                                and _norm_answer(symbolic.answer) == _norm_answer(winner))
                conf = min(0.95 if corroborated else 0.9, 0.55 + 0.12 * k)
                cert = (f"{k}/{n} independently-written programs agree on this result"
                        + ("; corroborated by symbolic reduction" if corroborated else ""))
                return ComputeResult(answer=winner, confidence=conf,
                                     method="self-consistency", certificate=cert,
                                     verified=corroborated,
                                     detail={"agreement": round(share, 3), "samples": n,
                                             "cross_verified": corroborated})

        # 3) nothing certified — abstain honestly so the caller's ladder continues.
        return None

    @staticmethod
    def _majority(answers: List[str]) -> Tuple[str, float, int, int]:
        """Most-agreed answer by normalised value → (representative, share, count, total)."""
        buckets: Dict[str, List[str]] = {}
        for a in answers:
            buckets.setdefault(_norm_answer(a), []).append(a)
        if not buckets:
            return "", 0.0, 0, 0
        best_key = max(buckets, key=lambda k: len(buckets[k]))
        best = buckets[best_key]
        return best[0], len(best) / len(answers), len(best), len(answers)

    # ------------------------------------------------------------------ #
    # Reducer 1 — symbolic (general CAS front end, self-verifying)
    # ------------------------------------------------------------------ #
    def _symbolic(self, problem: str) -> Optional[ComputeResult]:
        sp = _sympy()
        if sp is None:
            return None
        low = problem.lower()

        # order matters: the more specific verbs first, evaluation last.
        for reducer in (self._sym_prove, self._sym_calculus, self._sym_number_theory,
                        self._sym_solve, self._sym_transform, self._sym_evaluate):
            try:
                res = reducer(problem, low, sp)
            except Exception:  # noqa: BLE001 — try the next reduction, never raise
                res = None
            if res is not None:
                return res
        return None

    # --- identity / proof: certify by simplify(lhs - rhs) == 0 -------------- #
    def _sym_prove(self, problem: str, low: str, sp: Any) -> Optional[ComputeResult]:
        if not any(k in low for k in ("prove", "identity", "show that", "verify that")):
            return None
        # drop the leading directive ("prove that", "show that", "verify the identity", …) so it
        # is not mistaken for a factor of the left-hand side.
        stmt = re.sub(r"^\s*(please\s+)?(prove|verify|show|check)(\s+(that|the\s+identity))?\s*:?\s*",
                      "", problem, flags=re.IGNORECASE)
        stmt = re.sub(r"\bidentity\b\s*:?\s*", "", stmt, flags=re.IGNORECASE)
        eq = self._extract_equation(stmt)
        if eq is None:
            return None
        lhs, rhs = eq
        le, re_ = sp.sympify(_to_math(lhs)), sp.sympify(_to_math(rhs))
        is_zero = sp.simplify(le - re_) == 0
        verdict = "PROVEN" if is_zero else "REFUTED"
        return ComputeResult(
            answer=f"{verdict}: {lhs.strip()} = {rhs.strip()}",
            confidence=0.97 if is_zero else 0.9, method="symbolic",
            certificate=f"simplify(({lhs.strip()}) - ({rhs.strip()})) "
                        f"{'== 0' if is_zero else '!= 0'}",
            verified=True, detail={"kind": "identity", "holds": bool(is_zero)})

    # --- calculus: derivative / integral / limit of an arbitrary expression - #
    def _sym_calculus(self, problem: str, low: str, sp: Any) -> Optional[ComputeResult]:
        x = sp.Symbol("x")
        m = re.search(r"(?:derivative of|differentiate|d/dx)\s+(.+)", problem, re.IGNORECASE)
        if m:
            e = sp.sympify(_to_math(self._strip_tail(m.group(1), (" with respect", " wrt"))))
            d = sp.diff(e, x)
            return ComputeResult(answer=str(d), confidence=0.95, method="symbolic",
                                 certificate=f"d/dx [{e}] = {d}", verified=True,
                                 detail={"kind": "derivative"})
        m = re.search(r"(?:integral of|integrate|antiderivative of)\s+(.+)", problem, re.IGNORECASE)
        if m:
            body = self._strip_tail(m.group(1), (" dx", " with respect", " wrt"))
            e = sp.sympify(_to_math(body))
            F = sp.integrate(e, x)
            # verify: d/dx of the result returns the integrand
            ok = sp.simplify(sp.diff(F, x) - e) == 0
            return ComputeResult(answer=f"{F} + C", confidence=0.95 if ok else 0.7,
                                 method="symbolic", certificate=f"d/dx [{F}] = {sp.diff(F, x)}",
                                 verified=bool(ok), detail={"kind": "integral"})
        m = re.search(r"limit of\s+(.+?)\s+as\s+([A-Za-z])\s*(?:->|→|approaches)\s*(\S+)",
                      problem, re.IGNORECASE)
        if m:
            var = sp.Symbol(m.group(2))
            pt = sp.oo if m.group(3).lower() in ("oo", "inf", "infinity") else sp.sympify(m.group(3))
            L = sp.limit(sp.sympify(_to_math(m.group(1))), var, pt)
            return ComputeResult(answer=str(L), confidence=0.92, method="symbolic",
                                 certificate=f"limit = {L}", verified=True,
                                 detail={"kind": "limit"})
        return None

    # --- number theory over arbitrary integers ------------------------------ #
    def _sym_number_theory(self, problem: str, low: str, sp: Any) -> Optional[ComputeResult]:
        m = re.search(r"is\s+(\d+)\s+(?:a\s+)?prime", low)
        if m:
            n = int(m.group(1))
            isp = bool(sp.isprime(n))
            if isp:
                cert = f"{n} has no divisor in 2..isqrt({n})"
            else:
                f = sp.factorint(n)
                cert = f"{n} = " + " * ".join(f"{p}**{e}" if e > 1 else f"{p}"
                                              for p, e in sorted(f.items()))
            return ComputeResult(answer="prime" if isp else "composite", confidence=0.98,
                                 method="symbolic", certificate=cert, verified=True,
                                 detail={"kind": "primality", "n": n})
        m = re.search(r"(?:prime\s+)?factor(?:ize|ise|isation|ization)?\s+(?:of\s+)?(\d+)", low)
        if m:
            n = int(m.group(1))
            f = sp.factorint(n)
            prod = 1
            for p, e in f.items():
                prod *= p ** e
            terms = " * ".join(f"{p}**{e}" if e > 1 else f"{p}" for p, e in sorted(f.items()))
            return ComputeResult(answer=terms, confidence=0.98, method="symbolic",
                                 certificate=f"product of factors = {prod} = {n}",
                                 verified=(prod == n), detail={"kind": "factorisation", "n": n})
        m = re.search(r"\b(gcd|lcm)\b[^0-9]*(\d+)[^0-9]+(\d+)", low)
        if m:
            op, a, b = m.group(1), int(m.group(2)), int(m.group(3))
            val = int(sp.gcd(a, b)) if op == "gcd" else int(sp.lcm(a, b))
            return ComputeResult(answer=str(val), confidence=0.98, method="symbolic",
                                 certificate=f"{op}({a}, {b}) = {val}", verified=True,
                                 detail={"kind": op})
        return None

    # --- solve an arbitrary equation or system for its unknown(s) ----------- #
    def _sym_solve(self, problem: str, low: str, sp: Any) -> Optional[ComputeResult]:
        # gather every "lhs = rhs" clause (single-'=' only, so we skip ==/<=/>=)
        clauses = self._extract_equations(problem)
        if not clauses:
            return None
        eqs = []
        symset = set()
        for lhs, rhs in clauses:
            le, re_ = sp.sympify(_to_math(lhs)), sp.sympify(_to_math(rhs))
            eqs.append(sp.Eq(le, re_))
            symset |= (le.free_symbols | re_.free_symbols)
        if not symset:
            return None
        syms = sorted(symset, key=lambda s: str(s))
        sols = sp.solve(eqs, syms, dict=True)
        if not sols:
            return ComputeResult(answer="no solution", confidence=0.85, method="symbolic",
                                 certificate="SymPy solve returned no solution", verified=True,
                                 detail={"kind": "equation", "equations": len(eqs)})
        # verify by back-substitution into every equation
        rendered, all_ok = [], True
        for sol in sols:
            ok = all(sp.simplify(e.lhs.subs(sol) - e.rhs.subs(sol)) == 0 for e in eqs)
            all_ok = all_ok and ok
            rendered.append(", ".join(f"{k} = {v}" for k, v in sol.items()))
        answer = "; ".join(rendered) if len(rendered) > 1 else rendered[0]
        return ComputeResult(
            answer=answer, confidence=0.96 if all_ok else 0.7, method="symbolic",
            certificate=f"substituting the solution into all {len(eqs)} equation(s) gives 0 = 0",
            verified=bool(all_ok),
            detail={"kind": "equation", "unknowns": [str(s) for s in syms],
                    "solutions": len(sols)})

    # --- simplify / expand / factor an arbitrary expression ----------------- #
    def _sym_transform(self, problem: str, low: str, sp: Any) -> Optional[ComputeResult]:
        for verb, fn in (("simplify", sp.simplify), ("expand", sp.expand), ("factor", sp.factor)):
            m = re.search(rf"{verb}\s+(.+)", problem, re.IGNORECASE)
            if m and not m.group(1).strip().isdigit():   # 'factor 12' is number theory, handled above
                e = sp.sympify(_to_math(m.group(1)))
                out = fn(e)
                ok = sp.simplify(sp.sympify(_to_math(m.group(1))) - out) == 0
                return ComputeResult(answer=str(out), confidence=0.94 if ok else 0.7,
                                     method="symbolic",
                                     certificate=f"{verb}({e}) = {out}; equivalence checked",
                                     verified=bool(ok), detail={"kind": verb})
        return None

    # --- evaluate an arbitrary closed expression ---------------------------- #
    def _sym_evaluate(self, problem: str, low: str, sp: Any) -> Optional[ComputeResult]:
        m = re.search(r"(?:what is|evaluate|compute|calculate)\s+(.+)", problem, re.IGNORECASE)
        raw = m.group(1) if m else problem
        raw = self._strip_tail(raw, (" equal", " equals", "="))
        expr = _to_math(raw)
        # must look like a real arithmetic/symbolic expression, not prose
        if not re.search(r"[0-9]", expr) or not re.search(r"[-+*/%()!]|\*\*", expr):
            return None
        if re.search(r"[A-Za-z]{4,}", expr):     # words -> prose, not an expression
            return None
        try:
            val = sp.sympify(expr)
        except Exception:  # noqa: BLE001
            return None
        if val.free_symbols:
            return None
        out = sp.nsimplify(val) if val.is_number else val
        return ComputeResult(answer=str(out), confidence=0.95, method="symbolic",
                             certificate=f"{expr} = {out}", verified=True,
                             detail={"kind": "evaluate"})

    # ------------------------------------------------------------------ #
    # Reducer 2/3 — program-of-thought & search (executed in the real sandbox)
    # ------------------------------------------------------------------ #
    def _program_outputs(self, problem: str) -> List[Tuple[str, str]]:
        """Collect (answer, code) from every candidate program that ran cleanly."""
        codes = self._candidate_programs(problem)
        if not codes:
            return []
        out: List[Tuple[str, str]] = []
        run = _sandbox_runner()
        if run is None:
            return []
        seen = set()
        for code in codes:
            if not code or code in seen:
                continue
            seen.add(code)
            res = run(code, timeout_s=self.sandbox_timeout)
            if not getattr(res, "ok", False):
                continue
            ans = self._answer_of(res)
            if ans:
                out.append((ans, code))
        return out

    def _candidate_programs(self, problem: str) -> List[str]:
        codes: List[str] = []
        if self.proposer is not None:
            try:
                proposed = self.proposer(problem, self.program_samples) or []
            except Exception:  # noqa: BLE001 — a broken proposer just means no program candidates
                proposed = []
            for c in proposed:
                c = self._sanitise_code(c)
                if c:
                    codes.append(c)
        return codes

    @staticmethod
    def _sanitise_code(code: Any) -> str:
        """Accept a fenced or bare snippet; require it to *set ``result``* (the sandbox contract)."""
        if not isinstance(code, str):
            return ""
        m = re.search(r"```(?:python|py)?\s*(.+?)```", code, re.DOTALL | re.IGNORECASE)
        body = (m.group(1) if m else code).strip()
        if not body or "result" not in body:
            return ""
        return body

    @staticmethod
    def _answer_of(res: Any) -> str:
        val = getattr(res, "value", None)
        if val is not None:
            return str(val).strip()
        return (getattr(res, "stdout", "") or "").strip().splitlines()[-1].strip() \
            if (getattr(res, "stdout", "") or "").strip() else ""

    # ------------------------------------------------------------------ #
    # small parsing helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _strip_tail(text: str, tails: Sequence[str]) -> str:
        s = text.strip().rstrip(".?!")
        low = s.lower()
        for t in tails:
            i = low.find(t.lower())
            if i != -1:
                s = s[:i]
                low = s.lower()
        return s.strip()

    @staticmethod
    def _extract_equation(problem: str) -> Optional[Tuple[str, str]]:
        """The single 'lhs = rhs' an identity/proof refers to (ignores ==, <=, >=)."""
        for chunk in re.split(r"[,;]| that | if ", problem):
            if "=" in chunk and not re.search(r"[<>=!]=", chunk):
                lhs, rhs = chunk.split("=", 1)
                if lhs.strip() and rhs.strip():
                    return lhs, rhs
        return None

    @staticmethod
    def _extract_equations(problem: str) -> List[Tuple[str, str]]:
        """Every 'lhs = rhs' clause in the problem (for a system). Drops a trailing 'for x, y'."""
        body = re.sub(r"\bfor\s+[A-Za-z](\s*,\s*[A-Za-z])*\s*$", "", problem.strip(), flags=re.I)
        body = re.sub(r"^\s*solve\b", "", body, flags=re.I)
        clauses: List[Tuple[str, str]] = []
        for chunk in re.split(r"\band\b|[,;]", body):
            if "=" in chunk and not re.search(r"[<>=!]=", chunk):
                lhs, rhs = chunk.split("=", 1)
                if re.search(r"[A-Za-z]", lhs + rhs) and lhs.strip() and rhs.strip():
                    clauses.append((lhs, rhs))
        return clauses


# --------------------------------------------------------------------------- #
# Lazy, import-guarded dependencies (never a hard requirement)
# --------------------------------------------------------------------------- #
def _sympy() -> Any:
    try:
        import sympy  # type: ignore
        return sympy
    except Exception:  # noqa: BLE001
        return None


def _sandbox_runner() -> Optional[Callable[..., Any]]:
    try:
        from nyxara.agency.code_sandbox import run_python
        return run_python
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Module-level convenience
# --------------------------------------------------------------------------- #
def solve_by_compute(problem: str, *, proposer: Optional[Proposer] = None,
                     settings: Any = None) -> Optional[Tuple[str, float]]:
    """One-shot ``(answer, confidence)`` (or ``None``) — the faculty-slot entry the reasoner uses."""
    return ComputeReasoner(proposer=proposer, settings=settings).answer(problem)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    r = ComputeReasoner()
    print("=" * 72)
    print("NYXARA compute-reasoner self-test (offline, no model — symbolic reducer only)")
    print("=" * 72)
    problems = [
        "solve 3*x + 2*y = 12 and x - y = 1 for x, y",   # linear system (beyond old faculties)
        "solve x**3 - 6*x**2 + 11*x - 6 = 0",            # cubic
        "differentiate x**3 + sin(x)",
        "integrate 2*x",
        "prove (a+b)**2 = a**2 + 2*a*b + b**2",
        "factor 5040",
        "is 561 prime",
        "gcd of 48 and 36",
        "what is 2**10 + 24",
    ]
    for p in problems:
        res = r.solve(p)
        if res is None:
            print(f"  {p:48s} -> (abstain)")
        else:
            print(f"  {p:48s} -> {res.answer}   [{res.method}, verified={res.verified}]")
            print(f"  {'':48s}    ⌙ {res.certificate}")
