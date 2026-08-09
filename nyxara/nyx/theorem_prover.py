"""NYXARA · nyx/theorem_prover.py — a claim that carries its proof (⊢, NYX V.02).

The repo's control law is already **verifiable beats probabilistic**: a derived answer outranks
a confident guess (:mod:`nyxara.nyx.hybrid`). What "verifiable" meant until now was *her own
engines agreed with it*. This module gives that word a much harder meaning where the claim
admits one — a machine-checkable certificate from z3, and SymPy behind it for exact algebra.

    prove("x*x >= 0 for all real x")   →  PROVED,   certificate: ¬F is unsatisfiable
    prove("2 + 2 = 5")                 →  REFUTED,  counter-model included
    prove("x > 5")                     →  CONTINGENT, and named as such — it depends on x
    prove("gravity pulls the apple")   →  INEXPRESSIBLE, and no verdict is invented

Honest framing (the rule this repo keeps, see ``docs/CAPABILITIES.md``):

* **"Eliminate hallucination mathematically" cannot be fully true, and is not claimed here.**
  A proof runs on **formally expressible** claims. *"Gravity pulls the apple down"* is not an
  SMT formula and never will be; asked to prove it, this module returns ``INEXPRESSIBLE`` and
  refuses to produce a verdict at all. That refusal is the feature. What is real is that a
  provable claim now travels **with its certificate**, and everything else stays labelled
  exactly as before.
* **A timeout is a timeout.** On ``unknown`` or a budget overrun she says *"I could not
  prove this"* — never *"it is correct"*. Undecidable and hard are both reported as what they
  are.
* **Contingent is not the same as false.** ``x > 5`` is neither proved nor refuted, and saying
  "refuted" would be a lie about a claim that is simply about a free variable.
* The claim language is **closed and small** — arithmetic, comparison, boolean structure, and
  a universal/existential quantifier over declared reals or integers. Anything that does not
  parse under it is inexpressible, by construction rather than by judgement.

Pure standard library plus optional ``z3`` and ``sympy`` (both present in this environment:
z3 5.0.0, sympy 1.14.0). Every public method is fail-soft.
"""

from __future__ import annotations

import ast
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Verdict", "Certificate", "ProofCore", "Z3_LOCK"]

#: z3's Python API builds every expression in one **global context**, and that context is not
#: thread-safe: two threads asserting into it at the same time segfault the interpreter, which
#: is not something a fail-soft ``except`` can catch. Since V.02 the prover is reachable from
#: ``think()``, and the orchestrator reasons on a thread pool — so every z3 section in this repo
#: takes this one process-wide lock. Solving is milliseconds and bounded by an explicit timeout,
#: so serialising it costs far less than the crash it prevents.
Z3_LOCK = threading.RLock()


class Verdict:
    """What a proof attempt actually concluded. Five outcomes, and none of them is a guess."""

    PROVED = "proved"                 # ¬F is unsatisfiable — it holds for every model
    REFUTED = "refuted"               # F itself is unsatisfiable — it holds for none
    CONTINGENT = "contingent"         # both have models — it depends on the free variables
    INEXPRESSIBLE = "inexpressible"   # not a formula at all; no verdict is invented
    UNKNOWN = "unknown"               # the solver could not decide inside the budget

    #: The only two that may ever be treated as evidence downstream.
    DECISIVE = frozenset({PROVED, REFUTED})


@dataclass
class Certificate:
    """A proof attempt, and everything needed to check or reproduce it."""

    claim: str = ""
    verdict: str = Verdict.INEXPRESSIBLE
    formula: str = ""
    engine: str = ""                  # z3 | sympy | none
    witness: str = ""                 # a model, or a counter-model
    variables: Dict[str, str] = field(default_factory=dict)   # name → sort
    elapsed_ms: float = 0.0
    note: str = ""

    @property
    def proved(self) -> bool:
        return self.verdict == Verdict.PROVED

    @property
    def decisive(self) -> bool:
        return self.verdict in Verdict.DECISIVE

    def render(self) -> str:
        head = f"{self.verdict.upper():14} {self.claim}"
        lines = [head]
        if self.formula:
            lines.append(f"  formula   : {self.formula}")
        if self.variables:
            lines.append("  variables : " + ", ".join(f"{k}:{v}"
                                                      for k, v in self.variables.items()))
        if self.witness:
            label = "counter-model" if self.verdict == Verdict.CONTINGENT else "model"
            lines.append(f"  {label:10}: {self.witness}")
        lines.append(f"  by        : {self.engine or 'nothing'}  ({self.elapsed_ms:.1f} ms)")
        lines.append(f"  note      : {self.note}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"claim": self.claim, "verdict": self.verdict, "formula": self.formula,
                "engine": self.engine, "witness": self.witness,
                "variables": dict(self.variables), "proved": self.proved,
                "decisive": self.decisive,
                "elapsed_ms": round(self.elapsed_ms, 2), "note": self.note}


# --------------------------------------------------------------------------- #
# The claim language — closed, small, and stated as such.
# --------------------------------------------------------------------------- #
#: How a sort is declared in prose. Everything else defaults to Real.
_INT_HINT = re.compile(r"\b(?:integer|integers|int|natural|whole)\b", re.I)
#: Universal quantification, in the shapes people actually write it.
_FORALL = re.compile(r"\b(?:for\s+(?:all|every|each|any)|forall|∀)\b", re.I)
_EXISTS = re.compile(r"\b(?:there\s+(?:exists|is)|for\s+some|exists|∃)\b", re.I)
#: A quantifier clause, WITH the variables it declares — "for all real x", "for every integer
#: n, m". Captured as a unit so the clause can be removed whole: stripping only the words left
#: the declared variable dangling on the end, and "x*x >= 0 x" does not parse.
_QUANT_CLAUSE = re.compile(
    r"\b(?:for\s+(?:all|every|each|any|some)|forall|exists|there\s+(?:exists|is))\s+"
    r"(?:real|reals|integer|integers|int|natural|whole)?\s*"
    r"(?:numbers?|values?)?\s*"
    # Each declared name must NOT be followed by an operator: in "for all integer n, n*n >= n"
    # a greedy list happily eats the "n" of "n*n" as a second variable and leaves "*n >= n",
    # which does not parse. The lookahead stops the list at the formula's edge.
    r"(?P<vars>[A-Za-z_]\w*(?!\s*[-+*/%><=!(])"
    r"(?:\s*,\s*[A-Za-z_]\w*(?!\s*[-+*/%><=!(]))*)\s*[,:]?",
    re.I)
#: Prose that surrounds a formula and should be stripped before parsing.
_STRIP = re.compile(
    r"\b(?:real|reals|integer|integers|int|natural|whole|numbers?|values?|where|"
    r"such\s+that|holds?|is\s+true|prove\s+that|show\s+that|"
    r"it\s+is\s+the\s+case\s+that)\b", re.I)

_OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**",
        ast.Mod: "%", ast.FloorDiv: "//"}


class ProofCore:
    """Turn a claim into a formula, hand it to a solver, and report exactly what came back."""

    def __init__(self, *, timeout_ms: int = 3000, max_vars: int = 8) -> None:
        self.timeout_ms = max(50, int(timeout_ms))
        self.max_vars = max(1, int(max_vars))
        self.attempts = 0
        self.proved = 0
        self.refuted = 0
        self.contingent = 0
        self.inexpressible = 0
        self.unknown = 0

    # ---- what is actually installed ---------------------------------------- #
    @staticmethod
    def _z3() -> Any:
        try:
            import z3
            return z3
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _sympy() -> Any:
        try:
            import sympy
            return sympy
        except Exception:  # noqa: BLE001
            return None

    def available(self) -> Dict[str, bool]:
        return {"z3": self._z3() is not None, "sympy": self._sympy() is not None}

    # ---- the one call -------------------------------------------------------- #
    def prove(self, claim: str) -> Certificate:
        """Attempt a proof. An inexpressible claim gets **no verdict**, which is the point."""
        out = Certificate(claim=str(claim or "").strip())
        t0 = time.perf_counter()
        try:
            if not out.claim:
                out.note = "nothing was claimed"
                return out
            self.attempts += 1

            parsed = self.parse(out.claim)
            if parsed is None:
                self.inexpressible += 1
                out.verdict = Verdict.INEXPRESSIBLE
                out.note = (
                    "this is not a formally expressible claim, so there is nothing to prove or "
                    "refute. My claim language is arithmetic, comparison, boolean structure and "
                    "one quantifier over declared reals or integers. A sentence about the world "
                    "is not an SMT formula, and inventing a verdict for it would be exactly the "
                    "dishonesty a prover is supposed to prevent.")
                return out
            body, sorts, quantifier = parsed
            out.formula = body
            out.variables = dict(sorts)

            z3 = self._z3()
            if z3 is None:
                out.verdict = Verdict.UNKNOWN
                out.note = "z3 is not installed, so nothing here can be decided"
                self.unknown += 1
                return out

            with Z3_LOCK:      # z3's global context is not thread-safe; see Z3_LOCK
                verdict, witness, engine, note = self._decide(z3, body, sorts, quantifier)
            out.verdict, out.witness, out.engine, out.note = verdict, witness, engine, note
            self._count(verdict)
            return out
        except Exception as exc:  # noqa: BLE001 — a failed proof is never a proof
            out.verdict = Verdict.UNKNOWN
            out.note = f"the attempt failed ({type(exc).__name__}: {exc}) — that is not a proof"
            return out
        finally:
            out.elapsed_ms = (time.perf_counter() - t0) * 1000.0

    def _count(self, verdict: str) -> None:
        setattr(self, verdict if hasattr(self, verdict) else "unknown",
                getattr(self, verdict, self.unknown) + 1)

    # ---- claim → formula ------------------------------------------------------ #
    def parse(self, claim: str) -> Optional[Tuple[str, Dict[str, str], str]]:
        """``(python-ish formula, {var: sort}, quantifier)`` — or ``None`` if inexpressible."""
        try:
            text = str(claim or "").strip().rstrip(".")
            quantifier = ("forall" if _FORALL.search(text)
                          else "exists" if _EXISTS.search(text) else "")
            sort = "Int" if _INT_HINT.search(text) else "Real"

            # Remove the quantifier clause as a unit, keeping the variables it declared. z3
            # treats a free variable as universally quantified for validity checking anyway,
            # so the names are what matter, not the syntax that introduced them.
            declared: List[str] = []
            body = text
            match = _QUANT_CLAUSE.search(body)
            if match is not None:
                declared = [v.strip() for v in match.group("vars").split(",") if v.strip()]
                body = (body[: match.start()] + " " + body[match.end():])
            body = _STRIP.sub(" ", body)
            body = body.replace("≥", ">=").replace("≤", "<=").replace("≠", "!=")
            body = body.replace("→", "->").replace("¬", "not ").replace("∧", " and ")
            body = body.replace("∨", " or ")
            body = re.sub(r"->", ",", body) if body.count("->") == 1 and "implies" not in body \
                else body
            body = re.sub(r"[,;]\s*$", "", body).strip(" ,;")
            # A single "=" is equality in maths and assignment in Python. Everywhere a human
            # writes it in a claim, they mean the former.
            body = re.sub(r"(?<![=!<>])=(?!=)", "==", body)
            if not body:
                return None

            tree = ast.parse(body, mode="eval")
            names: Dict[str, str] = {name: sort for name in declared}
            if not self._is_boolean(tree.body):
                # A bare term ("gravity", "x + 1") is not a claim. It has no truth value, and
                # pretending it does is how a prover starts certifying nonsense.
                return None
            rendered = self._render(tree.body, names, sort, boolean=True)
            if rendered is None or len(names) > self.max_vars:
                return None
            return rendered, names, quantifier
        except Exception:  # noqa: BLE001 — anything that does not parse is inexpressible
            return None

    @staticmethod
    def _is_boolean(node: ast.AST) -> bool:
        if isinstance(node, (ast.Compare, ast.BoolOp)):
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id.lower() in ("implies", "and_", "or_", "not_")
        return False

    def _render(self, node: ast.AST, names: Dict[str, str], sort: str, *,
                boolean: bool = False) -> Optional[str]:
        """Re-render the tree into a form the z3 builder below understands. Whitelist only.

        ``boolean`` says this position expects a truth value. A bare name there is a *boolean*
        variable, not a number — without that, ``x and not x`` builds a Real inside And() and
        z3 refuses it, which is a parse bug reported as an unknown.
        """
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "True" if node.value else "False"
            if isinstance(node.value, (int, float)):
                return repr(node.value)
            return None
        if isinstance(node, ast.Name):
            if not node.id.isidentifier() or len(node.id) > 24:
                return None
            names[node.id] = "Bool" if boolean else names.get(node.id, sort)
            return node.id
        if isinstance(node, ast.UnaryOp):
            inner = self._render(node.operand, names, sort,
                                 boolean=isinstance(node.op, ast.Not))
            if inner is None:
                return None
            if isinstance(node.op, ast.USub):
                return f"(-{inner})"
            if isinstance(node.op, ast.UAdd):
                return inner
            if isinstance(node.op, ast.Not):
                return f"Not({inner})"
            return None
        if isinstance(node, ast.BinOp):
            op = _OPS.get(type(node.op))
            left = self._render(node.left, names, sort)
            right = self._render(node.right, names, sort)
            if op is None or left is None or right is None:
                return None
            return f"({left} {op} {right})"
        if isinstance(node, ast.Compare):
            parts: List[str] = []
            left = self._render(node.left, names, sort)
            if left is None:
                return None
            for op, comparator in zip(node.ops, node.comparators):
                symbol = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
                          ast.Gt: ">", ast.GtE: ">="}.get(type(op))
                right = self._render(comparator, names, sort)
                if symbol is None or right is None:
                    return None
                parts.append(f"({left} {symbol} {right})")
                left = right
            return parts[0] if len(parts) == 1 else "And(" + ", ".join(parts) + ")"
        if isinstance(node, ast.BoolOp):
            joined = [self._render(v, names, sort, boolean=True) for v in node.values]
            if any(v is None for v in joined):
                return None
            head = "And" if isinstance(node.op, ast.And) else "Or"
            return f"{head}(" + ", ".join(joined) + ")"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id.lower()
            args = [self._render(a, names, sort, boolean=(name != "implies" or True))
                    for a in node.args]
            if any(a is None for a in args):
                return None
            if name == "implies" and len(args) == 2:
                return f"Implies({args[0]}, {args[1]})"
            if name in ("and_", "or_") and args:
                return ("And(" if name == "and_" else "Or(") + ", ".join(args) + ")"
            if name == "not_" and len(args) == 1:
                return f"Not({args[0]})"
            return None
        return None

    # ---- formula → verdict ------------------------------------------------------ #
    def _decide(self, z3: Any, body: str, sorts: Dict[str, str],
                quantifier: str) -> Tuple[str, str, str, str]:
        env: Dict[str, Any] = {"And": z3.And, "Or": z3.Or, "Not": z3.Not,
                               "Implies": z3.Implies, "True": True, "False": False}
        for name, sort in sorts.items():
            env[name] = (z3.Bool(name) if sort == "Bool"
                         else z3.Int(name) if sort == "Int" else z3.Real(name))
        formula = eval(body, {"__builtins__": {}}, env)  # noqa: S307 — whitelisted render only

        # The quantifier is not decoration: it decides what a counter-model MEANS.
        if quantifier == "exists":
            satisfiable, model = self._unsat(z3, formula)
            if satisfiable is None:
                return self._sympy_fallback(body, sorts)
            if satisfiable:            # F is unsatisfiable ⇒ nothing witnesses it
                return (Verdict.REFUTED, "", "z3",
                        "there is no model at all, so the existential claim is false")
            return (Verdict.PROVED, model, "z3",
                    "a witness exists — the existential claim holds, and here it is")

        valid, counter = self._unsat(z3, z3.Not(formula))
        if valid is None:
            return self._sympy_fallback(body, sorts)
        if valid:
            return (Verdict.PROVED, "", "z3",
                    "¬F is unsatisfiable: it holds in every model — a real certificate")

        if quantifier == "forall":
            # "for all x, x*x >= 1" is FALSE, not contingent — the counter-model settles it.
            return (Verdict.REFUTED, counter, "z3",
                    "a counter-model exists, so the universal claim is false — here it is")

        unsatisfiable, model = self._unsat(z3, formula)
        if unsatisfiable is None:
            return self._sympy_fallback(body, sorts)
        if unsatisfiable:
            return (Verdict.REFUTED, counter, "z3",
                    "F itself is unsatisfiable: it holds in no model at all")
        # Both have models, and no quantifier said which way to read that. A real and common
        # answer, and it is not "false".
        return (Verdict.CONTINGENT, counter or model, "z3",
                "neither proved nor refuted: it depends on the free variables, and calling "
                "that 'false' would be a lie about a claim that is simply contingent")

    def _unsat(self, z3: Any, formula: Any) -> Tuple[Optional[bool], str]:
        """``(True, "")`` when unsatisfiable, ``(False, model)`` when satisfiable, ``None`` on unknown."""
        try:
            solver = z3.Solver()
            solver.set("timeout", self.timeout_ms)
            solver.add(formula)
            verdict = solver.check()
            if verdict == z3.unsat:
                return True, ""
            if verdict == z3.sat:
                try:
                    model = solver.model()
                    rendered = ", ".join(f"{d.name()}={model[d]}" for d in model.decls())
                except Exception:  # noqa: BLE001
                    rendered = ""
                return False, rendered
            return None, ""
        except Exception:  # noqa: BLE001
            return None, ""

    def _sympy_fallback(self, body: str, sorts: Dict[str, str]) -> Tuple[str, str, str, str]:
        """A second engine for exact algebra — and an honest ``unknown`` when it declines too."""
        sympy = self._sympy()
        if sympy is None:
            return (Verdict.UNKNOWN, "", "z3",
                    "the solver returned unknown inside the budget — I could not prove this, "
                    "which is not the same as it being correct")
        try:
            match = re.fullmatch(r"\((.+?) == (.+)\)", body.strip())
            if match is None:
                return (Verdict.UNKNOWN, "", "z3",
                        "the solver returned unknown and this is not an equation SymPy can "
                        "take on — I could not prove it, which is not the same as it holding")
            symbols = {n: sympy.Symbol(n, integer=(s == "Int"), real=True)
                       for n, s in sorts.items()}
            left = sympy.sympify(match.group(1), locals=symbols)
            right = sympy.sympify(match.group(2), locals=symbols)
            simplified = sympy.simplify(left - right)
            if simplified == 0:
                return (Verdict.PROVED, "", "sympy",
                        "SymPy simplified both sides to the same expression — exact algebra, "
                        "not a numeric check")
            return (Verdict.UNKNOWN, "", "sympy",
                    f"SymPy could not reduce the difference to zero (it left {simplified}) — "
                    f"that is a failure to prove, not a refutation")
        except Exception:  # noqa: BLE001
            return (Verdict.UNKNOWN, "", "sympy",
                    "neither engine could decide this inside the budget")

    # ---- what the rest of NYX asks for -------------------------------------------- #
    def certify(self, claim: str) -> Optional[Certificate]:
        """A certificate only when there genuinely is one. ``None`` means *do not upgrade this*."""
        try:
            got = self.prove(claim)
            return got if got.proved else None
        except Exception:  # noqa: BLE001
            return None

    def check_properties(self, claims: List[str]) -> List[Certificate]:
        """Property checks — what :mod:`nyxara.nyx.author` runs over code it just wrote."""
        out: List[Certificate] = []
        for claim in list(claims or [])[:16]:
            out.append(self.prove(claim))
        return out

    # ---- introspection --------------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        try:
            return {
                "available": self.available(),
                "attempts": self.attempts, "proved": self.proved, "refuted": self.refuted,
                "contingent": self.contingent, "inexpressible": self.inexpressible,
                "unknown": self.unknown, "timeout_ms": self.timeout_ms,
                "verdicts": [Verdict.PROVED, Verdict.REFUTED, Verdict.CONTINGENT,
                             Verdict.INEXPRESSIBLE, Verdict.UNKNOWN],
                "note": ("'eliminate hallucination mathematically' cannot be fully true and is "
                         "not claimed: a proof runs only on FORMALLY EXPRESSIBLE claims. "
                         "'Gravity pulls the apple down' is not an SMT formula, and asked to "
                         "prove it this returns INEXPRESSIBLE rather than inventing a verdict "
                         "— that refusal is the feature. A timeout is reported as 'I could not "
                         "prove this', never as 'it is correct', and contingent is never "
                         "reported as false"),
            }
        except Exception:  # noqa: BLE001
            return {}

    def describe(self) -> str:
        """What ``/nyx prove`` prints with no argument."""
        got = self.available()
        return "\n".join([
            f"z3    : {'available' if got['z3'] else 'ABSENT — nothing can be decided'}",
            f"sympy : {'available' if got['sympy'] else 'absent'}",
            "language: arithmetic, comparison, boolean structure, one quantifier over "
            "declared reals or integers.",
            "Anything outside that is INEXPRESSIBLE — I will not invent a verdict for a "
            "sentence about the world.",
            f"record: {self.proved} proved, {self.refuted} refuted, "
            f"{self.contingent} contingent, {self.inexpressible} inexpressible, "
            f"{self.unknown} undecided.",
        ])

    def to_dict(self) -> Dict[str, Any]:
        return {"attempts": self.attempts, "proved": self.proved, "refuted": self.refuted,
                "contingent": self.contingent, "inexpressible": self.inexpressible,
                "unknown": self.unknown}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            for key in ("attempts", "proved", "refuted", "contingent", "inexpressible",
                        "unknown"):
                setattr(self, key, int(data.get(key, 0)))
            return True
        except Exception:  # noqa: BLE001
            return False
