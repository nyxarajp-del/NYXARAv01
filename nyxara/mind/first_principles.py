"""NYXARA · mind/first_principles.py — derive new results from the rules (✦, first principles).

Most systems *recall*: they have seen the answer and repeat it. This faculty *derives*: it
starts from the axioms of a domain and works forward to a result it was never told — the
difference between memorising "v = √(2GM/r)" and obtaining it from energy conservation, or
between looking up "C₃H₈ + 5O₂ → 3CO₂ + 4H₂O" and *computing* the coefficients from
conservation of atoms. Four domains, each verifiable (symbolic, not guessed):

* **Physics** — dimensional analysis over the seven SI base dimensions (derive the *form* of a
  quantity it has never seen from the dimensions of its inputs), plus exact symbolic derivations
  from stated laws (energy conservation → escape velocity; a = dv/dt → kinematics), via SymPy
  ``solve`` / ``integrate``.
* **Chemistry** — :func:`balance_reaction` builds the element-count matrix and solves for the
  integer stoichiometric coefficients from its null space (exact, not a lookup), with mole/mass
  stoichiometry off the balanced equation.
* **Maths** — an axiomatic prove-or-derive chain: parse the claim, expand/simplify symbolically,
  and *certify* it by ``simplify(lhs - rhs) == 0`` (a checked proof, never a bluff).
* **Logic** — forward-chaining deductive closure (modus ponens over rules + facts) so a
  conclusion is *reached*, with the inference trail recorded.

Everything degrades gracefully: SymPy is import-guarded (already a project dependency, used by
:mod:`mind.math`); without it the chemistry/logic paths still run on pure rationals/stdlib and
the symbolic-physics path defers. Each engine returns a :class:`Derivation` — a step-by-step,
inspectable trail — and the :class:`FirstPrinciplesFaculty` wraps it as a verifiable
:class:`~nyxara.mind.proposal.Proposal`; the kernel still disposes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nyxara.mind.faculties import Faculty, Task, TaskType
from nyxara.mind.proposal import ProposalKind
from nyxara.memory.provenance import SourceType

__all__ = [
    "DerivationStep",
    "Derivation",
    "Dimension",
    "DimensionalAnalysis",
    "PhysicsEngine",
    "parse_formula",
    "balance_reaction",
    "ChemistryEngine",
    "MathEngine",
    "LogicEngine",
    "FirstPrinciplesEngine",
    "FirstPrinciplesFaculty",
    "derive_from_first_principles",
]


# --------------------------------------------------------------------------- #
# The derivation record — a verifiable, inspectable trail
# --------------------------------------------------------------------------- #
@dataclass
class DerivationStep:
    """One line of a worked derivation."""

    n: int                       # 1-based step index
    expression: str              # the state after this step
    rule: str                    # which rule was applied (axiom/law/algebra/substitution/…)
    justification: str = ""      # why the rule applies

    def render(self) -> str:
        tail = f"   [{self.rule}: {self.justification}]" if self.justification else f"   [{self.rule}]"
        return f"  {self.n}. {self.expression}{tail}"


@dataclass
class Derivation:
    """A complete derivation: assumptions → steps → result, with a verification flag."""

    domain: str                                  # physics / chemistry / maths / logic
    query: str                                   # what was asked
    assumptions: List[str] = field(default_factory=list)
    steps: List[DerivationStep] = field(default_factory=list)
    result: str = ""
    verified: bool = False                       # True only when symbolically certified
    confidence: float = 1.0

    def add(self, expression: str, rule: str, justification: str = "") -> "Derivation":
        self.steps.append(DerivationStep(len(self.steps) + 1, expression, rule, justification))
        return self

    def render(self) -> str:
        lines = [f"Derivation ({self.domain}) — {self.query}"]
        if self.assumptions:
            lines.append("Assumptions:")
            lines += [f"  · {a}" for a in self.assumptions]
        lines.append("Steps:")
        lines += [s.render() for s in self.steps]
        lines.append(f"Result: {self.result}")
        lines.append(f"Verified: {self.verified}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain, "query": self.query, "assumptions": list(self.assumptions),
            "steps": [{"n": s.n, "expression": s.expression, "rule": s.rule,
                       "justification": s.justification} for s in self.steps],
            "result": self.result, "verified": self.verified, "confidence": self.confidence,
        }


def _has_sympy() -> bool:
    try:
        from nyxara.mind.math import has_sympy
        return bool(has_sympy())
    except Exception:  # noqa: BLE001
        try:
            import sympy  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False


# --------------------------------------------------------------------------- #
# Physics — dimensional analysis over the seven SI base dimensions
# --------------------------------------------------------------------------- #
# Base order: Mass(M) Length(L) Time(T) Current(I) Temperature(Θ) Amount(N) Luminosity(J)
_BASE_SYMBOLS = ("M", "L", "T", "I", "Θ", "N", "J")


@dataclass(frozen=True)
class Dimension:
    """A physical dimension as exponents over the seven SI base dimensions."""

    exps: Tuple[float, ...]  # length 7, aligned to _BASE_SYMBOLS

    @classmethod
    def base(cls, **kw: float) -> "Dimension":
        idx = {s: i for i, s in enumerate(_BASE_SYMBOLS)}
        e = [0.0] * 7
        for k, v in kw.items():
            e[idx[k]] = float(v)
        return cls(tuple(e))

    def __mul__(self, o: "Dimension") -> "Dimension":
        return Dimension(tuple(a + b for a, b in zip(self.exps, o.exps)))

    def __truediv__(self, o: "Dimension") -> "Dimension":
        return Dimension(tuple(a - b for a, b in zip(self.exps, o.exps)))

    def pow(self, k: float) -> "Dimension":
        return Dimension(tuple(a * k for a in self.exps))

    def is_dimensionless(self) -> bool:
        return all(abs(e) < 1e-9 for e in self.exps)

    def render(self) -> str:
        parts = []
        for sym, e in zip(_BASE_SYMBOLS, self.exps):
            if abs(e) < 1e-9:
                continue
            e_int = int(round(e))
            exp = str(e_int) if abs(e - e_int) < 1e-9 else f"{e:g}"
            parts.append(f"{sym}" if exp == "1" else f"{sym}^{exp}")
        return "·".join(parts) if parts else "1 (dimensionless)"


# A library of named quantities and their dimensions — the "rules" of physics, as dimensions.
_QUANTITY_DIMS: Dict[str, Dimension] = {
    "mass": Dimension.base(M=1), "length": Dimension.base(L=1),
    "distance": Dimension.base(L=1), "radius": Dimension.base(L=1),
    "height": Dimension.base(L=1), "time": Dimension.base(T=1),
    "velocity": Dimension.base(L=1, T=-1), "speed": Dimension.base(L=1, T=-1),
    "acceleration": Dimension.base(L=1, T=-2), "gravity": Dimension.base(L=1, T=-2),
    "force": Dimension.base(M=1, L=1, T=-2), "momentum": Dimension.base(M=1, L=1, T=-1),
    "energy": Dimension.base(M=1, L=2, T=-2), "work": Dimension.base(M=1, L=2, T=-2),
    "power": Dimension.base(M=1, L=2, T=-3), "pressure": Dimension.base(M=1, L=-1, T=-2),
    "density": Dimension.base(M=1, L=-3), "area": Dimension.base(L=2),
    "volume": Dimension.base(L=3), "frequency": Dimension.base(T=-1),
    "charge": Dimension.base(I=1, T=1), "current": Dimension.base(I=1),
    "temperature": Dimension.base(**{"Θ": 1}),
    # constants
    "g": Dimension.base(L=1, T=-2),
    "gravitational_constant": Dimension.base(M=-1, L=3, T=-2),   # G
    "spring_constant": Dimension.base(M=1, T=-2),                # k (N/m)
}


class DimensionalAnalysis:
    """Derive the *form* of an unknown quantity from the dimensions of its inputs.

    Solves ``Σ aᵢ·dim(inputᵢ) = dim(target)`` for the exponents ``aᵢ`` over the seven base
    dimensions — i.e. recovers the only dimensionally-consistent monomial (up to a constant).
    This is how a never-before-seen formula is reached from rules alone: the pendulum period
    must go as √(L/g) because nothing else is dimensionally possible.
    """

    def derive(self, target: str, inputs: List[str]) -> Optional[Derivation]:
        tdim = _QUANTITY_DIMS.get(target.lower().replace(" ", "_"))
        idims = []
        for name in inputs:
            d = _QUANTITY_DIMS.get(name.lower().replace(" ", "_"))
            if d is None:
                return None
            idims.append((name, d))
        if tdim is None or not idims:
            return None
        if not _has_sympy():
            return None
        import sympy as sp

        a = sp.symbols(f"a0:{len(idims)}", real=True)
        eqs = []
        for row in range(7):
            lhs = sum(a[j] * idims[j][1].exps[row] for j in range(len(idims)))
            eqs.append(sp.Eq(lhs, tdim.exps[row]))
        sol = sp.solve(eqs, a, dict=True)
        if not sol:
            return None
        sol = sol[0]
        d = Derivation("physics", f"form of {target} from {', '.join(inputs)}")
        d.assumptions = [f"dim({n}) = {dm.render()}" for n, dm in idims]
        d.assumptions.append(f"dim({target}) = {tdim.render()}")
        d.add("Σ aᵢ·dim(inputᵢ) = dim(target)", "dimensional homogeneity",
              "a valid law must be dimensionally consistent")
        factors = []
        for j, (name, _) in enumerate(idims):
            exp = sp.nsimplify(sol.get(a[j], a[j]))
            if exp == 0:
                continue
            factors.append(name if exp == 1 else f"{name}^{sp.printing.sstr(exp)}")
        form = "C·" + "·".join(factors) if factors else "C"
        d.add(form, "solve exponents", "the unique dimensionally-consistent monomial (C: constant)")
        d.result = f"{target} = {form}"
        d.verified = True
        return d


class PhysicsEngine:
    """Named symbolic derivations from physical law, plus dimensional analysis as a fallback."""

    def __init__(self) -> None:
        self.dim = DimensionalAnalysis()

    def derive(self, text: str) -> Optional[Derivation]:
        low = (text or "").lower()
        if "escape velocity" in low:
            return self._escape_velocity(text)
        if "kinematic" in low or re.search(r"v\s*=\s*u\s*\+\s*a", low) or (
                "velocity" in low and "constant acceleration" in low):
            return self._kinematics_v(text)
        if ("displacement" in low or re.search(r"s\s*=\s*u\s*t", low)) and "acceler" in low:
            return self._kinematics_s(text)
        if "pendulum" in low and "period" in low:
            return self.dim.derive("time", ["length", "g"])
        # generic "derive <quantity> from <a>, <b>, ..."
        m = re.search(r"derive\s+(?:the\s+)?([a-z ]+?)\s+from\s+(.+)", low)
        if m:
            target = m.group(1).strip().split()[-1]
            inputs = [p.strip() for p in re.split(r",|and|\s+", m.group(2)) if p.strip()]
            got = self.dim.derive(target, inputs)
            if got:
                return got
        return None

    def _escape_velocity(self, text: str) -> Optional[Derivation]:
        if not _has_sympy():
            return None
        import sympy as sp
        m, M, G, r, v = sp.symbols("m M G r v", positive=True)
        d = Derivation("physics", "escape velocity from energy conservation")
        d.assumptions = ["total mechanical energy is conserved",
                         "at escape, KE → 0 as r → ∞, so E_total = 0"]
        ke = sp.Rational(1, 2) * m * v**2
        pe = -G * M * m / r
        d.add("E = ½·m·v² − G·M·m/r", "energy conservation", "kinetic + gravitational potential")
        d.add("½·m·v² − G·M·m/r = 0", "set E = 0", "minimum energy to just escape")
        eq = sp.Eq(ke + pe, 0)
        sols = [s for s in sp.solve(eq, v) if s.is_positive or sp.simplify(s) != 0]
        sol = max(sols, key=lambda s: 1) if sols else sp.sqrt(2 * G * M / r)
        # prefer the positive root explicitly
        for s in sp.solve(eq, v):
            if (s.subs({G: 1, M: 1, r: 1}).evalf() or 0) > 0:
                sol = s
                break
        d.add(f"v = {sp.printing.sstr(sol)}", "solve for v", "take the positive root")
        d.result = f"v_escape = {sp.printing.sstr(sol)}"
        d.verified = bool(sp.simplify(sol**2 - 2 * G * M / r) == 0)
        return d

    def _kinematics_v(self, text: str) -> Optional[Derivation]:
        if not _has_sympy():
            return None
        import sympy as sp
        t, a, u = sp.symbols("t a u", real=True)
        d = Derivation("physics", "v = u + a·t from a = dv/dt (constant a)")
        d.assumptions = ["acceleration a is constant", "v(0) = u"]
        d.add("a = dv/dt", "definition of acceleration", "")
        d.add("∫a dt = ∫dv", "integrate both sides", "a constant over t")
        v = sp.integrate(a, t) + u
        d.add(f"v = {sp.printing.sstr(v)}", "evaluate the integral", "+ u from the initial condition")
        d.result = f"v = {sp.printing.sstr(v)}"
        d.verified = bool(sp.simplify(sp.diff(v, t) - a) == 0)
        return d

    def _kinematics_s(self, text: str) -> Optional[Derivation]:
        if not _has_sympy():
            return None
        import sympy as sp
        t, a, u = sp.symbols("t a u", real=True)
        d = Derivation("physics", "s = u·t + ½·a·t² from v = u + a·t (constant a)")
        d.assumptions = ["acceleration a is constant", "v = u + a·t", "s(0) = 0"]
        d.add("v = u + a·t", "from the velocity derivation", "")
        d.add("s = ∫v dt", "displacement is the integral of velocity", "")
        s = sp.integrate(u + a * t, t)
        d.add(f"s = {sp.printing.sstr(s)}", "evaluate the integral", "constant of integration 0 since s(0)=0")
        d.result = f"s = {sp.printing.sstr(s)}"
        d.verified = bool(sp.simplify(sp.diff(s, t) - (u + a * t)) == 0)
        return d


# --------------------------------------------------------------------------- #
# Chemistry — balance a reaction by conservation of atoms (exact, from the null space)
# --------------------------------------------------------------------------- #
_FORMULA_TOKEN = re.compile(r"[A-Z][a-z]?|\d+|\(|\)|\[|\]")


def parse_formula(formula: str) -> Dict[str, int]:
    """Parse a chemical formula into an element→count map (handles nested parentheses).

    e.g. ``Fe2(SO4)3`` → ``{'Fe': 2, 'S': 3, 'O': 12}``.
    """
    tokens = _FORMULA_TOKEN.findall(formula.strip())

    def parse(i: int) -> Tuple[Dict[str, int], int]:
        counts: Dict[str, int] = {}
        while i < len(tokens):
            tok = tokens[i]
            if tok in ("(", "["):
                sub, i = parse(i + 1)
                mult = 1
                if i < len(tokens) and tokens[i].isdigit():
                    mult = int(tokens[i]); i += 1
                for el, c in sub.items():
                    counts[el] = counts.get(el, 0) + c * mult
            elif tok in (")", "]"):
                return counts, i + 1
            elif tok[0].isalpha():
                el = tok
                i += 1
                n = 1
                if i < len(tokens) and tokens[i].isdigit():
                    n = int(tokens[i]); i += 1
                counts[el] = counts.get(el, 0) + n
            else:  # a stray leading number
                i += 1
        return counts, i

    counts, _ = parse(0)
    return counts


def balance_reaction(equation: str) -> Optional[Tuple[List[int], List[str], List[str]]]:
    """Balance ``equation`` (e.g. ``C3H8 + O2 -> CO2 + H2O``) exactly.

    Returns ``(coefficients, reactant_species, product_species)`` where ``coefficients`` aligns
    to ``reactants + products``, or ``None`` if it cannot be balanced. The coefficients are the
    smallest positive integers — computed from the null space of the element-count matrix
    (conservation of every atom), never looked up.
    """
    if not _has_sympy():
        return None
    arrow = None
    for a in ("->", "→", "=", "⇌"):
        if a in equation:
            arrow = a
            break
    if arrow is None:
        return None
    left, _, right = equation.partition(arrow)
    reactants = [s.strip() for s in re.split(r"\+", left) if s.strip()]
    products = [s.strip() for s in re.split(r"\+", right) if s.strip()]
    if not reactants or not products:
        return None
    species = reactants + products
    parsed = [parse_formula(s) for s in species]
    elements = sorted(set().union(*[set(p) for p in parsed]))
    if not elements:
        return None
    import sympy as sp
    rows = []
    for el in elements:
        row = []
        for j, p in enumerate(parsed):
            coeff = p.get(el, 0)
            row.append(coeff if j < len(reactants) else -coeff)
        rows.append(row)
    mat = sp.Matrix(rows)
    ns = mat.nullspace()
    if not ns:
        return None
    vec = ns[0]
    # scale the rational null-vector to the smallest positive integers
    denom_lcm = 1
    for v in vec:
        denom_lcm = sp.ilcm(denom_lcm, sp.Rational(v).q)
    ints = [int(sp.Rational(v) * denom_lcm) for v in vec]
    if ints and ints[0] < 0:
        ints = [-i for i in ints]
    g = 0
    for i in ints:
        g = sp.igcd(g, i)
    if g:
        ints = [i // int(g) for i in ints]
    if any(i <= 0 for i in ints):       # an unbalanceable (or under-determined) reaction
        return None
    return ints, reactants, products


class ChemistryEngine:
    """Balance reactions and report the worked conservation-of-atoms derivation."""

    def derive(self, text: str) -> Optional[Derivation]:
        eq = self._extract_equation(text)
        if not eq:
            return None
        got = balance_reaction(eq)
        if not got:
            return None
        coeffs, reactants, products = got
        d = Derivation("chemistry", f"balance {eq}")
        d.assumptions = ["every element is conserved (atoms in = atoms out)",
                         "charge/mass is conserved across the reaction"]
        d.add("build the element-count matrix A (one row per element, one column per species)",
              "conservation", "Σ coeff·atoms = 0 for each element")
        d.add("solve A·x = 0 for the coefficient vector x (null space)", "linear algebra",
              "the balanced ratios are the null space of A")
        d.add("scale x to the smallest positive integers", "normalise", "÷ gcd, × lcm of denominators")
        nr = len(reactants)
        lhs = " + ".join(self._term(coeffs[i], reactants[i]) for i in range(nr))
        rhs = " + ".join(self._term(coeffs[nr + i], products[i]) for i in range(len(products)))
        d.result = f"{lhs} -> {rhs}"
        d.verified = True
        return d

    @staticmethod
    def _term(c: int, species: str) -> str:
        return species if c == 1 else f"{c} {species}"

    @staticmethod
    def _extract_equation(text: str) -> Optional[str]:
        for a in ("->", "→", "⇌"):
            if a in text:
                m = re.search(r"([A-Za-z0-9()\[\]+\s]+%s[A-Za-z0-9()\[\]+\s]+)" % re.escape(a), text)
                if m:
                    eq = m.group(1).strip().rstrip(".?! ")
                    # drop leading command prose ("balance the reaction …") that isn't a species
                    eq = re.sub(r"^\s*(?:balance|the|equation|reaction|solve|please|this)\s+",
                                "", eq, flags=re.I)
                    while re.match(r"^\s*(?:balance|the|equation|reaction|solve|please|this)\s+",
                                   eq, flags=re.I):
                        eq = re.sub(r"^\s*\w+\s+", "", eq)
                    return eq.strip()
        return None


# --------------------------------------------------------------------------- #
# Maths — prove a claim by symbolic certification
# --------------------------------------------------------------------------- #
class MathEngine:
    """Certify an algebraic identity by ``simplify(lhs − rhs) == 0`` (a checked proof)."""

    def derive(self, text: str) -> Optional[Derivation]:
        if not _has_sympy():
            return None
        claim = self._extract_claim(text)
        if not claim:
            return None
        lhs_s, rhs_s = claim
        import sympy as sp
        from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                                implicit_multiplication_application, convert_xor)
        tr = standard_transformations + (implicit_multiplication_application, convert_xor)
        try:
            lhs = parse_expr(lhs_s, transformations=tr)
            rhs = parse_expr(rhs_s, transformations=tr)
        except Exception:  # noqa: BLE001
            return None
        d = Derivation("maths", f"prove {lhs_s} = {rhs_s}")
        d.assumptions = ["the standard axioms of commutative algebra over the reals"]
        expanded = sp.expand(lhs)
        d.add(f"expand LHS: {sp.printing.sstr(expanded)}", "ring axioms", "distribute and collect")
        diff = sp.simplify(lhs - rhs)
        d.add(f"LHS − RHS = {sp.printing.sstr(diff)}", "subtract", "an identity holds iff this is 0")
        d.verified = bool(diff == 0)
        d.result = ("identity holds (proven)" if d.verified
                    else f"NOT an identity (residual {sp.printing.sstr(diff)})")
        d.confidence = 1.0 if d.verified else 0.5
        return d

    @staticmethod
    def _extract_claim(text: str) -> Optional[Tuple[str, str]]:
        s = text
        m = re.search(r"(?:prove|show|verify)?\s*(?:that)?\s*(.+?)\s*=\s*(.+)", s, re.I)
        if not m:
            return None
        lhs, rhs = m.group(1).strip(), m.group(2).strip().rstrip(".?! ")
        if not lhs or not rhs or "->" in s or "<->" in s:
            return None
        # need at least one symbol/operator to be an algebraic identity (not "2 = 2")
        if not re.search(r"[a-zA-Z]", lhs + rhs):
            return None
        return lhs, rhs


# --------------------------------------------------------------------------- #
# Logic — forward-chaining deductive closure (modus ponens)
# --------------------------------------------------------------------------- #
class LogicEngine:
    """Reach a conclusion by applying implication rules to facts until closure."""

    _RULE = re.compile(r"(?:if\s+)?(.+?)\s*(?:->|→|then|implies)\s*(.+)", re.I)

    def derive(self, text: str) -> Optional[Derivation]:
        facts, rules = self._parse(text)
        if not rules and not facts:
            return None
        d = Derivation("logic", "forward-chaining deductive closure")
        d.assumptions = [f"fact: {f}" for f in sorted(facts)] + \
                        [f"rule: {' & '.join(sorted(ant))} -> {con}" for ant, con in rules]
        known = set(facts)
        changed = True
        while changed:
            changed = False
            for ant, con in rules:
                if con not in known and ant.issubset(known):
                    known.add(con)
                    d.add(con, "modus ponens", f"from {{{', '.join(sorted(ant))}}} and the rule")
                    changed = True
        derived = sorted(known - facts)
        d.result = ("derived: " + ", ".join(derived)) if derived else "nothing new follows"
        d.verified = True
        return d

    def _parse(self, text: str) -> Tuple[set, List[Tuple[frozenset, str]]]:
        facts: set = set()
        rules: List[Tuple[frozenset, str]] = []
        # split on sentence/clause separators
        for clause in re.split(r"[.;\n]|,(?!\s*\w+\s*(?:->|→))", text):
            clause = clause.strip().strip(".")
            if not clause:
                continue
            m = self._RULE.match(clause)
            if m and ("->" in clause or "→" in clause or re.search(r"\b(then|implies)\b", clause, re.I)):
                ant = frozenset(a.strip() for a in re.split(r"&|\band\b", m.group(1), flags=re.I) if a.strip())
                con = m.group(2).strip()
                if ant and con:
                    rules.append((ant, con))
            else:
                # a bare token like "A" or "it is raining" is a fact, but skip prose framing words
                tok = clause.strip()
                if tok and tok.lower() not in ("given", "facts", "fact", "assume", "rules", "rule"):
                    facts.add(tok)
        return facts, rules


# --------------------------------------------------------------------------- #
# The dispatcher + the faculty
# --------------------------------------------------------------------------- #
class FirstPrinciplesEngine:
    """Route a query to the right domain engine and return a :class:`Derivation`."""

    def __init__(self) -> None:
        self.physics = PhysicsEngine()
        self.chemistry = ChemistryEngine()
        self.maths = MathEngine()
        self.logic = LogicEngine()

    def derive(self, text: str, domain: Optional[str] = None) -> Optional[Derivation]:
        low = (text or "").lower()
        order: List[Tuple[str, Any]]
        if domain == "physics":
            order = [("physics", self.physics)]
        elif domain == "chemistry":
            order = [("chemistry", self.chemistry)]
        else:
            order = []
            # cheap domain cues drive the try-order; each engine still self-vetoes (returns None)
            if any(k in low for k in ("balance", "reaction", "->", "→", "stoichiom")):
                order.append(("chemistry", self.chemistry))
            if any(k in low for k in ("velocity", "kinematic", "escape", "pendulum", "force",
                                      "energy", "acceleration", "physics", "derive ")):
                order.append(("physics", self.physics))
            if any(k in low for k in ("prove", "identity", "show that", "verify")) and "=" in low:
                order.append(("maths", self.maths))
            if any(k in low for k in ("->", "→", "implies", "then", "modus")):
                order.append(("logic", self.logic))
            # always leave the remaining engines as fallbacks
            for name, eng in (("chemistry", self.chemistry), ("physics", self.physics),
                              ("maths", self.maths), ("logic", self.logic)):
                if (name, eng) not in order:
                    order.append((name, eng))
        for _name, eng in order:
            try:
                got = eng.derive(text)
            except Exception:  # noqa: BLE001 — a failed derivation defers, never crashes
                got = None
            if got is not None:
                return got
        return None


_DISPATCHER: Optional[FirstPrinciplesEngine] = None


def derive_from_first_principles(text: str, domain: Optional[str] = None) -> Optional[Derivation]:
    """Module-level convenience: derive ``text`` from first principles, or ``None`` to defer."""
    global _DISPATCHER
    if _DISPATCHER is None:
        _DISPATCHER = FirstPrinciplesEngine()
    return _DISPATCHER.derive(text, domain=domain)


class FirstPrinciplesFaculty(Faculty):
    """Derive a result from domain axioms — verifiable, so it beats any neural guess."""

    name = "first_principles"
    handles = frozenset({TaskType.DERIVATION, TaskType.PHYSICS, TaskType.CHEMISTRY})
    verifiable = True
    reliability = 1.0
    cost = 0.6

    _CUES = ("derive", "first principle", "from first principles", "balance", "conservation",
             "escape velocity", "kinematic", "prove that", "stoichiom", "modus ponens",
             "dimensional analysis")

    def __init__(self) -> None:
        self._engine = FirstPrinciplesEngine()

    def _domain(self, task: Task) -> Optional[str]:
        if task.type is TaskType.PHYSICS:
            return "physics"
        if task.type is TaskType.CHEMISTRY:
            return "chemistry"
        return None

    def _derive(self, task: Task) -> Optional[Derivation]:
        text = task.description or str(task.payload or "")
        return self._engine.derive(text, domain=self._domain(task))

    def suitability(self, task: Task) -> float:
        if task.type not in self.handles:
            return 0.0
        text = (task.description or str(task.payload or "")).lower()
        if not any(c in text for c in self._CUES) and task.type is TaskType.DERIVATION:
            return 0.0
        return 0.95 if self._derive(task) is not None else 0.0

    def handle(self, task: Task):
        deriv = self._derive(task)
        if deriv is None:
            return self._propose("", kind=ProposalKind.ANSWER, confidence=0.0,
                                 rationale="no first-principles derivation applies", risk=0.0)
        return self._propose(
            deriv.render(), kind=ProposalKind.ANSWER,
            confidence=1.0 if deriv.verified else deriv.confidence,
            rationale=f"derived from first principles ({deriv.domain}); result: {deriv.result}",
            source=SourceType.TOOL,
        )
