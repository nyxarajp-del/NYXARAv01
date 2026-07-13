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
    "SymbolicEngine",
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
# Widened well past the original handful so dimensional derivation reaches the mechanics,
# thermodynamics and electromagnetism a real first-principles engine is expected to know. New
# quantities can also be *defined at runtime* from a stated equation (see ``DimensionalAnalysis``),
# so the table is a starting point, not a ceiling.
_QUANTITY_DIMS: Dict[str, Dimension] = {
    # --- kinematics / mechanics ---
    "mass": Dimension.base(M=1), "length": Dimension.base(L=1),
    "distance": Dimension.base(L=1), "radius": Dimension.base(L=1),
    "displacement": Dimension.base(L=1), "position": Dimension.base(L=1),
    "height": Dimension.base(L=1), "width": Dimension.base(L=1),
    "wavelength": Dimension.base(L=1), "time": Dimension.base(T=1),
    "period": Dimension.base(T=1), "velocity": Dimension.base(L=1, T=-1),
    "speed": Dimension.base(L=1, T=-1), "acceleration": Dimension.base(L=1, T=-2),
    "gravity": Dimension.base(L=1, T=-2), "jerk": Dimension.base(L=1, T=-3),
    "force": Dimension.base(M=1, L=1, T=-2), "weight": Dimension.base(M=1, L=1, T=-2),
    "momentum": Dimension.base(M=1, L=1, T=-1), "impulse": Dimension.base(M=1, L=1, T=-1),
    "energy": Dimension.base(M=1, L=2, T=-2), "work": Dimension.base(M=1, L=2, T=-2),
    "torque": Dimension.base(M=1, L=2, T=-2), "power": Dimension.base(M=1, L=2, T=-3),
    "pressure": Dimension.base(M=1, L=-1, T=-2), "stress": Dimension.base(M=1, L=-1, T=-2),
    "density": Dimension.base(M=1, L=-3), "area": Dimension.base(L=2),
    "volume": Dimension.base(L=3), "frequency": Dimension.base(T=-1),
    "angular_velocity": Dimension.base(T=-1), "angular_frequency": Dimension.base(T=-1),
    "angular_acceleration": Dimension.base(T=-2),
    "angular_momentum": Dimension.base(M=1, L=2, T=-1),
    "moment_of_inertia": Dimension.base(M=1, L=2),
    "action": Dimension.base(M=1, L=2, T=-1),
    "viscosity": Dimension.base(M=1, L=-1, T=-1),
    "surface_tension": Dimension.base(M=1, T=-2),
    "flow_rate": Dimension.base(L=3, T=-1),
    # --- electromagnetism ---
    "charge": Dimension.base(I=1, T=1), "current": Dimension.base(I=1),
    "voltage": Dimension.base(M=1, L=2, T=-3, I=-1),
    "potential": Dimension.base(M=1, L=2, T=-3, I=-1),
    "resistance": Dimension.base(M=1, L=2, T=-3, I=-2),
    "conductance": Dimension.base(M=-1, L=-2, T=3, I=2),
    "capacitance": Dimension.base(M=-1, L=-2, T=4, I=2),
    "inductance": Dimension.base(M=1, L=2, T=-2, I=-2),
    "magnetic_flux": Dimension.base(M=1, L=2, T=-2, I=-1),
    "magnetic_field": Dimension.base(M=1, T=-2, I=-1),
    "electric_field": Dimension.base(M=1, L=1, T=-3, I=-1),
    # --- thermodynamics ---
    "temperature": Dimension.base(**{"Θ": 1}),
    "heat": Dimension.base(M=1, L=2, T=-2),
    "entropy": Dimension.base(**{"M": 1, "L": 2, "T": -2, "Θ": -1}),
    "heat_capacity": Dimension.base(**{"M": 1, "L": 2, "T": -2, "Θ": -1}),
    # --- chemistry / amount / light ---
    "amount": Dimension.base(N=1), "concentration": Dimension.base(N=1, L=-3),
    "luminous_intensity": Dimension.base(J=1),
    # --- constants ---
    "g": Dimension.base(L=1, T=-2),
    "gravitational_constant": Dimension.base(M=-1, L=3, T=-2),   # G
    "spring_constant": Dimension.base(M=1, T=-2),                # k (N/m)
    "planck_constant": Dimension.base(M=1, L=2, T=-1),           # h
    "boltzmann_constant": Dimension.base(**{"M": 1, "L": 2, "T": -2, "Θ": -1}),
    "speed_of_light": Dimension.base(L=1, T=-1),
    "elementary_charge": Dimension.base(I=1, T=1),
    "coulomb_constant": Dimension.base(M=1, L=3, T=-4, I=-2),
}

# Common synonyms → canonical quantity name, so free-text derivations don't miss on wording.
_QUANTITY_ALIASES: Dict[str, str] = {
    "accel": "acceleration", "vel": "velocity", "temp": "temperature",
    "grav": "gravity", "freq": "frequency", "wavelength_lambda": "wavelength",
    "spring": "spring_constant", "voltage_v": "voltage", "emf": "voltage",
    "resistor": "resistance", "capacitor": "capacitance", "inductor": "inductance",
    "planck": "planck_constant", "boltzmann": "boltzmann_constant",
    "light_speed": "speed_of_light", "c": "speed_of_light",
}


def _canonical_quantity(name: str) -> str:
    key = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    return _QUANTITY_ALIASES.get(key, key)


class DimensionalAnalysis:
    """Derive the *form* of an unknown quantity from the dimensions of its inputs.

    Solves ``Σ aᵢ·dim(inputᵢ) = dim(target)`` for the exponents ``aᵢ`` over the seven base
    dimensions — i.e. recovers the only dimensionally-consistent monomial (up to a constant).
    This is how a never-before-seen formula is reached from rules alone: the pendulum period
    must go as √(L/g) because nothing else is dimensionally possible.

    The quantity table is *extensible*: ``define`` registers a new quantity's dimension from a
    stated defining equation (e.g. ``power = energy / time``), so a derivation is not confined to
    the built-in library — a genuinely new quantity can be introduced in the same turn and then
    reasoned about dimensionally.
    """

    def __init__(self, extra: Optional[Dict[str, "Dimension"]] = None) -> None:
        # instance overlay so a quantity defined for one query never mutates the shared table
        self._extra: Dict[str, Dimension] = dict(extra or {})

    def dim_of(self, name: str) -> Optional["Dimension"]:
        """Look up a quantity's dimension by canonical name (built-in or runtime-defined)."""
        key = _canonical_quantity(name)
        return self._extra.get(key) or _QUANTITY_DIMS.get(key)

    def define(self, name: str, expression: str) -> Optional["Dimension"]:
        """Register a new quantity from an expression over known quantities (``a*b/c``).

        Returns its dimension (also stored on this instance), or ``None`` when a factor's
        dimension is unknown — an honest decline rather than a guessed unit."""
        dim = self._dim_of_expression(expression)
        if dim is None:
            return None
        self._extra[_canonical_quantity(name)] = dim
        return dim

    def _dim_of_expression(self, expression: str) -> Optional["Dimension"]:
        """Dimension of a product/quotient/power expression over named quantities, or None."""
        if not _has_sympy():
            return None
        import sympy as sp
        try:
            expr = sp.sympify(expression, evaluate=False)
        except Exception:  # noqa: BLE001
            return None
        one = Dimension(tuple([0.0] * 7))

        def walk(node: Any) -> Optional[Dimension]:
            if node.is_Symbol:
                return self.dim_of(str(node))
            if node.is_Number:
                return one  # a bare constant is dimensionless
            if node.is_Add:
                dims = [walk(a) for a in node.args]
                if any(d is None for d in dims):
                    return None
                first = dims[0]
                return first if all(d.exps == first.exps for d in dims) else None
            if node.is_Mul:
                acc = one
                for a in node.args:
                    d = walk(a)
                    if d is None:
                        return None
                    acc = acc * d
                return acc
            if node.is_Pow:
                base, exp = node.args
                if not exp.is_Number:
                    return None
                d = walk(base)
                return d.pow(float(exp)) if d is not None else None
            return None

        return walk(expr)

    def derive(self, target: str, inputs: List[str]) -> Optional[Derivation]:
        tdim = self.dim_of(target)
        idims = []
        for name in inputs:
            d = self.dim_of(name)
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
        # register any new quantity introduced by a "where/with <q> = <expr>" defining clause,
        # so a quantity NYXARA has never seen can still be reasoned about dimensionally.
        self._register_defined_quantities(text)
        # generic "derive <quantity> from <a>, <b>, ..."
        m = re.search(r"derive\s+(?:the\s+)?([a-z ]+?)\s+from\s+(.+)", low)
        if m:
            target = m.group(1).strip().split()[-1]
            tail = re.split(r"\s+where\b|\s+with\b", m.group(2), maxsplit=1, flags=re.I)[0]
            inputs = [p.strip() for p in re.split(r",|and|\s+", tail) if p.strip()]
            got = self.dim.derive(target, inputs)
            if got:
                return got
        return None

    def _register_defined_quantities(self, text: str) -> None:
        """Parse ``where <q> = <expr over known quantities>`` clauses and register each new
        quantity's dimension on the analysis instance (best-effort, never fatal)."""
        for m in re.finditer(r"(?:where|with|define|let)\s+([A-Za-z][\w ]*?)\s*(?:=|is|as)\s*"
                             r"([A-Za-z0-9_.*/^()\s+-]+)", text, flags=re.I):
            name = m.group(1).strip().split()[-1]
            expr = m.group(2).strip().rstrip(".?! ")
            # translate an English "per" / "over" into division so "energy per time" reads as a law
            expr = re.sub(r"\s+(?:per|over)\s+", "/", expr, flags=re.I)
            try:
                self.dim.define(name, expr)
            except Exception:  # noqa: BLE001 — a definition that won't parse is simply skipped
                pass

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
        # require an explicit prove/show/verify framing — otherwise a bare "X = Y" is a
        # solve/rearrange request (SymbolicEngine's job), not an identity to certify.
        if not re.search(r"\b(prove|show|verify|identity|is it true)\b", s, re.I):
            return None
        m = re.search(r"(?:prove|show|verify)\s*(?:that)?\s*(.+?)\s*=\s*(.+)", s, re.I)
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
# Symbolic — general algebra & calculus: rearrange any stated law, solve any
# equation/system for any variable, and integrate a rate law (ODE). This is the
# reach that turns "only the four hand-written derivations" into a general engine.
# --------------------------------------------------------------------------- #
def _sympy_transforms():
    from sympy.parsing.sympy_parser import (standard_transformations,
                                            implicit_multiplication_application, convert_xor)
    return standard_transformations + (implicit_multiplication_application, convert_xor)


def _parse_expr(s: str):
    """Parse a mathematical expression with implicit multiplication (``m a`` = ``m*a``)."""
    from sympy.parsing.sympy_parser import parse_expr
    return parse_expr(s, transformations=_sympy_transforms())


class SymbolicEngine:
    """Rearrange a stated law for any variable, solve equations/systems, and integrate a
    rate law — all symbolic and self-verified (the solution is substituted back and checked).

    Where :class:`MathEngine` *proves an identity you assert*, this *derives a result you don't
    yet have*: given ``F = m*a`` and asked for ``a`` it returns ``a = F/m`` (verified), and given
    ``dv/dt = a`` (a constant) it integrates to ``v = a*t + C``. It fires only when it can recover
    a real equation and a target variable; otherwise it defers (returns ``None``)."""

    _DERIV = re.compile(r"d\s*([A-Za-z][A-Za-z0-9_]*)\s*/\s*d\s*([A-Za-z][A-Za-z0-9_]*)")

    def derive(self, text: str) -> Optional[Derivation]:
        if not _has_sympy():
            return None
        # an explicit chemistry arrow / identity-proof framing is another engine's job
        if any(a in text for a in ("->", "→", "⇌")):
            return None
        ode = self._as_ode(text)
        if ode is not None:
            return ode
        return self._solve_for(text)

    # ---- rate law → integrate (a general ODE path; subsumes the kinematics specials) ---- #
    def _as_ode(self, text: str) -> Optional[Derivation]:
        m = self._DERIV.search(text)
        if not m:
            return None
        import sympy as sp
        yname, xname = m.group(1), m.group(2)
        eqs = self._equations(text)
        if not eqs:
            return None
        # find the equation that states the derivative
        deriv_str = m.group(0)
        target_eq = None
        for lhs_s, rhs_s in eqs:
            if self._DERIV.search(lhs_s) or self._DERIV.search(rhs_s):
                target_eq = (lhs_s, rhs_s)
                break
        if target_eq is None:
            return None
        x = sp.Symbol(xname)
        y = sp.Function(yname)
        try:
            # build the ODE from the non-derivative side of the stated rate equation
            rhs_side = target_eq[1] if self._DERIV.search(target_eq[0]) else target_eq[0]
            rhs = _parse_expr(rhs_side).subs(sp.Symbol(yname), y(x))
            ode = sp.Eq(sp.Derivative(y(x), x), rhs)
            sol = sp.dsolve(ode, y(x))
        except Exception:  # noqa: BLE001
            return None
        d = Derivation("physics", f"integrate {deriv_str} = {rhs_side.strip()}")
        d.assumptions = [f"{deriv_str} = {rhs_side.strip()}",
                         f"integrate over {xname} (C is the constant of integration)"]
        d.add(f"d{yname}/d{xname} = {rhs_side.strip()}", "rate law", "the stated differential relation")
        d.add(f"∫ d{yname} = ∫ ({rhs_side.strip()}) d{xname}", "integrate both sides", "")
        expr = sol.rhs if isinstance(sol, sp.Equality) else sol
        d.add(f"{yname} = {sp.printing.sstr(expr)}", "evaluate the integral", "+ C: initial condition")
        d.result = f"{yname} = {sp.printing.sstr(expr)}"
        # verify: differentiating the solution reproduces the stated rate law
        try:
            check = sp.simplify(sp.diff(expr, x) - rhs.subs(y(x), expr))
            d.verified = bool(check == 0)
        except Exception:  # noqa: BLE001
            d.verified = False
        return d

    # ---- solve a law / system for a target variable ---- #
    _FOR = (r"\bfor\s+(?:the\s+)?([A-Za-z]\w*)", r"\bfind\s+(?:the\s+)?([A-Za-z]\w*)",
            r"\bderive\s+(?:the\s+)?([A-Za-z]\w*)", r"\bexpress\s+(?:the\s+)?([A-Za-z]\w*)",
            r"\bsolve\s+(?:for\s+)?(?:the\s+)?([A-Za-z]\w*)")

    def _target_var(self, text: str, symbols: set):
        import sympy as sp
        for pat in self._FOR:
            m = re.search(pat, text, re.I)
            if m and sp.Symbol(m.group(1)) in symbols:
                return sp.Symbol(m.group(1))
        return None

    def _solve_for(self, text: str) -> Optional[Derivation]:
        import sympy as sp
        eqs_s = self._equations(text)
        if not eqs_s:
            return None
        try:
            eqs = [sp.Eq(_parse_expr(l), _parse_expr(r)) for l, r in eqs_s]
        except Exception:  # noqa: BLE001
            return None
        symbols: set = set()
        for eq in eqs:
            symbols |= eq.free_symbols
        if not symbols:
            return None
        # a single equation in a single unknown ("solve 2x+3=9") is the dedicated AlgebraFaculty's
        # job — defer to it. This engine's unique reach is *symbolic rearrangement* of a law with
        # several symbols, and systems; keep it there so we don't shadow a cleaner numeric answer.
        if len(eqs) == 1 and len(symbols) == 1:
            return None
        target = self._target_var(text, symbols)
        if target is None:
            if len(symbols) == 1:               # "solve x**2 - 4 = 0" → the lone symbol
                target = next(iter(symbols))
            else:
                return None
        # to eliminate intermediate quantities in a system, solve for the target *plus* every
        # variable that appears alone as an equation's left side (the "defined" ones).
        defined = [eq.lhs for eq in eqs if eq.lhs.is_Symbol and eq.lhs != target]
        solve_for = [target] + [s for s in dict.fromkeys(defined) if s in symbols]
        try:
            sols = sp.solve(eqs, solve_for, dict=True)
        except Exception:  # noqa: BLE001
            return None
        if not sols:
            return None
        sol = sols[0]
        sol_val = sol.get(target)
        if sol_val is None:
            return None
        law_str = "; ".join(f"{l.strip()} = {r.strip()}" for l, r in eqs_s)
        d = Derivation("maths", f"solve {law_str} for {target}")
        d.assumptions = [f"given: {law_str}"]
        d.add(f"treat the relation(s) as equations in "
              f"{', '.join(str(s) for s in sorted(symbols, key=str))}",
              "algebra", "isolate the target by valid operations")
        d.add(f"{target} = {sp.printing.sstr(sol_val)}", "solve", f"rearranged for {target}")
        d.result = f"{target} = {sp.printing.sstr(sol_val)}"
        # verify: substitute the whole solution back into every equation → each residual is 0
        try:
            ok = all(sp.simplify(eq.lhs.subs(sol) - eq.rhs.subs(sol)) == 0 for eq in eqs)
            d.verified = bool(ok)
        except Exception:  # noqa: BLE001
            d.verified = False
        d.confidence = 1.0 if d.verified else 0.5
        return d

    @staticmethod
    def _equations(text: str) -> List[Tuple[str, str]]:
        """Recover ``lhs = rhs`` equations from free text (skips ``==``/``<=``/``>=``/``!=``)."""
        eqs: List[Tuple[str, str]] = []
        # split on connectives that separate multiple stated equations
        for clause in re.split(r"[,;\n]|\band\b|\bwhere\b|\bgiven\b", text, flags=re.I):
            clause = clause.strip()
            # a single '=' that is not part of a comparison operator
            m = re.search(r"(?<![<>=!])=(?![=])", clause)
            if not m:
                continue
            lhs, rhs = clause[:m.start()], clause[m.start() + 1:]
            lhs = re.sub(r"^\s*(?:solve|find|derive|express|compute|the|prove|show|that|"
                         r"rearrange|for|is|what|given)\s+", "", lhs, flags=re.I).strip()
            rhs = rhs.strip().rstrip(".?!")
            # cut a trailing natural-language tail ("… for constant acceleration") so prose words
            # never become spurious symbols — keep only the mathematical head of each side.
            rhs = re.split(r"\s+(?:for|where|when|with|to|that|so|which|assuming|if|under)\b",
                           rhs, maxsplit=1, flags=re.I)[0].strip()
            lhs = re.split(r"\s+(?:for|where|when|with|to|that|so|which|assuming|if|under)\b",
                           lhs, maxsplit=1, flags=re.I)[0].strip()
            # keep only clauses that look like real math (a letter and an operator or two terms)
            if lhs and rhs and re.search(r"[A-Za-z]", lhs + rhs) and not re.search(r"[<>]", clause):
                eqs.append((lhs, rhs))
        return eqs


# --------------------------------------------------------------------------- #
# Logic — forward-chaining deductive closure (modus ponens)
# --------------------------------------------------------------------------- #
def _norm_pred(phrase: str) -> str:
    """Fold a predicate phrase to a stable atom: lowercase, single word, singularised.

    ``"humans"`` → ``"human"``, ``"can fly"`` → ``"can_fly"`` — so *all humans are mortal* and
    *socrates is a human* share the atom ``human`` and the rule can fire."""
    w = (phrase or "").strip().lower().rstrip(".?!").strip()
    w = re.sub(r"^(?:a|an|the)\s+", "", w)
    w = re.sub(r"\s+", "_", w)
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w


def _pred_phrase(phrase: str) -> str:
    """Normalise a predicate that may lead with a copula: ``is a metal`` → ``metal``,
    ``conducts`` → ``conduct`` — so the antecedent/consequent of an ``if … then …`` rule and a
    plain ``X is a metal`` fact fold to the same atom."""
    p = (phrase or "").strip()
    p = re.sub(r"^(?:is|are)\s+(?:a\s+|an\s+)?", "", p, flags=re.I)
    return _norm_pred(p)


class LogicEngine:
    """Reach a conclusion by deductive closure.

    Two tiers, both exact and no LLM:

    * **First-order** — universally-quantified rules with a variable
      (*"all humans are mortal", "if X is a bird then X can fly"*) applied to unary facts
      (*"socrates is a human"*) to derive new facts (*"socrates is mortal"*) — the classic
      syllogism the plain propositional chainer cannot reach because it treats each sentence as
      an opaque atom.
    * **Propositional** — the original modus-ponens forward chaining over atomic propositions.

    First-order is tried first; if it recovers a genuine quantified rule and projects a new fact,
    that is the derivation, otherwise it falls back to the propositional closure."""

    _RULE = re.compile(r"(?:if\s+)?(.+?)\s*(?:->|→|then|implies)\s*(.+)", re.I)
    # "all/every/each <P> …" → ∀x. P(x) → Q(x); the copula/verb-phrase split is done in code.
    _ALL = re.compile(r"\b(?:all|every|each|any)\s+(.+)", re.I)
    # "if X <P> then X <Q>" (X a variable placeholder) → ∀x. P(x) → Q(x). Each side may be a
    # copula ("is a metal") or a verb phrase ("conducts"); the copula is stripped in code.
    _IFVAR = re.compile(r"if\s+(\w+)\s+(.+?)\s+then\s+\1\s+(.+)", re.I)
    # "<entity> is [a/an] <predicate>"  →  pred(entity)
    _FACT = re.compile(r"^(.+?)\s+is\s+(?:a\s+|an\s+)?(.+)$", re.I)

    def derive(self, text: str) -> Optional[Derivation]:
        fo = self._first_order(text)
        if fo is not None:
            return fo
        return self._propositional(text)

    @staticmethod
    def _split_rule(remainder: str) -> Optional[Tuple[str, str]]:
        """Split a universally-quantified body into (antecedent, consequent) predicates.

        ``humans are mortal`` → ``(human, mortal)`` (copula); ``birds can fly`` →
        ``(bird, can_fly)`` (verb phrase: first noun is the class, the rest the predicate)."""
        m = re.match(r"(.+?)\s+(?:are|is)\s+(?:a\s+|an\s+)?(.+)", remainder, re.I)
        if m:
            return _norm_pred(m.group(1)), _norm_pred(m.group(2))
        parts = remainder.split(None, 1)
        if len(parts) == 2:
            return _norm_pred(parts[0]), _norm_pred(parts[1])
        return None

    # ---- first-order (universally-quantified) chaining ---- #
    def _first_order(self, text: str) -> Optional[Derivation]:
        rules: List[Tuple[str, str]] = []   # (antecedent predicate, consequent predicate)
        facts: Dict[str, set] = {}          # entity -> set of predicates it satisfies
        clauses = re.split(r"[.;\n]", text)
        for clause in clauses:
            clause = clause.strip().rstrip(".")
            if not clause:
                continue
            mv = self._IFVAR.match(clause)
            if mv:
                rules.append((_pred_phrase(mv.group(2)), _pred_phrase(mv.group(3))))
                continue
            ma = self._ALL.match(clause)
            if ma:
                split = self._split_rule(ma.group(1).strip())
                if split:
                    rules.append(split)
                continue
            mf = self._FACT.match(clause)
            if mf and not re.search(r"\b(all|every|each|any|if|then)\b", clause, re.I):
                ent = mf.group(1).strip().lower()
                facts.setdefault(ent, set()).add(_norm_pred(mf.group(2)))
        if not rules or not facts:
            return None
        d = Derivation("logic", "first-order deductive closure (universal instantiation)")
        d.assumptions = [f"∀x. {a}(x) → {c}(x)" for a, c in rules] + \
                        [f"{p}({e})" for e in sorted(facts) for p in sorted(facts[e])]
        derived_any = False
        changed = True
        while changed:
            changed = False
            for ent, preds in facts.items():
                for ant, con in rules:
                    if ant in preds and con not in preds:
                        preds.add(con)
                        d.add(f"{con}({ent})", "universal instantiation + modus ponens",
                              f"from {ant}({ent}) and ∀x. {ant}(x) → {con}(x)")
                        derived_any = True
                        changed = True
        if not derived_any:
            return None
        news = [f"{p}({e})" for e in sorted(facts) for p in sorted(facts[e])
                if p in {c for _a, c in rules}]
        d.result = "derived: " + ", ".join(sorted(set(news)))
        d.verified = True
        return d

    # ---- propositional modus-ponens closure (the original engine) ---- #
    def _propositional(self, text: str) -> Optional[Derivation]:
        facts, rules = self._parse(text)
        # a real deduction needs implication rules; a bare list of "facts" is not a derivation,
        # so decline (defer) rather than return a vacuous "nothing new follows".
        if not rules:
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
        if not derived:
            return None  # rules present but nothing new follows → defer rather than assert
        d.result = "derived: " + ", ".join(derived)
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
        self.symbolic = SymbolicEngine()
        self.logic = LogicEngine()

    def derive(self, text: str, domain: Optional[str] = None) -> Optional[Derivation]:
        low = (text or "").lower()
        order: List[Tuple[str, Any]]
        if domain == "physics":
            # named physics derivations first, then the general symbolic solver (rearrange any law)
            order = [("physics", self.physics), ("symbolic", self.symbolic)]
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
            # general symbolic: rearrange a stated law, solve an equation/system, integrate a
            # rate law — the broad reach that keeps ordinary "solve … for …" off the LLM.
            if any(k in low for k in ("solve", "find", "rearrange", "express", "given",
                                      "make", "d/d")) and ("=" in low or "/d" in low):
                order.append(("symbolic", self.symbolic))
            if any(k in low for k in ("->", "→", "implies", "then", "modus", "all ", "every ",
                                      "each ")):
                order.append(("logic", self.logic))
            # always leave the remaining engines as fallbacks (symbolic before maths so a bare
            # "X = Y solve for Z" is rearranged, not mis-judged as a false identity)
            for name, eng in (("chemistry", self.chemistry), ("physics", self.physics),
                              ("symbolic", self.symbolic), ("maths", self.maths),
                              ("logic", self.logic)):
                if not any(name == n for n, _ in order):
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
             "dimensional analysis", "solve for", "rearrange", "express ", "integrate",
             "all ", "every ", "each ", "if ", "given ")

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
