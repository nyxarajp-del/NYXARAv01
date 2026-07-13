"""Tests for nyxara.mind.first_principles — derive results from domain axioms (verifiable)."""

from __future__ import annotations


from nyxara.mind.first_principles import (
    FirstPrinciplesEngine,
    FirstPrinciplesFaculty,
    balance_reaction,
    derive_from_first_principles,
    parse_formula,
)
from nyxara.mind.faculties import Task, TaskType


# --------------------------------------------------------------------------- #
# Chemistry — exact balancing from the null space (not a lookup)
# --------------------------------------------------------------------------- #
def test_parse_formula_nested_parentheses():
    assert parse_formula("Fe2(SO4)3") == {"Fe": 2, "S": 3, "O": 12}
    assert parse_formula("C3H8") == {"C": 3, "H": 8}
    assert parse_formula("Ca(OH)2") == {"Ca": 1, "O": 2, "H": 2}


def test_balance_combustion_propane():
    coeffs, reactants, products = balance_reaction("C3H8 + O2 -> CO2 + H2O")
    assert reactants == ["C3H8", "O2"] and products == ["CO2", "H2O"]
    assert coeffs == [1, 5, 3, 4]


def test_balance_iron_oxidation():
    coeffs, _r, _p = balance_reaction("Fe + O2 -> Fe2O3")
    assert coeffs == [4, 3, 2]


def test_balance_smallest_integers():
    # H2 + O2 -> H2O must reduce to 2,1,2 (not 4,2,4)
    coeffs, _r, _p = balance_reaction("H2 + O2 -> H2O")
    assert coeffs == [2, 1, 2]


def test_balance_conserves_every_atom():
    coeffs, reactants, products = balance_reaction("C3H8 + O2 -> CO2 + H2O")
    nr = len(reactants)
    left, right = {}, {}
    for i, sp in enumerate(reactants):
        for el, n in parse_formula(sp).items():
            left[el] = left.get(el, 0) + coeffs[i] * n
    for j, sp in enumerate(products):
        for el, n in parse_formula(sp).items():
            right[el] = right.get(el, 0) + coeffs[nr + j] * n
    assert left == right


# --------------------------------------------------------------------------- #
# Physics — symbolic derivation + dimensional analysis
# --------------------------------------------------------------------------- #
def test_escape_velocity_derivation_verified():
    d = derive_from_first_principles("derive the escape velocity from energy conservation")
    assert d is not None and d.domain == "physics" and d.verified
    # result must equal sqrt(2 G M / r) up to formatting
    assert "sqrt" in d.result and "G" in d.result and "M" in d.result and "r" in d.result


def test_kinematics_velocity_derivation_verified():
    d = derive_from_first_principles("derive v = u + a t for constant acceleration kinematics")
    assert d is not None and d.verified
    assert "a*t" in d.result.replace(" ", "") and "u" in d.result


def test_dimensional_analysis_pendulum_period():
    d = derive_from_first_principles("derive the period of a pendulum")
    assert d is not None and d.verified
    # T ~ sqrt(length / g): length^(1/2) and g^(-1/2)
    assert "length" in d.result and "g" in d.result


def test_dimensional_analysis_generic_form():
    eng = FirstPrinciplesEngine()
    d = eng.physics.dim.derive("force", ["mass", "acceleration"])
    assert d is not None and d.verified
    assert "mass" in d.result and "acceleration" in d.result


# --------------------------------------------------------------------------- #
# Maths — certified identity
# --------------------------------------------------------------------------- #
def test_math_proof_identity_holds():
    d = derive_from_first_principles("prove that (a+b)^2 = a^2 + 2*a*b + b^2")
    assert d is not None and d.domain == "maths" and d.verified
    assert "proven" in d.result


def test_math_proof_rejects_false_claim():
    d = derive_from_first_principles("prove that (a+b)^2 = a^2 + b^2")
    assert d is not None and not d.verified
    assert "NOT" in d.result


# --------------------------------------------------------------------------- #
# Logic — forward-chaining deductive closure
# --------------------------------------------------------------------------- #
def test_logic_forward_chaining():
    d = derive_from_first_principles("A. A -> B. B -> C.")
    assert d is not None and d.domain == "logic" and d.verified
    assert "B" in d.result and "C" in d.result


# --------------------------------------------------------------------------- #
# Faculty wiring
# --------------------------------------------------------------------------- #
def test_faculty_is_verifiable_and_handles_new_types():
    fac = FirstPrinciplesFaculty()
    assert fac.verifiable and fac.reliability == 1.0
    assert TaskType.DERIVATION in fac.handles
    assert TaskType.CHEMISTRY in fac.handles


def test_faculty_suitability_fires_and_defers():
    fac = FirstPrinciplesFaculty()
    hot = Task(TaskType.CHEMISTRY, description="balance C3H8 + O2 -> CO2 + H2O")
    cold = Task(TaskType.DERIVATION, description="tell me a joke")
    assert fac.suitability(hot) > 0.0
    assert fac.suitability(cold) == 0.0


def test_faculty_handle_returns_verifiable_proposal():
    fac = FirstPrinciplesFaculty()
    task = Task(TaskType.CHEMISTRY, description="balance H2 + O2 -> H2O")
    prop = fac.handle(task)
    assert prop.confidence == 1.0
    assert "2 H2O" in prop.content


def test_solve_with_faculties_routes_to_first_principles():
    from nyxara.mind.reasoning_faculties import solve_with_faculties
    out = solve_with_faculties("balance C3H8 + O2 -> CO2 + H2O")
    assert out is not None
    answer, conf = out
    assert conf == 1.0 and "5 O2" in answer


# --------------------------------------------------------------------------- #
# Symbolic — general law rearrangement (solve any stated law for any variable)
# --------------------------------------------------------------------------- #
def test_symbolic_rearranges_a_stated_law_and_verifies():
    d = derive_from_first_principles("given F = m*a, solve for a")
    assert d is not None and d.verified
    assert d.result.replace(" ", "") == "a=F/m"


def test_symbolic_solves_a_famous_relation():
    d = derive_from_first_principles("solve E = m*c^2 for m")
    assert d is not None and d.verified
    assert "m = E/c**2" in d.result


def test_symbolic_solves_a_system_eliminating_the_intermediate():
    d = derive_from_first_principles("given p = f/A and f = m*a, find p")
    assert d is not None and d.verified
    # p is expressed in the base quantities m, a, A (f eliminated)
    r = d.result.replace(" ", "")
    assert "f" not in r.split("=", 1)[1] and "m" in r and "A" in r


def test_symbolic_integrates_a_rate_law_ode():
    d = derive_from_first_principles("given dv/dt = a, find v")
    assert d is not None and d.verified and d.domain == "physics"
    assert "a*t" in d.result.replace(" ", "")


def test_symbolic_defers_to_algebra_faculty_for_single_unknown():
    # a single equation in one unknown is the AlgebraFaculty's job — the symbolic engine defers
    from nyxara.mind.first_principles import SymbolicEngine
    assert SymbolicEngine().derive("solve 2x + 3 = 9") is None


# --------------------------------------------------------------------------- #
# Logic — first-order (universally-quantified) chaining, the classic syllogism
# --------------------------------------------------------------------------- #
def test_first_order_syllogism_copula():
    d = derive_from_first_principles("all humans are mortal. socrates is a human.")
    assert d is not None and d.domain == "logic" and d.verified
    assert "mortal(socrates)" in d.result


def test_first_order_verb_phrase_predicate():
    d = derive_from_first_principles("every bird can fly. tweety is a bird.")
    assert d is not None and "can_fly(tweety)" in d.result


def test_first_order_if_then_variable_rule():
    d = derive_from_first_principles("if X is a metal then X conducts. copper is a metal.")
    assert d is not None and "conduct(copper)" in d.result


def test_logic_defers_on_bare_facts_and_chat():
    # a bare list of facts with no rule is not a derivation → defer (None), not "nothing follows"
    assert derive_from_first_principles("hi how are you today") is None
    assert derive_from_first_principles("the sky is blue") is None


# --------------------------------------------------------------------------- #
# Dimensional analysis — a new quantity defined from a stated equation
# --------------------------------------------------------------------------- #
def test_dimensional_define_new_quantity_from_equation():
    from nyxara.mind.first_principles import DimensionalAnalysis
    da = DimensionalAnalysis()
    # power is not asserted directly — it is defined from energy/time and then dimensionally placed
    dim = da.define("thrust_power", "energy / time")
    assert dim is not None
    assert da.dim_of("thrust_power") == da.dim_of("power")


def test_arithmetic_still_routes_correctly_no_regression():
    from nyxara.mind.reasoning_faculties import solve_with_faculties
    out = solve_with_faculties("2 + 3 * 4")
    assert out is not None and out[0] == "14"
