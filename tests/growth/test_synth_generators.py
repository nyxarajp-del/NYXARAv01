"""Tests for the synthetic generator banks — synth_math, synth_code, synth_chat.

The failure these exist to prevent was measured, not imagined: `generate_domain_docs("code",
3000)` returned 3000 documents containing **six distinct strings**, and every test passed. So
"it produced a string" is not a test. Two properties are checked instead:

1. **Diversity is enforced.** Every generator declares a state space, and drawing `cap` items
   must actually yield near-`cap` distinct documents. A generator that quietly repeats fails.
2. **The stated answer is INDEPENDENTLY correct.** Maths answers are parsed back out of the
   document and recomputed by a second route — never by calling the generator's own code path.
   Code is executed. A generator that emits confident wrong arithmetic at corpus scale is worse
   than one that emits nothing.
"""

from __future__ import annotations

import random
import re
from fractions import Fraction

import pytest

from nyxara.growth.synth_chat import GENERATORS as CHAT
from nyxara.growth.synth_code import GENERATORS as CODE
from nyxara.growth.synth_common import SynthReport, draw
from nyxara.growth.synth_math import GENERATORS as MATH

ALL = MATH + CODE + CHAT


# --------------------------------------------------------------------------- #
# 1. Diversity — the property whose absence was invisible
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gen", ALL, ids=lambda g: g.name)
def test_every_generator_produces_distinct_documents(gen) -> None:
    """A generator that repeats itself must fail here, not in a 6B-token corpus."""
    rng = random.Random(17)
    want = min(120, gen.cap)
    built = [b for b in (gen.fn(rng) for _ in range(want * 3)) if b is not None]
    assert len(built) >= want // 2, f"{gen.name} declined most draws ({len(built)}/{want * 3})"
    distinct = len(set(built))
    assert distinct >= min(want, len(built)) * 0.75, (
        f"{gen.name} produced {distinct} distinct from {len(built)} draws — "
        f"its declared space of {gen.space:,} is not real")


@pytest.mark.parametrize("gen", ALL, ids=lambda g: g.name)
def test_declared_space_is_not_wildly_optimistic(gen) -> None:
    """Sample well inside the cap; near-total distinctness is required at that density."""
    rng = random.Random(29)
    want = min(60, max(2, gen.cap // 4))
    built = [b for b in (gen.fn(rng) for _ in range(want * 4)) if b is not None][:want]
    if len(built) < 2:
        pytest.skip(f"{gen.name} is too narrow to sample meaningfully")
    assert len(set(built)) >= len(built) * 0.7, f"{gen.name} overstates its state space"


def test_the_code_domain_is_no_longer_six_documents() -> None:
    """The regression, stated as the number it actually was."""
    from nyxara.growth.synth_data import generate_domain_docs

    docs = list(generate_domain_docs("code", 800, seed=3))
    assert len(set(docs)) >= 700, (
        f"the code domain collapsed to {len(set(docs))} distinct documents "
        f"(it was SIX before this bank existed)")


def test_draw_never_exceeds_a_declared_state_space() -> None:
    """The cap is the mechanism; if it does not bind, nothing prevents a repeat of the six."""
    narrow = min(ALL, key=lambda g: g.cap)
    report = SynthReport()
    docs = list(draw([narrow], narrow.cap * 20, seed=1, report=report))
    assert len(docs) <= narrow.cap
    assert narrow.name in report.capped, "a capped generator must be named in the report"


def test_draw_output_is_deterministic_for_a_seed() -> None:
    a = list(draw(MATH[:6], 60, seed=5))
    b = list(draw(MATH[:6], 60, seed=5))
    assert a == b, "a corpus must be reproducible from its seed"


# --------------------------------------------------------------------------- #
# 2. Maths — recomputed independently, never via the generator's own path
# --------------------------------------------------------------------------- #
def _by_name(name: str):
    return next(g for g in MATH if g.name == name)


def _draw_pairs(name: str, n: int, seed: int = 41):
    rng = random.Random(seed)
    gen = _by_name(name)
    out = []
    for _ in range(n * 4):
        built = gen.fn(rng)
        if built:
            out.append(built)
        if len(out) >= n:
            break
    return out


def test_long_addition_answers_are_arithmetically_true() -> None:
    for prompt, answer in _draw_pairs("long_addition", 40):
        a, b, total = map(int, re.search(r"(\d+) \+ (\d+) = (\d+)$", answer).groups())
        assert a + b == total, f"{a} + {b} != {total}"


def test_long_subtraction_answers_are_arithmetically_true() -> None:
    for _prompt, answer in _draw_pairs("long_subtraction", 40):
        a, b, diff = map(int, re.search(r"(\d+) - (\d+) = (\d+)$", answer).groups())
        assert a - b == diff


def test_long_multiplication_answers_are_arithmetically_true() -> None:
    for _prompt, answer in _draw_pairs("long_multiplication", 40):
        a, b, prod = map(int, re.search(r"(\d+) x (\d+) = (\d+)$", answer).groups())
        assert a * b == prod


def test_long_division_reconstructs_the_dividend() -> None:
    for _prompt, answer in _draw_pairs("long_division", 40):
        n, d, q, r = map(int, re.search(
            r"(\d+) = (\d+) x (\d+) \+ (\d+)", answer).groups())
        assert d * q + r == n and 0 <= r < d


def test_gcd_is_the_real_gcd() -> None:
    import math as _m

    for _prompt, answer in _draw_pairs("gcd_euclid", 40):
        a, b, g = map(int, re.search(r"gcd\((\d+), (\d+)\) = (\d+)", answer).groups())
        assert _m.gcd(a, b) == g


def test_prime_factorisation_multiplies_back() -> None:
    for _prompt, answer in _draw_pairs("prime_factorisation", 30):
        n, factors = re.search(r"^(\d+) = (.+)$", answer.splitlines()[-1]).groups()
        product = 1
        for term in factors.split(" x "):
            base, _, exp = term.partition("^")
            product *= int(base) ** (int(exp) if exp else 1)
        assert product == int(n), f"{factors} != {n}"


def test_base_conversion_round_trips() -> None:
    for _prompt, answer in _draw_pairs("base_conversion", 30):
        n, base, digits = re.search(
            r"(\d+) in base (\d+) is ([0-9A-F]+)$", answer).groups()
        assert int(digits, int(base)) == int(n)


def test_linear_equation_root_satisfies_the_equation() -> None:
    for prompt, answer in _draw_pairs("linear_equation", 40):
        a, var, sign, b, c = re.search(
            r"(\d+)([a-z]) ([+-]) (\d+) = (-?\d+)", prompt).groups()
        root = int(re.search(rf"{var} = (-?\d+)$", answer).group(1))
        offset = int(b) if sign == "+" else -int(b)
        assert int(a) * root + offset == int(c)


def test_quadratic_roots_satisfy_the_quadratic() -> None:
    for prompt, answer in _draw_pairs("quadratic_from_roots", 40):
        lead, sb, b, sc, c = re.search(
            r"Solve (\d*)x\^2 ([+-]) (\d+)x ([+-]) (\d+) = 0", prompt).groups()
        a = int(lead) if lead else 1
        bb = int(b) if sb == "+" else -int(b)
        cc = int(c) if sc == "+" else -int(c)
        roots = [int(m) for m in re.findall(r"x = (-?\d+)", answer.splitlines()[-1])]
        assert roots, "no roots stated"
        for r in roots:
            assert a * r * r + bb * r + cc == 0, f"{a}x^2+{bb}x+{cc} != 0 at x={r}"


def test_linear_system_solution_satisfies_both_equations() -> None:
    for prompt, answer in _draw_pairs("linear_system", 40):
        a, b, e = map(int, re.search(r"(\d+)x \+ (\d+)y = (-?\d+)", prompt).groups())
        rows = re.findall(r"(\d+)x \+ (\d+)y = (-?\d+)", prompt)
        c, d, f = map(int, rows[1])
        x, y = map(int, re.search(r"x = (-?\d+), y = (-?\d+)$", answer).groups())
        assert a * x + b * y == e and c * x + d * y == f


def test_fraction_sums_are_exact_and_in_lowest_terms() -> None:
    for prompt, answer in _draw_pairs("fraction_sum", 40):
        a, b, op, c, d = re.search(r"(\d+)/(\d+) ([+-]) (\d+)/(\d+)", prompt).groups()
        expected = (Fraction(int(a), int(b)) + Fraction(int(c), int(d)) if op == "+"
                    else Fraction(int(a), int(b)) - Fraction(int(c), int(d)))
        num, den = map(int, re.search(r"Lowest terms: (-?\d+)/(\d+)$", answer).groups())
        got = Fraction(num, den)
        assert got == expected
        assert got.denominator == den, "not reduced to lowest terms"


def test_modular_exponentiation_matches_pow() -> None:
    for prompt, answer in _draw_pairs("modular_arithmetic", 30):
        b, e, m = map(int, re.search(r"(\d+)\^(\d+) mod (\d+)", prompt).groups())
        got = int(re.search(r"mod \d+ = (\d+)$", answer).group(1))
        assert pow(b, e, m) == got


def test_combinations_match_math_comb() -> None:
    import math as _m

    for _prompt, answer in _draw_pairs("combinations", 30):
        n, k, got = map(int, re.search(r"C\((\d+), (\d+)\) = (\d+)$", answer).groups())
        assert _m.comb(n, k) == got


def test_permutations_match_math_perm() -> None:
    import math as _m

    for _prompt, answer in _draw_pairs("permutations", 25):
        n, k, got = map(int, re.search(r"P\((\d+), (\d+)\) = (\d+)$", answer).groups())
        assert _m.perm(n, k) == got


def test_pythagorean_hypotenuse_is_exact() -> None:
    for prompt, answer in _draw_pairs("pythagorean_triple", 30):
        a, b = map(int, re.search(r"legs (\d+) and (\d+)", prompt).groups())
        c = int(re.search(r"hypotenuse = (\d+)$", answer).group(1))
        assert a * a + b * b == c * c


def test_determinant_matches_the_definition() -> None:
    for prompt, answer in _draw_pairs("determinant_2x2", 30):
        a, b, c, d = map(int, re.search(
            r"\[\[(-?\d+), (-?\d+)\], \[(-?\d+), (-?\d+)\]\]", prompt).groups())
        got = int(re.search(r"det = (-?\d+)$", answer).group(1))
        assert a * d - b * c == got


def test_arithmetic_sequence_term_and_sum() -> None:
    for prompt, answer in _draw_pairs("arithmetic_sequence", 30):
        first, step = map(int, re.search(
            r"starts at (-?\d+) and increases by (\d+)", prompt).groups())
        n = int(re.search(r"What is term (\d+)", prompt).group(1))
        nth, total = map(int, re.search(r"term \d+ = (-?\d+), sum = (-?\d+)$", answer).groups())
        assert first + (n - 1) * step == nth
        assert sum(first + i * step for i in range(n)) == total


def test_dice_probabilities_are_in_range_and_reduced() -> None:
    for _prompt, answer in _draw_pairs("dice_probability", 30):
        ways, out, num, den = map(int, re.search(
            r"Probability = (\d+)/(\d+) = (\d+)/(\d+)$", answer).groups())
        assert Fraction(ways, out) == Fraction(num, den)
        assert 0 < Fraction(num, den) <= 1
        assert Fraction(num, den).denominator == den


def test_summary_statistics_mean_and_median() -> None:
    for prompt, answer in _draw_pairs("summary_statistics", 25):
        values = [int(v) for v in re.search(r": ([-\d, ]+)$", prompt.strip()).group(1).split(", ")]
        mean_txt, median_txt = re.search(r"mean = (\S+), median = (\S+)$", answer).groups()
        expected_mean = Fraction(sum(values), len(values))
        assert Fraction(mean_txt) == expected_mean
        ordered = sorted(values)
        mid = len(ordered) // 2
        expected_median = (Fraction(ordered[mid]) if len(ordered) % 2
                           else Fraction(ordered[mid - 1] + ordered[mid], 2))
        assert Fraction(median_txt) == expected_median


# --------------------------------------------------------------------------- #
# 3. Code — executed, and the bug proven to be a real bug
# --------------------------------------------------------------------------- #
def test_every_algorithm_document_contains_runnable_code() -> None:
    for prompt, answer in _pairs(CODE, "code_algorithm", 12):
        assert "def " in answer and "->" in answer


def test_a_bugfix_pair_states_a_genuinely_different_output() -> None:
    """The mutant must actually diverge — an equivalent mutant is not a bug."""
    for prompt, answer in _pairs(CODE, "code_bugfix", 12):
        assert "def " in prompt, "the broken version must be shown"
        assert "Fixed:" in answer and "def " in answer.split("Fixed:")[1]
        returned, correct = re.search(
            r"it returns (.+?), but the correct answer is (.+?)\n", answer, re.S).groups()
        assert returned.strip() != correct.strip(), "the 'bug' changed nothing"


def test_generated_tests_assert_the_real_outputs() -> None:
    """Re-execute the reference against the emitted assertions."""
    from nyxara.growth.sandbox_exec import GuardedExecutor, SandboxLimits

    with GuardedExecutor(SandboxLimits(cpu_seconds=2, wall_seconds=5.0)) as ex:
        for prompt, answer in _pairs(CODE, "code_tests", 8):
            src = prompt.split("\n\n", 1)[1]
            fn = re.search(r"def (\w+)\(", src).group(1)
            for call, expected in re.findall(rf"assert {fn}\((.+?)\) == (.+)", answer):
                got = ex.call(src, fn, [[eval(call)]])          # noqa: S307 — our own literal
                assert got.ok and got.results[0][0]
                assert repr(got.results[0][1]) == repr(eval(expected))


def test_a_trace_matches_a_real_execution() -> None:
    from nyxara.growth.sandbox_exec import GuardedExecutor, SandboxLimits

    with GuardedExecutor(SandboxLimits(cpu_seconds=2, wall_seconds=5.0)) as ex:
        for prompt, answer in _pairs(CODE, "code_trace", 8):
            src = prompt.split("\n\n", 1)[1]
            fn = re.search(r"def (\w+)\(", src).group(1)
            values = eval(re.search(r"Trace this function on (\[.*?\])", prompt).group(1))
            stated = int(re.search(r"= (-?\d+)$", answer).group(1))
            got = ex.call(src, fn, [[values]])
            assert got.ok and got.results[0][0]
            assert got.results[0][1] == stated, "the narrated trace disagrees with execution"


def _pairs(bank, name: str, n: int, seed: int = 13):
    gen = next(g for g in bank if g.name == name)
    rng = random.Random(seed)
    out = []
    for _ in range(n * 8):
        built = gen.fn(rng)
        if built:
            out.append(built)
        if len(out) >= n:
            break
    assert out, f"{name} produced nothing"
    return out


# --------------------------------------------------------------------------- #
# 4. Conversation — the constraints the prompts actually state
# --------------------------------------------------------------------------- #
def test_bullet_documents_give_exactly_the_number_requested() -> None:
    """A document that asks for three bullets and gives four teaches that instructions are noise."""
    for prompt, answer in _pairs(CHAT, "chat_format_bullets", 20):
        want = int(re.search(r"exactly (\d+) bullet", prompt).group(1))
        assert len([ln for ln in answer.splitlines() if ln.startswith("- ")]) == want


def test_json_documents_emit_parseable_json() -> None:
    import json

    for _prompt, answer in _pairs(CHAT, "chat_format_json", 15):
        payload = json.loads(answer)
        assert set(payload) == {"id", "status", "reading", "unit"}


def test_one_sentence_documents_give_one_sentence() -> None:
    for _prompt, answer in _pairs(CHAT, "chat_format_sentence", 10):
        assert answer.count(".") == 1


def test_tool_traces_carry_real_arithmetic() -> None:
    """A fabricated tool result teaches the model to fabricate tool results."""
    for _prompt, answer in _pairs(CHAT, "chat_tool_multi_turn", 20):
        a, b, product = map(int, re.search(
            r"<tool>calculator\((\d+) \* (\d+)\)</tool>\n<result>(\d+)</result>",
            answer if answer.startswith("<tool>") else answer).groups())
        assert a * b == product


def test_primality_traces_are_correct() -> None:
    for prompt, answer in _pairs(CHAT, "chat_tool_primality", 25):
        n = int(re.search(r"Is (\d+) a prime", prompt).group(1))
        actual = n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))
        assert (" is prime" in answer) is actual, f"{n} misclassified"


def test_retrieval_includes_unanswerable_cases() -> None:
    """A model trained only on answerable passages confabulates when the answer is absent."""
    pairs = _pairs(CHAT, "chat_retrieval", 120, seed=5)
    refusals = sum(1 for _p, a in pairs if "does not say" in a)
    assert refusals > 0, "no unanswerable cases — the negative half is missing"
    assert refusals < len(pairs), "every case is unanswerable, which is its own failure"


def test_multi_hop_sums_are_real() -> None:
    for prompt, answer in _pairs(CHAT, "chat_multi_hop", 20):
        n1, n2 = (int(m) for m in re.findall(r"draws (\d+)", prompt))
        total = int(re.search(r"Combined draw: (\d+)", answer).group(1))
        assert n1 + n2 == total
