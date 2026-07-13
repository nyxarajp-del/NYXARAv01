"""Tests for nyxara.growth.open_world — open-world generalization from first principles.

NYXARA is pointed at a black box she has *never seen* and must crack it by the real loop:
Observe (probe) → Hypothesize (induce laws) → Test (discriminating experiments) → Model
(a predictor that generalizes). The whole inquiry runs offline (no LLM, no network), driven by
deterministic synthetic systems whose hidden rules the engine has no prior knowledge of — so these
tests are hermetic and reproducible. The non-negotiable property is honesty: when nothing fits she
must say ``UNMODELLED``, never bluff a model of a system she cannot explain.
"""

from __future__ import annotations

import random

import pytest

from nyxara.growth.open_world import (
    DomainSpec,
    LawFamily,
    OpenWorldGeneralizer,
    UnderstandingReport,
    Verdict,
    build_alien_machine,
)


def _g(**kw) -> OpenWorldGeneralizer:
    kw.setdefault("seed", 7)
    return OpenWorldGeneralizer(**kw)


# --------------------------------------------------------------------------- #
# Recovering hidden laws — each system's rule is unknown to the engine
# --------------------------------------------------------------------------- #
def test_recovers_affine_law_and_predicts_unseen_input():
    rep = _g().understand(lambda x: 3 * x - 2, domain=DomainSpec(1, "real"), label="affine")
    assert rep.verdict is Verdict.MODELLED
    assert rep.law_family == LawFamily.AFFINE.value
    # generalizes to an input far outside everything it probed
    assert rep.predict(1000) == pytest.approx(2998, abs=1e-3)


def test_recovers_modular_law():
    rep = _g().understand(lambda x: x % 5, domain=DomainSpec(1, "int", -10, 10), label="mod")
    assert rep.verdict is Verdict.MODELLED
    assert rep.law_family == LawFamily.MODULAR.value
    assert rep.predict(23) == 23 % 5


def test_recovers_quadratic_polynomial():
    rep = _g().understand(lambda x: 2 * x * x + x - 5, domain=DomainSpec(1, "real"), label="quad")
    assert rep.verdict is Verdict.MODELLED
    assert rep.law_family == LawFamily.POLYNOMIAL.value
    assert rep.predict(12) == pytest.approx(2 * 144 + 12 - 5, abs=1e-2)


def test_recovers_multiplicative_law():
    rep = _g().understand(lambda xy: 4 * xy[0] * xy[1],
                          domain=DomainSpec(2, "real", scalar=False), label="prod")
    assert rep.verdict is Verdict.MODELLED
    assert rep.predict((3, 5)) == pytest.approx(60, abs=1e-2)


def test_recovers_threshold_law():
    rep = _g().understand(lambda x: 1 if x < 0 else 8, domain=DomainSpec(1, "real"), label="thr")
    assert rep.verdict is Verdict.MODELLED
    assert rep.law_family == LawFamily.THRESHOLD.value
    assert rep.predict(-5) == 1 and rep.predict(5) == 8


def test_recovers_boolean_xor():
    rep = _g().understand(lambda ab: bool(ab[0]) ^ bool(ab[1]),
                          domain=DomainSpec(2, "bool", scalar=False), label="xor")
    assert rep.verdict is Verdict.MODELLED
    assert rep.predict((True, False)) == 1 and rep.predict((True, True)) == 0


def test_recovers_boolean_and_as_named_operator():
    rep = _g().understand(lambda ab: bool(ab[0] and ab[1]),
                          domain=DomainSpec(2, "bool", scalar=False), label="and")
    assert rep.verdict is Verdict.MODELLED
    # the named AND (2 params) must beat the memorising lookup (4 rows) by Occam/MDL
    assert rep.law_family == LawFamily.BOOLEAN.value and rep.law == "out = AND"


# --------------------------------------------------------------------------- #
# Max-level law families — the shapes a polynomial can't reach
# --------------------------------------------------------------------------- #
import math  # noqa: E402


def test_recovers_power_law():
    rep = _g().understand(lambda x: 3.0 * x ** 1.5, domain=DomainSpec(1, "real", 0.5, 8),
                          label="power")
    assert rep.verdict is Verdict.MODELLED
    assert rep.law_family == LawFamily.POWER.value
    assert rep.predict(4.0) == pytest.approx(3.0 * 4.0 ** 1.5, rel=1e-2)


def test_recovers_exponential_law():
    rep = _g().understand(lambda x: 2.0 * math.exp(0.5 * x), domain=DomainSpec(1, "real", 0, 5),
                          label="exp")
    assert rep.verdict is Verdict.MODELLED
    assert rep.law_family == LawFamily.EXPONENTIAL.value
    assert rep.predict(4.0) == pytest.approx(2.0 * math.exp(0.5 * 4.0), rel=1e-2)


def test_recovers_logarithmic_law():
    rep = _g().understand(lambda x: 2.5 * math.log(abs(x)) + 1 if x else 1,
                          domain=DomainSpec(1, "real", 1, 8), label="log")
    assert rep.verdict is Verdict.MODELLED
    assert rep.law_family == LawFamily.LOGARITHMIC.value


def test_recovers_rational_law():
    rep = _g().understand(lambda x: 30.0 / (x + 2.0) + 1.0, domain=DomainSpec(1, "real", 0, 9),
                          label="rational")
    assert rep.verdict is Verdict.MODELLED
    assert rep.law_family == LawFamily.RATIONAL.value
    assert rep.predict(4.0) == pytest.approx(30.0 / 6.0 + 1.0, rel=1e-2)


def test_recovers_sinusoidal_law():
    rep = _g().understand(lambda x: 3.0 * math.sin(1.0 * x) + 2.0, domain=DomainSpec(1, "real"),
                          label="sine")
    assert rep.verdict is Verdict.MODELLED
    assert rep.law_family == LawFamily.SINUSOIDAL.value


def test_recovers_gcd_over_integers():
    rep = _g().understand(lambda xy: math.gcd(int(xy[0]), int(xy[1])),
                          domain=DomainSpec(2, "int", 1, 12, scalar=False), label="gcd")
    assert rep.verdict is Verdict.MODELLED
    assert rep.law_family == LawFamily.GCD.value
    assert rep.predict((8, 12)) == 4


def test_models_a_stateful_recurrence_fibonacci():
    """The first family that models a system WITH MEMORY: out[n] = out[n-1] + out[n-2]."""
    memo = {0: 1, 1: 1}

    def fib(n):
        n = int(round(n))
        if n < 0:
            return 1
        while n not in memo:
            k = max(memo)
            memo[k + 1] = memo[k] + memo[k - 1]
        return memo[n]

    rep = _g().understand(fib, domain=DomainSpec(1, "int", 0, 20), label="fib")
    assert rep.verdict is Verdict.MODELLED
    assert rep.law_family == LawFamily.RECURRENCE.value
    assert rep.predict(21) == pytest.approx(fib(21), rel=1e-6)


def test_arithmetic_sequence_stays_affine_not_recurrence():
    """A clean line must NOT be over-explained as a memory law — Occam keeps it affine."""
    rep = _g().understand(lambda n: 2 * int(round(n)) + 3, domain=DomainSpec(1, "int", 0, 20),
                          label="arith")
    assert rep.verdict is Verdict.MODELLED
    assert rep.law_family == LawFamily.AFFINE.value


def test_new_families_serialize_and_rebuild():
    from nyxara.growth.open_world import rebuild_predict, _to_vec
    rep = _g().understand(lambda x: 2.0 * math.exp(0.5 * x), domain=DomainSpec(1, "real", 0, 5),
                          label="exp")
    w = rep.winner
    assert w is not None and w.coeffs
    pred = rebuild_predict(w.family, w.coeffs, dims=1)
    assert pred is not None
    spec = DomainSpec(1, "real")
    assert pred(_to_vec(3.0, spec)) == pytest.approx(rep.predict(3.0), rel=1e-9)


# --------------------------------------------------------------------------- #
# Honesty — when she cannot crack it, she must say so
# --------------------------------------------------------------------------- #
def test_unmodellable_random_system_is_honestly_unmodelled():
    rng = random.Random(123)
    rep = _g().understand(lambda x: rng.random() * 100, domain=DomainSpec(1, "real"), label="rng")
    assert rep.verdict is Verdict.UNMODELLED
    assert rep.confidence < 0.3
    assert rep.holdout_accuracy < 0.6


def test_rejecting_box_is_inconclusive_not_bluffed():
    def always_rejects(_action):
        raise ValueError("input rejected")

    rep = _g().understand(always_rejects, domain=DomainSpec(1, "real"), label="closed")
    assert rep.verdict in (Verdict.INCONCLUSIVE, Verdict.UNMODELLED)
    assert rep.confidence < 0.3


# --------------------------------------------------------------------------- #
# Efficiency, determinism, serialization
# --------------------------------------------------------------------------- #
def test_active_discrimination_stays_within_budget():
    budget = 40
    rep = _g().understand(lambda x: 2 * x + 1, domain=DomainSpec(1, "real"),
                          budget=budget, label="line")
    # probes are bounded: the discovery budget plus a small fixed validation set
    assert rep.probes_used <= budget + 24
    assert rep.verdict is Verdict.MODELLED


def test_deterministic_under_fixed_seed():
    sys = lambda x: 5 * x - 3  # noqa: E731
    a = OpenWorldGeneralizer(seed=42).understand(sys, domain=DomainSpec(1, "real"))
    b = OpenWorldGeneralizer(seed=42).understand(sys, domain=DomainSpec(1, "real"))
    assert a.law == b.law
    assert a.probes_used == b.probes_used
    assert a.confidence == b.confidence


def test_report_to_dict_roundtrips():
    rep = _g().understand(lambda x: 3 * x - 2, domain=DomainSpec(1, "real"), label="affine")
    d = rep.to_dict()
    assert d["system"] == "affine"
    assert d["verdict"] == "modelled"
    assert d["law_family"] == "affine"
    assert isinstance(d["candidates"], list) and d["candidates"]
    assert 0.0 <= d["confidence"] <= 1.0


# --------------------------------------------------------------------------- #
# Graceful degradation + folding into the wider mind
# --------------------------------------------------------------------------- #
def test_works_offline_with_no_optional_faculties():
    g = OpenWorldGeneralizer(world_model=None, causal_model=None, memory=None,
                             knowledge=None, novelty=None, seed=1)
    rep = g.understand(lambda x: -x + 4, domain=DomainSpec(1, "real"))
    assert isinstance(rep, UnderstandingReport)
    assert rep.verdict is Verdict.MODELLED


def test_folds_a_belief_with_honest_confidence():
    from nyxara.growth.autonomous_scientist import BeliefModel

    bm = BeliefModel()
    g = OpenWorldGeneralizer(belief_model=bm, seed=3)
    g.understand(lambda x: 2 * x + 1, domain=DomainSpec(1, "real"), label="line")
    assert len(bm) == 1
    belief = next(iter(bm.beliefs.values()))
    assert belief.verdict == "supported"          # she modelled it → the belief is supported


def test_folds_probes_into_provided_world_model():
    from nyxara.mind.world_model import WorldModel

    wm = WorldModel()
    g = OpenWorldGeneralizer(world_model=wm, seed=3)
    g.understand(lambda x: 2 * x + 1, domain=DomainSpec(1, "real"), label="line")
    # every probe is recorded as a real (state, "probe", next_state) transition
    assert "probe" in wm.actions()


def test_unmodelled_box_folds_a_refuted_belief():
    from nyxara.growth.autonomous_scientist import BeliefModel

    rng = random.Random(99)
    bm = BeliefModel()
    g = OpenWorldGeneralizer(belief_model=bm, seed=5)
    g.understand(lambda x: rng.random(), domain=DomainSpec(1, "real"), label="noise")
    belief = next(iter(bm.beliefs.values()))
    assert belief.verdict == "refuted"            # honest: "I could not model this"


# --------------------------------------------------------------------------- #
# The hidden "alien machine" demo path + core integration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", list(range(6)))
def test_cracks_a_freshly_built_alien_machine(seed):
    system, domain, _secret = build_alien_machine(seed)
    rep = OpenWorldGeneralizer(seed=seed + 1).understand(system, domain=domain, label="alien")
    assert rep.verdict is Verdict.MODELLED
    assert rep.confidence >= 0.5


def test_core_generalize_returns_a_well_formed_report():
    from nyxara import NyxaraCore

    core = NyxaraCore()
    out = core.generalize()
    assert "verdict" in out and "law" in out and "confidence" in out
    # an explicit unknown system, too
    out2 = core.generalize(lambda x: 9 * x - 1, label="line")
    assert out2["verdict"] == "modelled"
