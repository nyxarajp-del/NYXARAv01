"""Tests for the real-experiment-engine upgrade of the Scientist + Autonomous Scientist.

Covers the new capabilities — statistical (Monte-Carlo + significance) experiments, the
expanded deterministic propositions, TF-IDF semantic grounding, the AST-restricted LLM
experiment path (with a stub LLM, fully offline), and Bayesian belief updating.
"""

from __future__ import annotations

from nyxara.growth.autonomous_scientist import BeliefModel, _apply_evidence, Belief
from nyxara.growth.scientist import (ExperimentKind, Scientist, Verdict,
                                     _is_safe_experiment_code, _run_safe_experiment,
                                     _semantic_affirm, _significance_test, _simulate_statistic)


# --------------------------------------------------------------------------- #
# Statistical experiments (Monte-Carlo + a real significance test)
# --------------------------------------------------------------------------- #
def test_fair_coin_claim_supported():
    rep = Scientist().investigate("Is a fair coin's probability of heads 0.5?")
    assert rep.experiment.kind is ExperimentKind.STATISTICAL
    assert rep.conclusion.verdict is Verdict.SUPPORTED
    assert rep.observation.metrics["pvalue"] >= 0.05


def test_biased_coin_claim_refuted():
    rep = Scientist().investigate("Is a fair coin's probability of heads 0.7?")
    assert rep.experiment.kind is ExperimentKind.STATISTICAL
    assert rep.conclusion.verdict is Verdict.REFUTED
    assert rep.observation.metrics["pvalue"] < 0.05


def test_dice_sum_probability_supported():
    rep = Scientist().investigate("Do two dice sum to 7 with probability 1/6?")
    assert rep.experiment.kind is ExperimentKind.STATISTICAL
    assert rep.conclusion.verdict is Verdict.SUPPORTED


def test_die_expected_value_supported():
    rep = Scientist().investigate("Does a fair die have expected value 3.5?")
    assert rep.experiment.kind is ExperimentKind.STATISTICAL
    assert rep.conclusion.verdict is Verdict.SUPPORTED


def test_simulate_is_deterministic():
    a = _simulate_statistic("stat:die_mean:3.5", 500)
    b = _simulate_statistic("stat:die_mean:3.5", 500)
    assert a["samples"] == b["samples"]            # seeded ⇒ reproducible


def test_significance_test_detects_difference():
    p_same, _ = _significance_test([3.5] * 50, 3.5, "ttest")
    p_diff, _ = _significance_test([float(i % 2) for i in range(50)], 3.5, "ttest")
    assert p_same > p_diff and p_diff < 0.05


# --------------------------------------------------------------------------- #
# Expanded deterministic propositions
# --------------------------------------------------------------------------- #
def test_divisibility_true_and_false():
    sci = Scientist()
    assert sci.investigate("Is 100 divisible by 4?").conclusion.verdict is Verdict.SUPPORTED
    assert sci.investigate("Is 7 divisible by 2?").conclusion.verdict is Verdict.REFUTED


def test_even_odd_and_perfect_square():
    sci = Scientist()
    assert sci.investigate("Is 8 even?").conclusion.verdict is Verdict.SUPPORTED
    assert sci.investigate("Is 9 a perfect square?").conclusion.verdict is Verdict.SUPPORTED
    assert sci.investigate("Is 10 a perfect square?").conclusion.verdict is Verdict.REFUTED


def test_numeric_comparison():
    sci = Scientist()
    assert sci.investigate("Is 5 greater than 3?").conclusion.verdict is Verdict.SUPPORTED
    assert sci.investigate("Is 2 greater than 8?").conclusion.verdict is Verdict.REFUTED


# --------------------------------------------------------------------------- #
# TF-IDF semantic grounding
# --------------------------------------------------------------------------- #
def test_semantic_affirm_matches_related_not_unrelated():
    stmt = "the moon reflects sunlight"
    affirming, scores = _semantic_affirm(
        stmt, ["The moon does not emit its own light; it reflects sunlight.",
               "The Eiffel Tower is in Paris and is made of iron."])
    assert len(affirming) == 1                     # only the genuinely-related chunk
    assert scores[0] > scores[1]


def test_semantic_affirm_empty():
    assert _semantic_affirm("anything", []) == ([], [])


# --------------------------------------------------------------------------- #
# LLM-authored, AST-restricted, sandboxed experiments
# --------------------------------------------------------------------------- #
class _StubProvider:
    name = "tinyllama"

    def available(self) -> bool:
        return True


class _StubLLM:
    def __init__(self, text: str) -> None:
        self._text = text

    def chosen_provider(self):
        return _StubProvider()

    def generate(self, prompt: str, **kw) -> str:
        return self._text


def test_safe_experiment_validator_accepts_safe_code():
    code = "def experiment():\n    return all((n % 2 == 0) for n in range(0, 10, 2))\n"
    assert _is_safe_experiment_code(code)
    assert _run_safe_experiment(code) is True


def test_safe_experiment_validator_rejects_dangerous_code():
    assert not _is_safe_experiment_code("import os\ndef experiment():\n    return True\n")
    assert not _is_safe_experiment_code(
        "def experiment():\n    return open('/etc/passwd').read()\n")
    assert not _is_safe_experiment_code("def experiment():\n    return ().__class__\n")
    assert not _is_safe_experiment_code(
        "def experiment():\n    while True:\n        pass\n")     # unbounded loop


def test_llm_experiment_disabled_without_flag():
    sci = Scientist(llm=_StubLLM("def experiment():\n    return True\n"))
    # default allow_llm_experiments=False ⇒ open question routes to KNOWLEDGE, not LLM code
    rep = sci.investigate("Is the Collatz conjecture true for the first 20 integers?")
    assert rep.experiment.kind is ExperimentKind.KNOWLEDGE


def test_llm_experiment_runs_when_enabled():
    code = ("def experiment():\n"
            "    return all((n * 2) % 2 == 0 for n in range(20))\n")
    sci = Scientist(llm=_StubLLM(code), allow_llm_experiments=True)
    rep = sci.investigate("Is twice any integer always even?")
    assert rep.experiment.kind is ExperimentKind.COMPUTATIONAL
    assert "LLM-authored" in rep.experiment.method
    assert rep.conclusion.verdict is Verdict.SUPPORTED


def test_llm_experiment_rejects_unsafe_code_and_falls_back():
    sci = Scientist(llm=_StubLLM("import os\ndef experiment():\n    return True\n"),
                    allow_llm_experiments=True)
    rep = sci.investigate("Does some open-ended thing hold?")
    assert rep.experiment.kind is ExperimentKind.KNOWLEDGE   # unsafe ⇒ not executed


# --------------------------------------------------------------------------- #
# Bayesian belief updating
# --------------------------------------------------------------------------- #
def test_bayesian_consistent_evidence_sharpens():
    b = Belief(variable="v", statement="s", verdict="inconclusive", confidence=0.5)
    _apply_evidence(b, "supported", 0.9)
    first = b.confidence
    _apply_evidence(b, "supported", 0.9)
    _apply_evidence(b, "supported", 0.9)
    assert b.verdict == "supported"
    assert b.confidence > first                    # repeated support sharpens toward 1


def test_bayesian_conflicting_evidence_stays_uncertain():
    b = Belief(variable="v", statement="s", verdict="inconclusive", confidence=0.5)
    _apply_evidence(b, "supported", 0.9)
    _apply_evidence(b, "refuted", 0.9)
    assert b.verdict == "inconclusive"             # a clean contradiction → honest 0.5
    assert abs(b.support_prob - 0.5) < 0.05


def test_bayesian_does_not_hard_flip_on_one_reading():
    b = Belief(variable="v", statement="s", verdict="inconclusive", confidence=0.5)
    for _ in range(5):
        _apply_evidence(b, "supported", 0.9)       # build a strong prior
    assert b.verdict == "supported"
    _apply_evidence(b, "refuted", 0.6)             # one weaker contradiction
    assert b.verdict == "supported"                # belief bends, doesn't snap


def test_belief_model_update_is_bayesian():
    from nyxara.growth.scientist import Scientist as _S
    rep = _S().investigate("Is 17 a prime number?")
    model = BeliefModel()
    d = model.update(rep)
    b = model.get("is_prime(17)")
    assert b is not None and b.verdict == "supported"
    assert 0.0 <= b.confidence <= 1.0 and b.alpha > b.beta
