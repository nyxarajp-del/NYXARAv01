"""Tests for the law recall/apply bridge (Gap B, engine side).

A discovered law is useless if it only sits on disk. ``recall`` finds the law relevant to a
question and ``apply`` evaluates it at concrete inputs — the link that lets self-discovered
science ground an answer. Hermetic, no LLM.
"""

from __future__ import annotations

from nyxara.growth.law_discovery import LawDiscoveryEngine


def _engine_with_product_law() -> LawDiscoveryEngine:
    eng = LawDiscoveryEngine(seed=7)
    X, y = [], []
    for m in range(1, 9):
        for a in range(1, 9):
            X.append([float(m), float(a)])
            y.append(float(m * a))
    eng.discover_from_data(X, y, var_names=["m", "a"], target_name="force")
    return eng


def test_a_law_is_discovered_and_held():
    eng = _engine_with_product_law()
    assert len(eng.known_laws()) >= 1


def test_recall_finds_the_relevant_law_by_variables():
    eng = _engine_with_product_law()
    law = eng.recall("what is the force from mass m and acceleration a?")
    assert law is not None
    assert law.target == "force"


def test_recall_returns_none_when_nothing_matches():
    eng = _engine_with_product_law()
    assert eng.recall("what is the capital of France?") is None


def test_apply_evaluates_the_law_at_concrete_inputs():
    eng = _engine_with_product_law()
    res = eng.apply({"m": 3.0, "a": 4.0}, query="force from m and a")
    assert res is not None
    assert res["target"] == "force"
    assert abs(res["value"] - 12.0) < 1e-6


def test_apply_returns_none_without_the_needed_variables():
    eng = _engine_with_product_law()
    # the law needs both m and a; only m supplied
    assert eng.apply({"m": 3.0}) is None


def test_apply_ignores_non_numeric_inputs():
    eng = _engine_with_product_law()
    assert eng.apply({"m": "not a number", "a": 4.0}) is None


# --------------------------------------------------------------------------- #
# Relevance — a law must answer the question that was asked
# --------------------------------------------------------------------------- #
# Observed in a live session: `"the status is good"` was answered with *"From my own experiments
# I discovered the law fall_distance = 0.5197*((t)^2 * g) + ..."*. Not a dishonesty bug — she
# really had discovered that law — a relevance bug. The match was an unanchored substring in
# either direction, so the token `is` matched the target `fall_d(is)tance`, and any sentence
# containing a short common word recalled a physics law.
from nyxara.growth.law_discovery import Law, _word_parts        # noqa: E402


def _engine_with(target: str, var_names: list) -> LawDiscoveryEngine:
    """An engine holding exactly one law, so a recall names it or nothing."""
    eng = LawDiscoveryEngine.__new__(LawDiscoveryEngine)
    eng._laws = [Law(target=target, expression="0.5197*((t)^2 * g)",
                     var_names=list(var_names), complexity=5)]
    return eng


def _falling() -> LawDiscoveryEngine:
    return _engine_with("fall_distance", ["t", "g"])


def test_a_common_word_inside_the_target_does_not_recall_a_law():
    """`is` lives inside `fall_d-is-tance`. That is a spelling coincidence, not a question
    about falling bodies."""
    assert _falling().recall("the status is good") is None


def test_arithmetic_does_not_summon_physics():
    assert _falling().recall("what is 2+2?") is None


def test_a_greeting_does_not_summon_physics():
    assert _falling().recall("Hii") is None
    assert _falling().recall("hello there") is None


def test_a_one_letter_variable_is_not_evidence_of_relevance():
    """Laws name their inputs `t`, `g`, `a` — and `a` is an English word. Concrete inputs reach
    the law through ``apply``/``_extract_var_values``, which does not depend on this scoring."""
    eng = _engine_with("velocity", ["a", "t"])
    assert eng.recall("give me a hand") is None


def test_the_law_is_still_recalled_by_its_own_words():
    """The fix must not make the faculty unreachable — that would trade a wrong answer for no
    answer, which is the other way to be useless."""
    eng = _falling()
    assert eng.recall("how far does it fall in time t under gravity g?") is not None
    assert eng.recall("compute the fall distance") is not None
    assert eng.recall("what is the fall_distance?") is not None


def test_a_multi_character_variable_still_counts():
    eng = _engine_with("force", ["mass", "accel"])
    assert eng.recall("what force comes from mass and accel?") is not None


def test_the_original_reported_query_returns_nothing():
    """The exact string from the session that surfaced this."""
    assert _falling().recall("the status is good") is None


# ---- the helper the match is built on ---- #
def test_word_parts_splits_on_underscores():
    assert _word_parts("fall_distance") == {"fall", "distance"}


def test_word_parts_splits_camel_case():
    assert _word_parts("fallDistance") == {"fall", "distance"}


def test_word_parts_drops_single_characters():
    """A one-character part is not evidence — it is how `t` matched half the dictionary."""
    assert _word_parts("x_velocity") == {"velocity"}


def test_word_parts_survives_odd_input():
    assert _word_parts("") == set() and _word_parts(None) == set()
