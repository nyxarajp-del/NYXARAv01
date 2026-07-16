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
