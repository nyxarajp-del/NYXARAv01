"""Tests for the bounded internal civilization — agency/hive_council.py (Part F)."""
from __future__ import annotations

from nyxara.agency.hive_council import HiveCouncil


def _wrong(_p: str) -> str:
    return "41"


def _right(_p: str) -> str:
    return "42"


def test_certified_winner_passes():
    hive = HiveCouncil(solvers=[_wrong, _right], max_agents=8)
    res = hive.solve("what is 6*7?", difficulty=0.5, budget=1.0)
    assert res.decided is True
    assert res.answer == "42" and "42" in res.certificate  # a re-checkable proof
    assert res.population >= 2


def test_no_certificate_abstains():
    # every solver is wrong -> nothing is provable -> fail-closed abstention
    hive = HiveCouncil(solvers=[_wrong, lambda p: "40"], max_agents=8)
    res = hive.solve("what is 6*7?")
    assert res.decided is False and res.reason.startswith("no proof")


def test_population_is_self_sized_and_bounded():
    hive = HiveCouncil(solvers=[_right], max_agents=10)
    # easy + low budget -> small; hard + full budget -> larger, never exceeding the cap
    assert hive.population_size(difficulty=0.0, budget=0.0) == 1
    assert hive.population_size(difficulty=1.0, budget=1.0) == 10
    assert hive.population_size(difficulty=0.5, budget=1.0) <= 10


def test_never_exceeds_resource_cap():
    hive = HiveCouncil(solvers=[_right], max_agents=4)
    res = hive.solve("what is 6*7?", difficulty=1.0, budget=1.0)
    assert res.population <= 4
