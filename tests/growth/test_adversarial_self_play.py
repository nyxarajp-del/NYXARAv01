"""Tests for nyxara.growth.adversarial_self_play — clone-vs-clone self-play (Pillar F)."""

from __future__ import annotations

from pathlib import Path

from nyxara.growth.adversarial_self_play import (
    AdversarialSelfPlay,
    Elo,
    StrategyBandit,
    _match,
    _norm,
)
from nyxara.growth.flywheel import DataFlywheel
from nyxara.growth.weakness import WeaknessReport


def _asp(tmp_path: Path, **kw) -> AdversarialSelfPlay:
    fw = DataFlywheel(store_path=tmp_path / "adversarial.jsonl")
    return AdversarialSelfPlay(flywheel=fw, seed=11, **kw)


# --------------------------------------------------------------------------- #
# grading helpers
# --------------------------------------------------------------------------- #
def test_norm_is_numeric_and_punctuation_insensitive():
    assert _norm("42.") == "42"
    assert _norm(" 42 ") == _norm("42")
    assert _norm("1,000") == "1000"
    assert _norm(None) is None


def test_match_grades_by_exact_value():
    assert _match("42", "42.0") is True
    assert _match("The answer is 42", "42") is False   # raw text != bare value
    assert _match(None, "42") is False


# --------------------------------------------------------------------------- #
# the contest is real and oracle-graded
# --------------------------------------------------------------------------- #
def test_compete_runs_and_returns_wellformed_stats(tmp_path):
    rep = _asp(tmp_path).compete(rounds=120)
    assert rep["played"] >= 100
    assert 0.0 <= rep["defender_win_rate"] <= 1.0
    assert set(rep["per_category"]).issubset(
        {"arithmetic", "percent", "algebra", "sequence", "date", "multi_step"})
    assert rep["elo"]["defender"] != rep["elo"]["attacker"]  # the rating actually moved


def test_defender_solves_verifiable_problems_without_a_teacher(tmp_path):
    rep = _asp(tmp_path).compete(rounds=200)
    # the oracle is exact (no LLM), so a competent defender wins the majority verifiably
    assert rep["defender_win_rate"] > 0.5
    # and the verified-solved pairs were folded into the foundry corpus
    assert rep["collected"] > 0


def test_strong_generalist_strategy_is_learned_to_dominate(tmp_path):
    rep = _asp(tmp_path).compete(rounds=250)
    defend = {s["name"]: s["win_rate"] for s in rep["discovered_strategies"]["defender"]}
    # the bandit should discover that the strong faculties→chain router beats the narrow tools
    assert defend.get("auto", 0.0) >= defend.get("faculties", 0.0)
    assert defend.get("auto", 0.0) >= defend.get("chain", 0.0)


def test_lost_categories_become_weaknesses_and_topics(tmp_path):
    asp = _asp(tmp_path)
    asp.compete(rounds=150)
    report = asp.to_weaknesses()
    assert isinstance(report, WeaknessReport)
    for w in report.weaknesses:
        assert w.id.startswith("adversarial:")
        assert 0.0 <= w.severity <= 1.0
    # every surfaced topic corresponds to a not-yet-mastered category
    for topic in asp.to_topics():
        assert topic.startswith("hard ") and topic.endswith(" problems")


def test_contest_is_deterministic_for_a_fixed_seed(tmp_path):
    a = AdversarialSelfPlay(flywheel=DataFlywheel(store_path=tmp_path / "a.jsonl"),
                            seed=11).compete(rounds=180)
    b = AdversarialSelfPlay(flywheel=DataFlywheel(store_path=tmp_path / "b.jsonl"),
                            seed=11).compete(rounds=180)
    assert a["defender_wins"] == b["defender_wins"]
    assert a["per_category"] == b["per_category"]


def test_max_seconds_bounds_the_contest(tmp_path):
    rep = _asp(tmp_path).compete(rounds=10_000_000, max_seconds=0.25)
    assert rep["played"] < 10_000_000   # stopped on the time budget, not the round budget


# --------------------------------------------------------------------------- #
# the learning primitives
# --------------------------------------------------------------------------- #
def test_elo_rewards_the_winner():
    elo = Elo()
    for _ in range(20):
        elo.update(defender_won=True)
    assert elo.defender > 1000.0 > elo.attacker


def test_bandit_learns_a_winning_arm():
    import random
    bandit = StrategyBandit(rng=random.Random(0))
    for _ in range(40):
        bandit.record("defend:good", 1.0)
        bandit.record("defend:bad", 0.0)
    assert bandit.mean("defend:good") > 0.8 > bandit.mean("defend:bad")
    wins = sum(1 for _ in range(200)
               if bandit.choose("defend:", ["good", "bad"]) == "good")
    assert wins > 150
