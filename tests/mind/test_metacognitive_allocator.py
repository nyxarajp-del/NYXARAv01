"""Tests for nyxara.mind.metacognitive_allocator — first-class metacognition.

Calibrated uncertainty must DRIVE compute allocation: an unknown/hard problem earns the full
ladder (up to the ten-minute search); a lived-easy one earns a single forward pass; high stakes
never get the shortcut; the value-of-computation stopper halts a fruitless climb and escalation
raises a budget when a turn measures harder than predicted; and the whole thing is calibrated,
persistent, and never raises. No network, no weights — her own signals only.
"""

from __future__ import annotations

from pathlib import Path

from nyxara.mind.effort_memory import signature
from nyxara.mind.metacognitive_allocator import (MetacognitiveAllocator, TurnBudget,
                                                 get_allocator)


class _Ceiling:
    max_rung, samples, max_seconds = 4, 9, 120.0


def _teach_easy(a, q, n=12):
    for _ in range(n):
        a.record(q, predicted_uncertainty=0.8, target_score=0.8, scores=[(1, 0.9)],
                 winning_rung=1)


def _teach_hard(a, q, n=12):
    for _ in range(n):
        a.record(q, predicted_uncertainty=0.3, target_score=0.8,
                 scores=[(1, 0.3), (2, 0.5), (3, 0.85)], winning_rung=3)


def test_unknown_problem_defaults_to_thinking_hard():
    a = MetacognitiveAllocator()
    b = a.allocate("why is the sky blue?", ceiling=_Ceiling())
    assert b.predicted_uncertainty >= 0.70   # epistemic humility, never assumes cheap
    assert b.max_rung == 4 and b.max_seconds >= 600.0


def test_lived_easy_gets_one_forward_pass():
    a = MetacognitiveAllocator(easy_uncertainty=0.25, hard_uncertainty=0.70)
    q = "why is the sky blue?"
    _teach_easy(a, q)
    b = a.allocate(q, ceiling=_Ceiling(), mems_count=5)
    assert b.predicted_uncertainty <= 0.25
    assert b.max_rung == 1 and b.samples == 1 and b.max_seconds <= 5.0
    assert "1 pass" in b.reason


def test_lived_hard_gets_full_ladder_and_ten_minute_search():
    a = MetacognitiveAllocator(easy_uncertainty=0.25, hard_uncertainty=0.70)
    q = "prove the polynomial identity holds for every positive integer n"
    _teach_hard(a, q)
    b = a.allocate(q, ceiling=_Ceiling(), mems_count=5)
    assert b.predicted_uncertainty >= 0.70
    assert b.max_rung == 4 and b.max_seconds >= 600.0


def test_high_stakes_never_shortcut_to_one_pass():
    a = MetacognitiveAllocator(easy_uncertainty=0.25, hard_uncertainty=0.70)
    risky = "delete the production database now"
    _teach_easy(a, risky)   # even a perfectly easy history
    b = a.allocate(risky, ceiling=_Ceiling(), mems_count=5)
    assert b.max_rung == 4, "high stakes must climb the ladder, not shortcut"


def test_hierarchical_backoff_borrows_from_shape():
    a = MetacognitiveAllocator(easy_uncertainty=0.25, hard_uncertainty=0.70)
    q = "why is the sky blue?"
    _teach_easy(a, q, n=20)
    # a genuinely NEW signature of the same 'question' shape borrows the shape's easy evidence
    cousin = a.assess("what makes the grass appear green, and why do autumn leaves turn a "
                      "warm shade of yellow and red across the whole hillside every year?",
                      mems_count=5)
    assert cousin.signature != signature(q)
    assert cousin.uncertainty < 0.7   # not treated as fully unknown


def test_calibration_is_measured_and_recorded():
    a = MetacognitiveAllocator()
    q = "why is the sky blue?"
    _teach_easy(a, q, n=20)
    rep = a.report()
    assert rep["calibration"]["samples"] >= 20
    assert 0.0 <= rep["calibration"]["ece"] <= 1.0
    assert rep["turns"] >= 20 and rep["difficulty"]


def test_voc_stops_when_marginal_value_collapses():
    a = MetacognitiveAllocator(voc_stopping=True)
    q = "why is the sky blue?"
    # a shape where the deeper rung never improves the answer → its expected gain collapses
    for _ in range(8):
        a.record(q, predicted_uncertainty=0.5, target_score=0.95,
                 scores=[(1, 0.85), (2, 0.85), (3, 0.85)], winning_rung=1)
    budget = TurnBudget(max_rung=4, samples=3, max_seconds=120.0, target_score=0.95,
                        predicted_uncertainty=0.5, signature=signature(q))
    d = a.should_continue(budget=budget, next_rung=2, best_score=0.85, elapsed=1.0)
    assert d.stop and "value" in d.reason


def test_target_met_stops_immediately():
    a = MetacognitiveAllocator()
    budget = TurnBudget(max_rung=4, samples=3, max_seconds=120.0, target_score=0.8,
                        predicted_uncertainty=0.5, signature="question/short/normal")
    d = a.should_continue(budget=budget, next_rung=2, best_score=0.9, elapsed=1.0)
    assert d.stop and not d.escalate and "target met" in d.reason


def test_live_escalation_on_weak_first_pass():
    a = MetacognitiveAllocator(live_escalation=True)
    cheap = TurnBudget(max_rung=1, samples=1, max_seconds=5.0, target_score=0.8,
                       predicted_uncertainty=0.1, signature="question/short/normal")
    d = a.should_continue(budget=cheap, next_rung=2, best_score=0.3, elapsed=0.5)
    assert d.escalate and not d.stop
    raised = a.escalate(cheap, ceiling=_Ceiling())
    assert raised.escalated and raised.max_rung == 4 and raised.max_seconds >= 600.0


def test_time_budget_stops_the_climb():
    a = MetacognitiveAllocator()
    budget = TurnBudget(max_rung=4, samples=3, max_seconds=10.0, target_score=0.9,
                        predicted_uncertainty=0.8, signature="math/long/normal")
    # already 9s spent; the next rung's estimated cost would blow the 10s budget
    d = a.should_continue(budget=budget, next_rung=3, best_score=0.5, elapsed=9.0)
    assert d.stop and "time budget" in d.reason


def test_calibration_shifts_prediction_toward_reality():
    """After many outcomes, predicted hardness tracks empirical hardness (calibration works)."""
    a = MetacognitiveAllocator()
    q = "explain the cause of the daytime sky colour"
    # a kind that LOOKS distinct each turn but is consistently easy — the calibrator learns it
    for _ in range(30):
        a.record(q, predicted_uncertainty=0.6, target_score=0.8, scores=[(1, 0.95)],
                 winning_rung=1)
    # the difficulty belief for this signature is now firmly "easy"
    assert a.assess(q).uncertainty < 0.5


def test_persistence_round_trip(tmp_path):
    p = tmp_path / "meta.json"
    a = MetacognitiveAllocator(path=p)
    q = "why is the sky blue?"
    _teach_easy(a, q)
    a.save()
    again = MetacognitiveAllocator(path=p)
    assert again.allocate(q, ceiling=_Ceiling(), mems_count=5).max_rung == 1
    assert again.calibrator.samples == a.calibrator.samples


def test_shared_allocator_is_one_instance_per_datadir():
    # two calls with default settings return the SAME shared object (one calibration source)
    a1 = get_allocator()
    a2 = get_allocator()
    assert a1 is a2
    assert hasattr(a1, "answer_calibrator")


def test_never_raises_on_garbage():
    a = MetacognitiveAllocator()
    a.record("", predicted_uncertainty=float("nan"), target_score=0.8, scores=[], winning_rung=0)
    b = a.allocate("", ceiling=object())
    assert b.max_rung >= 1
    assert a.assess("").uncertainty >= 0.0
