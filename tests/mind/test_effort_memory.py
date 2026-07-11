"""Tests for nyxara.mind.effort_memory — learned, compounding effort allocation.

The store must (a) bucket problems by *shape* not content, (b) learn from verified outcomes which
reasoning rung pays off for a signature, (c) refuse to suggest until it has enough evidence, and
(d) persist that lesson across a restart. No model weights, no network — pure learning.
"""

from __future__ import annotations

from pathlib import Path

from nyxara.mind.effort_memory import EffortMemory, signature


def test_signature_is_shape_stable_and_low_cardinality():
    # paraphrases of the same shape collapse to the same bucket
    assert signature("why is the sky blue?") == signature("why is the grass green?")
    # the shape axis separates the obvious kinds
    assert signature("what is 2+2").startswith("math/")
    assert signature("delete the production database").startswith("command/")
    assert signature("why is the sky blue?").startswith("question/")
    # stakes axis fires on dangerous content
    assert "high" in signature("shutdown the production server")
    assert "normal" in signature("what is your name?")
    # carries no raw stimulus text
    assert "sky" not in signature("why is the sky blue?")


def test_learns_winning_rung_from_verified_outcomes():
    em = EffortMemory(min_observations=3.0, success_floor=0.5)
    sig = signature("explain thermodynamics")
    assert em.suggest(sig) is None  # cold: nothing learned, controller spreads effort evenly

    for _ in range(5):
        em.record(sig, 3, 0.9)   # MCTS keeps producing strong verified answers
        em.record(sig, 1, 0.2)   # self-consistency keeps failing
    assert em.suggest(sig) == 3
    assert em.mean_for(sig, 3) > em.mean_for(sig, 1)


def test_suggestion_gated_on_evidence_and_floor():
    # too little evidence -> no suggestion even if the one sample was perfect
    em = EffortMemory(min_observations=5.0, success_floor=0.5)
    sig = signature("hello there")
    em.record(sig, 2, 1.0)
    assert em.suggest(sig) is None

    # enough evidence but everything is weak -> still no suggestion (nothing worth aiming at)
    em2 = EffortMemory(min_observations=2.0, success_floor=0.7)
    sig2 = signature("some hard prompt here friend")
    for _ in range(4):
        em2.record(sig2, 2, 0.3)
    assert em2.suggest(sig2) is None


def test_persistence_round_trips(tmp_path: Path):
    p = tmp_path / "effort.json"
    em = EffortMemory(min_observations=2.0, success_floor=0.4, path=p)
    sig = signature("why is the ocean salty?")
    for _ in range(4):
        em.record(sig, 3, 0.85)
    em.save()
    assert p.exists()

    reloaded = EffortMemory(min_observations=2.0, success_floor=0.4, path=p)
    assert reloaded.suggest(sig) == 3
    assert abs(reloaded.mean_for(sig, 3) - em.mean_for(sig, 3)) < 1e-9


def test_corrupt_file_never_breaks(tmp_path: Path):
    p = tmp_path / "effort.json"
    p.write_text("{ not valid json", encoding="utf-8")
    em = EffortMemory(path=p)  # must not raise
    assert em.suggest(signature("anything at all")) is None
