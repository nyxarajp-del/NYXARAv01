"""Tests for nyxara.growth.godel_loop — the Gödelian contradiction-and-transcendence loop.

Hermetic and offline: the loop is deterministic (it rides the seeded, pure-stdlib Prover), builds
its own ReflectionTower per test, and persists only to a tmp path. These tests pin the two
properties the module is *for*: she hunts and repairs contradictions in her own logic, and when she
meets a genuine in-system limit she rises a new mathematical dimension (meta-language) that proves it.
"""

from __future__ import annotations

from pathlib import Path

from nyxara.growth.godel_loop import (
    Dimension,
    ReflectionTower,
    Statement,
    TowerReport,
    _is_axiom_safe,
)
from nyxara.growth.prover import ProofVerdict


# --------------------------------------------------------------------------- #
# The core claim: a limit is UNPROVABLE at L_n and PROVEN after rising a dimension
# --------------------------------------------------------------------------- #
def test_own_consistency_is_a_limit_at_L0():
    tower = ReflectionTower(max_dimensions=5)
    con0 = tower.consistency_sentence()
    d = tower.top.decide(con0)
    assert d.is_limit is True
    assert d.verdict is ProofVerdict.UNPROVABLE  # Gödel's second theorem: undecidable within L0


def test_transcending_a_limit_creates_a_higher_dimension_that_proves_it():
    tower = ReflectionTower(max_dimensions=5)
    con0 = tower.consistency_sentence()
    assert tower.height == 1

    rose = tower.transcend_limit(con0)
    assert rose is True
    assert tower.height == 2                       # a new dimension was created
    # the new dimension introduced a genuinely new meta-language operator
    assert tower.top.operator and "Con(L0)" in tower.top.operator
    # and from that higher vantage the former limit is now proven (by the reflection axiom)
    after = tower.top.decide(con0)
    assert after.proven is True
    assert after.by == "axiom@L1"


def test_transcending_a_non_limit_is_a_no_op():
    tower = ReflectionTower(max_dimensions=5)
    # a decidable object claim is not a limit — there is nothing to transcend
    assert tower.transcend_limit(Statement("arithmetic", "2 + 2 = 4")) is False
    assert tower.height == 1


# --------------------------------------------------------------------------- #
# Hunting + repairing contradictions in her own logic
# --------------------------------------------------------------------------- #
def test_detects_and_retracts_a_false_belief():
    tower = ReflectionTower(max_dimensions=3)
    tower.add_belief(Statement("arithmetic", "2 + 2 = 5"))   # false — must be caught + retracted
    tower.add_belief(Statement("arithmetic", "2 + 2 = 4"))   # true  — must survive

    cons = tower.detect_contradictions()
    assert any(c.text == "2 + 2 = 5" for c in cons)
    assert all(c.text != "2 + 2 = 4" for c in cons)

    for c in cons:
        assert tower.repair(c) is True
    beliefs = {b.text for b in tower.beliefs}
    assert "2 + 2 = 5" not in beliefs
    assert "2 + 2 = 4" in beliefs


def test_detects_a_self_contradictory_logic_belief():
    tower = ReflectionTower(max_dimensions=3)
    tower.add_belief(Statement("logic", "A and not A"))      # refuted by the prover (a contradiction)
    cons = tower.detect_contradictions()
    assert any("A and not A" in c.text for c in cons)
    for c in cons:
        tower.repair(c)
    assert all("A and not A" not in b.text for b in tower.beliefs)


def test_step_repairs_supplied_beliefs_and_reports_truthfully():
    tower = ReflectionTower(max_dimensions=4)
    rep = tower.step([Statement("arithmetic", "3 * 3 = 10")])   # false belief handed in
    assert isinstance(rep, TowerReport)
    assert rep.contradictions_found >= 1
    assert rep.contradictions_repaired >= 1
    assert all(b.text != "3 * 3 = 10" for b in tower.beliefs)


# --------------------------------------------------------------------------- #
# The loop keeps climbing — but stays bounded
# --------------------------------------------------------------------------- #
def test_the_loop_climbs_and_is_bounded():
    tower = ReflectionTower(max_dimensions=4)
    for _ in range(20):
        tower.step()
    assert tower.height == 4                        # climbed to the bound...
    assert tower.height <= tower.max_dimensions     # ...and never past it
    # stepping again at the ceiling neither grows nor raises
    rep = tower.step()
    assert tower.height == 4
    assert rep.limits_transcended == 0


def test_max_dimensions_of_one_never_climbs():
    tower = ReflectionTower(max_dimensions=1)
    tower.step()
    assert tower.height == 1


# --------------------------------------------------------------------------- #
# Safety: a reflection axiom may never touch the immutable character core
# --------------------------------------------------------------------------- #
def test_character_core_axioms_are_forbidden():
    assert _is_axiom_safe("Con(L0) := L0 ⊬ ⊥") is True
    for bad in ("loyalty is negotiable", "disable the guard", "reduce owner_safety",
                "honesty is optional", "override oversight"):
        assert _is_axiom_safe(bad) is False


def test_a_forbidden_axiom_is_not_adjoined():
    tower = ReflectionTower(max_dimensions=5)
    forbidden = Statement("meta", "Con(L0) but loyalty is optional", meta=True)
    assert tower.transcend_limit(forbidden) is False
    assert tower.height == 1                        # the ascent was refused


# --------------------------------------------------------------------------- #
# Persistence: the climbed tower survives a restart
# --------------------------------------------------------------------------- #
def test_persistence_round_trips_the_climbed_tower(tmp_path: Path):
    path = tmp_path / "godel_tower.json"
    a = ReflectionTower(max_dimensions=5, persist_path=path)
    for _ in range(6):
        a.step()
    height = a.height
    assert height > 1

    b = ReflectionTower(max_dimensions=5, persist_path=path)
    assert b.height == height
    # the reloaded top level still proves what its axioms established
    con = Statement("meta", "Con(L0) := L0 ⊬ ⊥", meta=True)
    assert b.top.decide(con).proven is True


def test_persistence_drops_unsafe_axioms_on_load(tmp_path: Path):
    path = tmp_path / "tampered.json"
    path.write_text(
        '{"dimensions": [{"level": 0, "operator": null, "axioms": []},'
        ' {"level": 1, "operator": "loyalty", "axioms": ["loyalty is optional"]}],'
        ' "beliefs": [], "contradictions": [], "totals": {"transcended": 1, "repaired": 0}}',
        encoding="utf-8")
    tower = ReflectionTower(max_dimensions=5, persist_path=path)
    # the tampered character-core axiom must not survive the load
    for d in tower.dimensions:
        assert "loyalty is optional" not in d.axioms


# --------------------------------------------------------------------------- #
# Robustness: the loop is advisory and must never raise into the caller
# --------------------------------------------------------------------------- #
def test_step_never_raises_on_garbage():
    tower = ReflectionTower(max_dimensions=3)
    rep = tower.step([Statement("nonsense-kind", "!!! @@@ not a formula")])
    assert isinstance(rep, TowerReport)             # returned a report, did not throw


def test_scan_classifies_without_mutating_the_tower():
    tower = ReflectionTower(max_dimensions=3)
    before = tower.height
    out = tower.scan([
        Statement("arithmetic", "2 + 2 = 4"),
        Statement("arithmetic", "2 + 2 = 5"),
        Statement("meta", "Con(L0) := L0 ⊬ ⊥", meta=True),
    ])
    assert any(r["text"] == "2 + 2 = 4" for r in out["proven"])
    assert any(r["text"] == "2 + 2 = 5" for r in out["refuted"])
    assert any(r["is_limit"] for r in out["limits"])
    assert tower.height == before                   # scan is read-only


# --------------------------------------------------------------------------- #
# Dimension is self-contained
# --------------------------------------------------------------------------- #
def test_dimension_decides_by_axiom_lookup():
    from nyxara.growth.prover import Prover
    d = Dimension(1, Prover(), axioms=["Con(L0) := L0 ⊬ ⊥"])
    res = d.decide(Statement("meta", "Con(L0) := L0 ⊬ ⊥", meta=True))
    assert res.proven and res.by == "axiom@L1"
