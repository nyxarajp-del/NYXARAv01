"""Tests for nyxara.growth.meta_meta — the recursive meta tower (improving HOW she improves).

Hermetic and offline: every genome is bounded, the 1+1 selection is seeded for determinism, and a
synthetic fitness environment stands in for real index gain. These tests pin the two properties the
redesign is *for*: the genome evolves the improvement ALGORITHM (not just execution knobs), and the
recursion compounds at every level.
"""

from __future__ import annotations

import random
from pathlib import Path

from nyxara.growth.meta_meta import (
    MetaGenome,
    MetaImprovementController,
    RecursiveMetaController,
    SearchGenome,
)


# --------------------------------------------------------------------------- #
# MetaGenome — bounded, mutatable, and applies the WHOLE algorithm
# --------------------------------------------------------------------------- #
def test_meta_genome_clamps_to_safe_envelope():
    # out-of-range values on construction are pulled back to the bounds (int + float knobs)
    g = MetaGenome(recursion_depth=99, max_edits_per_cycle=-5, ledger_reward_scale=9999.0,
                   ledger_prior_strength=-1.0, bandit_severity_blend=3.0,
                   intelligence_momentum=2.0)
    assert g.recursion_depth == 5 and g.max_edits_per_cycle == 1
    assert g.ledger_reward_scale == 200.0 and g.ledger_prior_strength == 0.25
    assert g.bandit_severity_blend == 1.0 and g.intelligence_momentum == 0.95


def test_meta_genome_mutate_is_a_real_neighbour():
    rng = random.Random(0)
    g = MetaGenome()
    for _ in range(50):
        m = g.mutate(rng, SearchGenome())
        assert m._values() != g._values()      # never burns a window on an identical twin
        # and the neighbour itself is in-envelope
        assert 1 <= m.recursion_depth <= 5 and 0.0 <= m.bandit_severity_blend <= 1.0


def test_meta_genome_apply_writes_every_knob():
    class _SI:
        llm_edit_recursion_depth = 0
        max_edits_per_cycle = 0
        ledger_reward_scale = 0.0
        ledger_prior_strength = 0.0
        bandit_severity_blend = 0.0
        intelligence_momentum = 0.0

    class _S:
        self_improvement = _SI()

    s = _S()
    g = MetaGenome(recursion_depth=4, max_edits_per_cycle=5, ledger_reward_scale=80.0,
                   ledger_prior_strength=2.0, bandit_severity_blend=0.4,
                   intelligence_momentum=0.5)
    g.apply(s)
    si = s.self_improvement
    assert si.llm_edit_recursion_depth == 4 and si.max_edits_per_cycle == 5
    assert si.ledger_reward_scale == 80.0 and si.ledger_prior_strength == 2.0
    assert si.bandit_severity_blend == 0.4 and si.intelligence_momentum == 0.5


# --------------------------------------------------------------------------- #
# A single rung climbs toward where reality pays off
# --------------------------------------------------------------------------- #
def _algo_env(g: MetaGenome) -> float:
    """A landscape where a HIGHER reward-scale + DEEPER recursion genuinely raise capability."""
    return 0.01 * g.recursion_depth + 0.0005 * g.ledger_reward_scale


def test_single_rung_evolves_the_algorithm():
    rung = MetaImprovementController(MetaGenome(recursion_depth=2, ledger_reward_scale=20.0),
                                     search=SearchGenome(window=2), seed=1)
    start_scale = rung.champion.ledger_reward_scale
    for _ in range(80):
        for _ in range(rung.window):
            rung.record(_algo_env(rung.active) + 0.0002 * (rung._rng.random() - 0.5))
        rung.maybe_evolve()
    # the ALGORITHM knob (reward scale) climbed, not only an execution knob — the whole point
    assert rung.champion.ledger_reward_scale > start_scale
    assert rung.champion.recursion_depth >= 2


# --------------------------------------------------------------------------- #
# The tower — real recursion at every level, compounding
# --------------------------------------------------------------------------- #
def test_tower_has_configured_height():
    assert len(RecursiveMetaController(height=1).levels) == 1
    assert len(RecursiveMetaController(height=3).levels) == 3
    # bounded — never starts past the safe ceiling (now the configurable hard_max_height, default 6)
    assert len(RecursiveMetaController(height=99).levels) == 6
    assert len(RecursiveMetaController(height=99, hard_max_height=3).levels) == 3


def _drive(tower: RecursiveMetaController, n: int) -> None:
    rng = tower.levels[0]._rng
    for _ in range(n):
        tower.record(_algo_env(tower.active) + 0.0002 * (rng.random() - 0.5))
        tower.maybe_evolve()


def test_tower_compounds_to_every_level():
    tower = RecursiveMetaController(height=3, seed=7)
    _drive(tower, 500)
    # the compounding signal must reach EVERY level — each higher rung actually evolved, which can
    # only happen if its level-below fed it selection-quality fitness across full windows
    for k, ctrl in enumerate(tower.levels):
        assert ctrl.generation >= 1, f"level {k} never evolved — recursion did not compound"
    # and the bottom champion climbed toward the paying region
    assert tower.champion.recursion_depth >= 3
    assert tower.champion.ledger_reward_scale > MetaGenome().ledger_reward_scale


def test_tower_levels_evolve_the_right_genome_types():
    tower = RecursiveMetaController(height=3, seed=2)
    assert isinstance(tower.levels[0].champion, MetaGenome)        # engine algorithm at the bottom
    for k in range(1, 3):
        assert isinstance(tower.levels[k].champion, SearchGenome)  # search-about-search above


def test_tower_apply_binds_active_engine_genome(tmp_path):
    class _SI:
        llm_edit_recursion_depth = 0
        max_edits_per_cycle = 0
        ledger_reward_scale = 0.0
        ledger_prior_strength = 0.0
        bandit_severity_blend = 0.0
        intelligence_momentum = 0.0

    class _S:
        self_improvement = _SI()

    tower = RecursiveMetaController(height=2, seed=4)
    s = _S()
    tower.apply(s)
    # the active (on-trial) engine genome is bound live into the config
    assert s.self_improvement.llm_edit_recursion_depth == tower.active.recursion_depth
    assert s.self_improvement.ledger_reward_scale == tower.active.ledger_reward_scale


def test_tower_persistence_round_trips_all_levels(tmp_path):
    path = Path(tmp_path) / "meta_meta.json"
    t1 = RecursiveMetaController(height=3, seed=3, persist_path=path)
    _drive(t1, 200)
    assert path.exists()
    gens = [c.generation for c in t1.levels]
    champ0 = t1.champion.to_dict()

    t2 = RecursiveMetaController(height=3, seed=99, persist_path=path)
    assert [c.generation for c in t2.levels] == gens
    assert t2.champion.to_dict() == champ0
    # the persisted search champions came back too
    assert t2.levels[1].champion.to_dict() == t1.levels[1].champion.to_dict()


# --------------------------------------------------------------------------- #
# Master-raisable growth ceilings (#6: bounds are configurable, never removed)
# --------------------------------------------------------------------------- #
def test_master_can_raise_ceilings_but_never_past_the_hard_guard():
    """The Master may grow more aggressively, but the immovable hard guard always caps it."""
    snap = (MetaGenome._ABS_MAX_RECURSION, MetaGenome._ABS_MAX_EDITS)
    try:
        # raising lifts the operating absolute ceiling
        r, e = MetaGenome.set_absolute_ceilings(recursion=32, edits=48)
        assert (r, e) == (32, 48)
        # the hard guard is immovable — no dial can push past it
        r, e = MetaGenome.set_absolute_ceilings(recursion=10_000, edits=10_000)
        assert r == MetaGenome._HARD_MAX_RECURSION and e == MetaGenome._HARD_MAX_EDITS
        # monotonic: a request below the current ceiling is ignored (never lowers a granted bound)
        r, e = MetaGenome.set_absolute_ceilings(recursion=1, edits=1)
        assert r == MetaGenome._HARD_MAX_RECURSION and e == MetaGenome._HARD_MAX_EDITS
    finally:
        MetaGenome._ABS_MAX_RECURSION, MetaGenome._ABS_MAX_EDITS = snap
