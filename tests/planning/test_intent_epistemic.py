"""Tests: IntentSystem's epistemic value is computed from the shared objective."""

from __future__ import annotations

import random

from nyxara.identity.affect import AffectSystem
from nyxara.mind.free_energy import FreeEnergyEngine
from nyxara.mind.world_model import WorldModel
from nyxara.planning.intent import IntentSystem


def _sparse_world() -> WorldModel:
    """A world model with real experience but a large unknown region."""
    wm = WorldModel(k=3, distance_scale=2.0)
    rng = random.Random(0)
    for _ in range(60):
        x = rng.uniform(-3.0, 0.0)
        a = rng.choice(["probe", "wait"])
        wm.observe((x,), a, (x + (1.0 if a == "probe" else 0.0),))
    return wm


def _deplete(affect: AffectSystem) -> None:
    for drive in affect.drives.values():
        drive.level = 0.1


def test_epistemic_computed_when_engine_wired():
    affect = AffectSystem()
    _deplete(affect)
    wm = _sparse_world()
    eng = FreeEnergyEngine(world_model=wm)
    intent = IntentSystem(affect, world_model=wm, free_energy=eng)
    cands = intent.generate_candidates()
    assert cands
    for dg in cands:
        assert 0.0 <= dg.epistemic <= 1.0
        assert dg.value == -dg.expected_free_energy    # pinned identity intact


def test_template_fallback_without_engine():
    affect = AffectSystem()
    _deplete(affect)
    plain = IntentSystem(affect)
    wired = IntentSystem(affect, world_model=_sparse_world(),
                         free_energy=FreeEnergyEngine(world_model=_sparse_world()))
    plain_epi = {c.drive: c.epistemic for c in plain.generate_candidates()}
    wired_epi = {c.drive: c.epistemic for c in wired.generate_candidates()}
    # the computed path respects the floor: never below 0.25 × template
    for drive, e in wired_epi.items():
        assert e >= 0.25 * plain_epi[drive] - 1e-9


def test_owner_first_selection_untouched():
    affect = AffectSystem()
    _deplete(affect)
    intent = IntentSystem(affect, world_model=_sparse_world(),
                          free_energy=FreeEnergyEngine(world_model=_sparse_world()))
    top = intent.select(k=1)
    assert top                                          # something was adopted
    assert top[0].drive in ("owner_connection", "safety")
