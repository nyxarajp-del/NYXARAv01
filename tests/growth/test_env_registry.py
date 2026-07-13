"""Tests for nyxara.growth.env_registry — the persistent memory of cracked environments.

Adaptation is remembering, not amnesia: the second time NYXARA meets a black box she already cracked,
she should recognise it (a few cheap confirmation probes) and reuse the saved law instead of
re-deriving it. These tests are hermetic (a tmp store) and offline (no LLM, no network).
"""
from __future__ import annotations

from nyxara.growth.env_registry import EnvironmentProfile, EnvironmentRegistry
from nyxara.growth.open_world import DomainSpec, OpenWorldGeneralizer, Verdict


def _line(x):  # a simple, cleanly-modellable box
    return 3 * x - 2


def test_profile_round_trips():
    prof = EnvironmentProfile(label="line", domain=DomainSpec(1, "real").to_dict(),
                              law_family="affine", law_description="out = -2 + 3·x[0]",
                              coeffs={"w": [-2.0, 3.0]}, confidence=0.9,
                              fingerprint=[[1.0, 1.0], [2.0, 4.0], [3.0, 7.0]])
    back = EnvironmentProfile.from_dict(prof.to_dict())
    assert back.label == "line" and back.law_family == "affine"
    assert back.coeffs == {"w": [-2.0, 3.0]}
    pred = back.rebuild()
    assert pred is not None and abs(pred((10.0,)) - (3 * 10 - 2)) < 1e-9


def test_build_persist_and_reload(tmp_path):
    store = tmp_path / "env_registry.json"
    reg = EnvironmentRegistry(path=store, min_checks=3)
    assert len(reg) == 0
    spec = DomainSpec(1, "real")
    rep = OpenWorldGeneralizer(seed=7).understand(_line, domain=spec, label="line-A")
    prof = reg.build_profile(rep)
    assert prof is not None and prof.law_family == "affine"
    reg.save(prof)
    assert store.exists()
    # a brand-new registry reads it back from disk
    reg2 = EnvironmentRegistry(path=store, min_checks=3)
    assert len(reg2) == 1 and reg2.all()[0].label == "line-A"


def test_same_box_is_recognized_cheaply(tmp_path):
    store = tmp_path / "env_registry.json"
    reg = EnvironmentRegistry(path=store, min_checks=3)
    spec = DomainSpec(1, "real")
    # first encounter: full inquiry (many probes) and a saved profile
    first = OpenWorldGeneralizer(seed=7, registry=reg).understand(_line, domain=spec, label="line")
    assert first.verdict is Verdict.MODELLED and first.recognized is False
    assert len(reg) == 1

    # second encounter with a fresh generalizer bound to the SAME registry: recognised, not re-cracked
    second = OpenWorldGeneralizer(seed=1, registry=reg).understand(_line, domain=spec, label="line")
    assert second.verdict is Verdict.MODELLED
    assert second.recognized is True
    assert second.probes_used < first.probes_used          # recognition is far cheaper
    assert second.predict(1000) == first.predict(1000)     # the reused law still generalizes


def test_different_box_is_not_a_false_match(tmp_path):
    store = tmp_path / "env_registry.json"
    reg = EnvironmentRegistry(path=store, min_checks=3)
    spec = DomainSpec(1, "real")
    OpenWorldGeneralizer(seed=7, registry=reg).understand(_line, domain=spec, label="line")
    # a genuinely different system must NOT be mistaken for the remembered one
    assert reg.match(lambda x: 9 * x + 5, spec) is None
    other = OpenWorldGeneralizer(seed=2, registry=reg).understand(
        lambda x: 9 * x + 5, domain=spec, label="other")
    assert other.recognized is False
    assert len(reg) == 2          # it was cracked fresh and remembered separately


def test_unmodellable_box_is_not_remembered(tmp_path):
    import random
    store = tmp_path / "env_registry.json"
    reg = EnvironmentRegistry(path=store, min_checks=3)
    rng = random.Random(123)
    rep = OpenWorldGeneralizer(seed=7, registry=reg).understand(
        lambda x: rng.random() * 100, domain=DomainSpec(1, "real"), label="noise")
    assert rep.verdict is Verdict.UNMODELLED
    assert len(reg) == 0          # she never persists a memory she could not model
