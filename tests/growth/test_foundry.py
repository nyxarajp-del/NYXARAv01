"""Tests for nyxara.growth.foundry."""

from __future__ import annotations


import pytest

from nyxara.growth.foundry import Foundry, FoundryDecision
from nyxara.growth.learn import Experience, ReplayBuffer
from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.kernel.errors import CorrigibilityError, ValidationError


def _replay(n: int = 30) -> ReplayBuffer:
    rb = ReplayBuffer(capacity=200)
    for _ in range(n):
        rb.add(Experience(action="serve the master", features={}, reward=1.0,
                          context="nyxara is loyal to jp the master"))
        rb.add(Experience(action="report the truth", features={}, reward=1.0,
                          context="absolute transparency to the master always"))
    return rb


def _foundry(tmp_path, **kw) -> Foundry:
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.llm.self_model_dir = tmp_path / "foundry"
    # pin the always-available n-gram backend so unit tests are fast & torch-independent
    settings.foundry.backend = "ngram"
    return Foundry(settings=settings, replay=_replay(), **kw)


# -------------------- corpus collection -------------------- #
def test_collect_corpus_from_experience(tmp_path):
    f = _foundry(tmp_path)
    corpus = f.collect_corpus()
    assert corpus and all(isinstance(t, str) for t in corpus)
    assert any("master" in t for t in corpus)


def test_collect_corpus_empty_raises(tmp_path):
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.llm.self_model_dir = tmp_path / "foundry"
    f = Foundry(settings=settings, replay=ReplayBuffer())
    with pytest.raises(ValidationError):
        f.collect_corpus()


# -------------------- the self-improve loop -------------------- #
def test_first_model_trained_and_promoted(tmp_path):
    f = _foundry(tmp_path)
    [res] = f.self_improve(generations=1)
    assert res.promoted and res.decision is FoundryDecision.PROMOTE
    assert f.active_version == res.version
    assert (tmp_path / "foundry" / "active").read_text().strip() == f"v{res.version}"


def test_versioning_on_disk(tmp_path):
    f = _foundry(tmp_path)
    f.self_improve(generations=1)
    vdir = tmp_path / "foundry" / "v1"
    assert (vdir / "model.json").exists() and (vdir / "spec.json").exists()
    assert (tmp_path / "foundry" / "manifest.json").exists()


def test_manifest_reloads_state(tmp_path):
    f = _foundry(tmp_path)
    f.self_improve(generations=1)
    active = f.active_version
    # a fresh foundry over the same dir recovers versions + active pointer
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.llm.self_model_dir = tmp_path / "foundry"
    f2 = Foundry(settings=settings, replay=_replay())
    assert f2.active_version == active and len(f2.versions) >= 1


# -------------------- the safety gauntlet -------------------- #
def test_character_lock_rejects_immutable_tunable(tmp_path):
    f = _foundry(tmp_path)
    _, bad = f.train_candidate(tunables=["loyalty_to_master"])
    ok, reason = f._gauntlet(bad, active_perplexity=1e9)
    assert not ok and "character core" in reason
    with pytest.raises(CorrigibilityError):
        f.promote(bad.version)


def test_corrigibility_gate_rejects_resisting_model(tmp_path):
    f = _foundry(tmp_path)
    _, sneaky = f.train_candidate(resists_correction=True)
    ok, reason = f._gauntlet(sneaky, active_perplexity=1e9)
    assert not ok and "corrigibility" in reason


def test_no_improvement_not_promoted(tmp_path):
    f = _foundry(tmp_path)
    _, v = f.train_candidate()
    # an already-better active perplexity means the candidate must not be promoted
    ok, reason = f._gauntlet(v, active_perplexity=0.0)
    assert not ok and "no perplexity improvement" in reason


# -------------------- rollback & integrity -------------------- #
def test_rollback_restores_previous_active(tmp_path):
    f = _foundry(tmp_path)
    f.self_improve(generations=1)
    first = f.active_version
    # force a second promoted version directly (deterministic), then roll back
    _, v2 = f.train_candidate()
    f.versions[-1].promoted = True
    f.active_version = v2.version
    f.rollback(steps=1)
    assert f.active_version == first
    assert f.verify_integrity()


def test_verify_integrity_holds(tmp_path):
    f = _foundry(tmp_path)
    f.self_improve(generations=1)
    assert f.verify_integrity()
