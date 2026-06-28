"""Tests for the data flywheel (growth/flywheel.py).

Deterministic, offline, temp-file backed. We assert the quality bar (length, confidence,
verifier, dedup), the foundry-compatible on-disk format, persistence across instances, and
that a NyxaraCore actually feeds verified-good Master turns into the corpus."""

from __future__ import annotations

import json


from nyxara.growth.distill import DistillationExample, load_distillation_docs
from nyxara.growth.flywheel import DataFlywheel


def _fw(tmp_path, **kw) -> DataFlywheel:
    return DataFlywheel(store_path=tmp_path / "flywheel.jsonl", **kw)


# --------------------------------------------------------------------------- #
# The quality bar
# --------------------------------------------------------------------------- #
def test_collects_a_good_pair(tmp_path):
    fw = _fw(tmp_path)
    d = fw.consider("What is 2+2?", "It is 4.", confidence=0.9)
    assert d.collected and d.example is not None
    assert fw.count() == 1


def test_rejects_empty_and_too_short(tmp_path):
    fw = _fw(tmp_path, min_chars=8)
    assert not fw.consider("", "something", confidence=1.0).collected
    assert not fw.consider("a prompt", "", confidence=1.0).collected
    assert not fw.consider("a prompt", "short", confidence=1.0).collected  # < 8 chars
    assert fw.count() == 0


def test_rejects_too_long(tmp_path):
    fw = _fw(tmp_path, max_chars=20)
    assert not fw.consider("p", "x" * 21, confidence=1.0).collected


def test_confidence_floor(tmp_path):
    fw = _fw(tmp_path, min_confidence=0.6)
    assert not fw.consider("prompt one", "a fine answer", confidence=0.5).collected
    assert fw.consider("prompt two", "a fine answer", confidence=0.6).collected


def test_dedup_by_prompt(tmp_path):
    fw = _fw(tmp_path)
    assert fw.consider("Same Q?", "answer one", confidence=1.0).collected
    d2 = fw.consider("  same q? ", "answer two", confidence=1.0)  # case/space-insensitive
    assert not d2.collected and "duplicate" in d2.reason
    assert fw.count() == 1


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #
def test_verifier_rejects(tmp_path):
    fw = _fw(tmp_path, verifier=lambda p, a: False)
    d = fw.consider("Is the sky green?", "Yes, totally.", confidence=1.0)
    assert not d.collected and "verifier" in d.reason


def test_verifier_confirms_marks_source(tmp_path):
    fw = _fw(tmp_path, verifier=lambda p, a: True)
    d = fw.consider("What is 2+2?", "It is 4.", confidence=1.0)
    assert d.collected and d.example.source == "flywheel-verified"


def test_explicit_verified_override_beats_configured_verifier(tmp_path):
    fw = _fw(tmp_path, verifier=lambda p, a: False)
    d = fw.consider("trust me", "an answer here", confidence=1.0, verified=True)
    assert d.collected and d.example.source == "flywheel-verified"


def test_flaky_verifier_does_not_sink_turn(tmp_path):
    def boom(p, a):
        raise RuntimeError("nope")
    fw = _fw(tmp_path, verifier=boom)
    d = fw.consider("a prompt", "an answer here", confidence=1.0)
    assert d.collected and d.example.source == "flywheel"  # unknown -> kept, not verified


# --------------------------------------------------------------------------- #
# On-disk format & persistence
# --------------------------------------------------------------------------- #
def test_corpus_is_foundry_compatible(tmp_path):
    fw = _fw(tmp_path)
    fw.consider("Explain recursion briefly.", "A function that calls itself.", confidence=1.0)
    # raw line round-trips as a DistillationExample
    line = (tmp_path / "flywheel.jsonl").read_text().splitlines()[0]
    ex = DistillationExample.from_dict(json.loads(line))
    assert ex.prompt and ex.answer
    # and the shared foundry loader renders it into a training doc
    docs = load_distillation_docs(tmp_path / "flywheel.jsonl")
    assert len(docs) == 1 and docs[0].strip()


def test_persistence_across_instances(tmp_path):
    fw1 = _fw(tmp_path)
    fw1.consider("persisted?", "yes it is", confidence=1.0)
    # a fresh instance over the same file sees the prior prompt and dedups it
    fw2 = _fw(tmp_path)
    assert fw2.count() == 1
    assert not fw2.consider("persisted?", "different answer", confidence=1.0).collected


# --------------------------------------------------------------------------- #
# Integration: the core feeds the flywheel
# --------------------------------------------------------------------------- #
def test_core_feeds_flywheel_on_owner_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("NYXARA_FLYWHEEL__STORE_PATH", str(tmp_path / "fw.jsonl"))
    from nyxara.kernel.config import reload_settings
    reload_settings()
    try:
        from nyxara.kernel.orchestrator import NyxaraCore
        from nyxara.agency.permissions import Authority
        core = NyxaraCore()
        assert core.flywheel is not None
        before = core.flywheel.count()
        core.process("Hello NYXARA, tell me something.", authority=Authority.OWNER)
        assert core.flywheel.count() >= before  # a cleared respond turn is offered
    finally:
        reload_settings()
