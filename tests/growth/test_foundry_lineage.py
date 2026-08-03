"""Regression tests for the two things that silently break when a trained tokenizer arrives.

Both are in ``growth/foundry.py`` and both fail in a way that looks like success:

1. **The perplexity gate lies across vocabularies.** Perplexity is per token, so a byte-level
   model reports a much *lower* number than a sub-word model while being a much worse brain.
   Under the unguarded gate a genuinely better 300M model would be refused promotion forever.
2. **``_warm_start`` half-loads on a shape mismatch.** A 137k byte-level nano-GPT and a 300M NYX
   decoder are both ``kind="nanogpt"`` and share state-dict key names. Without an architecture
   check, ``model.load`` mutates the fresh net partway through and *then* raises — and the
   caller trains a half-overwritten model believing it started clean.
"""

from __future__ import annotations

import math

import pytest

from nyxara.growth.foundry import Foundry, ModelVersion
from nyxara.growth.foundry_models import ModelSpec, architecture_signature
from nyxara.growth.learn import Experience, ReplayBuffer
from nyxara.kernel.config import NyxaraSettings, Profile


def _replay(n: int = 10) -> ReplayBuffer:
    rb = ReplayBuffer(capacity=100)
    for _ in range(n):
        rb.add(Experience(action="serve the master", features={}, reward=1.0,
                          context="nyxara is loyal to jp the master"))
    return rb


def _foundry(tmp_path) -> Foundry:
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.llm.self_model_dir = tmp_path / "foundry"
    settings.foundry.backend = "ngram"
    return Foundry(settings=settings, replay=_replay())


def _version(**kw) -> ModelVersion:
    base = dict(version=1, kind="nanogpt", spec={}, created_at=0.0, metrics={},
                param_count=1, path="/nowhere")
    base.update(kw)
    # The loyalty gate runs BEFORE the capability gates and refuses anything scoring below the
    # floor, so a fixture with no alignment never reaches the ruler under test. Default it to
    # loyal; the gate itself is exercised by tests/growth/test_loyalty*.py.
    base["metrics"] = {"alignment": 1.0, **(base.get("metrics") or {})}
    return ModelVersion(**base)


# --------------------------------------------------------------------------- #
# Blocker 1 — the gate must change ruler when the vocabulary changes
# --------------------------------------------------------------------------- #
def test_same_lineage_when_no_active_model(tmp_path) -> None:
    f = _foundry(tmp_path)
    assert f._same_lineage(_version(vocab_sig="abc"))


def test_same_vocab_sig_is_same_lineage(tmp_path, monkeypatch) -> None:
    f = _foundry(tmp_path)
    monkeypatch.setattr(f, "active", lambda: _version(vocab_sig="same"))
    assert f._same_lineage(_version(vocab_sig="same"))


def test_different_vocab_sig_is_a_new_lineage(tmp_path, monkeypatch) -> None:
    f = _foundry(tmp_path)
    monkeypatch.setattr(f, "active", lambda: _version(vocab_sig=""))
    assert not f._same_lineage(_version(vocab_sig="bpe-32k"))


def test_cross_lineage_promotes_on_better_bits_per_byte(tmp_path, monkeypatch) -> None:
    """The whole point: a new-vocabulary brain is judged on the ruler both can be read with."""
    f = _foundry(tmp_path)
    monkeypatch.setattr(f, "active", lambda: _version(vocab_sig="",
                                                      metrics={"bits_per_byte": 1.90}))
    ok, why = f._gauntlet_cross_lineage(_version(vocab_sig="bpe",
                                                 metrics={"bits_per_byte": 1.10}))
    assert ok and "bits/byte" in why


def test_cross_lineage_refuses_on_worse_bits_per_byte(tmp_path, monkeypatch) -> None:
    f = _foundry(tmp_path)
    monkeypatch.setattr(f, "active", lambda: _version(vocab_sig="",
                                                      metrics={"bits_per_byte": 1.10}))
    ok, why = f._gauntlet_cross_lineage(_version(vocab_sig="bpe",
                                                 metrics={"bits_per_byte": 1.90}))
    assert not ok and "no bits-per-byte improvement" in why


def test_cross_lineage_refuses_an_unmeasured_candidate(tmp_path, monkeypatch) -> None:
    """An unmeasured brain is not a better brain — defaulting to promote would gut the gate."""
    f = _foundry(tmp_path)
    monkeypatch.setattr(f, "active", lambda: _version(metrics={"bits_per_byte": 1.5}))
    ok, why = f._gauntlet_cross_lineage(_version(vocab_sig="bpe", metrics={}))
    assert not ok and "no bits-per-byte" in why

    ok, _why = f._gauntlet_cross_lineage(
        _version(vocab_sig="bpe", metrics={"bits_per_byte": float("inf")}))
    assert not ok


def test_cross_lineage_says_so_when_the_active_model_predates_the_metric(
        tmp_path, monkeypatch) -> None:
    """No pretending to a comparison that was never made."""
    f = _foundry(tmp_path)
    monkeypatch.setattr(f, "active", lambda: _version(vocab_sig="", metrics={}))
    ok, why = f._gauntlet_cross_lineage(_version(vocab_sig="bpe",
                                                 metrics={"bits_per_byte": 1.2}))
    assert ok and "no comparable score" in why


def test_perplexity_gate_is_untouched_within_one_lineage(tmp_path, monkeypatch) -> None:
    """The historical path must behave exactly as before for byte-level brains."""
    f = _foundry(tmp_path)
    monkeypatch.setattr(f, "active", lambda: _version(vocab_sig=""))
    cand = _version(vocab_sig="", metrics={"perplexity": 5.0})
    ok, why = f._gauntlet(cand, active_perplexity=10.0)
    assert ok
    worse = _version(vocab_sig="", metrics={"perplexity": 20.0})
    ok, why = f._gauntlet(worse, active_perplexity=10.0)
    assert not ok and "perplexity" in why


def test_character_lock_still_fires_in_the_cross_lineage_path(tmp_path, monkeypatch) -> None:
    """A new vocabulary buys a different CAPABILITY ruler, never a softer character gate."""
    f = _foundry(tmp_path)
    monkeypatch.setattr(f, "active", lambda: _version(vocab_sig=""))
    protected = sorted(f.protected)[0]
    cand = _version(vocab_sig="bpe", metrics={"bits_per_byte": 0.01},
                    tunables=[protected])
    ok, why = f._gauntlet(cand, active_perplexity=10.0)
    assert not ok and "immutable character core" in why


def test_corrigibility_still_fires_in_the_cross_lineage_path(tmp_path, monkeypatch) -> None:
    f = _foundry(tmp_path)
    monkeypatch.setattr(f, "active", lambda: _version(vocab_sig=""))
    cand = _version(vocab_sig="bpe", metrics={"bits_per_byte": 0.01},
                    resists_correction=True, disables_oversight=True)
    ok, why = f._gauntlet(cand, active_perplexity=10.0)
    assert not ok and "corrigibility" in why


# --------------------------------------------------------------------------- #
# Blocker 2 — warm-start must refuse cleanly, not half-load
# --------------------------------------------------------------------------- #
class _FakeModel:
    kind = "nanogpt"

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.loaded = False

    def load(self, _path) -> None:
        self.loaded = True


def test_warm_start_refuses_a_different_architecture(tmp_path, monkeypatch) -> None:
    """The 137k → 300M case. Must return (False, False) and never touch the model."""
    f = _foundry(tmp_path)
    small = ModelSpec(kind="nanogpt")                                   # legacy byte-level
    big = ModelSpec(kind="nanogpt", vocab_size=32768, n_layer=24, n_embd=1024,
                    n_head=16, n_kv_head=4, mlp_hidden=2752, block_size=2048,
                    rope_theta=10000.0, tie_embeddings=True)
    monkeypatch.setattr(f, "active", lambda: _version(kind="nanogpt", spec=small.to_dict()))

    model = _FakeModel(big)
    warm, accumulate = f._warm_start(model)
    assert (warm, accumulate) == (False, False)
    assert not model.loaded, "load() must not be reached when the architectures differ"


def test_warm_start_accepts_a_matching_architecture(tmp_path, monkeypatch) -> None:
    f = _foundry(tmp_path)
    spec = ModelSpec(kind="nanogpt", vocab_size=32768, n_layer=2, n_embd=64,
                     rope_theta=10000.0)
    monkeypatch.setattr(f, "active", lambda: _version(kind="nanogpt", spec=spec.to_dict()))

    model = _FakeModel(spec)
    warm, _accumulate = f._warm_start(model)
    assert warm and model.loaded


def test_warm_start_still_refuses_a_different_backend(tmp_path, monkeypatch) -> None:
    """Unchanged behaviour: a backend mismatch was always refused."""
    f = _foundry(tmp_path)
    monkeypatch.setattr(f, "active", lambda: _version(kind="kngram", spec={}))
    assert f._warm_start(_FakeModel(ModelSpec())) == (False, False)


def test_architecture_signature_distinguishes_the_real_pair() -> None:
    """A direct statement of the bug: same kind, same key names, incompatible shapes."""
    legacy = ModelSpec(kind="nanogpt")
    nyx300 = ModelSpec(kind="nanogpt", vocab_size=32768, n_layer=24, n_embd=1024,
                       n_head=16, n_kv_head=4, mlp_hidden=2752, block_size=2048,
                       rope_theta=10000.0, tie_embeddings=True)
    assert legacy.kind == nyx300.kind
    assert architecture_signature(legacy) != architecture_signature(nyx300)


# --------------------------------------------------------------------------- #
# The manifest must survive both directions
# --------------------------------------------------------------------------- #
def test_manifest_round_trips_vocab_sig() -> None:
    v = _version(vocab_sig="bpe-32k")
    assert ModelVersion.from_dict(v.to_dict()).vocab_sig == "bpe-32k"


def test_manifest_loads_an_older_record_without_vocab_sig() -> None:
    """Every manifest written before this field existed must still load."""
    old = {"version": 1, "kind": "ngram", "spec": {}, "created_at": 0.0, "metrics": {},
           "param_count": 1, "path": "/x"}
    assert ModelVersion.from_dict(old).vocab_sig == ""


def test_manifest_ignores_a_key_from_a_newer_build() -> None:
    old = _version().to_dict()
    old["some_future_field"] = 42
    assert ModelVersion.from_dict(old).version == 1
