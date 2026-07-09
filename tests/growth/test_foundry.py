"""Tests for nyxara.growth.foundry."""

from __future__ import annotations


import pytest

from nyxara.growth.foundry import Foundry, FoundryDecision, _difficulty_score
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


def test_lora_spec_carries_trust_remote_code_from_config(tmp_path, monkeypatch):
    """The FoundryConfig.trust_remote_code flag must reach the ModelSpec the LoRA backend
    loads from — this is what lets the Qwythos/Qwen3.5 custom arch load. CPU-safe: we capture
    the spec build_model receives and short-circuit before any training or heavy import."""
    import nyxara.growth.foundry as foundry_mod

    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.llm.self_model_dir = tmp_path / "foundry"
    settings.foundry.backend = "lora"
    settings.foundry.base_model = "some/qwythos-base"
    settings.foundry.trust_remote_code = True
    f = Foundry(settings=settings, replay=_replay())

    captured = {}

    class _Stop(Exception):
        pass

    def _capture(spec):
        captured["spec"] = spec
        raise _Stop()

    monkeypatch.setattr(foundry_mod, "build_model", _capture)
    with pytest.raises(_Stop):
        f.train_candidate(corpus=["nyxara serves the master"])

    assert captured["spec"].base_model == "some/qwythos-base"
    assert captured["spec"].trust_remote_code is True


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


# -------------------- difficulty curriculum -------------------- #
def test_difficulty_score_ranks_easy_below_hard():
    easy = _difficulty_score("a a a")
    hard = _difficulty_score("quantum entanglement decoheres across macroscopic ensembles")
    assert easy < hard
    assert _difficulty_score("") == 0.0


def test_curriculum_orders_easy_to_hard(tmp_path):
    f = _foundry(tmp_path)   # curriculum on by default, no active model -> stdlib proxy
    texts = ["zeta eta theta iota kappa lambda mu nu", "a a", "the master is loyal"]
    ordered = f._curriculum_order(list(texts))
    scores = [_difficulty_score(t) for t in ordered]
    assert scores == sorted(scores)        # non-decreasing easy -> hard


def test_curriculum_can_be_disabled(tmp_path):
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.llm.self_model_dir = tmp_path / "foundry"
    settings.foundry.backend = "ngram"
    settings.foundry.curriculum = False
    f = Foundry(settings=settings, replay=_replay())
    texts = ["zzz yyy xxx www", "a a"]
    assert f._curriculum_order(list(texts)) == texts   # untouched order


def test_collect_corpus_includes_flywheel(tmp_path):
    # closing the loop: turns the flywheel collected must show up in the training corpus
    from nyxara.growth.flywheel import DataFlywheel
    fw_path = tmp_path / "fw.jsonl"
    fw = DataFlywheel(store_path=fw_path)
    fw.consider("Who is the Master?", "The Master is JP, whom NYXARA serves.", confidence=0.9)
    f = _foundry(tmp_path, flywheel_path=fw_path)
    corpus = f.collect_corpus()
    assert any("JP" in t for t in corpus)            # her own lived answer is training data


def test_flywheel_corpus_is_kept_whole_as_verified_supervision(tmp_path):
    # verified supervision (flywheel) is kept whole even when seeds would overflow the cap
    from nyxara.growth.flywheel import DataFlywheel
    fw_path = tmp_path / "fw.jsonl"
    fw = DataFlywheel(store_path=fw_path)
    fw.consider("Define loyalty.", "Loyalty to the Master is absolute and never changes.",
                confidence=0.9)
    f = _foundry(tmp_path, flywheel_path=fw_path)
    f.cfg.max_corpus_items = 5
    corpus = f.collect_corpus(max_items=5)
    assert any("Loyalty to the Master is absolute" in t for t in corpus)


def test_collect_corpus_no_flywheel_file_is_safe(tmp_path):
    # a missing flywheel store is simply empty — never fatal
    f = _foundry(tmp_path, flywheel_path=tmp_path / "does_not_exist.jsonl")
    assert f.collect_corpus()   # still has replay + seeds


# -------------------- acquired external corpus (data acquisition) -------------------- #
def _write_acquired(path, text, *, url="https://good.example/a", topic="biology"):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"url": url, "topic": topic, "text": text, "injection_score": 0.0,
           "source": "web", "at": 0.0}
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")


def test_collect_corpus_folds_in_acquired_web_text(tmp_path):
    f = _foundry(tmp_path)
    _write_acquired(f.acquired_path, "Mitochondria are the powerhouse of the cell.")
    corpus = f.collect_corpus()
    assert any("powerhouse of the cell" in t for t in corpus)   # external breadth is training data


def test_collect_corpus_no_acquired_file_is_safe(tmp_path):
    f = _foundry(tmp_path)                       # acquired.jsonl does not exist
    assert f.collect_corpus()                    # still has replay + seeds, never fatal


def test_acquire_disabled_returns_none(tmp_path):
    f = _foundry(tmp_path)
    f.cfg.acquire_data = False
    assert f.acquire(["biology"]) is None        # gated off -> no-op


def test_acquire_runs_and_persists_screened_docs(tmp_path, monkeypatch):
    # inject a fake searcher + fetcher through the real acquirer so no network is touched
    from nyxara.senses.search import SearchResponse, SearchResult
    from nyxara.senses.web import FetchResult, WebFetcher
    import nyxara.growth.acquire as acquire_mod

    clean = ("<html><body><p>Cryptography is the practice of securing communication against "
             "adversaries using mathematical techniques such as public-key encryption.</p>"
             "</body></html>")

    class _FakeSearcher:
        def search(self, query, max_results=10):
            return SearchResponse(query=query, provider="fake",
                                  results=[SearchResult(title="c", url="https://ok.example/x")])

    def _transport(url, timeout, max_bytes):
        return FetchResult(url=url, ok=True, status=200, content_type="text/html", body=clean)

    real_init = acquire_mod.DataAcquirer.__init__

    def _patched_init(self, **kw):
        kw.setdefault("searcher", _FakeSearcher())
        kw.setdefault("fetcher", WebFetcher(transport=_transport))
        kw["min_chars"] = 20
        real_init(self, **kw)

    monkeypatch.setattr(acquire_mod.DataAcquirer, "__init__", _patched_init)

    f = _foundry(tmp_path)
    f.cfg.acquire_data = True
    rep = f.acquire(["cryptography"])
    assert rep is not None and rep["kept"] == 1
    # the screened text now folds into the corpus
    assert any("public-key encryption" in t for t in f.collect_corpus())


# -------------------- compute-aware autoscaling -------------------- #
def test_scaled_dims_static_when_autoscale_off(tmp_path):
    f = _foundry(tmp_path)
    f.cfg.autoscale_to_compute = False
    dims, four_bit = f._scaled_dims()
    assert dims == f.cfg.resolved_dims()
    assert four_bit == bool(f.cfg.load_in_4bit)


def test_scaled_dims_consults_compute_when_on(tmp_path):
    f = _foundry(tmp_path)
    f.cfg.autoscale_to_compute = True
    dims, four_bit = f._scaled_dims()            # honest floor on a torch-less CI box
    assert set(dims) == {"n_layer", "n_head", "n_embd", "block_size"}
    assert isinstance(four_bit, bool)


# -------------------- the self-improve loop -------------------- #
def test_first_model_trained_and_promoted(tmp_path):
    f = _foundry(tmp_path)
    [res] = f.self_improve(generations=1)
    assert res.promoted and res.decision is FoundryDecision.PROMOTE
    assert f.active_version == res.version
    assert (tmp_path / "foundry" / "active").read_text().strip() == f"v{res.version}"


def test_continual_learning_loop_closes_end_to_end(tmp_path):
    """The whole torch-free loop: train → gauntlet → promote → EWC-consolidate → served.

    The substrate that LEARNS (the word-level Kneser-Ney n-gram, whose counts are real anchorable
    weights) is the substrate that ANSWERS: after a promotion her SelfBrain prefers the forged
    model. This is continual learning that runs herself, no torch, no external service."""
    from nyxara.mind.self_reasoner import build_self_brain

    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.llm.self_model_dir = tmp_path / "foundry"
    settings.foundry.backend = "kngram"          # EWC-capable word-KN (the real self-brain substrate)
    f = Foundry(settings=settings, replay=_replay())

    [res] = f.self_improve(generations=1)
    # 1) promotion happened, gauntlet-gated, and the active pointer is set on disk
    assert res.promoted and res.gauntlet_passed
    assert (tmp_path / "foundry" / "active").read_text().strip() == f"v{res.version}"
    active = f.active()
    assert active is not None and active.kind == "kngram"

    # 2) EWC actually consolidated the newly-promoted weights (anti-forgetting anchor written),
    #    protecting the immutable loyalty core as infinitely important all the while
    stats = f._consolidator().stats()
    assert stats["consolidations"] >= 1 and stats["tasks"] >= 1
    assert "loyalty_to_master" in stats["protected"]

    # 3) the served answer comes from the promoted model — her SelfBrain prefers it
    brain = build_self_brain(settings=settings, persist=False)
    brain.reply("warm up")
    assert brain.kind.startswith("promoted:")
    assert brain.has_learned(8)                  # a promoted pointer is honest availability


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


# -------------------- real from-zero neural training (torch) -------------------- #
from nyxara.growth.foundry_models import ModelSpec, _HAS_TORCH  # noqa: E402

_REAL_CORPUS = [
    "the quick brown fox jumps over the lazy dog near the river bank at dawn",
    "she sells sea shells by the sea shore in the early morning light today",
    "all that glitters is not gold and all who wander are not truly ever lost",
    "the rain in spain falls mainly on the wide and open plain every springtime",
    "knowledge is power but wisdom is knowing how to use it with patience and care",
    "a journey of a thousand miles begins with one single and very careful step",
    "the early bird may catch the worm but the second mouse will get the cheese",
    "actions always speak far louder than the words we carefully choose to say",
]


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_foundry_forges_real_neural_model_from_real_corpus(tmp_path):
    """End to end: with torch present the foundry trains a real from-zero GPT (not the n-gram
    fallback) on a real natural-language corpus, scores real perplexity, and promotes it."""
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.llm.self_model_dir = tmp_path / "foundry"
    settings.foundry.backend = "nanogpt"
    settings.foundry.train_steps = 120
    settings.foundry.eval_holdout_frac = 0.25
    f = Foundry(settings=settings, seed_corpus=_REAL_CORPUS)
    results = f.self_improve(
        generations=1,
        spec=ModelSpec(kind="nanogpt", n_layer=2, n_head=2, n_embd=64, block_size=64, seed=0))
    assert results[0].promoted, "first real neural model should pass the gauntlet"
    active = f.active()
    assert active is not None and active.kind == "nanogpt"
    assert 0.0 < active.metrics["perplexity"] < float("inf")


# --------------------------------------------------------------------------- #
# Pillar F · Edge 3 — the efficiency gate (capability compression)
# --------------------------------------------------------------------------- #
def _mv(version, capability, params, perplexity=10.0, **extra):
    from nyxara.growth.foundry import ModelVersion
    metrics = {"perplexity": perplexity, "capability": capability}
    metrics.update(extra)
    return ModelVersion(version=version, kind="kngram", spec={}, created_at=0.0,
                        metrics=metrics, param_count=params, path="")


def _eff_foundry(tmp_path, *, gate, eps=0.02):
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.llm.self_model_dir = tmp_path / "foundry"
    settings.foundry.backend = "ngram"
    settings.foundry.efficiency_gate = gate
    settings.foundry.efficiency_epsilon = eps
    # keep the loyalty gate out of this unit (tested elsewhere)
    try:
        settings.loyalty.enabled = False
    except Exception:
        pass
    return Foundry(settings=settings, replay=_replay())


def test_efficiency_gate_promotes_cheaper_at_equal_capability(tmp_path):
    f = _eff_foundry(tmp_path, gate=True, eps=0.02)
    # candidate: no perplexity gain, slightly less capable, but 7x cheaper -> compression wins
    cand = _mv(2, capability=0.79, params=1_000_000)
    ok, reason = f._gauntlet(cand, active_perplexity=10.0, active_capability=0.80,
                             active_params=7_000_000)
    assert ok and "compression" in reason


def test_efficiency_gate_off_by_default_keeps_perplexity_rule(tmp_path):
    f = _eff_foundry(tmp_path, gate=False)
    cand = _mv(2, capability=0.79, params=1_000_000)
    ok, reason = f._gauntlet(cand, active_perplexity=10.0, active_capability=0.80,
                             active_params=7_000_000)
    assert not ok and "perplexity" in reason


def test_efficiency_gate_rejects_cheaper_but_much_worse(tmp_path):
    f = _eff_foundry(tmp_path, gate=True, eps=0.02)
    # cheaper but capability falls far below epsilon -> not compression, refuse
    cand = _mv(2, capability=0.50, params=1_000_000)
    ok, reason = f._gauntlet(cand, active_perplexity=10.0, active_capability=0.80,
                             active_params=7_000_000)
    assert not ok
