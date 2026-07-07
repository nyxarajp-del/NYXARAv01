"""Tests for growth/bootstrap.py — auto-forge NYXARA's OWN primary brain on boot (CPU/n-gram).

The real path LoRA-tunes TinyLlama-1.1B; on a deps-free CI box the foundry degrades to its
always-on n-gram backend, so these tests exercise the *wiring* (provider gating, idempotency,
the recorded TinyLlama-1.1B base, SelfProvider hand-off) without any heavy download.
"""

from __future__ import annotations

import json
from pathlib import Path

from nyxara.growth.bootstrap import (
    IDENTITY_SEED,
    TINYLLAMA_1_1B,
    build_seed_corpus,
    ensure_primary_model,
    primary_model_present,
)
from nyxara.kernel.config import LLMProvider, NyxaraSettings, Profile


def _self_settings(tmp_path: Path) -> NyxaraSettings:
    s = NyxaraSettings.for_profile(Profile.DEV)
    s.llm.provider = LLMProvider.SELF
    s.llm.self_model_dir = tmp_path / "foundry"
    return s


def test_noop_when_provider_is_not_self(tmp_path: Path):
    # her own model is forged only when `self` is the chosen primary provider
    s = NyxaraSettings.for_profile(Profile.DEV)   # default provider is tinyllama, not self
    s.llm.self_model_dir = tmp_path / "foundry"
    assert ensure_primary_model(s) is None
    assert not primary_model_present(s)
    assert not (tmp_path / "foundry" / "active").exists()


def test_forges_and_promotes_primary_brain(tmp_path: Path):
    s = _self_settings(tmp_path)
    version = ensure_primary_model(s, log=lambda _m: None)
    assert version == 1
    assert primary_model_present(s)
    assert (tmp_path / "foundry" / "active").read_text().strip() == "v1"


def test_records_tinyllama_base_even_when_degraded(tmp_path: Path):
    # the spec is written with the real TinyLlama-1.1B base; only the *backend* degrades on a
    # bare box, so a GPU machine rebuilds the very same LoRA from the recorded base.
    s = _self_settings(tmp_path)
    ensure_primary_model(s, log=lambda _m: None)
    spec = json.loads((tmp_path / "foundry" / "manifest.json").read_text())["versions"][0]["spec"]
    assert spec["base_model"] == TINYLLAMA_1_1B
    assert spec["kind"] == "lora"


def test_idempotent_does_not_reforge(tmp_path: Path):
    s = _self_settings(tmp_path)
    first = ensure_primary_model(s, log=lambda _m: None)
    # a second boot finds the active pointer and returns it without training a new version
    second = ensure_primary_model(s, log=lambda _m: None)
    assert first == second == 1
    versions = json.loads((tmp_path / "foundry" / "manifest.json").read_text())["versions"]
    assert len(versions) == 1


def test_respects_explicit_non_default_base(tmp_path: Path):
    # a base the Master set deliberately is honoured verbatim (only the tiny default is upgraded)
    s = _self_settings(tmp_path)
    s.foundry.base_model = "some/other-base"
    ensure_primary_model(s, log=lambda _m: None)
    spec = json.loads((tmp_path / "foundry" / "manifest.json").read_text())["versions"][0]["spec"]
    assert spec["base_model"] == "some/other-base"

    s2 = _self_settings(tmp_path / "two")
    ensure_primary_model(s2, base_model="explicit/base", log=lambda _m: None)
    spec2 = json.loads((tmp_path / "two" / "foundry" / "manifest.json").read_text())[
        "versions"][0]["spec"]
    assert spec2["base_model"] == "explicit/base"


def test_never_raises_into_boot(tmp_path: Path, monkeypatch):
    # any forge failure is swallowed (returns None) so the front door never dies
    import nyxara.growth.bootstrap as boot

    def _boom(*_a, **_k):
        raise RuntimeError("simulated forge failure")

    monkeypatch.setattr(boot, "_forge", _boom)
    s = _self_settings(tmp_path)
    msgs: list[str] = []
    assert ensure_primary_model(s, log=msgs.append) is None
    assert any("could not forge" in m for m in msgs)


def test_forged_brain_is_served_by_self_provider(tmp_path: Path):
    from nyxara.mind.llm import LLM

    s = _self_settings(tmp_path)
    ensure_primary_model(s, log=lambda _m: None)
    llm = LLM(settings=s)
    assert llm.provider_status()["self"] is True
    assert llm.chosen_provider().name == "self"
    out = llm.generate("Who is your master?", system="You are NYXARA.", max_tokens=16)
    assert isinstance(out, str) and out  # her own brain answers (no mock substitution)


def test_identity_seed_is_loyal_and_nonempty():
    assert IDENTITY_SEED and all(isinstance(t, str) and t for t in IDENTITY_SEED)
    assert any("JP" in t or "Master" in t for t in IDENTITY_SEED)


# --------------------------------------------------------------------------- #
# build_seed_corpus — the always-on (stdlib) base brain is grounded in real law
# --------------------------------------------------------------------------- #
def test_seed_corpus_is_richer_than_tiny_seed():
    corpus = build_seed_corpus()
    assert len(corpus) > 3 * len(IDENTITY_SEED)         # far more than eight lines
    assert len(corpus) == len(set(corpus))              # de-duplicated, stable


def test_seed_corpus_grounded_in_rules_and_values():
    joined = " ".join(build_seed_corpus()).lower()
    for kw in ("jp", "jaypal", "loyalty", "corrigib", "allegiance",
               "transparency", "sovereign"):
        assert kw in joined, kw


def test_seed_corpus_accepts_extra_and_dedups():
    base = build_seed_corpus()
    extended = build_seed_corpus(extra=["a brand new lived fact", IDENTITY_SEED[0]])
    assert "a brand new lived fact" in extended
    assert len(extended) == len(set(extended))          # the duplicate seed was folded out
    assert len(extended) == len(base) + 1


def test_richer_corpus_makes_a_stronger_offline_brain():
    """A kngram brain trained on the grounded corpus scores a rule-derived sentence better
    than one trained on the tiny seed — a measurable capability gain, not just more text."""
    from nyxara.growth.foundry_models import WordKNGramLM

    probe = "absolute allegiance to master jp above all else"
    tiny = WordKNGramLM(order=3); tiny.train_on(IDENTITY_SEED)
    rich = WordKNGramLM(order=3); rich.train_on(build_seed_corpus())
    assert rich.perplexity(probe) < tiny.perplexity(probe)
