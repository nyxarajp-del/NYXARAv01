"""Tests for growth/bootstrap.py — auto-forge NYXARA's OWN primary brain on boot (CPU/n-gram).

The real path LoRA-tunes the DistilGPT-2 base; on a deps-free CI box the foundry degrades to its
always-on n-gram backend, so these tests exercise the *wiring* (provider gating, idempotency,
the recorded base, SelfProvider hand-off) without any heavy download.
"""

from __future__ import annotations

import json
from pathlib import Path

from nyxara.growth.bootstrap import (
    DISTILGPT2,
    IDENTITY_SEED,
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
    # her own model is forged only when it would actually serve (`self`, or the `auto` ladder
    # once its serve gate is open). An explicit non-self provider is a clean no-op. This used to
    # name `airouter`; the cloud rungs were removed from LLMProvider and the test kept naming one,
    # so it died in the enum lookup before it could assert anything about the forge at all.
    s = NyxaraSettings.for_profile(Profile.DEV)
    s.llm.provider = LLMProvider.NATIVE
    s.llm.self_model_dir = tmp_path / "foundry"
    assert ensure_primary_model(s) is None
    assert not primary_model_present(s)
    assert not (tmp_path / "foundry" / "active").exists()


def test_auto_forges_her_own_brain_by_default(tmp_path: Path):
    # the shipped posture: under `auto` with the serve gate open (self_serve_any_backend, ON by
    # default) she forges & promotes her OWN brain on first boot — no manual provider flip.
    s = NyxaraSettings.for_profile(Profile.DEV)   # provider=auto, self_serve_any_backend=True
    s.llm.self_model_dir = tmp_path / "foundry"
    assert ensure_primary_model(s, log=lambda _m: None) == 1
    assert primary_model_present(s)


def test_forges_and_promotes_primary_brain(tmp_path: Path):
    s = _self_settings(tmp_path)
    version = ensure_primary_model(s, log=lambda _m: None)
    assert version == 1
    assert primary_model_present(s)
    assert (tmp_path / "foundry" / "active").read_text().strip() == "v1"


def test_records_distilgpt2_base_even_when_degraded(tmp_path: Path):
    # the spec is written with the real DistilGPT-2 base; only the *backend* degrades on a bare
    # box, so a machine with the foundry stack rebuilds the very same LoRA from the recorded base.
    s = _self_settings(tmp_path)
    ensure_primary_model(s, log=lambda _m: None)
    spec = json.loads((tmp_path / "foundry" / "manifest.json").read_text())["versions"][0]["spec"]
    assert spec["base_model"] == DISTILGPT2
    assert spec["kind"] == "lora"
    # The spec must record the CONFIGURED posture, whatever it is. This asserted a literal False,
    # which was the shipped default until `foundry.trust_remote_code` was deliberately turned on
    # (config.py: "ON by Master JP's explicit instruction"), and then failed on every run without
    # telling anyone which of the two had moved. What is worth pinning is that the flag survives
    # the trip into the manifest in BOTH directions — a spec that silently dropped it would hand a
    # rebuilding machine a different security posture than the one that was configured.
    assert spec["trust_remote_code"] is s.foundry.trust_remote_code


def test_spec_records_the_configured_remote_code_posture(tmp_path: Path):
    # Both directions, because only one of them is the dangerous one to get wrong: a manifest that
    # recorded True when False was configured would hand a rebuilding machine permission to execute
    # a model repo's own Python, unreviewed, that the Master never granted.
    for posture in (False, True):
        root = tmp_path / f"foundry-{posture}"
        s = NyxaraSettings.for_profile(Profile.DEV)
        s.llm.provider = LLMProvider.SELF
        s.llm.self_model_dir = root
        s.foundry.trust_remote_code = posture
        ensure_primary_model(s, log=lambda _m: None)
        spec = json.loads((root / "manifest.json").read_text())["versions"][0]["spec"]
        assert spec["trust_remote_code"] is posture


def test_tiny_placeholder_base_is_upgraded_to_distilgpt2(tmp_path: Path):
    # a stale config still pinned to the tiny CPU/CI placeholder is upgraded to the real
    # DistilGPT-2 for the *primary* forge (a deliberate non-default base is left untouched).
    s = _self_settings(tmp_path)
    s.foundry.base_model = "sshleifer/tiny-gpt2"
    ensure_primary_model(s, log=lambda _m: None)
    spec = json.loads((tmp_path / "foundry" / "manifest.json").read_text())["versions"][0]["spec"]
    assert spec["base_model"] == DISTILGPT2


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
    assert isinstance(out, str) and out  # her own forged brain answers (no substitution)


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


# --------------------------------------------------------------------------- #
# The cortex runtime — provisioned on boot, and bounded at every step
# --------------------------------------------------------------------------- #
def _cortex_settings(**over) -> NyxaraSettings:
    """DEV, with every gate open. Each test then closes exactly one and checks the refusal."""
    s = NyxaraSettings.for_profile(Profile.DEV)
    s.llm.qwythos_enabled = True
    s.llm.qwythos_auto_install = True
    s.explorer.autonomous_install = True
    s.features.web_access = True
    for k, v in over.items():
        target, _, field = k.partition("__")
        setattr(getattr(s, target), field, v)
    return s


def _no_real_pip(monkeypatch):
    """Fail loudly if a test reaches pip. Nothing here may install anything."""
    import nyxara.growth.bootstrap as boot

    def _boom(*a, **k):
        raise AssertionError("a test reached pip")

    monkeypatch.setattr(boot, "_cortex_runtime_present", lambda: False)
    monkeypatch.setattr("nyxara.agency.code_sandbox.safe_shell", _boom)


def test_the_runtime_is_a_no_op_when_it_is_already_importable(monkeypatch):
    import nyxara.growth.bootstrap as boot

    monkeypatch.setattr(boot, "_cortex_runtime_present", lambda: True)
    monkeypatch.setattr("nyxara.agency.code_sandbox.safe_shell",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("reached pip")))
    said: list[str] = []
    assert boot._ensure_cortex_runtime(_cortex_settings(), said.append) is True
    assert said == []                      # a present runtime is not news


def test_the_test_profile_never_installs(monkeypatch):
    """A suite that installs a package is not hermetic, and this wheel may build from source."""
    import nyxara.growth.bootstrap as boot

    _no_real_pip(monkeypatch)
    said: list[str] = []
    assert boot._ensure_cortex_runtime(
        NyxaraSettings.for_profile(Profile.TEST), said.append) is False
    assert said == []                      # silent, because under TEST it is not a degradation


def test_each_closed_gate_refuses_with_its_own_reason(monkeypatch):
    """Four different things stop the install, and an operator must be told which one."""
    import nyxara.growth.bootstrap as boot

    _no_real_pip(monkeypatch)
    for override, expect in (
        ({"llm__qwythos_auto_install": False}, "auto-install is off"),
        ({"explorer__autonomous_install": False}, "not authorised"),
        ({"features__web_access": False}, "no network"),
    ):
        said: list[str] = []
        assert boot._ensure_cortex_runtime(_cortex_settings(**override), said.append) is False
        assert any(expect in line for line in said), (override, said)
        # and every refusal still names the one-line manual fix
        assert any("nyxara[qwythos]" in line for line in said), override

    said = []
    assert boot._ensure_cortex_runtime(
        _cortex_settings(llm__qwythos_enabled=False), said.append) is False


def test_it_installs_exactly_one_named_distribution(monkeypatch):
    """Never a resolved list, never a shell string — one distribution, argv-style."""
    import nyxara.growth.bootstrap as boot

    calls: list[list[str]] = []
    monkeypatch.setattr(boot, "_cortex_runtime_present",
                        lambda: len(calls) > 0)          # absent, then present after the install

    def _fake_shell(argv, **kwargs):
        calls.append(list(argv))
        return type("_R", (), {"ok": True, "stderr": "", "error": ""})()

    monkeypatch.setattr("nyxara.agency.code_sandbox.safe_shell", _fake_shell)
    said: list[str] = []
    assert boot._ensure_cortex_runtime(_cortex_settings(), said.append) is True
    assert len(calls) == 1
    assert calls[0][-3:] == ["install", "--quiet", "llama-cpp-python"]
    assert "-m" in calls[0] and "pip" in calls[0]         # the running interpreter's pip


def test_a_failed_install_is_reported_and_the_boot_continues(monkeypatch):
    import nyxara.growth.bootstrap as boot

    monkeypatch.setattr(boot, "_cortex_runtime_present", lambda: False)
    monkeypatch.setattr(
        "nyxara.agency.code_sandbox.safe_shell",
        lambda argv, **k: type("_R", (), {"ok": False, "stderr": "error: no C compiler",
                                          "error": ""})())
    said: list[str] = []
    assert boot._ensure_cortex_runtime(_cortex_settings(), said.append) is False
    assert any("no C compiler" in line for line in said)
    assert any("one rung lower" in line for line in said)


def test_a_raising_installer_never_takes_the_boot_down(monkeypatch):
    import nyxara.growth.bootstrap as boot

    monkeypatch.setattr(boot, "_cortex_runtime_present", lambda: False)
    monkeypatch.setattr("nyxara.agency.code_sandbox.safe_shell",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("pip is not on PATH")))
    said: list[str] = []
    assert boot._ensure_cortex_runtime(_cortex_settings(), said.append) is False
    assert any("skipped" in line for line in said)


def test_the_runtime_is_provisioned_before_the_weights(tmp_path, monkeypatch):
    """6.5 GB of GGUF with nothing able to load it is the one outcome worth avoiding."""
    import nyxara.growth.bootstrap as boot

    order: list[str] = []
    monkeypatch.setattr(boot, "_ensure_cortex_runtime",
                        lambda s, say: order.append("runtime") or False)
    monkeypatch.setattr("nyxara.mind.gguf_assets.model_present",
                        lambda s=None: order.append("weights") or False)
    s = _self_settings(tmp_path)
    s.llm.qwythos_enabled = True
    boot._ensure_qwythos(s, lambda _m: None)
    assert order[:2] == ["runtime", "weights"]
