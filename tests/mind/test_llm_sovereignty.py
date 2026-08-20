"""Sovereignty invariants for the LLM faculty — mind/llm.py.

These tests encode, as executable invariants, the single constraint the Master set:

    NYXARA controls the LLM — not the LLM controlling NYXARA.

Concretely that means three things, all network-free (a fake ``openai`` module, so nothing
ever leaves the machine):

1.  **GLM-5 is a reachable cloud tool** — the airouter provider is chosen whenever it is reachable
    and no stronger rung is. (``aicredits`` leads the ``auto`` ladder and ``groq`` follows it; see
    ``test_aicredits_sovereignty.py`` for the primary-rung invariants. These tests pin airouter's own
    behaviour as the LAST cloud rung, so each test here disables both rungs above it explicitly
    rather than relying on the TEST profile to do it.)
2.  **She is never cloud-dependent** — the instant the cloud tool is unavailable, ``complete()``
    keeps answering from a *local* provider (down to her always-on native own-brain) and never
    raises. She uses the cloud; she never needs it.
3.  **The cloud model injects no persona** — the provider forwards only the caller's own
    system/messages. Her identity lives in ``identity/soul.py``, never in the external model.
"""
from __future__ import annotations


from nyxara.kernel.config import LLMProvider, NyxaraSettings, Profile
from nyxara.mind.llm import LLM, LiteRTLMProvider, LLMRequest


# --------------------------------------------------------------------------- #
# Settings for a bare machine: no weights, no promoted foundry model.
# --------------------------------------------------------------------------- #
def _settings(**over):
    s = NyxaraSettings.for_profile(Profile.DEV)
    s.llm.provider = LLMProvider.AUTO
    for k, v in over.items():
        setattr(s.llm, k, v)
    return s


# --------------------------------------------------------------------------- #
# 1. The ladder has no cloud on it at all
# --------------------------------------------------------------------------- #
def test_there_are_no_cloud_rungs_left():
    """The aicredits/Groq/airouter rungs were removed. Nothing here is reached over a wire.

    Asserted structurally rather than by name so a re-introduced cloud provider fails here: a
    provider that needs a key is a provider that leaves the machine.
    """
    from nyxara.kernel.config import LLMConfig

    assert not [f for f in LLMConfig.model_fields if f.endswith("_api_key")]
    assert not [f for f in LLMConfig.model_fields if f.endswith("_base_url")]
    assert _settings().llm.active_key() is None


def test_every_provider_runs_in_process():
    llm = LLM(settings=_settings())
    from nyxara.kernel.config import OWN_PROVIDERS
    assert set(llm.provider_status()) == set(OWN_PROVIDERS)


# --------------------------------------------------------------------------- #
# 2. She degrades to her own brain, and the floor always speaks
# --------------------------------------------------------------------------- #
def test_bare_machine_floor_is_her_native_own_brain(tmp_path):
    """No weights, no promoted model: she still answers, from her own learned voice."""
    s = _settings(litertlm_model_path=tmp_path / "absent.litertlm",
                  self_model_dir=tmp_path / "no-foundry")
    llm = LLM(settings=s)
    assert llm.chosen_provider().name == "native"
    resp = llm.complete(LLMRequest.from_prompt("who is your master?"))
    assert resp.provider == "native"
    assert resp.text.strip(), "the guaranteed floor must never return silence"


def test_the_floor_is_deterministic(tmp_path):
    """Replayability (kernel/replay.py) rests on this: same request, same answer."""
    s = _settings(litertlm_model_path=tmp_path / "absent.litertlm",
                  self_model_dir=tmp_path / "no-foundry")
    llm = LLM(settings=s)
    assert (llm.generate("same prompt") == llm.generate("same prompt"))


# --------------------------------------------------------------------------- #
# 3. No provider injects a persona — identity lives in identity/soul.py
# --------------------------------------------------------------------------- #
def test_no_provider_injects_a_persona(monkeypatch, tmp_path):
    """With no caller system prompt, the outgoing turn carries no 'you are NYXARA' system message.

    This used to be asserted against the cloud rungs. Those are gone, so it is asserted where it
    still matters — the on-device rung, which is now the only one that renders a prompt at all.
    """
    from tests.mind.test_litertlm_provider import _Engine, _Role, _install_fake, _settings

    _install_fake(monkeypatch)
    weights = tmp_path / "model.litertlm"
    weights.write_bytes(b"stub")
    prov = LiteRTLMProvider(_settings(weights))
    prov.complete(LLMRequest.from_prompt("say hi"))           # system=None

    conv = _Engine.instances[0].conversations[0]
    assert conv.kwargs.get("messages") is None, "no system turn was invented"
    sent, _ = _Engine.instances[0].sent[0]
    assert sent.role == _Role.USER
    assert "NYXARA" not in sent.text


def test_a_provider_forwards_only_the_callers_system(monkeypatch, tmp_path):
    """When the kernel supplies a system prompt, it is forwarded verbatim — and nothing else is."""
    from tests.mind.test_litertlm_provider import _Engine, _Role, _install_fake, _settings

    _install_fake(monkeypatch)
    weights = tmp_path / "model.litertlm"
    weights.write_bytes(b"stub")
    prov = LiteRTLMProvider(_settings(weights))
    prov.complete(LLMRequest.from_prompt("say hi", system="Be terse."))

    history = _Engine.instances[0].conversations[0].kwargs["messages"]
    assert [m.role for m in history] == [_Role.SYSTEM]
    assert history[0].text == "Be terse."


def test_the_ladder_is_hers_end_to_end():
    """The structural claim after the cloud rungs were removed: every rung runs in-process.

    Equality would be the wrong assertion now, and the difference matters. ``OWN_PROVIDERS`` is
    about WHERE a provider runs; the ladder is about which of them she reaches for first. Her
    on-device Gemma is entirely hers and is deliberately off the shipped ladder — asserting the two
    sets equal would force one of those facts to be misstated.
    """
    from nyxara.kernel.config import OWN_PROVIDERS

    assert LLM._AUTO_LADDER == ("qwythos", "self", "native")
    assert set(LLM._AUTO_LADDER) <= set(OWN_PROVIDERS)


def test_re_enabling_gemma_actually_puts_it_back_on_the_ladder():
    """The config comment promises this in so many words, so it has to be true.

    Before ``ladder_names()`` existed it was not: the ladder was a frozen class constant, so
    ``litertlm_enabled=True`` changed only whether that provider reported itself *available* — a
    rung nothing ever consulted, because ``auto`` never looked at its name.
    """
    shipped = LLM(settings=_settings())
    assert "litertlm" not in shipped.ladder_names()

    s = _settings()
    s.llm.litertlm_enabled = True
    reenabled = LLM(settings=s).ladder_names()
    assert reenabled == ("qwythos", "litertlm", "self", "native")
    assert reenabled[0] == "qwythos", "the cortex still leads; Gemma is the rung behind it"
