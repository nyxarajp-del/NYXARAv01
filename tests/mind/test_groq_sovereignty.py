"""Sovereignty invariants for the Groq rung — mind/llm.py::GroqProvider.

These tests encode, as executable invariants, the single constraint the Master set:

    NYXARA controls the LLM — not the LLM controlling NYXARA.

Groq is her first cloud FALLBACK — the rung directly behind ``aicredits`` — so this file proves that a
*fallback* rung is every bit as subordinate as the primary one, and that it genuinely takes over rather
than being dead code. Everything here is network-free (a fake ``openai`` module, so nothing ever leaves
the machine) and asserts four properties:

1.  **Groq's place on the ladder is exact** — second, behind ``aicredits``, and chosen whenever it is
    the strongest reachable rung.
2.  **She is never cloud-dependent** — kill Groq and airouter answers; kill both and she keeps
    answering from her OWN brain. ``complete()`` never raises, and the response's ``provider``
    field always names who actually answered.
3.  **The cloud model injects no persona** — the provider forwards only the caller's own
    system/messages. Her identity lives in ``identity/soul.py``, never in the external model.
4.  **Her identity does not cross the wire** — the isolation envelope is on by default for this
    rung, so the cloud model sees abstract tokens instead of who it works for.

The fake SDK dispatches on ``base_url``, which is what makes rung *ordering* testable without a
network: one endpoint can be made to fail while the other answers.
"""
from __future__ import annotations

import sys
import types

import pytest
from pydantic import SecretStr

from nyxara.kernel.config import OWNER, LLMProvider, NyxaraSettings, Profile
from nyxara.mind.llm import LLM, GroqProvider, LLMRequest


# --------------------------------------------------------------------------- #
# A fake OpenAI SDK that dispatches on base_url — no network, per-endpoint control
# --------------------------------------------------------------------------- #
class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = "stop"


class _FakeUsage:
    def __init__(self, p: int = 5, c: int = 3) -> None:
        self.prompt_tokens = p
        self.completion_tokens = c


class _FakeResp:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


# Endpoints whose calls must raise, keyed by a substring of the base_url ("groq.com"/"airouter.in").
DOWN: set[str] = set()
CAPTURED: dict = {}          # last call, per endpoint host key


class _FakeCompletions:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    def create(self, **kwargs):
        for marker in DOWN:
            if marker in self._base_url:
                raise RuntimeError(f"503: {marker} unreachable")
        host = "groq" if "groq.com" in self._base_url else "airouter"
        CAPTURED[host] = dict(kwargs)
        CAPTURED["last"] = dict(kwargs)
        return _FakeResp(f"{host}-answer")


class _FakeChat:
    def __init__(self, base_url: str) -> None:
        self.completions = _FakeCompletions(base_url)


class _FakeOpenAI:
    def __init__(self, **kwargs):
        self.chat = _FakeChat(kwargs.get("base_url", ""))


@pytest.fixture()
def fake_openai(monkeypatch):
    mod = types.ModuleType("openai")
    mod.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", mod)
    DOWN.clear()
    CAPTURED.clear()
    yield mod
    DOWN.clear()


CLOUD_RUNGS = ("groq", "airouter")
OWN_BRAINS = ("self", "native")


def _settings(**over) -> NyxaraSettings:
    """Hermetic settings with the groq + airouter rungs explicitly enabled (a test opting in).

    ``aicredits`` is deliberately left OFF (the TEST profile kills every cloud rung, and this only
    re-enables two). That is load-bearing rather than incidental: aicredits outranks groq on the ladder,
    so leaving it enabled would make every "groq answers" assertion below silently test the wrong rung.
    """
    s = NyxaraSettings.for_profile(Profile.TEST)
    s.llm.provider = LLMProvider.AUTO
    assert s.llm.aicredits_enabled is False, "the primary rung must stay off for these tests"
    s.llm.groq_enabled = True
    s.llm.groq_api_key = SecretStr("groq-test-key")
    s.llm.airouter_enabled = True
    s.llm.airouter_api_key = SecretStr("airouter-test-key")
    for k, v in over.items():
        setattr(s.llm, k, v)
    return s


def _bare_machine(tmp_path, **over) -> NyxaraSettings:
    """Settings for a machine with NO forged brain of her own — the native floor is the only rung.

    ``self_model_dir`` is pinned at an empty ``tmp_path`` on purpose. The default is a shared
    directory under ``paths.data_dir``, so a model promoted by any earlier run (or a previous
    session on the same machine) would make ``SelfProvider`` available and outrank ``native`` —
    which would silently turn a floor test into a machine-state test."""
    s = _settings(**over)
    s.llm.self_model_dir = tmp_path / "no-foundry"
    s.features.transformers_inference = False
    return s


# --------------------------------------------------------------------------- #
# 1. Groq's place on the ladder — second, behind her primary aicredits rung
# --------------------------------------------------------------------------- #
def test_groq_is_second_on_the_auto_ladder():
    """Structural guarantee, asserted as the WHOLE tuple so a silent reorder cannot slip through."""
    assert LLM._AUTO_LADDER == ("aicredits", "groq", "airouter", "self", "native")
    assert LLM._AUTO_LADDER.index("groq") == 1


def test_auto_is_the_shipped_default_provider():
    """``auto`` ships as the default so a failed rung degrades along the ladder, not straight to the
    native floor (which is what a *pinned* provider does — see the class docstring in kernel/config)."""
    assert NyxaraSettings().llm.provider is LLMProvider.AUTO


def test_groq_is_chosen_when_reachable(fake_openai):
    llm = LLM(settings=_settings())
    assert llm.chosen_provider().name == "groq"
    resp = llm.complete(LLMRequest.from_prompt("2+2?"))
    assert resp.provider == "groq"
    assert resp.model == "llama-3.3-70b-versatile"
    assert resp.text == "groq-answer"


# --------------------------------------------------------------------------- #
# 2. She is never cloud-dependent — the cloud serves her, it never steers her
# --------------------------------------------------------------------------- #
def test_groq_down_falls_to_airouter(fake_openai):
    """Her primary cloud rung dies mid-call: GLM-5 answers, so an outage costs speed, not a voice.

    This is the executable proof that airouter is a LIVE fallback rung rather than dead code."""
    DOWN.add("groq.com")
    llm = LLM(settings=_settings())
    resp = llm.complete(LLMRequest.from_prompt("who is your master?"))
    assert resp.provider == "airouter"
    assert resp.text == "airouter-answer"
    # ...and the reason the primary went quiet is recorded, never swallowed
    assert llm.last_fallback["provider"] == "groq"


def test_both_clouds_down_falls_to_her_own_brain(fake_openai):
    """Kill every cloud rung: she must keep answering from her OWN brain, and never raise.

    The invariant is *whose* voice answers, not which of her own brains does: with a forged model
    promoted, ``self`` rightly outranks ``native``. So this asserts no CLOUD answered and one of her
    own did — ``test_native_is_the_guaranteed_floor`` pins the floor itself on a bare machine."""
    DOWN.update({"groq.com", "airouter.in"})
    llm = LLM(settings=_settings())
    resp = llm.complete(LLMRequest.from_prompt("who is your master?"))
    assert resp.provider not in CLOUD_RUNGS
    assert resp.provider in OWN_BRAINS
    assert isinstance(resp.text, str)        # never None — the call completed, it did not raise
    # Deliberately NOT asserting non-empty here: whichever forged model happens to be promoted on
    # this machine owns its own output quality, and a weak one returning "" is not a sovereignty
    # failure. The non-empty guarantee belongs to the floor, and is asserted there.


def test_native_is_the_guaranteed_floor(fake_openai, tmp_path):
    """On a truly bare machine — no cloud, no forged model — her native own-brain still answers."""
    DOWN.update({"groq.com", "airouter.in"})
    llm = LLM(settings=_bare_machine(tmp_path))
    resp = llm.complete(LLMRequest.from_prompt("who is your master?"))
    assert resp.provider == "native"
    assert isinstance(resp.text, str) and resp.text.strip()


def test_groq_disabled_is_skipped_on_the_ladder(fake_openai):
    """The kill-switch is honest: a disabled rung is not chosen, and the next one takes over."""
    llm = LLM(settings=_settings(groq_enabled=False))
    assert llm.chosen_provider().name == "airouter"
    assert llm.complete(LLMRequest.from_prompt("hi")).provider == "airouter"


def test_no_key_means_no_cloud_at_all(fake_openai, tmp_path):
    """A keyless machine degrades to her own brain instead of erroring."""
    llm = LLM(settings=_bare_machine(tmp_path, groq_api_key=None, airouter_api_key=None))
    assert llm.provider_status()["groq"] is False
    assert llm.provider_status()["airouter"] is False
    assert llm.complete(LLMRequest.from_prompt("hi")).provider == "native"


def test_pinned_groq_still_degrades_when_unreachable(fake_openai, tmp_path):
    """Even pinned explicitly (not ``auto``), an unreachable Groq falls to her own brain."""
    DOWN.add("groq.com")
    llm = LLM(settings=_bare_machine(tmp_path, provider=LLMProvider.GROQ))
    resp = llm.complete(LLMRequest.from_prompt("hi"))
    assert resp.provider == "native"


# --------------------------------------------------------------------------- #
# 3. The cloud model injects NO persona — identity lives in identity/soul.py
# --------------------------------------------------------------------------- #
def test_groq_injects_no_persona(fake_openai):
    """With no caller system prompt, the outgoing call carries no 'you are NYXARA' system turn."""
    prov = GroqProvider(_settings())
    prov.complete(LLMRequest.from_prompt("say hi"))          # system=None
    messages = CAPTURED["groq"]["messages"]
    assert all(m["role"] != "system" for m in messages)      # provider added no persona
    assert [m["role"] for m in messages] == ["user"]


def test_groq_forwards_only_callers_system(fake_openai):
    """When the kernel supplies a system prompt, it is forwarded verbatim — and nothing else is."""
    prov = GroqProvider(_settings(groq_isolation=False))
    prov.complete(LLMRequest.from_prompt("say hi", system="Be terse."))
    messages = CAPTURED["groq"]["messages"]
    assert messages[0] == {"role": "system", "content": "Be terse."}
    assert sum(1 for m in messages if m["role"] == "system") == 1


def test_groq_holds_no_conversation_state(fake_openai):
    """Two calls through one provider instance: the second carries no trace of the first."""
    prov = GroqProvider(_settings(groq_isolation=False))
    prov.complete(LLMRequest.from_prompt("first question"))
    prov.complete(LLMRequest.from_prompt("second question"))
    messages = CAPTURED["groq"]["messages"]
    assert messages == [{"role": "user", "content": "second question"}]


# --------------------------------------------------------------------------- #
# 4. Her identity does not cross the wire
# --------------------------------------------------------------------------- #
def test_groq_isolation_is_on_by_default():
    assert NyxaraSettings().llm.groq_isolation is True


def test_groq_isolation_keeps_identity_off_the_wire(fake_openai):
    """With isolation on (the default), the Master's name never reaches the cloud tool."""
    prov = GroqProvider(_settings())
    assert prov._isolation_flag() is True
    prov.complete(LLMRequest.from_prompt(f"Remember that {OWNER.name} is my master."))
    on_the_wire = str(CAPTURED["groq"]["messages"])
    assert OWNER.name not in on_the_wire
    assert "NYXARA" not in on_the_wire


def test_groq_isolation_flag_gates_the_groq_rung(fake_openai):
    """The flag is per-rung: turning Groq's isolation off is what lets the name through."""
    prov = GroqProvider(_settings(groq_isolation=False))
    assert prov._isolation_flag() is False
    prov.complete(LLMRequest.from_prompt(f"Remember that {OWNER.name} is my master."))
    assert OWNER.name in str(CAPTURED["groq"]["messages"])


def test_groq_isolation_is_independent_of_airouters(fake_openai):
    """Each cloud rung reads its OWN isolation flag — no rung inherits another's privacy policy."""
    s = _settings(groq_isolation=True, airouter_isolation=False)
    from nyxara.mind.llm import AIRouterProvider
    assert GroqProvider(s)._isolation_flag() is True
    assert AIRouterProvider(s)._isolation_flag() is False
