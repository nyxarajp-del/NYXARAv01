"""Tests for the air-gapped mind — guard/isolation_envelope.py (Part K).

Verifies that the envelope abstracts named identity/secrets bijectively and re-hydrates losslessly,
and that a caller decides per-provider whether it applies.

The old end-to-end "only abstract tokens cross the wire" test is gone with the wire: the cloud rungs
this guarded were removed from mind/llm.py, so the module currently has no caller. It is kept
because the threat returns the moment anything external is added back.
"""
from __future__ import annotations



from nyxara.guard.isolation_envelope import IsolationEnvelope
from nyxara.kernel.config import OWNER, NyxaraSettings, Profile


def _settings(**over) -> NyxaraSettings:
    s = NyxaraSettings.for_profile(Profile.TEST)
    for k, v in over.items():
        setattr(s.llm, k, v)
    return s


# --------------------------------------------------------------------------- #
# Unit — bijective abstraction / rehydration
# --------------------------------------------------------------------------- #
def test_abstract_hides_identity_and_secrets():
    env = IsolationEnvelope(_settings(), extra_secrets=["CorePassphrase42"])
    text = (f"I am NYXARA, serving {OWNER.name} ({OWNER.handle}, {OWNER.email}). "
            f"secret=CorePassphrase42")
    abstract = env.abstract(text)
    for leaked in ("NYXARA", OWNER.name, OWNER.handle, OWNER.email, "CorePassphrase42"):
        assert leaked not in abstract, f"{leaked!r} leaked past the air-gap"
    # round-trip is lossless
    assert env.rehydrate(abstract) == text


def test_rehydrate_maps_tokens_back():
    env = IsolationEnvelope(_settings())
    abstract = env.abstract("NYXARA")
    assert abstract != "NYXARA" and abstract in env.mapping().values()
    assert env.rehydrate(abstract) == "NYXARA"


def test_disabled_is_passthrough():
    env = IsolationEnvelope(_settings(), enabled=False)
    assert env.enabled() is False
    assert env.abstract("NYXARA and Jaypal Khoja") == "NYXARA and Jaypal Khoja"


def test_the_caller_decides_per_provider():
    """A calling provider supplies its OWN isolation flag via ``enabled=``, so privacy stays a
    per-provider decision rather than one global switch."""
    assert IsolationEnvelope(_settings(), enabled=False).enabled() is False
    assert IsolationEnvelope(_settings(), enabled=True).enabled() is True
    assert IsolationEnvelope(_settings(), enabled=None).enabled() is True   # no scope -> protect


def test_the_envelope_defaults_to_protecting():
    """It has no caller today — the cloud rungs it guarded are gone from mind/llm.py. The safe
    default is therefore ON: anything that builds one without a provider scope wants protection."""
    s = _settings()
    assert IsolationEnvelope(s).enabled() is True
    assert IsolationEnvelope(s, enabled=False).enabled() is False


def test_word_boundary_avoids_false_hits():
    # "JP" (the handle) must not be abstracted inside an unrelated word like "JPEG".
    env = IsolationEnvelope(_settings())
    out = env.abstract("Open the JPEG file")
    assert out == "Open the JPEG file"


def test_the_round_trip_is_lossless_without_any_provider():
    """What the old on-the-wire test proved, asserted where it still holds: abstract then rehydrate
    returns the original terms exactly. There is no wire left to send them over."""
    env = IsolationEnvelope(_settings())
    original = "NYXARA reports to Jaypal Khoja about the kernel."
    abstracted = env.abstract(original)
    assert "NYXARA" not in abstracted and "Jaypal" not in abstracted
    assert env.rehydrate(abstracted) == original
