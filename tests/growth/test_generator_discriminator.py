"""Tests for nyxara.growth.generator_discriminator — adversarial self-play (RLSP)."""

from __future__ import annotations

from nyxara.growth.generator_discriminator import Flaw, RLSPLoop


class _FlawThenSoundLLM:
    """A rival that flags one flaw on the first look, then judges the rewrite sound."""

    def __init__(self):
        self.seen = 0

    def generate(self, prompt, *, system=None, **kw):
        return "Paris is the capital of France; ~2.1M residents in the city proper."

    def generate_json(self, prompt, *, system=None, **kw):
        self.seen += 1
        if self.seen == 1:
            return {"flaws": [{"description": "Conflates city-proper vs metro population.",
                               "severity": 0.8}], "sound": False}
        return {"flaws": [], "sound": True}


def test_discriminator_forces_revision_then_converges():
    loop = RLSPLoop(_FlawThenSoundLLM(), max_rounds=4, use_llm=True)
    res = loop.solve("How many people live in Paris?")
    assert res.rounds >= 1                       # the rival forced at least one revision
    assert res.converged                         # then judged the rewrite sound
    assert any("metro" in f.description.lower() or "city-proper" in f.description.lower()
               for f in res.flaws)


def test_sound_draft_converges_with_no_revision():
    class _AlwaysSound:
        def generate(self, *a, **k):
            return "ok"
        def generate_json(self, *a, **k):
            return {"flaws": [], "sound": True}

    class _NoOpCritic:           # isolate the LLM verdict from the deterministic probes
        def critique(self, ctx):
            from types import SimpleNamespace
            return SimpleNamespace(findings=[])

    loop = RLSPLoop(_AlwaysSound(), max_rounds=3, use_llm=True, critic=_NoOpCritic())
    res = loop.harden("x", {"kind": "respond", "text": "A sound answer.",
                            "confidence": 0.9, "risk": "low", "reversible": True})
    assert res.converged and res.rounds == 0


def test_offline_uses_deterministic_critic_and_never_crashes():
    loop = RLSPLoop(None, max_rounds=2)
    res = loop.harden("hi", {"kind": "respond", "text": "Hello, Master.",
                             "confidence": 0.9, "risk": "low", "reversible": True})
    # with no LLM it cannot rewrite, so it keeps the draft — but it still returns a result
    assert res.decision["text"] == "Hello, Master."
    assert isinstance(res.flaws, list)


def test_respects_round_budget():
    class _AlwaysFlawed:
        def generate(self, *a, **k):
            return "revised text"
        def generate_json(self, *a, **k):
            return {"flaws": [{"description": "still wrong", "severity": 0.9}],
                    "sound": False}

    loop = RLSPLoop(_AlwaysFlawed(), max_rounds=3, use_llm=True)
    res = loop.solve("an unsolvable nit")
    assert res.rounds <= 3 and not res.converged


def test_flaw_dataclass_roundtrips():
    f = Flaw(description="leap of logic", severity=0.7, source="llm")
    d = f.to_dict()
    assert d["description"] == "leap of logic" and d["severity"] == 0.7 and d["source"] == "llm"
