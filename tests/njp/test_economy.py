"""The cognitive budget, and the one place a cheap turn could still reach an expensive organ.

**What was already governed.** A greeting cannot reach the cortex and could not before this
module: ``CognitivePolicy.forbids_world_knowledge`` refuses it the world-knowledge pathways, and
``Router.route`` reports ``cortex_permitted=False`` and ``extraction_permitted=False`` for every
social act. The first test here pins that, because a budget layer that claimed credit for it would
be claiming credit for someone else's work.

**What was not.** ``Dialogue.respond`` called ``_render`` on every turn that produced content, so
``"Hii Master 👋"`` — a finished sentence her own social path wrote — was sent to a language model
to be "rewritten as one natural, direct reply". Reasoning is 2-12 ms on this machine; that call is
seconds. The tests below drive the whole path with a counting fake model, because the claim is
about calls avoided and a claim about calls has to be counted rather than argued.
"""

from __future__ import annotations

import pytest

from nyxara.njp.economy import Budget, CognitiveEconomy, Tier
from nyxara.njp.voice import Dialogue


class _Act:
    def __init__(self, kind: str) -> None:
        self.kind = kind


class _CountingLLM:
    """A fluent surface that records being asked to do work."""

    def __init__(self) -> None:
        self.calls = 0
        self.caps = []

    def generate(self, prompt, system=None, max_tokens=220):
        self.calls += 1
        self.caps.append(max_tokens)
        return "rendered: " + prompt.split("Conclusion: ")[-1].split("\n")[0]

    def provider_status(self):
        return {"litertlm": True}

    def chosen_provider(self):
        return type("_P", (), {"name": "litertlm"})()


# --------------------------------------------------------------------------- #
# what was already governed — pinned so the budget cannot take credit for it
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("greeting", ["Hii", "thanks", "bye", "how are you"])
def test_a_social_turn_could_never_reach_the_cortex(greeting):
    from nyxara.njp.brain import NJPBrain
    routing = NJPBrain().think(greeting).routing
    assert routing is not None
    assert not routing.cortex_permitted
    assert not routing.extraction_permitted


# --------------------------------------------------------------------------- #
# the rule — stated on the content, not on a list of acts
# --------------------------------------------------------------------------- #

def test_content_that_is_already_an_utterance_is_not_sent_to_be_reworded():
    economy = CognitiveEconomy()
    assert not economy.assess(content="Hii Master 👋", act=_Act("greeting")).render


def test_a_bare_fact_is_a_conclusion_and_still_has_to_be_phrased():
    """The case rendering exists for: "water" becomes "A sparrow needs water.\""""
    economy = CognitiveEconomy()
    assert economy.assess(content="water", act=_Act("knowledge_query")).render


def test_an_unfamiliar_act_is_governed_by_the_rule_rather_than_by_a_list():
    """A finished sentence needs no renderer whoever wrote it, under any act kind."""
    economy = CognitiveEconomy()
    finished = economy.assess(content="The reactor scrams when coolant pressure falls.",
                              act=_Act("some_act_nobody_has_invented"))
    assert not finished.render


@pytest.mark.parametrize("kind,tier", [
    ("greeting", Tier.TRIVIAL),
    ("thanks", Tier.TRIVIAL),
    ("causal_query", Tier.HIGH),
])
def test_the_tier_follows_the_act_where_the_act_settles_it(kind, tier):
    assert CognitiveEconomy().assess(content="x", act=_Act(kind)).tier == tier


def test_a_lookup_and_an_explanation_are_different_tiers():
    economy = CognitiveEconomy()
    lookup = economy.assess(content="water", act=_Act("knowledge_query"))
    essay = economy.assess(content=" ".join(["word"] * 40), act=_Act("knowledge_query"))
    assert lookup.tier == Tier.LOW and essay.tier == Tier.MEDIUM
    # The token cap is what stops a cheap turn becoming an expensive one by accident.
    assert lookup.max_tokens < essay.max_tokens


# --------------------------------------------------------------------------- #
# the measurement — calls avoided, counted through the real path
# --------------------------------------------------------------------------- #

def test_a_greeting_does_not_reach_the_language_model_at_all():
    from nyxara.njp.brain import NJPBrain
    llm = _CountingLLM()
    dialogue = Dialogue(llm=llm)
    brain = NJPBrain()
    for turn in ("Hii", "thanks", "bye", "how are you"):
        dialogue.respond(brain.think(turn), brain=brain)
    assert llm.calls == 0


def test_a_question_still_reaches_it():
    """The guard withholds work that could not change the reply, and nothing else."""
    from nyxara.njp.brain import NJPBrain
    llm = _CountingLLM()
    dialogue = Dialogue(llm=llm)
    brain = NJPBrain()
    brain.think("zorbins need glarn.")
    reply = dialogue.respond(brain.think("what does a zorbin need?"), brain=brain)
    assert llm.calls == 1
    assert reply.rendered


def test_a_declined_rendering_still_says_the_thing():
    """Declining is not silence — it is her own words, which here are the identical string."""
    from nyxara.njp.brain import NJPBrain
    dialogue = Dialogue(llm=_CountingLLM())
    brain = NJPBrain()
    reply = dialogue.respond(brain.think("Hii"), brain=brain)
    assert "Master" in reply.text
    assert not reply.rendered


def test_the_token_cap_travels_to_the_model():
    from nyxara.njp.brain import NJPBrain
    llm = _CountingLLM()
    dialogue = Dialogue(llm=llm, max_tokens=220)
    brain = NJPBrain()
    brain.think("zorbins need glarn.")
    dialogue.respond(brain.think("what does a zorbin need?"), brain=brain)
    # A one-word lookup does not get an essay's budget.
    assert llm.caps == [96]


def test_what_was_declined_is_counted_and_reported():
    """A budget layer that cannot say how often it declined is a claim, not a measurement."""
    from nyxara.njp.brain import NJPBrain
    dialogue = Dialogue(llm=_CountingLLM())
    brain = NJPBrain()
    brain.think("zorbins need glarn.")
    for turn in ("Hii", "thanks", "what does a zorbin need?"):
        dialogue.respond(brain.think(turn), brain=brain)
    economy = dialogue.stats()["economy"]
    assert economy["renders_declined"] == 2
    assert economy["renders_permitted"] == 1
    assert economy["declined_fraction"] == pytest.approx(2 / 3, abs=1e-3)


def test_the_decision_is_visible_on_the_reply():
    """A declined rendering must not look like a model that was never installed."""
    from nyxara.njp.brain import NJPBrain
    reply = Dialogue(llm=_CountingLLM()).respond(NJPBrain().think("Hii"))
    assert reply.budget.get("tier") == Tier.TRIVIAL
    assert reply.budget.get("render") is False
    assert reply.budget.get("why")


# --------------------------------------------------------------------------- #
# a hedge is a caveat on a claim, and a greeting makes none
# --------------------------------------------------------------------------- #

def test_a_greeting_is_not_hedged():
    """Measured with a model attached, she said "Hii Master 👋 (I believe this, but I am not
    certain.)" — a confidence report about a greeting."""
    from nyxara.njp.brain import NJPBrain
    reply = Dialogue(llm=_CountingLLM()).respond(NJPBrain().think("Hii"))
    assert "believe" not in reply.text.lower()
    assert "confidence" not in reply.text.lower()


def test_a_believed_claim_is_still_hedged():
    """The caveat is withheld from turns that make no claim, and from nothing else."""
    from nyxara.njp.brain import NJPBrain
    brain = NJPBrain()
    brain.think("zorbins need glarn.")
    reply = Dialogue(llm=_CountingLLM()).respond(brain.think("what does a zorbin need?"),
                                                 brain=brain)
    assert "confidence" in reply.text.lower()


# --------------------------------------------------------------------------- #
# a broken accountant may not silence her
# --------------------------------------------------------------------------- #

def test_an_absent_economy_withholds_nothing():
    from nyxara.njp.brain import NJPBrain
    llm = _CountingLLM()
    dialogue = Dialogue(llm=llm, economy=None)
    dialogue.economy = None            # as if the module could not be imported
    brain = NJPBrain()
    reply = dialogue.respond(brain.think("Hii"), brain=brain)
    assert llm.calls == 1 and reply.rendered


def test_a_raising_economy_withholds_nothing():
    from nyxara.njp.brain import NJPBrain

    class _Broken:
        def assess(self, **_kwargs):
            raise RuntimeError("boom")

    llm = _CountingLLM()
    dialogue = Dialogue(llm=llm)
    dialogue.economy = _Broken()
    assert dialogue.respond(NJPBrain().think("Hii")).text
    assert llm.calls == 1


def test_the_economy_never_raises():
    economy = CognitiveEconomy()
    for content in ("", "   ", "x" * 10000, "\x00"):
        for act in (None, _Act(""), object()):
            assert isinstance(economy.assess(content=content, act=act), Budget)
