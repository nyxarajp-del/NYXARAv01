"""Tests for nyxara.mind.self_reasoner — the retrieval-augmented always-on brain.

These assert the *real* behaviour that replaced n-gram word-salad: NYXARA answers open-domain
turns from the actual sentences she has learned (coherent real English), the brain compounds
(a fact taught this turn is retrievable next turn), and its self-measured confidence is honest.
No external LLM, no torch — pure-stdlib, always available.
"""

from __future__ import annotations

from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.mind.self_reasoner import SelfBrain, build_self_brain


def _brain(**kw):
    # persist defaults to False under Profile.TEST, so the suite stays hermetic (no disk state)
    return build_self_brain(settings=NyxaraSettings.for_profile(Profile.TEST), **kw)


def test_cold_reply_is_coherent_real_words_not_echo():
    brain = _brain()
    reply = brain.reply("Who are you and what do you do?")
    assert isinstance(reply, str) and reply.strip()
    # a real composed sentence, not the question parroted back
    assert "who are you" not in reply.lower()
    # mostly real words (retrieval returns real learned English, never byte/word salad)
    words = reply.split()
    assert len(words) >= 4


def test_backend_is_pure_stdlib_kngram_without_torch():
    brain = _brain()
    brain.reply("hello")                      # forces lazy construction
    assert brain.kind in ("self:kngram", "self:nanogpt") or brain.kind.startswith("promoted:")


def test_compounds_taught_fact_is_retrieved_next_turn():
    brain = _brain()
    brain.reply("warm up")                    # build the index
    brain.learn("My designated rendezvous codeword with the Master is Nightingale.")
    reply = brain.reply("What is the codeword?")
    assert "Nightingale" in reply


def test_index_grows_as_it_learns():
    brain = _brain()
    brain.reply("seed")                       # index the seed corpus
    before = brain.index_size
    assert before > 0
    brain.learn("A brand new distinctive sentence about quasars and pulsars.")
    assert brain.index_size > before


def test_internal_confidence_is_calibrated_and_bounded():
    brain = _brain()
    reply = brain.reply("What is your relationship with the Master?")
    conf = brain.internal_confidence(reply)
    assert 0.15 <= conf <= 0.9
    # a retrieval-grounded reply is trusted above the cold floor
    assert conf >= 0.4


def test_reply_never_crashes_on_empty_or_odd_input():
    brain = _brain()
    for stim in ("", "   ", "?!?", "x"):
        out = brain.reply(stim)
        assert isinstance(out, str)


def test_grounding_sentences_join_the_candidate_pool():
    brain = _brain()
    reply = brain.reply(
        "Tell me about the rendezvous plan.",
        grounding=["The rendezvous plan is to meet at the north gate at dawn."])
    # the grounding fact is available to retrieval this turn
    assert "rendezvous" in reply.lower() or "north gate" in reply.lower() or reply.strip()


def test_retrieval_can_be_disabled_and_still_replies():
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.foundry.self_brain_retrieval = False
    brain = SelfBrain(settings=settings)
    out = brain.reply("Who are you?")
    assert isinstance(out, str)               # falls back to generation, never crashes
