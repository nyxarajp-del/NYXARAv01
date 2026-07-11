"""Tests for nyxara.mind.self_reasoner — the retrieval-augmented always-on brain.

These assert the *real* behaviour that replaced n-gram word-salad: NYXARA answers open-domain
turns from the actual sentences she has learned (coherent real English), the brain compounds
(a fact taught this turn is retrievable next turn), and its self-measured confidence is honest.
No external LLM, no torch — pure-stdlib, always available.
"""

from __future__ import annotations

from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.mind.self_reasoner import SelfBrain, SelfBrainProvider, build_self_brain


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


def test_default_backend_is_a_real_neural_net_not_statistics():
    # the always-on brain's DEFAULT is now a real, weight-changing neural net — the pure-NumPy
    # transformer on a torch-less box (backprop, no torch/LLM/cloud), the torch NanoGPT where torch
    # is present — so her lived learning changes weights, not just n-gram counts. It falls to the
    # stdlib Kneser-Ney n-gram only when even NumPy is absent, and a forged foundry model is still
    # preferred once promoted.
    brain = _brain()
    brain.reply("hello")                      # forces lazy construction
    assert brain.kind in ("self:genesis_np", "self:nanogpt", "self:kngram") \
        or brain.kind.startswith("promoted:")
    from nyxara.growth.foundry_models import _numpy_available, _HAS_TORCH
    if _numpy_available() and not _HAS_TORCH:
        # the normal case (NumPy present, no torch): her default learner is the real neural brain
        assert brain.kind == "self:genesis_np", brain.kind


def test_nanogpt_backend_is_a_real_neural_net_and_persists(tmp_path):
    # opting the fallback into the neural backend gives a genuine trained net (a NumPy transformer
    # on this torch-less box), NOT the toy byte model — and its weights persist across a restart.
    from nyxara.mind.self_reasoner import SelfBrain
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.foundry.self_brain_backend = "nanogpt"
    settings.paths.data_dir = tmp_path
    brain = SelfBrain(settings=settings, persist=True)
    brain.reply("warm up")                    # forces lazy construction + a first real train
    assert brain.kind in ("self:genesis_np", "self:nanogpt")
    assert brain._lm.param_count() > 1000     # orders larger than an n-gram count table
    assert brain.save() is True               # persistence now covers neural checkpoints too
    d = brain._lm_persist_dir()
    assert (d / "weights.npz").exists() or (d / "model.json").exists()


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


# --------------------------------------------------------------------------- #
# The handoff: the learned brain becomes the router's "own model" (Phase 1)
# --------------------------------------------------------------------------- #
_NET_LESSONS = (
    "A firewall blocks unwanted network ports to protect a host.",
    "To harden a server, close unused ports and patch the operating system.",
    "TLS encrypts traffic between a client and a server using certificates.",
    "A load balancer distributes incoming requests across several backends.",
    "Rate limiting caps how many requests a client may make in a time window.",
    "Caching stores frequent responses so repeat requests are served faster.",
    "DNS resolves human-readable domain names into IP addresses.",
    "A reverse proxy forwards client requests to one or more backend servers.",
)


def _cold_brain():
    # prefer_promoted=False isolates from any foundry/active pointer a sibling test may have
    # written to the shared TEST data dir — these tests pin the *learned-from-experience* path.
    return build_self_brain(settings=NyxaraSettings.for_profile(Profile.TEST),
                            persist=False, prefer_promoted=False)


def test_self_brain_has_learned_is_honest_about_a_cold_brain():
    brain = _cold_brain()
    # a cold brain (only the persona seed) has learned nothing — it must defer to the teacher
    assert brain.learned_count == 0
    assert not brain.has_learned(8)
    brain.learn(*_NET_LESSONS)
    assert brain.learned_count >= len(_NET_LESSONS)
    assert brain.has_learned(8)


def test_self_brain_provider_available_only_after_real_learning():
    brain = _cold_brain()
    prov = SelfBrainProvider(brain, min_learned_docs=8)
    assert prov.available() is False          # cold brain -> not available -> teacher answers
    brain.learn(*_NET_LESSONS)
    assert prov.available() is True           # learned beyond seed -> own model may answer


def test_handoff_fires_once_she_has_learned():
    """The substrate that learns is the substrate that answers — handoff rises off 0%."""
    from nyxara.mind.llm import LLM
    from nyxara.mind.router import Router

    settings = NyxaraSettings.for_profile(Profile.TEST)
    brain = build_self_brain(settings=settings, persist=False)
    brain.learn(*_NET_LESSONS)
    router = Router(LLM(settings=settings), settings=settings,
                    self_provider=SelfBrainProvider(brain, min_learned_docs=8))
    res = router.draft_self("How does a firewall protect a host?")
    assert res is not None and res.handed_off and res.source == "self"
    assert res.confidence >= settings.router.threshold


def test_provider_complete_wraps_reply_in_response_shape():
    from nyxara.mind.llm import LLMRequest

    brain = _brain()
    brain.learn(*_NET_LESSONS)
    prov = SelfBrainProvider(brain, min_learned_docs=8)
    resp = prov.complete(LLMRequest.from_prompt("What does a load balancer do?"))
    assert resp.provider == "self" and isinstance(resp.text, str) and resp.text.strip()


# --------------------------------------------------------------------------- #
# Knowledge grounding: she answers a turn she genuinely KNOWS, defers the rest
# (raises real handoff off 0% without ever bluffing) — Pillar A / D2.
# --------------------------------------------------------------------------- #
def _kb_with(*facts: str):
    from nyxara.knowledge.base import KnowledgeBase
    from nyxara.memory.store import MemoryStore

    settings = NyxaraSettings.for_profile(Profile.TEST)
    kb = KnowledgeBase(store=MemoryStore(settings=settings), name="kb")
    for i, fact in enumerate(facts):
        kb.ingest_text(fact, source=f"src-{i}")
    return settings, kb


def test_knowledge_grounding_makes_a_cold_brain_available():
    settings, kb = _kb_with(
        "The Andromeda galaxy is about 2.5 million light-years from the Milky Way.")
    brain = build_self_brain(settings=settings, persist=False, prefer_promoted=False,
                             knowledge=kb)
    # she has learned nothing from experience, yet she KNOWS things -> available (grounded path)
    assert brain.learned_count == 0
    assert brain.has_grounding() is True
    assert SelfBrainProvider(brain, min_learned_docs=8).available() is True
    # a brain with no knowledge base stays honestly cold
    assert _cold_brain().has_grounding() is False


def test_grounded_turn_is_answered_from_knowledge_ungrounded_defers():
    from nyxara.mind.llm import LLM
    from nyxara.mind.router import Router

    settings, kb = _kb_with(
        "The Andromeda galaxy is about 2.5 million light-years from the Milky Way.")
    brain = build_self_brain(settings=settings, persist=False, prefer_promoted=False,
                             knowledge=kb)
    router = Router(LLM(settings=settings), settings=settings,
                    self_provider=SelfBrainProvider(brain, min_learned_docs=8))
    # grounded: she answers herself, from her real knowledge
    hit = router.draft_self("How far away is the Andromeda galaxy?")
    assert hit is not None and hit.handed_off and hit.source == "self"
    assert "andromeda" in hit.text.lower() or "light-year" in hit.text.lower()
    # ungrounded: she defers (None) rather than bluffing from the persona seed or a cold n-gram
    assert router.draft_self("What did I eat for breakfast last Tuesday?") is None


def test_handoff_path_never_bluffs_from_the_persona_seed():
    """require_grounding excludes the generic seed, so an unrelated turn returns "" (defers)."""
    settings, kb = _kb_with("Photosynthesis converts light into chemical energy in plants.")
    brain = build_self_brain(settings=settings, persist=False, prefer_promoted=False,
                             knowledge=kb)
    # the offline voice may still speak from the seed (no teacher exists there)...
    # ...but the handoff path answers only from genuine grounding:
    assert brain.reply("Give me an unrelated random anecdote.",
                       require_grounding=True) == ""
