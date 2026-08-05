"""Tests for nyxara.mind.swarm — the self-improving Society of Mind (debate + RSI)."""

from __future__ import annotations

from typing import List

from nyxara.agency.permissions import Capability
from nyxara.kernel.config import LLMProvider as ProviderName
from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.mind.llm import LLM, LLMProviderBase, Usage
from nyxara.mind.swarm import DeliberativeSwarm, PersonaScore
from nyxara.memory.store import MemoryStore


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class _Recording(LLMProviderBase):
    """A real (non-mock) provider that records every prompt and returns a fixed answer."""

    def __init__(self, answer: str = "a balanced position on the matter") -> None:
        super().__init__()
        self.name = "litertlm"
        self._answer = answer
        self.prompts: List[str] = []

    def available(self) -> bool:
        return True

    def default_model(self) -> str:
        return "litertlm-1"

    def _complete(self, req, model):
        try:
            self.prompts.append(req.last_user())
        except Exception:  # noqa: BLE001 — recording is best-effort in the test
            self.prompts.append(str(req))
        return (self._answer, "stop", Usage(1, 1), None)


def _llm_backed():
    """An LLM facade whose chosen provider is a real recording provider (not mock)."""
    settings = NyxaraSettings.for_profile(Profile.DEV)
    settings.llm.provider = ProviderName.LITERTLM
    provider = _Recording()
    return LLM(settings=settings, providers={"litertlm": provider}), settings, provider


def _offline(*, rounds: int = 3, memory=None, **swarm_overrides) -> DeliberativeSwarm:
    """An offline swarm on its OWN settings, so per-test cfg tweaks never leak (get_settings
    returns a process-wide singleton)."""
    settings = NyxaraSettings.for_profile(Profile.DEV)
    for key, value in swarm_overrides.items():
        setattr(settings.swarm, key, value)
    return DeliberativeSwarm(llm=None, memory=memory, settings=settings, rounds=rounds)


# --------------------------------------------------------------------------- #
# multi-round debate
# --------------------------------------------------------------------------- #
def test_multi_round_runs_offline():
    swarm = _offline(rounds=3)
    result = swarm.deliberate("Should NYXARA prioritise speed or correctness?")
    assert len(result.rounds) == 3
    # every round proposes for each selected persona
    for r in result.rounds:
        assert set(r.proposals.keys()) == set(result.personas_used)
        assert all(text for text in r.proposals.values())
    assert result.answer
    assert 0.0 < result.confidence <= 1.0


def test_round_count_follows_config():
    swarm = _offline(rounds=5)
    assert len(swarm.deliberate("any problem").rounds) == 5


def test_convene_returns_candidate():
    swarm = _offline(rounds=2)
    cand = swarm.convene("What is a robust path to AGI safety?")
    assert cand.kind == "respond"
    assert cand.capability == Capability.MESSAGE_SEND
    assert "swarm" in cand.rationale.lower()
    assert 0.0 < cand.confidence <= 1.0


def test_quality_signal_bounded():
    result = _offline(rounds=3).deliberate("bound me")
    assert 0.0 <= result.quality <= 1.0


def test_synthesis_offline_deterministic():
    a = _offline(rounds=3).deliberate("define fairness").answer
    b = _offline(rounds=3).deliberate("define fairness").answer
    assert a == b


# --------------------------------------------------------------------------- #
# rounds 2..N actually see peers (LLM-backed)
# --------------------------------------------------------------------------- #
def test_refine_rounds_see_peers():
    llm, _settings, provider = _llm_backed()
    swarm = DeliberativeSwarm(llm=llm, rounds=2)
    swarm.deliberate("a debatable problem")
    # round-2 persona prompts must contain the peer-proposals block
    assert any("Your peers in the swarm proposed" in p for p in provider.prompts)


# --------------------------------------------------------------------------- #
# scoring + self-improvement of the roster
# --------------------------------------------------------------------------- #
def test_persona_scoring_updates():
    swarm = _offline(rounds=2)
    result = swarm.deliberate("score me")
    for name in result.personas_used:
        assert name in result.contributions
        assert 0.0 <= result.contributions[name] <= 1.0
        assert swarm._scores[name].problems_seen == 1


def test_scores_persist_and_reload(tmp_path):
    mem = MemoryStore()
    swarm = _offline(rounds=2, memory=mem)
    for q in ("problem A", "problem B", "problem C"):
        swarm.deliberate(q)
    before = swarm.scores()
    assert swarm._problems == 3

    # round-trip the whole store through disk, then a fresh swarm recovers the learned table
    path = mem.save(str(tmp_path / "mem.json"))
    mem2 = MemoryStore()
    mem2.load(path)
    swarm2 = DeliberativeSwarm(llm=None, memory=mem2, rounds=2)
    swarm2._ensure_loaded()
    assert swarm2._problems == 3
    assert swarm2.scores() == before


def test_useless_persona_dropped():
    swarm = _offline(rounds=1)
    # a persona that has contributed nothing over many problems is below the retention floor
    swarm._scores["Philosopher"] = PersonaScore(
        name="Philosopher", problems_seen=10, cumulative_contribution=0.0)
    selected, dropped, _ = swarm._adapt_roster("any problem")
    assert "Philosopher" in dropped
    assert "Philosopher" not in [p.name for p in selected]
    assert len(selected) >= 2          # the floor keeps a real debate alive


def test_roster_never_collapses_below_two():
    swarm = _offline(rounds=1)
    # mark EVERY persona useless — the floor must still seat at least two
    for p in swarm.roster:
        swarm._scores[p.name] = PersonaScore(
            name=p.name, problems_seen=10, cumulative_contribution=0.0)
    selected, dropped, _ = swarm._adapt_roster("any problem")
    assert len(selected) >= 2
    assert dropped == []               # collapse-guard revives the whole roster


def test_high_contributor_survives_truncation():
    swarm = _offline(rounds=1)
    swarm.cfg.max_personas = 2
    swarm._scores["Critic"] = PersonaScore(
        name="Critic", problems_seen=5, cumulative_contribution=5.0)   # mean 1.0
    selected, _dropped, _ = swarm._adapt_roster("a problem")
    assert len(selected) == 2
    assert "Critic" in [p.name for p in selected]


# --------------------------------------------------------------------------- #
# dynamic specialist spawning
# --------------------------------------------------------------------------- #
def test_spawn_specialist_offline():
    swarm = _offline(rounds=2)
    result = swarm.deliberate("Design an encryption cipher resistant to attack.")
    assert result.spawned == "Cryptographer"
    assert "Cryptographer" in result.personas_used


def test_no_spawn_when_disabled():
    swarm = _offline(rounds=2)
    swarm.cfg.allow_spawn = False
    result = swarm.deliberate("Design an encryption cipher resistant to attack.")
    assert result.spawned is None


# --------------------------------------------------------------------------- #
# graceful degradation
# --------------------------------------------------------------------------- #
def test_no_memory_is_noop():
    swarm = _offline(rounds=2, memory=None)
    result = swarm.deliberate("no memory here")   # must not raise
    assert result.answer
    assert swarm._find_record() is None


# --------------------------------------------------------------------------- #
# offline proposals are REAL lens-specific analysis, not placeholder filler
# --------------------------------------------------------------------------- #
def test_offline_proposals_are_grounded_not_filler():
    from nyxara.mind.swarm import _offline_persona_proposal, Persona

    p = Persona(name="Engineer", system="…", weight=0.85)
    text = _offline_persona_proposal(p, "How should we encrypt user data at rest?", 1)
    assert "encrypt" in text.lower() or "data" in text.lower()   # grounded in the topic
    assert "build it and where it would fail" in text            # the Engineer's real lens
    assert "grounded in engineer principles" not in text         # old filler is gone


def test_offline_proposal_adapts_to_question_type():
    from nyxara.mind.swarm import _offline_persona_proposal, Persona

    p = Persona(name="Scientist", system="…", weight=0.9)
    why = _offline_persona_proposal(p, "Why does the cache thrash?", 1)
    how = _offline_persona_proposal(p, "How do I fix the cache?", 1)
    assert "mechanism" in why
    assert "steps" in how
    assert why != how


def test_offline_rounds_visibly_differ():
    from nyxara.mind.swarm import _offline_persona_proposal, Persona

    p = Persona(name="Critic", system="…", weight=0.8)
    r1 = _offline_persona_proposal(p, "Should we ship now?", 1)
    r2 = _offline_persona_proposal(p, "Should we ship now?", 2)
    assert r1 != r2
    assert "Refining after the debate" in r2


def test_offline_synthesis_is_structured_multi_lens():
    swarm = _offline(rounds=2)
    answer = swarm.deliberate("How should we encrypt user data at rest?").answer
    assert "lenses" in answer
    assert "NYXARA's synthesis:" in answer
    assert answer.count("•") >= 3                       # several persona positions integrated
