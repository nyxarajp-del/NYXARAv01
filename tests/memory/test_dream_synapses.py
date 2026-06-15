"""Tests for the Dream State in nyxara.memory.dream — distillation + log pruning +
Deep Memory Synapses.

Hermetic and offline: a real MindScope + MemoryStore are populated, then a dream session
distils recurring principles, deletes useless logs, and fixes durable synapses.
"""

from __future__ import annotations

import time

from nyxara.memory.consolidation import Consolidator
from nyxara.memory.dream import DreamReport, DreamSession
from nyxara.memory.provenance import Provenance, SourceType
from nyxara.memory.store import MemoryStore, MemoryType
from nyxara.observe.mindscope import MindScope, ThoughtKind


def _mind_with_pattern(repeats: int = 3, noise: int = 250) -> MindScope:
    ms = MindScope()
    for _ in range(repeats):
        ms.record(ThoughtKind.INFERENCE, "rotating logs keeps the disk healthy", salience=0.6)
    for i in range(noise):
        ms.record(ThoughtKind.PERCEPTION, f"tick {i}", salience=0.05)
    return ms


# --------------------------------------------------------------------------- #
# distillation + synapses
# --------------------------------------------------------------------------- #
def test_distills_principles_and_fixes_synapses():
    store = MemoryStore()
    ds = DreamSession(mind=_mind_with_pattern(), memory=store, distill_min_support=2)
    rep = ds.dream_state(deep=True)
    assert isinstance(rep, DreamReport)
    assert rep.deep_state
    assert rep.principles_distilled >= 1
    assert rep.synapses_fixed >= 1
    # the synapses are high-importance, deep-synapse-tagged SEMANTIC memories
    synapses = [r for r in store._kv.values() if "deep-synapse" in r.tags]
    assert synapses
    assert all(r.importance >= 0.9 and r.mem_type is MemoryType.SEMANTIC for r in synapses)
    assert ds.deep_synapse_count() == rep.synapses_fixed


def test_synapses_survive_forgetting():
    store = MemoryStore()
    ds = DreamSession(mind=_mind_with_pattern(), memory=store, distill_min_support=2)
    rep = ds.dream_state(deep=True)
    assert rep.synapses_fixed >= 1
    # run the forgetting curve far into the future — protected synapses must remain
    Consolidator(store).forget_pass(now=time.time() + 86400 * 3650)
    assert ds.deep_synapse_count() == rep.synapses_fixed


def test_repeated_dreams_do_not_duplicate_synapses():
    store = MemoryStore()
    ds = DreamSession(mind=_mind_with_pattern(), memory=store, distill_min_support=2)
    first = ds.dream_state(deep=True)
    second = ds.dream_state(deep=True)
    assert first.synapses_fixed >= 1
    assert second.synapses_fixed == 0          # same principles -> revise, not pile up


# --------------------------------------------------------------------------- #
# deleting useless logs, respecting protection
# --------------------------------------------------------------------------- #
def test_deletes_low_salience_logs():
    ms = _mind_with_pattern(noise=300)
    before = len(ms)
    ds = DreamSession(mind=ms, memory=MemoryStore())
    rep = ds.dream_state(deep=True)
    assert rep.logs_deleted >= 1
    assert len(ms) < before                    # noise pruned


def test_deletes_transient_unprotected_memories_only():
    store = MemoryStore()
    # a transient, low-importance scaffolding memory (deletable)
    junk = store.remember("idle scratch note", mem_type=MemoryType.SEMANTIC,
                          importance=0.1, tags=["idle"])
    # an owner memory and a high-importance memory (must be protected)
    owner = store.remember("the Master's standing order", mem_type=MemoryType.SEMANTIC,
                           provenance=Provenance(SourceType.OWNER, confidence=1.0),
                           importance=0.5, tags=["owner"])
    precious = store.remember("a hard-won lesson", mem_type=MemoryType.SEMANTIC,
                              importance=0.95, tags=["lesson"])

    ds = DreamSession(memory=store)
    ds.dream_state(deep=True)

    ids = set(store._kv.keys())
    assert junk.mem_id not in ids              # transient junk deleted
    assert owner.mem_id in ids                 # owner protected
    assert precious.mem_id in ids              # high-importance protected


# --------------------------------------------------------------------------- #
# all passes + graceful degradation
# --------------------------------------------------------------------------- #
def test_dream_state_runs_all_passes():
    from nyxara.growth.skill_memory import SkillMemory

    sm = SkillMemory()
    sm.learn("rotate logs", "run logrotate", tools=["shell"])
    ds = DreamSession(skill_memory=sm, mind=_mind_with_pattern(), memory=MemoryStore())
    rep = ds.dream_state(deep=True)
    d = rep.to_dict()
    assert d["deep_state"] is True
    assert d["skills_replayed"] >= 1
    assert d["reasoning_replayed"] >= 1        # the fixed import path makes this work now
    assert d["principles_distilled"] >= 1
    assert d["synapses_fixed"] >= 1


def test_degrades_with_no_subsystems():
    rep = DreamSession().dream()
    assert isinstance(rep, DreamReport)
    assert rep.principles_distilled == 0
    assert rep.synapses_fixed == 0
    assert rep.logs_deleted == 0


def test_shallow_dream_skips_consolidation_passes():
    ds = DreamSession(mind=_mind_with_pattern(), memory=MemoryStore())
    rep = ds.dream(deep=False)
    assert not rep.deep_state
    assert rep.principles_distilled == 0
    assert rep.synapses_fixed == 0


# --------------------------------------------------------------------------- #
# orchestrator wiring
# --------------------------------------------------------------------------- #
def test_dream_session_wired_to_memory():
    from nyxara.kernel.orchestrator import NyxaraCore

    core = NyxaraCore(enable_tools=False, enable_skills=False)
    if core.dream_session is None:
        return  # memory capability degraded in this environment; nothing to assert
    assert core.dream_session.memory is core.memory
    rep = core.report()
    assert "dream_sessions" in rep
