"""End-to-end: NYXARA's learned state is lifelong — it survives a process restart.

The defect these tests guard against is *forgetting-by-restart*: before this, ``save_state``
persisted only memory + self-model + prior, so the reward learner's weights, the EWC
anti-forgetting anchors, and the trained embedder all reverted to boot state on every reboot.
Here we prove the whole learned state now round-trips through a *fresh* core, and that the
append-only journal recovers any learning that happened after the last checkpoint (a crash).
"""

from __future__ import annotations

import os

import pytest

from nyxara.agency.permissions import Authority
from nyxara.kernel.orchestrator import NyxaraCore


def _weights(core) -> dict:
    return core.learner.to_dict()["model"]["w"]


def _weights_close(a: dict, b: dict, tol: float = 1e-9) -> bool:
    if set(a) != set(b):
        return False
    for action, row in a.items():
        other = b[action]
        if set(row) != set(other):
            return False
        for f, v in row.items():
            if abs(v - other[f]) > tol:
                return False
    return True


def test_learner_and_anchors_survive_a_fresh_core(tmp_path):
    target = str(tmp_path / "longterm.json")
    core = NyxaraCore()
    assert core.learner is not None
    for _ in range(50):
        core.learner.record("greet_warmly", {"with_master": 1.0}, reward=1.0, task="greet")
    core.learner.consolidate()
    for _ in range(30):
        core.learner.record("be_terse", {"task_urgent": 1.0}, reward=1.0, task="urgent")
    v_greet = core.learner.value("greet_warmly", {"with_master": 1.0})
    imp = core.learner.model.importance("greet_warmly", "with_master")
    before = _weights(core)

    saved = core.save_state(target)
    assert saved and os.path.exists(str(tmp_path / "learner.json"))
    assert os.path.exists(str(tmp_path / "synapses.json"))

    # a brand-new process: nothing in memory, load from disk
    core2 = NyxaraCore()
    core2.load_state(target)
    assert _weights_close(before, _weights(core2)), "learner weights not restored across restart"
    assert abs(core2.learner.value("greet_warmly", {"with_master": 1.0}) - v_greet) < 1e-9
    assert abs(core2.learner.model.importance("greet_warmly", "with_master") - imp) < 1e-9
    # an old skill is *not* forgotten after the restart
    assert core2.learner.value("greet_warmly", {"with_master": 1.0}) > 0.5


def test_journal_recovers_learning_after_a_crash(tmp_path):
    target = str(tmp_path / "longterm.json")
    jpath = str(tmp_path / "learning_journal.jsonl")

    # core A: journal on, learn through real turns, checkpoint, then learn MORE without saving
    core_a = NyxaraCore()
    core_a._journal_enabled = True
    core_a._journal_path = jpath
    for msg in ("hello", "what can you do", "thanks", "tell me a fact", "good"):
        core_a.process(msg, authority=Authority.OWNER)
    core_a.save_state(target)                      # watermark captures these events
    steps_at_checkpoint = core_a.learner._steps
    for msg in ("more", "keep going", "again"):    # learned AFTER the checkpoint (the "lost" work)
        core_a.process(msg, authority=Authority.OWNER)
    assert core_a.learner._steps > steps_at_checkpoint
    weights_live = _weights(core_a)

    # ---- crash: process dies here, the post-checkpoint learning was never saved to learner.json ----

    # core B: a fresh process pointed at the same checkpoint + journal
    core_b = NyxaraCore()
    core_b._journal_enabled = True
    core_b._journal_path = jpath
    core_b.load_state(target)
    # the journal replayed the post-checkpoint events, so B recovered what the crash "lost"
    assert getattr(core_b, "_journal_replayed", 0) >= 1
    assert core_b.learner._steps == core_a.learner._steps
    assert _weights_close(weights_live, _weights(core_b)), \
        "journal did not fully recover post-checkpoint learning"


def test_daemon_lifespan_restores_and_checkpoints(tmp_path, monkeypatch):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from nyxara.server.app import create_app

    monkeypatch.setenv("NYXARA_SERVER__AUTONOMIC", "false")
    core = NyxaraCore()
    calls = {"load": 0, "save": 0}
    orig_load, orig_save = core.load_state, core.save_state
    core.load_state = lambda *a, **k: (calls.__setitem__("load", calls["load"] + 1) or orig_load(*a, **k))
    core.save_state = lambda *a, **k: (calls.__setitem__("save", calls["save"] + 1) or orig_save(*a, **k))

    app = create_app(core=core)
    with fastapi_testclient.TestClient(app):
        pass                                       # entering/exiting drives lifespan startup+shutdown
    assert calls["load"] >= 1, "daemon did not restore learned state on startup"
    assert calls["save"] >= 1, "daemon did not checkpoint learned state on shutdown"
