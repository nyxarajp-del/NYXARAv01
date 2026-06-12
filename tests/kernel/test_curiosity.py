"""Tests for Step-6 the curiosity loop — NYXARA closes her own known-unknowns.

When idle she notices what she knows she does not know, values those gaps (value of
information), and runs a *safe, internal* experiment on the most valuable one — consulting
her grounded knowledge — folding the answer back as a belief and a memory. The loop is
strictly internal: it asks the world nothing and acts on nothing.
"""

from __future__ import annotations

from nyxara.kernel.orchestrator import NyxaraCore


def _core(**kw):
    return NyxaraCore(**kw)


# -------------------- a gap exists at boot, and curiosity closes it -------------------- #
def test_boot_has_self_knowledge_gaps():
    nyx = _core()
    gaps = nyx.known_unknowns()
    # she boots not yet having formed confident beliefs about her own / the Master's name
    assert "NYXARA.name" in gaps or "Master.name" in gaps


def test_curiosity_resolves_a_gap():
    nyx = _core()
    before = len(nyx.known_unknowns())
    report = nyx.curiosity_pass()
    assert report["gaps"] == before
    assert report["resolved"] is True
    assert report["investigated"] is not None
    # the gap she investigated is now a confident belief -> fewer known-unknowns
    assert len(nyx.known_unknowns()) < before


def test_curiosity_learns_the_masters_name():
    from nyxara.kernel.config import get_settings
    nyx = _core()
    # run a few passes; one of them settles Master.name from grounded config
    for _ in range(5):
        nyx.curiosity_pass()
    b = nyx.self_model.beliefs.best("Master", "name")
    assert b is not None and b.object == get_settings().owner.name


def test_curiosity_drains_all_resolvable_gaps():
    nyx = _core()
    for _ in range(6):
        nyx.curiosity_pass()
    gaps = nyx.known_unknowns()
    # the three grounded self-facts (NYXARA.name, NYXARA.is_a, Master.name) are settled
    assert "NYXARA.name" not in gaps
    assert "NYXARA.is_a" not in gaps
    assert "Master.name" not in gaps


def test_discovery_is_remembered():
    nyx = _core()
    nyx.curiosity_pass()
    hits = nyx.memory.recall("learned", k=5, strengthen=False)
    assert any("curiosity" in (rec.tags or set()) for rec, _ in hits)


# -------------------- it is part of the idle pass -------------------- #
def test_idle_maintenance_runs_curiosity():
    nyx = _core()
    report = nyx.idle_maintenance()
    # the first idle pass resolves a gap and records which one under "curiosity"
    assert "curiosity" in report


# -------------------- gated, internal, robust -------------------- #
def test_scram_halts_curiosity():
    nyx = _core()
    nyx.oversight.scram(reason="test halt")
    report = nyx.curiosity_pass()
    assert report["resolved"] is False


def test_curiosity_safe_without_self_model():
    nyx = _core(enable_memory=False, enable_growth=False)
    assert nyx.self_model is None
    report = nyx.curiosity_pass()
    assert report == {"gaps": 0, "investigated": None, "resolved": False}


def test_curiosity_idempotent_when_no_gaps_remain():
    nyx = _core()
    for _ in range(8):
        nyx.curiosity_pass()
    # once the resolvable gaps are gone, a further pass resolves nothing (no crash, no churn)
    report = nyx.curiosity_pass()
    assert report["resolved"] is False
