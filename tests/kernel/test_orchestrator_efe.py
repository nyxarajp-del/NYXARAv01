"""Tests: the free-energy engine is wired through the orchestrator (advisory,
fail-soft, gate-sovereign)."""

from __future__ import annotations

from nyxara.kernel.orchestrator import Candidate, NyxaraCore


def _core() -> NyxaraCore:
    return NyxaraCore(enable_memory=False, enable_social=False)


def test_free_energy_engine_built():
    core = _core()
    if core.predictive is None:          # growth disabled in this build — nothing to wire
        return
    assert core.free_energy is not None
    assert core.free_energy.predictive is core.predictive    # ONE shared instance
    assert core.free_energy.world_model is core.world_model


def test_efe_appraise_fail_soft():
    core = _core()
    cand = Candidate(text="do the thing", kind="act", tool="echo")
    out = core._efe_appraise(cand, "please do the thing")
    assert isinstance(out, Candidate)                        # never raises
    # respond candidates pass through untouched
    resp = Candidate(text="hello", kind="respond")
    assert core._efe_appraise(resp, "hi").efe is None


def test_update_preference_prior_fail_soft():
    core = _core()
    core._update_preference_prior()                          # no goals — no-op, no raise
    if core.goals is not None and core.predictive is not None:
        core.goals.create("serve the Master well", {"owner_benefit": 1.0}, priority=0.9)
        core._update_preference_prior()
        assert core.predictive.preference is not None        # C now carries the goal


def test_candidate_efe_field_default():
    assert Candidate(text="x").efe is None


def test_report_exposes_free_energy():
    core = _core()
    rep = core.report()
    if core.free_energy is not None:
        assert "free_energy" in rep
        assert "gamma" in rep["free_energy"]
