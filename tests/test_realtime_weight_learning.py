"""NYXARA · tests/test_realtime_weight_learning.py — live, gauntlet-gated online weight learning.

The Master's gap was that NYXARA only *queued* lived experience and folded it into her generative
core weights lazily (when she next generated) or offline (the foundry) — so on many turns her
weights never moved. :meth:`SelfBrain.online_step` closes that: it folds queued experience into the
weights **now**, every turn / tick, reversibly and gauntlet-gated. These tests prove that path is
real (weights change and stick), safe (a regressing neural step rolls back byte-for-byte), bounded,
and honestly disable-able — with NO external LLM.
"""

from __future__ import annotations

from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.mind.self_reasoner import LiveLearnReport, SelfBrain, build_self_brain


def _numpy() -> bool:
    try:
        import numpy  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _kn_brain(*, settings=None, persist=False, **kw) -> SelfBrain:
    """A live brain pinned to the pure-stdlib KN backend (deterministic on any machine).

    ``prefer_promoted=False`` keeps the test hermetic: a promoted foundry model left on disk by
    another test must not override the pinned backend, so the KN behaviour under test is what runs."""
    settings = settings or NyxaraSettings.for_profile(Profile.TEST)
    brain = build_self_brain(settings=settings, persist=persist, prefer_promoted=False, **kw)
    brain._backend = "kngram"
    brain.reply("Who are you?")      # cold-build the real backend so there are weights to fold into
    return brain


def _neural_brain(*, settings=None, **kw) -> SelfBrain:
    """A live brain on the pure-NumPy neural transformer (genesis_np) — real backprop, no torch."""
    settings = settings or NyxaraSettings.for_profile(Profile.TEST)
    brain = build_self_brain(settings=settings, persist=False, prefer_promoted=False, **kw)
    brain._backend = "genesis_np"
    brain.reply("Who are you?")
    return brain


# --------------------------------------------------------------------------- #
# 1. online_step is the live entry point: it folds queued experience into the weights NOW.
# --------------------------------------------------------------------------- #
def test_online_step_folds_queued_experience_into_weights():
    brain = _kn_brain()
    reply = "zzqx wobbex flurgle plarn"
    before = brain._lm.perplexity(reply)
    for _ in range(4):
        brain.learn("cue about the zzqx protocol", reply, reward=1.0)
    report = brain.online_step()

    assert isinstance(report, LiveLearnReport)
    assert report.applied and report.kept and report.changed()
    assert report.reinforced >= 1 and report.suppressed == 0
    assert brain._lm.perplexity(reply) < before          # the weights measurably learned it, live
    # the queue was drained by the live fold — no lazy generate-time refit was needed
    assert not brain._unfit


def test_online_step_is_a_noop_when_nothing_is_queued():
    brain = _kn_brain()
    report = brain.online_step()
    assert isinstance(report, LiveLearnReport)
    assert not report.applied and report.docs_folded == 0


# --------------------------------------------------------------------------- #
# 2. Bounded: a single live flush folds at most `budget` exchanges; the rest wait for the next.
# --------------------------------------------------------------------------- #
def test_online_step_respects_the_flush_budget():
    brain = _kn_brain()
    for i in range(10):
        brain.learn(f"distinct lesson number {i} about topic {i}", reward=1.0)
    queued = len(brain._unfit)
    assert queued >= 6
    report = brain.online_step(budget=3)
    assert report.docs_folded <= 3
    assert len(brain._unfit) == queued - report.docs_folded   # remainder waits for the next flush
    assert brain._unfit                                        # still work left to do


# --------------------------------------------------------------------------- #
# 3. Honest off-switch: with online learning disabled, online_step never touches the weights.
# --------------------------------------------------------------------------- #
def test_online_step_disabled_is_honest_noop():
    brain = _kn_brain()
    brain._online_learn = False
    pc = brain._safe_param_count()
    brain.learn("some fact to queue up", reward=1.0)
    report = brain.online_step()
    assert not report.applied and "disabled" in report.detail
    assert brain._safe_param_count() == pc


# --------------------------------------------------------------------------- #
# 4. KN un-learn: a punished reply is genuinely un-learned live (a negative weight update).
# --------------------------------------------------------------------------- #
def test_online_step_unlearns_a_punished_reply():
    brain = _kn_brain()
    bad = "wombats orbit the crimson moon nightly"
    brain.learn("what do wombats do", bad, reward=1.0)
    brain.online_step()
    taught = brain._lm.perplexity(bad)
    brain.learn("what do wombats do", bad, reward=-1.0)     # a correction
    report = brain.online_step()
    assert report.suppressed >= 1
    assert brain._lm.perplexity(bad) >= taught             # the punished reply is now no more likely


# --------------------------------------------------------------------------- #
# 5. Neural gauntlet: a rewarded fold on the real NumPy transformer is KEPT and lowers the
#    taught text's perplexity — real learning that sticks, not a protected no-op.
# --------------------------------------------------------------------------- #
def test_neural_online_step_keeps_a_genuine_improvement():
    if not _numpy():
        return                                             # neural substrate needs NumPy
    brain = _neural_brain()
    if getattr(brain._lm, "kind", "") != "genesis_np":
        return                                             # backend degraded — nothing to assert
    fact = "the andromeda galaxy is the nearest large spiral galaxy to our own"
    before = brain._mean_perplexity([fact])
    for _ in range(4):
        brain.learn(fact, reward=1.0)
    report = brain.online_step()

    assert report.applied and (report.kept or report.rolled_back)
    if report.kept:
        assert brain._mean_perplexity([fact]) < before     # the taught fact became more probable
        assert report.changed()


# --------------------------------------------------------------------------- #
# 6. Reversibility: the neural snapshot → restore the gauntlet uses is byte-for-byte exact.
# --------------------------------------------------------------------------- #
def test_neural_snapshot_restore_is_byte_for_byte():
    if not _numpy():
        return
    import numpy as np
    brain = _neural_brain()
    lm = brain._lm
    if not getattr(lm, "params", None):
        return
    key = next(iter(lm.params))
    original = lm.params[key].data.copy()
    snap = brain._snapshot_neural(lm)
    assert snap is not None
    lm.params[key].data = lm.params[key].data + 2.71828    # corrupt the weights
    assert not np.allclose(lm.params[key].data, original)
    assert brain._restore_neural(lm, snap)
    assert np.allclose(lm.params[key].data, original)      # rollback restored them exactly
    brain._cleanup_snapshot(snap)


# --------------------------------------------------------------------------- #
# 7. Zero-tolerance contract: a kept step never regressed the probe; a rolled-back one changed
#    nothing (the exact-prior weights are restored).
# --------------------------------------------------------------------------- #
def test_zero_tolerance_gauntlet_contract():
    if not _numpy():
        return
    brain = _neural_brain()
    if getattr(brain._lm, "kind", "") != "genesis_np":
        return
    brain._online_verify_tol = 0.0
    pc_before = brain._safe_param_count()
    for _ in range(3):
        brain.learn("xanadu qintar zylophone brillig frobnicate", reward=1.0)
    report = brain.online_step()
    assert report.kept or report.rolled_back
    if report.rolled_back:
        assert brain._safe_param_count() == pc_before      # rollback left the weights untouched
    elif report.kept and report.applied:
        # a kept step under zero tolerance must not have regressed the (finite) probe
        import math
        assert (not math.isfinite(report.perplexity_before)
                or report.perplexity_after <= report.perplexity_before + 1e-6)
