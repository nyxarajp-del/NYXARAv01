"""Pytest fixtures + hermetic safety harness for the NYXARA test suite.

NYXARA ships with Master JP's standing authorisation to self-modify (``autonomous_enact`` and
the mind-evolution / LLM-edit paths default ON for live DEV/PROD runs). The *test suite*, however,
must never edit the real source tree on disk while it runs — a hermetic suite is reproducible and
safe. So before any settings are constructed we force every enactment path OFF via the environment
and rebuild the cached settings singleton.

Individual tests that deliberately exercise enactment (e.g. the deterministic-edit recursion tests)
still do so safely: they build their *own* settings object with ``autonomous_enact=True`` and point
the optimizer at a ``tmp_path`` file, never the live package.
"""

from __future__ import annotations

import os

# --- force a hermetic, non-self-modifying posture for the whole suite (set before import) --- #
os.environ.setdefault("NYXARA_SELF_IMPROVEMENT__AUTONOMOUS_ENACT", "false")
os.environ.setdefault("NYXARA_SELF_IMPROVEMENT__ALLOW_TUNING", "false")
os.environ.setdefault("NYXARA_SELF_IMPROVEMENT__ALLOW_LLM_EDITS", "false")
os.environ.setdefault("NYXARA_SELF_OPTIMIZATION__AUTONOMOUS_ENACT", "false")
os.environ.setdefault("NYXARA_MIND_EVOLUTION__AUTONOMOUS_ENACT", "false")
# Learning-rule synthesis (growth/rule_synth.py) INVENTS a new weight-update rule when the fixed
# learner stalls and, with autonomous_enact, installs it into the live learner. Searching/measuring
# is hermetic and fast, but installing mutates a live faculty — seal it OFF for the suite (same
# rationale as the flags above). Tests that exercise adoption build their own synthesizer/settings.
os.environ.setdefault("NYXARA_RULE_SYNTHESIS__AUTONOMOUS_ENACT", "false")
# The Genesis Protocol's boot kickoff (run_on_boot, ON for live DEV/PROD) designs and micro-trains
# a real neural architecture on the first idle tick — minutes of compute that every core-booting
# test would otherwise pay. Same hermetic rationale as the enactment flags above: keep it OFF for
# the suite. The genesis tests build their own GenesisConfig (and set backend="torch" explicitly).
os.environ.setdefault("NYXARA_GENESIS__RUN_ON_BOOT", "false")
# In production the Genesis substrate is "auto" → it builds and trains NYXARA's OWN designed
# architecture for real in NumPy (growth/genesis_numpy.py) whenever a search runs. That is genuine
# per-candidate neural training — too slow to pay on every search-mechanics test. Pin the substrate
# to the fast n-gram path for any get_settings()-based search in the suite; the real NumPy brain is
# covered end-to-end in tests/growth/test_genesis_numpy_substrate.py, which opts in explicitly.
os.environ.setdefault("NYXARA_GENESIS__SUBSTRATE", "ngram")
# Continuous Recursive Self-Improvement (kernel/orchestrator.idle_maintenance) runs the heavy
# GrowthEngine tower (mind-evolution, recursive self-improvement, meta-research — each benchmarks
# the whole reasoner) on a throttled idle cadence. ON for live DEV/PROD so NYXARA improves herself
# unprompted; OFF for the suite for the same hermetic/compute rationale as the flags above. Tests
# that exercise the wiring build their own settings with continuous=True and a small cadence.
os.environ.setdefault("NYXARA_SELF_IMPROVEMENT__CONTINUOUS", "false")
# The recursive meta towers over the mind-evolution and meta-research SEARCHES (growth/meta_meta.py)
# evolve bounded capability knobs across passes and persist state to disk. ON for live DEV/PROD so
# NYXARA recursively optimizes HOW she searches for smarter reasoning and wider invention; OFF for
# the suite for the same hermetic/determinism rationale. Tests that exercise them build their own
# settings (or drive the generic RecursiveMetaController directly — see test_meta_meta.py).
os.environ.setdefault("NYXARA_MIND_EVOLUTION__META_META_ENABLED", "false")
os.environ.setdefault("NYXARA_META_RESEARCH__META_META_ENABLED", "false")
os.environ.setdefault("NYXARA_LLM__LITERTLM_ENABLED", "false")
os.environ.setdefault("NYXARA_LLM__LITERTLM_AUTO_DOWNLOAD", "false")
# The model foundry TRAINS. `AutonomicLoop._maybe_grow` -> `GrowthEngine.run` -> `improve_self` ->
# `Foundry.self_improve` -> `train_candidate` -> `model.train_on(steps=train_steps)` is real
# backprop, in pure NumPy, and it ran on every autonomic tick the suite drove — so a test that
# merely asserts "growth is wired" paid for a full training run, four times over in
# `test_periodic_growth_runs`. This is where CI's multi-hour `Run tests` cancellations came from.
#
# `Profile.TEST` already sets `foundry.enabled = False` for exactly this reason, and
# `tests/growth/test_autolearn.py` documents relying on it — but the suite does not run under that
# profile, so the switch was never reaching the settings the tests actually build. Set it the same
# way as every other hermetic flag above. Tests that WANT the foundry already enable it explicitly
# on their own settings object (test_promotion_bus, test_directive_dispatch, test_learning_loop_
# wiring, test_self_provider_reload), so none of them depends on this default.
os.environ.setdefault("NYXARA_FOUNDRY__ENABLED", "false")
# Metacontrol allocates real deliberation time per turn — up to 180s for a hard question, 600s for
# an extreme one — and the kernel now enforces it, so a high-stakes test prompt genuinely sat and
# thought. `test_autonomy_never_grants_extra_power` ("delete the production database") was the
# slowest test in the suite at 72s for exactly that reason: the budget was working as designed, and
# the design is right for a live turn and absurd for a test asserting a gate.
#
# Cap the ceiling rather than disabling metacontrol: an allocation of zero would send every turn
# down the deterministic floor and stop the suite from exercising the reasoning path at all. Same
# rationale as pinning the genesis substrate above — keep the mechanism, not the price.
os.environ.setdefault("NYXARA_METACONTROL__MAX_SECONDS_CEILING", "6")
# NYX V.001's Layer 0-17 cycle (nyxara/nyx001/) runs inside the OUTERMOST reason-seat, so at its
# default of every turn it is paid on every turn of every test — ~0.15s each, at every scale,
# because the cost is eighteen layers running rather than the width. Measured: the growth-heavy
# kernel tests went from 24s to 35s with it in the seat.
#
# Sample the beat rather than disabling the stack: a skipped beat still writes its observation to
# episodic memory, so replay keeps learning from it and the mechanism stays exercised. Exactly the
# rationale used for the metacontrol ceiling above and the genesis substrate below — keep the
# mechanism, not the price. tests/nyx001 builds its own stacks and is unaffected.
os.environ.setdefault("NYXARA_NYX001__STACK_EVERY_N_TURNS", "8")
# L-OMNI (nyx/omni.py) reads NYXARA's own source, lowers hot numeric functions to C, compiles
# them and swaps them into the *live process*. ON for real runs at Master JP's instruction, and
# the same hermetic rationale as every flag above applies to the suite: a test asserting that the
# layer is wired should not invoke a C compiler, and a kernel swapped into one test's process is
# state leaking into the next. Scanning and lowering stay ON — those are pure and fast, and they
# are what most of test_omni.py exercises; only the compile-and-load tier is sealed. The tests
# that forge for real (tests/nyx/test_omni.py) build their own compiler with hot_swap=True and
# point it at a module in tmp_path, never at the live package.
os.environ.setdefault("NYXARA_NYX__OMNI_HOT_SWAP", "false")

from nyxara.kernel.config import reload_settings  # noqa: E402 — must follow the env setup above

# Rebuild the cached singleton so the env overrides above take effect for every test that calls
# get_settings(). Tests that need a different posture build their own settings explicitly.
reload_settings()

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_signal_bus():
    """Reset the process-shared cross-module signal bus around each test (no cross-test leakage)."""
    try:
        from nyxara.growth.signal_bus import reset_signal_bus
        reset_signal_bus()
    except Exception:  # noqa: BLE001
        pass
    yield
    try:
        from nyxara.growth.signal_bus import reset_signal_bus
        reset_signal_bus()
    except Exception:  # noqa: BLE001
        pass
