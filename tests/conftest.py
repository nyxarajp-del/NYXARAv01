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
