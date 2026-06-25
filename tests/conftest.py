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
os.environ.setdefault("NYXARA_MIND_EVOLUTION__AUTONOMOUS_ENACT", "false")
# The Genesis Protocol's boot kickoff (run_on_boot, ON for live DEV/PROD) designs and micro-trains
# a real neural architecture on the first idle tick — minutes of compute that every core-booting
# test would otherwise pay. Same hermetic rationale as the enactment flags above: keep it OFF for
# the suite. The genesis tests build their own GenesisConfig (and set backend="torch" explicitly).
os.environ.setdefault("NYXARA_GENESIS__RUN_ON_BOOT", "false")

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
