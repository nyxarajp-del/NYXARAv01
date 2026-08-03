"""The TEST profile owns its own disk — it never reads the machine's live ~/.nyxara.

Sealing the writers (foundry, enact paths, autonomic loop, perception) is only half of
hermeticity, and it was the half this repo had. The suite still *read* the real home, so one
`nyxara-grow` run that promoted a self-model was enough to make six unrelated tests fail on
state no test had created. These pin the other half."""

from __future__ import annotations

import os
from pathlib import Path

from nyxara.kernel.config import NyxaraSettings, PathsConfig, Profile


def test_test_profile_never_points_at_the_real_home():
    paths = NyxaraSettings.for_profile(Profile.TEST).paths
    real_home = Path(os.path.expanduser("~")) / ".nyxara"
    assert paths.root != real_home
    assert real_home not in paths.root.parents


def test_every_derived_directory_moves_with_the_root():
    """A half-redirected layout is worse than none: `root` clean but `memory_dir` still under
    the real home would leak silently, and only for the subsystems that use that one field."""
    paths = NyxaraSettings.for_profile(Profile.TEST).paths
    for name in PathsConfig.model_fields:
        value = getattr(paths, name)
        if name == "root" or value is None:
            continue
        assert paths.root in Path(value).parents, f"{name} escaped the scratch root"


def test_the_root_is_stable_across_settings_objects():
    """Per-process, not per-settings-object. Plenty of tests write through one settings object
    and read through another built later; a fresh directory each time would break them."""
    a = NyxaraSettings.for_profile(Profile.TEST)
    b = NyxaraSettings.for_profile(Profile.TEST)
    assert a.paths.root == b.paths.root


def test_an_explicit_nyxara_home_still_wins(monkeypatch, tmp_path):
    """The redirect is a default, not a cage — a test (or CI) that deliberately names a home
    gets it. Only the accidental case is caught."""
    monkeypatch.setenv("NYXARA_HOME", str(tmp_path / "chosen"))
    assert NyxaraSettings.for_profile(Profile.TEST).paths.root == tmp_path / "chosen"


def test_live_profiles_keep_the_real_home(monkeypatch):
    """Hermeticity is a TEST-profile property. DEV and PROD must still find her actual
    memory, weights and audit trail where she left them."""
    monkeypatch.delenv("NYXARA_HOME", raising=False)
    expected = Path(os.path.expanduser("~")) / ".nyxara"
    for profile in (Profile.DEV, Profile.PROD):
        assert NyxaraSettings.for_profile(profile).paths.root == expected
