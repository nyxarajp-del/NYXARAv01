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


# --------------------------------------------------------------------------- #
# The half the tests above could not see
# --------------------------------------------------------------------------- #
# Every test above asks `NyxaraSettings.for_profile(Profile.TEST)` and confirms it is redirected.
# It is. But the suite does not run under that profile — `get_settings()` returns Profile.DEV in
# here — so the redirect was never reaching the settings the tests actually build, and these passed
# while the property they name was false. They assert "if you asked for TEST you would be safe";
# nothing was asking.
#
# The consequence was not theoretical: with `~/.nyxara/data/foundry/` holding promoted models,
# every test that built a `NyxaraCore()` loaded a real genesis_np model and ran NumPy transformer
# inference on the turn path, and `tests/eval/` hit the 300 s per-test ceiling inside
# `growth/genesis_numpy.py`. The suite's behaviour depended on undeclared local state.
#
# So the checks below ask about the settings the suite is ACTUALLY using, not about a profile it
# could have used.
def test_the_running_suite_does_not_read_the_real_home():
    from nyxara.kernel.config import get_settings

    real_home = Path(os.path.expanduser("~")) / ".nyxara"
    root = get_settings().paths.root
    assert root != real_home, (
        "the suite is reading the machine's live ~/.nyxara — a promoted self-model or a learned "
        "graph left there silently changes what these tests observe")
    assert real_home not in root.parents


def test_every_live_derived_directory_also_escaped_the_real_home():
    """A half-redirected layout leaks silently, and only for the subsystems using that one field."""
    from nyxara.kernel.config import get_settings

    real_home = Path(os.path.expanduser("~")) / ".nyxara"
    paths = get_settings().paths
    for name in PathsConfig.model_fields:
        value = getattr(paths, name)
        if value is None:
            continue
        assert real_home not in Path(value).parents and Path(value) != real_home, (
            f"{name} still points inside the real home")


def test_the_live_root_is_stable_across_settings_objects():
    """Same requirement as the TEST-profile root: tests write through one settings object and
    read through another built later, so a fresh directory per object would break them."""
    from nyxara.kernel.config import get_settings, NyxaraSettings

    assert get_settings().paths.root == NyxaraSettings().paths.root
