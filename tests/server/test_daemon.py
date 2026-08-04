"""``nyxara-daemon`` — the always-on entry point.

:mod:`nyxara.daemon` is a real ``[project.scripts]`` console script, so nothing imports it and
it had no test either. That combination is how an entry point rots silently: the one thing it
does — default the background mind ON — could break and no run would notice.
"""

from __future__ import annotations

import os

import nyxara.daemon as daemon


def test_daemon_defaults_the_background_mind_on(monkeypatch) -> None:
    """The daemon's whole reason to exist: one process that is both API and continuous mind."""
    monkeypatch.delenv("NYXARA_SERVER__AUTONOMIC", raising=False)
    seen: dict = {}

    def _fake_main(argv=None) -> int:
        seen["autonomic"] = os.environ.get("NYXARA_SERVER__AUTONOMIC")
        seen["argv"] = argv
        return 0

    monkeypatch.setattr("nyxara.server.__main__.main", _fake_main)
    assert daemon.main([]) == 0
    assert seen["autonomic"] == "true"
    assert seen["argv"] == []


def test_daemon_respects_an_explicit_master_override(monkeypatch) -> None:
    """``setdefault``, not assignment — the Master's explicit choice must win."""
    monkeypatch.setenv("NYXARA_SERVER__AUTONOMIC", "false")
    seen: dict = {}

    def _fake_main(argv=None) -> int:
        seen["autonomic"] = os.environ.get("NYXARA_SERVER__AUTONOMIC")
        return 0

    monkeypatch.setattr("nyxara.server.__main__.main", _fake_main)
    assert daemon.main([]) == 0
    assert seen["autonomic"] == "false", "the daemon overrode an explicit Master setting"
