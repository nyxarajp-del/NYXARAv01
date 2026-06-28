"""Tests for the server CLI entry point (nyxara/server/__main__.py).

The real ``main()`` hands off to uvicorn (a blocking call), so we monkeypatch the
``run`` it imports and assert the wiring: settings are read, the banner is printed,
``run`` is invoked, and a clean exit code is returned. No socket is ever opened."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from nyxara.server.__main__ import main  # noqa: E402


def test_main_invokes_run_and_returns_zero(monkeypatch, capsys):
    called = {}

    def _fake_run(*args, **kwargs):
        called["kwargs"] = kwargs

    # main() does `from nyxara.server.app import run` at call time, so patching the
    # attribute on the module is enough to intercept it.
    monkeypatch.setattr("nyxara.server.app.run", _fake_run)

    rc = main()
    out = capsys.readouterr().out

    assert rc == 0
    assert "settings" in called["kwargs"]  # main forwards the resolved settings
    assert "NYXARA serving on" in out
    assert "/health" in out


def test_main_reports_missing_dependency(monkeypatch, capsys):
    # simulate the optional [server] extra being absent: `from ...app import run`
    # raises ImportError when the name is gone, which is the branch main() guards.
    monkeypatch.delattr("nyxara.server.app.run")

    rc = main()
    err = capsys.readouterr().err
    assert rc == 1
    assert err.strip()  # the import error message is surfaced on stderr
