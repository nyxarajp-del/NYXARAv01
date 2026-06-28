"""Tests for growth/autodeps.py — first-run auto-install of the heavy LoRA stack.

These exercise the *wiring* — the opt-out, the deps-present short-circuit, the one-shot
sentinel, and the never-raises-into-boot contract — without ever shelling out to pip or
re-exec'ing the interpreter (both are monkeypatched).
"""

from __future__ import annotations

from pathlib import Path

import nyxara.growth.autodeps as ad


def test_short_circuits_when_deps_present(monkeypatch):
    # already importable -> no install attempt, no sentinel, returns True
    monkeypatch.setattr(ad, "runtime_deps_present", lambda: True)
    calls: list = []
    monkeypatch.setattr(ad.subprocess, "run", lambda *a, **k: calls.append(a))
    assert ad.ensure_runtime_deps(log=lambda _m: None) is True
    assert calls == []


def test_opt_out_via_env(monkeypatch):
    monkeypatch.setattr(ad, "runtime_deps_present", lambda: False)
    monkeypatch.setenv("NYXARA_AUTO_INSTALL", "0")
    msgs: list[str] = []
    assert ad.ensure_runtime_deps(log=msgs.append) is False
    assert any("disabled" in m for m in msgs)


def test_skips_under_test_profile(monkeypatch):
    monkeypatch.setattr(ad, "runtime_deps_present", lambda: False)
    monkeypatch.setenv("NYXARA_AUTO_INSTALL", "1")
    monkeypatch.setenv("NYXARA_PROFILE", "test")
    assert ad.ensure_runtime_deps(log=lambda _m: None) is False


def test_sentinel_blocks_retry(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(ad, "runtime_deps_present", lambda: False)
    monkeypatch.setenv("NYXARA_AUTO_INSTALL", "1")
    monkeypatch.delenv("NYXARA_PROFILE", raising=False)
    sentinel = tmp_path / ".deps_attempted"
    sentinel.touch()
    monkeypatch.setattr(ad, "_ATTEMPT_SENTINEL", sentinel)
    ran: list = []
    monkeypatch.setattr(ad.subprocess, "run", lambda *a, **k: ran.append(a))
    msgs: list[str] = []
    assert ad.ensure_runtime_deps(log=msgs.append) is False
    assert ran == []  # the prior-attempt sentinel prevents a second pip run
    assert any("prior auto-install" in m for m in msgs)


def test_successful_install_reexecs(monkeypatch, tmp_path: Path):
    # deps absent at first, present after the (faked) install -> we re-exec to load them
    present = {"v": False}
    monkeypatch.setattr(ad, "runtime_deps_present", lambda: present["v"])
    monkeypatch.setenv("NYXARA_AUTO_INSTALL", "1")
    monkeypatch.delenv("NYXARA_PROFILE", raising=False)
    monkeypatch.delenv("NYXARA_AUTODEPS_REEXECED", raising=False)
    monkeypatch.setattr(ad, "_ATTEMPT_SENTINEL", tmp_path / ".deps_attempted")

    class _Proc:
        returncode = 0
        stdout = "ok"

    def _fake_run(*_a, **_k):
        present["v"] = True  # the install "worked"
        return _Proc()

    reexec_calls: list = []
    monkeypatch.setattr(ad.subprocess, "run", _fake_run)
    monkeypatch.setattr(ad.os, "execv", lambda *a: reexec_calls.append(a))

    assert ad.ensure_runtime_deps(log=lambda _m: None) is True
    assert reexec_calls, "a successful first install should re-exec to load the new stack"


def test_failed_install_never_raises(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(ad, "runtime_deps_present", lambda: False)
    monkeypatch.setenv("NYXARA_AUTO_INSTALL", "1")
    monkeypatch.delenv("NYXARA_PROFILE", raising=False)
    monkeypatch.setattr(ad, "_ATTEMPT_SENTINEL", tmp_path / ".deps_attempted")

    def _boom(*_a, **_k):
        raise OSError("pip exploded")

    monkeypatch.setattr(ad.subprocess, "run", _boom)
    msgs: list[str] = []
    # never raises; falls back honestly
    assert ad.ensure_runtime_deps(log=msgs.append) is False
    assert any("could not start" in m for m in msgs)
