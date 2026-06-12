"""Tests for nyxara.agency.code_sandbox — sandboxed code and shell execution."""

from __future__ import annotations

from nyxara.agency.code_sandbox import run_python, safe_shell


def test_run_python_simple_value():
    res = run_python("result = 2 + 2", timeout_s=5.0)
    assert res.ok
    assert res.value == 4 or "4" in (res.stdout or "")


def test_run_python_stdout_captured():
    res = run_python("print('hello sandbox')", timeout_s=5.0)
    assert res.ok and "hello sandbox" in res.stdout


def test_run_python_exception_is_data_not_raise():
    res = run_python("raise ValueError('boom')", timeout_s=5.0)
    assert not res.ok
    assert "boom" in (res.stderr or "") or "boom" in (res.error or "")


def test_run_python_times_out():
    res = run_python("while True:\n    pass", timeout_s=1.0)
    assert res.timed_out and not res.ok


def test_run_python_truncates_output():
    res = run_python("print('x' * 1000000)", timeout_s=5.0, max_output=1000)
    assert len(res.stdout) <= 1000 + 64


def test_safe_shell_echo():
    res = safe_shell(["echo", "hi-there"], timeout_s=5.0)
    assert res.ok and "hi-there" in res.stdout


def test_safe_shell_nonzero_exit():
    res = safe_shell("false", timeout_s=5.0)
    assert not res.ok and res.returncode != 0


def test_safe_shell_times_out():
    res = safe_shell(["sleep", "10"], timeout_s=1.0)
    assert res.timed_out and not res.ok
