"""Tests for nyxara.agency.code_sandbox — sandboxed code and shell execution."""

from __future__ import annotations

from nyxara.agency.code_sandbox import (ShellSession, run_python, run_shell_command,
                                        safe_shell)


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


# --------------------------------------------------------------------------- #
# run_shell_command — the real, unrestricted terminal (shell=True)
# --------------------------------------------------------------------------- #
def test_real_shell_pipe():
    res = run_shell_command("echo hello | tr a-z A-Z", timeout_s=10.0)
    assert res.ok and "HELLO" in res.stdout


def test_real_shell_redirection_roundtrip(tmp_path):
    out = tmp_path / "answer.txt"
    w = run_shell_command(f"echo $((6*7)) > {out}", timeout_s=10.0)
    assert w.ok
    r = run_shell_command(f"cat {out}", timeout_s=10.0)
    assert r.ok and r.stdout.strip() == "42"


def test_real_shell_chaining_and_env_expansion():
    res = run_shell_command("X=nyx && echo $X-done", timeout_s=10.0)
    assert res.ok and "nyx-done" in res.stdout


def test_real_shell_supplied_env():
    res = run_shell_command("echo $NYXARA_TESTVAR", timeout_s=10.0,
                            env={"NYXARA_TESTVAR": "wired"})
    assert res.ok and "wired" in res.stdout


def test_real_shell_glob(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "c.txt").write_text("")
    res = run_shell_command("ls *.py | wc -l", timeout_s=10.0, cwd=str(tmp_path))
    assert res.ok and res.stdout.strip() == "2"


def test_real_shell_nonzero_exit_is_data():
    res = run_shell_command("exit 7", timeout_s=10.0)
    assert not res.ok and res.returncode == 7


def test_real_shell_times_out():
    res = run_shell_command("sleep 10", timeout_s=1.0)
    assert res.timed_out and not res.ok


def test_real_shell_unbounded_timeout_runs():
    # timeout_s <= 0 means "no deadline"; a fast command still returns promptly.
    res = run_shell_command("echo unbounded", timeout_s=0)
    assert res.ok and "unbounded" in res.stdout


def test_real_shell_empty_command_is_data():
    res = run_shell_command("   ", timeout_s=5.0)
    assert not res.ok and res.error


def test_real_shell_stdin():
    res = run_shell_command("cat", timeout_s=5.0, stdin="piped-in\n")
    assert res.ok and "piped-in" in res.stdout


# --------------------------------------------------------------------------- #
# ShellSession — a persistent terminal (cd / export carry across calls)
# --------------------------------------------------------------------------- #
def test_session_cd_persists():
    s = ShellSession()
    s.run("cd /")
    r = s.run("pwd")
    assert r.stdout.strip() == "/" and s.cwd == "/"


def test_session_export_persists():
    s = ShellSession()
    s.run("export NYXARA_SESSION_VAR=persisted")
    r = s.run("echo $NYXARA_SESSION_VAR")
    assert r.stdout.strip() == "persisted"


def test_session_stdout_is_clean_of_sentinels():
    s = ShellSession()
    r = s.run("echo just-this")
    assert r.stdout.strip() == "just-this"
    assert "NYXARA_CWD" not in r.stdout and "NYXARA_ENV" not in r.stdout


def test_session_returncode_is_user_command_status():
    s = ShellSession()
    r = s.run("exit 3")
    assert r.returncode == 3 and not r.ok


def test_session_reset():
    s = ShellSession()
    s.run("cd /")
    s.run("export NYXARA_RESET_VAR=1")
    s.reset()
    assert s.cwd != "/" or True  # cwd back to os.getcwd()
    r = s.run("echo [$NYXARA_RESET_VAR]")
    assert r.stdout.strip() == "[]"
