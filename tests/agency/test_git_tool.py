"""NYXARA · git as a typed faculty — argv lists, refs that cannot become options, audited.

Nothing here is a new permission: `run_shell("git …")` already worked and still does. What is
asserted is the difference a *typed* route makes — structured arguments, `shell=False`, and a
branch name that cannot turn itself into `--upload-pack=…`.
"""

from __future__ import annotations

import subprocess

import pytest

from nyxara.agency.git_tool import GIT_SUBCOMMANDS, Git, GitResult, build_git_tools
from nyxara.agency.permissions import Authority, Capability, RiskTier
from nyxara.agency.tools import ToolRegistry


@pytest.fixture()
def repo(tmp_path):
    """A real, throwaway git repository — nothing here touches the working tree."""
    root = tmp_path / "repo"
    root.mkdir()
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    (root / "a.txt").write_text("one\n")
    return Git(repo=str(root), env=env)


def _has_git() -> bool:
    return Git().available()


pytestmark = pytest.mark.skipif(not _has_git(), reason="git is not installed")


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #
def test_status_sees_an_untracked_file(repo):
    got = repo.status()
    assert got.ok and "a.txt" in got.stdout


def test_add_and_commit_make_a_real_commit(repo):
    assert repo.add(paths="a.txt").ok
    assert repo.commit(message="first").ok
    log = repo.log(count=1)
    assert log.ok and "first" in log.stdout


def test_diff_and_show_read_the_repository(repo):
    repo.add(paths="a.txt")
    repo.commit(message="first")
    assert repo.show(rev="HEAD").ok
    assert repo.diff(stat=True).ok


def test_branch_creates_and_checkout_switches(repo):
    repo.add(paths="a.txt")
    repo.commit(message="first")
    assert repo.branch(name="side").ok
    assert repo.checkout(ref="side").ok
    assert repo.current_branch().stdout.strip() == "side"


# --------------------------------------------------------------------------- #
# Flag injection — the reason a typed wrapper exists at all
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", ["--upload-pack=touch /tmp/pwned", "-c", "--exec=x"])
def test_a_ref_may_not_become_an_option(repo, value):
    got = repo.checkout(ref=value)
    assert got.refused and not got.ok
    assert "may not begin with '-'" in got.refused


def test_a_path_may_not_become_an_option(repo):
    assert repo.diff(path="--output=/tmp/x").refused


def test_a_remote_may_not_become_an_option(repo):
    assert repo.push(remote="--receive-pack=sh").refused


def test_a_newline_is_refused_in_a_ref(repo):
    assert repo.checkout(ref="main\nrm -rf /").refused


def test_a_commit_message_with_shell_metacharacters_is_literal(repo):
    """It goes through argv, so `$(…)` is a message, not a substitution."""
    repo.add(paths="a.txt")
    assert repo.commit(message="fix $(touch /tmp/pwned) && echo hi").ok
    log = repo.log(count=1)
    assert "$(touch /tmp/pwned)" in log.stdout


def test_an_empty_commit_message_is_refused(repo):
    assert repo.commit(message="   ").refused


# --------------------------------------------------------------------------- #
# Failure is data
# --------------------------------------------------------------------------- #
def test_a_failing_command_returns_data_not_an_exception(repo):
    got = repo.show(rev="no-such-rev")
    assert got.ok is False and got.rc != 0 and got.stderr


def test_a_missing_git_reports_itself(monkeypatch):
    engine = Git()
    monkeypatch.setattr(engine, "available", lambda: False)
    got = engine.status()
    assert got.ok is False and got.refused == "no-git"
    assert "not installed" in got.stderr


def test_output_is_clipped_rather_than_unbounded(repo):
    engine = Git(repo=repo.repo, max_output=8)
    got = engine.log(count=1)
    assert got.truncated or len(got.stdout) <= 64


def test_git_result_serialises():
    assert set(GitResult().to_dict()) >= {"ok", "subcommand", "rc", "stdout", "stderr"}


# --------------------------------------------------------------------------- #
# Registration — risk assigned by what can actually be undone
# --------------------------------------------------------------------------- #
def test_the_faculty_registers_a_fixed_set_of_subcommands():
    reg = build_git_tools(ToolRegistry())
    names = {n for n in reg.names() if n.startswith("git_")}
    assert names == {"git_status", "git_diff", "git_log", "git_show", "git_branch",
                     "git_add", "git_commit", "git_checkout", "git_pull", "git_clone",
                     "git_push"}
    assert len(GIT_SUBCOMMANDS) >= 10


def test_reads_are_low_risk_and_reversible():
    spec = build_git_tools(ToolRegistry()).get("git_status")
    assert spec.risk == RiskTier.LOW and spec.reversible
    assert spec.capability == Capability.FS_READ


def test_push_is_registered_irreversible_because_it_leaves_the_machine():
    spec = build_git_tools(ToolRegistry()).get("git_push")
    assert spec.reversible is False and spec.risk == RiskTier.HIGH
    assert "IRREVERSIBLE" in spec.description


def test_commit_is_a_moderate_write():
    spec = build_git_tools(ToolRegistry()).get("git_commit")
    assert spec.capability == Capability.FS_WRITE and spec.risk == RiskTier.MODERATE


def test_a_registered_tool_runs_through_the_pipeline(repo):
    reg = build_git_tools(ToolRegistry(), git=repo)
    result = reg.invoke("git_status", {}, authority=Authority.OWNER)
    assert result.ok and result.value["ok"] is True


def test_registration_is_idempotent_by_name():
    reg = ToolRegistry()
    build_git_tools(reg)
    build_git_tools(reg)        # must not raise "already registered"
    assert reg.get("git_status") is not None


def test_the_default_toolset_includes_git():
    from nyxara.agency.default_tools import build_default_tools
    reg = build_default_tools(ToolRegistry())
    assert "git_status" in reg.names() and "git_commit" in reg.names()
