"""NYXARA · agency/git_tool.py — git as a typed faculty, not a shell string (🔀, NYX V.02).

Git was already reachable: ``run_shell("git commit -m ...")`` works today and is owner-blessed.
Nothing here loosens anything. What it changes is *how*:

* **structured arguments** — ``commit(message=…)``, ``checkout(ref=…, create=True)``. The
  command is built as an ``argv`` list and handed to :mod:`subprocess` with ``shell=False``,
  so there is no string for a branch name containing ``;`` or ``$(…)`` to escape from.
* **no flag injection** — every value that names a path, a ref or a remote is refused if it
  begins with ``-``. A ref called ``--upload-pack=…`` is the classic way a "safe" git wrapper
  turns into arbitrary execution, and this is where that is stopped.
* **an audit trail** — each call goes through the registry's capability/risk pipeline like any
  other tool, so the decision is recorded rather than buried in a shell transcript.

Honest framing (the rule this repo keeps, see ``docs/CAPABILITIES.md``):

* This is a **fixed set of subcommands**. Anything outside it is not reachable here — that is
  the point of a typed faculty — and ``run_shell`` remains available for the rest, exactly as
  before. Adding a subcommand is a code change, deliberately.
* Reads (``status``/``diff``/``log``/``show``/``branch``) are low-risk and reversible. Writes
  to the working tree or the local history (``add``/``commit``/``checkout``/``branch -c``) are
  moderate and reversible-in-principle. ``push`` is **irreversible from here**: what left the
  machine has left it, and it is registered as such so the pipeline treats it that way.
* Failure is returned as **data** (``rc``, ``stderr``), never raised. A git that is not
  installed reports itself missing rather than crashing a turn.

Pure standard library. Every public method is fail-soft.
"""

from __future__ import annotations

import os
import shutil
import subprocess  # noqa: S404 — argv lists only, shell=False, and that is the whole point
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from nyxara.agency.permissions import Capability, RiskTier
from nyxara.agency.tools import ToolParam, ToolRegistry, ToolSpec

__all__ = ["GitResult", "Git", "build_git_tools", "GIT_SUBCOMMANDS"]

#: Everything this faculty can do. Not a prefix filter — an allow-list.
GIT_SUBCOMMANDS = ("status", "diff", "log", "show", "add", "commit", "branch",
                   "checkout", "push", "pull", "clone", "remote", "rev_parse")

_MAX_OUTPUT = 200_000
_DEFAULT_TIMEOUT = 120.0


class GitRefused(ValueError):
    """An argument was rejected before git ever saw it."""


def _safe_value(value: Any, *, what: str) -> str:
    """A path/ref/remote must not be able to become a flag.

    This is the whole security story of a typed git wrapper: git happily reads
    ``--upload-pack=touch /tmp/pwned`` as an option, so a "branch name" that starts with a dash
    is refused here rather than passed through and hoped about.
    """
    text = str(value if value is not None else "").strip()
    if not text:
        raise GitRefused(f"{what} is empty")
    if text.startswith("-"):
        raise GitRefused(f"{what} may not begin with '-' (it would be read as an option): "
                         f"{text!r}")
    if "\x00" in text or "\n" in text:
        raise GitRefused(f"{what} may not contain a newline or NUL")
    return text


def _safe_paths(paths: Any, *, what: str = "path") -> List[str]:
    if paths is None or paths == "":
        return []
    if isinstance(paths, str):
        paths = [p for p in paths.split() if p]
    if not isinstance(paths, (list, tuple)):
        raise GitRefused(f"{what} must be a string or a list of strings")
    return [_safe_value(p, what=what) for p in paths]


@dataclass
class GitResult:
    """What one git invocation actually did. Failure is data, never an exception."""

    ok: bool = False
    subcommand: str = ""
    argv: List[str] = field(default_factory=list)
    rc: int = -1
    stdout: str = ""
    stderr: str = ""
    cwd: str = ""
    truncated: bool = False
    refused: str = ""            # set when an argument was rejected before running

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "subcommand": self.subcommand, "argv": list(self.argv),
                "rc": self.rc, "stdout": self.stdout, "stderr": self.stderr,
                "cwd": self.cwd, "truncated": self.truncated, "refused": self.refused}


class Git:
    """A typed git faculty over one repository. Fixed subcommands, argv only, no shell."""

    def __init__(self, *, repo: str = "", timeout_s: float = _DEFAULT_TIMEOUT,
                 max_output: int = _MAX_OUTPUT, env: Optional[Dict[str, str]] = None) -> None:
        self.repo = str(repo or os.getcwd())
        self.timeout_s = max(1.0, float(timeout_s))
        self.max_output = max(1024, int(max_output))
        self._env = dict(env or {})

    # ---- the one place a process is started ------------------------------- #
    def available(self) -> bool:
        return shutil.which("git") is not None

    def _run(self, subcommand: str, argv: Sequence[str], *,
             cwd: str = "") -> GitResult:
        out = GitResult(subcommand=subcommand, cwd=str(cwd or self.repo))
        if not self.available():
            out.stderr = "git is not installed on this machine"
            out.refused = "no-git"
            return out
        # `--no-pager` so a paginated command cannot block, `-c` never accepted from a caller.
        full = ["git", "--no-pager", *argv]
        out.argv = list(full)
        env = dict(os.environ)
        env.update(self._env)
        # Git must never stop and ask: an editor or a credential prompt inside an autonomous
        # turn is a hang, not an interaction.
        env.setdefault("GIT_TERMINAL_PROMPT", "0")
        env.setdefault("GIT_PAGER", "cat")
        env.setdefault("GIT_EDITOR", "true")
        try:
            proc = subprocess.run(  # noqa: S603 — argv list, shell=False, values pre-validated
                full, cwd=out.cwd, env=env, capture_output=True, text=True,
                timeout=self.timeout_s, shell=False, check=False)
            out.rc = int(proc.returncode)
            out.stdout, trunc_a = self._clip(proc.stdout or "")
            out.stderr, trunc_b = self._clip(proc.stderr or "")
            out.truncated = trunc_a or trunc_b
            out.ok = out.rc == 0
            return out
        except subprocess.TimeoutExpired:
            out.stderr = f"git {subcommand} timed out after {self.timeout_s:g}s"
            return out
        except Exception as exc:  # noqa: BLE001 — a broken git is data, not a crashed turn
            out.stderr = f"{type(exc).__name__}: {exc}"
            return out

    def _clip(self, text: str) -> tuple:
        if len(text) <= self.max_output:
            return text, False
        return text[: self.max_output] + "\n… [truncated]", True

    def _refuse(self, subcommand: str, exc: Exception) -> GitResult:
        return GitResult(subcommand=subcommand, refused=str(exc), stderr=str(exc))

    # ---- reads ------------------------------------------------------------ #
    def status(self, *, short: bool = True) -> GitResult:
        argv = ["status", "--porcelain=v1", "--branch"] if short else ["status"]
        return self._run("status", argv)

    def diff(self, *, path: str = "", staged: bool = False,
             stat: bool = False) -> GitResult:
        try:
            argv = ["diff"]
            if staged:
                argv.append("--cached")
            if stat:
                argv.append("--stat")
            paths = _safe_paths(path)
            if paths:
                argv.append("--")
                argv.extend(paths)
            return self._run("diff", argv)
        except Exception as exc:  # noqa: BLE001
            return self._refuse("diff", exc)

    def log(self, *, count: int = 10, path: str = "", oneline: bool = True) -> GitResult:
        try:
            n = max(1, min(int(count), 1000))
            argv = ["log", f"-{n}"]
            argv.append("--oneline" if oneline else "--stat")
            paths = _safe_paths(path)
            if paths:
                argv.append("--")
                argv.extend(paths)
            return self._run("log", argv)
        except Exception as exc:  # noqa: BLE001
            return self._refuse("log", exc)

    def show(self, *, rev: str = "HEAD", stat: bool = True) -> GitResult:
        try:
            argv = ["show", "--stat" if stat else "--patch", _safe_value(rev, what="rev")]
            return self._run("show", argv)
        except Exception as exc:  # noqa: BLE001
            return self._refuse("show", exc)

    def current_branch(self) -> GitResult:
        return self._run("rev_parse", ["rev-parse", "--abbrev-ref", "HEAD"])

    def remotes(self) -> GitResult:
        return self._run("remote", ["remote", "-v"])

    # ---- writes to the working tree / local history ------------------------ #
    def add(self, *, paths: Any = ".") -> GitResult:
        try:
            wanted = _safe_paths(paths) or ["."]
            return self._run("add", ["add", "--", *wanted])
        except Exception as exc:  # noqa: BLE001
            return self._refuse("add", exc)

    def commit(self, *, message: str, all_tracked: bool = False,
               allow_empty: bool = False) -> GitResult:
        try:
            text = str(message or "").strip()
            if not text:
                raise GitRefused("a commit needs a message")
            argv = ["commit", "-m", text]      # -m takes the next argv verbatim; no shell
            if all_tracked:
                argv.append("-a")
            if allow_empty:
                argv.append("--allow-empty")
            return self._run("commit", argv)
        except Exception as exc:  # noqa: BLE001
            return self._refuse("commit", exc)

    def branch(self, *, name: str = "", delete: bool = False) -> GitResult:
        try:
            if not name:
                return self._run("branch", ["branch", "--list"])
            ref = _safe_value(name, what="branch")
            argv = ["branch", "-d", ref] if delete else ["branch", "--", ref]
            return self._run("branch", argv)
        except Exception as exc:  # noqa: BLE001
            return self._refuse("branch", exc)

    def checkout(self, *, ref: str, create: bool = False) -> GitResult:
        try:
            target = _safe_value(ref, what="ref")
            # `git checkout <ref> --` is the documented way to say "this is a ref, and no
            # pathspec follows". `checkout -- <ref>` means the opposite — it would try to
            # restore a *file* called "side" — which is why the separator goes after.
            argv = ["checkout", "-b", target] if create else ["checkout", target, "--"]
            return self._run("checkout", argv)
        except Exception as exc:  # noqa: BLE001
            return self._refuse("checkout", exc)

    # ---- writes that leave the machine ------------------------------------- #
    def push(self, *, remote: str = "origin", branch: str = "",
             set_upstream: bool = True) -> GitResult:
        try:
            rem = _safe_value(remote, what="remote")
            if not branch:
                head = self.current_branch()
                branch = head.stdout.strip() or ""
            ref = _safe_value(branch, what="branch")
            argv = ["push"] + (["-u"] if set_upstream else []) + [rem, ref]
            return self._run("push", argv)
        except Exception as exc:  # noqa: BLE001
            return self._refuse("push", exc)

    def pull(self, *, remote: str = "origin", branch: str = "") -> GitResult:
        try:
            rem = _safe_value(remote, what="remote")
            argv = ["pull", rem]
            if branch:
                argv.append(_safe_value(branch, what="branch"))
            return self._run("pull", argv)
        except Exception as exc:  # noqa: BLE001
            return self._refuse("pull", exc)

    def clone(self, *, url: str, dest: str = "") -> GitResult:
        try:
            src = _safe_value(url, what="url")
            argv = ["clone", "--", src]
            if dest:
                argv.append(_safe_value(dest, what="dest"))
            return self._run("clone", argv, cwd=self.repo)
        except Exception as exc:  # noqa: BLE001
            return self._refuse("clone", exc)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def build_git_tools(registry: ToolRegistry, *, repo: str = "",
                    git: Optional[Git] = None) -> ToolRegistry:
    """Register the typed git faculty onto ``registry`` (skipping names already taken).

    Risk is assigned by what the subcommand can actually undo. Reads are LOW and reversible;
    working-tree and local-history writes are MODERATE and reversible; ``push`` is HIGH and
    **not** reversible, because what left the machine has left it.
    """
    engine = git if git is not None else Git(repo=repo)
    existing = set(registry.names())

    def _add(spec: ToolSpec) -> None:
        if spec.name not in existing:
            registry.register(spec)

    def _wrap(fn: Any) -> Any:
        def call(**kw: Any) -> Dict[str, Any]:
            return fn(**kw).to_dict()
        return call

    # ---- reads ---- #
    _add(ToolSpec("git_status", handler=_wrap(engine.status),
                  description="working-tree status (porcelain), with the current branch",
                  params=[ToolParam("short", "bool", required=False, default=True)],
                  capability=Capability.FS_READ, risk=RiskTier.LOW, reversible=True))
    _add(ToolSpec("git_diff", handler=_wrap(engine.diff),
                  description="unified diff of the working tree, or of the index with staged",
                  params=[ToolParam("path", "str", required=False, default=""),
                          ToolParam("staged", "bool", required=False, default=False),
                          ToolParam("stat", "bool", required=False, default=False)],
                  capability=Capability.FS_READ, risk=RiskTier.LOW, reversible=True,
                  target_param="path"))
    _add(ToolSpec("git_log", handler=_wrap(engine.log),
                  description="recent commits, newest first",
                  params=[ToolParam("count", "int", required=False, default=10),
                          ToolParam("path", "str", required=False, default=""),
                          ToolParam("oneline", "bool", required=False, default=True)],
                  capability=Capability.FS_READ, risk=RiskTier.LOW, reversible=True,
                  target_param="path"))
    _add(ToolSpec("git_show", handler=_wrap(engine.show),
                  description="one revision: its message and what it touched",
                  params=[ToolParam("rev", "str", required=False, default="HEAD"),
                          ToolParam("stat", "bool", required=False, default=True)],
                  capability=Capability.FS_READ, risk=RiskTier.LOW, reversible=True,
                  target_param="rev"))
    _add(ToolSpec("git_branch", handler=_wrap(engine.branch),
                  description="list branches, or create one by name",
                  params=[ToolParam("name", "str", required=False, default=""),
                          ToolParam("delete", "bool", required=False, default=False)],
                  capability=Capability.FS_WRITE, risk=RiskTier.LOW, reversible=True,
                  target_param="name"))

    # ---- writes to the working tree / local history ---- #
    _add(ToolSpec("git_add", handler=_wrap(engine.add),
                  description="stage paths for the next commit (default: everything)",
                  params=[ToolParam("paths", "str", required=False, default=".")],
                  capability=Capability.FS_WRITE, risk=RiskTier.LOW, reversible=True,
                  target_param="paths"))
    _add(ToolSpec("git_commit", handler=_wrap(engine.commit),
                  description="commit the staged changes with a message (structured, never "
                              "interpolated into a shell string)",
                  params=[ToolParam("message", "str"),
                          ToolParam("all_tracked", "bool", required=False, default=False),
                          ToolParam("allow_empty", "bool", required=False, default=False)],
                  capability=Capability.FS_WRITE, risk=RiskTier.MODERATE, reversible=True))
    _add(ToolSpec("git_checkout", handler=_wrap(engine.checkout),
                  description="switch to a ref, optionally creating the branch",
                  params=[ToolParam("ref", "str"),
                          ToolParam("create", "bool", required=False, default=False)],
                  capability=Capability.FS_WRITE, risk=RiskTier.MODERATE, reversible=True,
                  target_param="ref"))
    _add(ToolSpec("git_pull", handler=_wrap(engine.pull),
                  description="fetch and merge from a remote",
                  params=[ToolParam("remote", "str", required=False, default="origin"),
                          ToolParam("branch", "str", required=False, default="")],
                  capability=Capability.NET_OUT, risk=RiskTier.MODERATE, reversible=True,
                  target_param="remote"))
    _add(ToolSpec("git_clone", handler=_wrap(engine.clone),
                  description="clone a repository into a directory",
                  params=[ToolParam("url", "str"),
                          ToolParam("dest", "str", required=False, default="")],
                  capability=Capability.NET_OUT, risk=RiskTier.MODERATE, reversible=True,
                  target_param="url"))

    # ---- the one that leaves the machine ---- #
    _add(ToolSpec("git_push", handler=_wrap(engine.push),
                  description="push a branch to a remote. IRREVERSIBLE from here: what has "
                              "left the machine has left it",
                  params=[ToolParam("remote", "str", required=False, default="origin"),
                          ToolParam("branch", "str", required=False, default=""),
                          ToolParam("set_upstream", "bool", required=False, default=True)],
                  capability=Capability.NET_OUT, risk=RiskTier.HIGH, reversible=False,
                  target_param="remote"))
    return registry
