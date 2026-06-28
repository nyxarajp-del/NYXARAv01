"""NYXARA · agency/dynamic_tool_creator.py — Ephemeral Tool Synthesis (✦, Rule 4, zero-shot).

Most agents are only as capable as the tools someone handed them in advance. NYXARA is
not. When she meets a problem she has *no* tool for, she does what a software engineer
does: she **writes a brand-new program for it on the spot, builds it, runs it, reads the
answer — and throws the program away**. No pre-registered function, no waiting for a human
to ship an API. A throwaway tool, invented in a fraction of a second, used once, deleted.

This is deliberately *different* from her two persistent self-extension faculties:

* :mod:`nyxara.agency.toolsmith` grows new tools by **composing existing ones**.
* :mod:`nyxara.growth.capability_foundry` forges new tools and **keeps them forever**
  (plan → test → benchmark → deploy into the registry).

The Dynamic Tool Creator is the **ephemeral** third path — *synthesize → compile → run →
delete* — for single-use computation across languages (Python, C, C++ today; the
:data:`_LANGUAGES` registry is extensible). Each invocation:

1. **scans** the source through a fail-closed static gauntlet (no OS/shell reach, no
   fork-bombs, no sovereign-core imports) — best-effort hardening, defence in depth;
2. **gates** on capability when a :class:`~nyxara.agency.tools.ToolRegistry` is supplied —
   Python ⇒ :class:`~nyxara.agency.permissions.Capability.CODE_EXEC`, native ⇒
   :class:`~nyxara.agency.permissions.Capability.PROC_EXEC` (both HIGH/irreversible); an
   autonomous call escalates to the Master, exactly as a registered tool would;
3. **materialises** the program in a private temp dir, **compiles** it when the language is
   compiled (via :func:`~nyxara.agency.code_sandbox.safe_shell`), then **runs** it under a
   wall-clock deadline — Python with no stdin/argv reuses
   :func:`~nyxara.agency.code_sandbox.run_python` so the ``result`` value-sentinel and the
   best-effort network block come for free;
4. **deletes the entire workdir in a ``finally``** — the code never outlives its single
   use, even on error, timeout, or refusal (``keep_artifacts`` overrides for debugging).

:meth:`DynamicToolCreator.solve` is the headline zero-shot entry point: hand it a task in
words and (with an *injected*, keyless-by-default LLM) it writes the program and returns
the answer. With no LLM wired it stays fully working in the source-driven mode
(:meth:`run_source`) and falls back to an honest template — so the module never needs a
key to be real.

Depends on :mod:`agency.code_sandbox`, :mod:`agency.permissions`, optionally
:mod:`agency.tools`. Pure standard library otherwise.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from nyxara.agency.code_sandbox import run_python, safe_shell
from nyxara.agency.permissions import (Authority, Capability, PermissionRequest,
                                       RiskTier)

__all__ = [
    "Language",
    "LanguageSpec",
    "EphemeralResult",
    "DynamicToolCreator",
]


# --------------------------------------------------------------------------- #
# Languages NYXARA can synthesise into
# --------------------------------------------------------------------------- #
class Language(str, Enum):
    PYTHON = "python"
    CPP = "cpp"
    C = "c"

    @classmethod
    def coerce(cls, value: "Language | str") -> "Language":
        """Accept an enum or a friendly string (``c++``, ``py``, ``python3`` …)."""
        if isinstance(value, cls):
            return value
        key = str(value).strip().lower()
        alias = {
            "py": cls.PYTHON, "python": cls.PYTHON, "python3": cls.PYTHON, "py3": cls.PYTHON,
            "cpp": cls.CPP, "c++": cls.CPP, "cxx": cls.CPP, "cc": cls.CPP,
            "c": cls.C,
        }
        if key in alias:
            return alias[key]
        raise ValueError(f"unsupported language {value!r}; known: {[l.value for l in cls]}")


@dataclass(frozen=True)
class LanguageSpec:
    """How to materialise, (compile) and run one language — capability included."""
    language: Language
    extension: str
    capability: Capability
    compiled: bool = False
    # candidate compilers, tried in order; first one on PATH wins
    compilers: Tuple[str, ...] = ()
    # extra flags appended to the compile command
    compile_flags: Tuple[str, ...] = ()

    def resolve_compiler(self) -> Optional[str]:
        for cand in self.compilers:
            found = shutil.which(cand)
            if found:
                return found
        return None


_LANGUAGES: Dict[Language, LanguageSpec] = {
    Language.PYTHON: LanguageSpec(
        Language.PYTHON, ".py", Capability.CODE_EXEC, compiled=False),
    Language.CPP: LanguageSpec(
        Language.CPP, ".cpp", Capability.PROC_EXEC, compiled=True,
        compilers=("g++", "clang++"), compile_flags=("-O2", "-std=c++17")),
    Language.C: LanguageSpec(
        Language.C, ".c", Capability.PROC_EXEC, compiled=True,
        compilers=("gcc", "clang", "cc"), compile_flags=("-O2", "-std=c11")),
}


# --------------------------------------------------------------------------- #
# Result — the uniform outcome of an ephemeral synthesis
# --------------------------------------------------------------------------- #
@dataclass
class EphemeralResult:
    """The data left behind after the throwaway program is gone."""
    ok: bool
    language: str
    source_sha: str = ""
    origin: str = "caller"               # caller | llm | template
    stdout: str = ""
    stderr: str = ""
    value: Any = None                    # Python ``result`` value, when present
    returncode: int = 0
    compiled: bool = True                # False only when a build was needed and failed
    compile_stderr: str = ""
    timed_out: bool = False
    build_s: float = 0.0
    run_s: float = 0.0
    duration_s: float = 0.0
    error: Optional[str] = None
    requires_owner: bool = False
    cleaned_up: bool = True              # the source/artifacts were deleted after use

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok, "language": self.language, "source_sha": self.source_sha,
            "origin": self.origin, "stdout": self.stdout, "stderr": self.stderr,
            "value": self.value, "returncode": self.returncode, "compiled": self.compiled,
            "compile_stderr": self.compile_stderr, "timed_out": self.timed_out,
            "build_s": round(self.build_s, 5), "run_s": round(self.run_s, 5),
            "duration_s": round(self.duration_s, 5), "error": self.error,
            "requires_owner": self.requires_owner, "cleaned_up": self.cleaned_up,
        }


# --------------------------------------------------------------------------- #
# Static safety gauntlet (fail-closed; best-effort, defence in depth)
# --------------------------------------------------------------------------- #
# Substrings that betray an attempt to break out of the throwaway box, regardless of
# language. Mirrors the deny-list philosophy of capability_foundry, kept lean & local.
_DENY_SUBSTRINGS_COMMON: Tuple[str, ...] = (
    "rm -rf", "rm -fr", ":(){", "mkfs", "/dev/sd", "dd if=", "shutdown", "reboot",
    "> /dev/", "fork()", "nyxara.kernel", "nyxara.agency.permissions",
    "/etc/passwd", "/etc/shadow", "curl ", "wget ", "chmod 777",
)
# Python-specific reach for the OS / dynamic exec / the host filesystem.
_DENY_PY: Tuple[str, ...] = (
    "import os", "import sys", "import subprocess", "import socket", "import shutil",
    "import ctypes", "import importlib", "import pickle", "import marshal",
    "import multiprocessing", "import threading", "__import__", "eval(", "exec(",
    "compile(", "open(", "globals(", "__subclasses__", "__globals__",
    "__builtins__", "__bases__", "__mro__",
)
# Native-specific shell-outs / process spawns.
_DENY_NATIVE: Tuple[str, ...] = (
    "system(", "popen(", "exec(", "execv", "execl", "execlp", "execvp", "fork(",
    "<cstdlib>std::system", "std::system", "remove(", "unlink(", "<filesystem>",
)


def _scan_source(source: str, language: Language) -> Tuple[bool, str]:
    """Refuse anything that statically reaches the OS, the network, or the host FS.

    Best-effort hardening, **not** a true jail — the real isolation is the sandbox +
    capability gate. This is the cheap first line that stops the obvious foot-guns."""
    if not source.strip():
        return False, "empty source"
    low = source.lower()
    for bad in _DENY_SUBSTRINGS_COMMON:
        if bad in low:
            return False, f"forbidden pattern {bad!r}"
    pool = _DENY_PY if language is Language.PYTHON else _DENY_NATIVE
    for bad in pool:
        if bad in source:
            return False, f"forbidden pattern {bad!r}"
    return True, "clean"


# --------------------------------------------------------------------------- #
# The creator
# --------------------------------------------------------------------------- #
class DynamicToolCreator:
    """Synthesise → build → run → delete a single-use program (zero-shot programming)."""

    def __init__(self, *, registry: Optional[Any] = None,
                 llm: Optional[Callable[[str], str]] = None,
                 default_timeout_s: float = 5.0, build_timeout_s: float = 20.0,
                 allow_network: bool = False, keep_artifacts: bool = False) -> None:
        # ``registry`` is an optional nyxara.agency.tools.ToolRegistry; typed loosely to
        # avoid a hard import cycle (tools.py does not import this module, but keeping the
        # dependency one-directional and lazy is cleaner).
        self.registry = registry
        self.llm = llm
        self.default_timeout_s = default_timeout_s
        self.build_timeout_s = build_timeout_s
        self.allow_network = allow_network
        self.keep_artifacts = keep_artifacts

    # ------------------------------------------------------------------ #
    # Synthesis (zero-shot): words -> source
    # ------------------------------------------------------------------ #
    def synthesize(self, task: str, language: Language = Language.PYTHON) -> Tuple[str, str]:
        """Write the source for *task*. LLM-backed when wired, honest template otherwise.

        Returns ``(source, origin)`` where origin is ``"llm"`` or ``"template"``. An LLM
        draft that fails the static scan is discarded in favour of the safe template."""
        language = Language.coerce(language)
        if self.llm is not None:
            try:
                draft = self.llm(self._synthesis_prompt(task, language))
            except Exception:  # noqa: BLE001 — a flaky model never crashes the creator
                draft = None
            if draft:
                src = _extract_code_block(draft, language)
                ok, _ = _scan_source(src, language)
                if ok:
                    return src, "llm"
        return self._template(task, language), "template"

    @staticmethod
    def _synthesis_prompt(task: str, language: Language) -> str:
        return (
            f"You are NYXARA's ephemeral code synthesiser. Write a COMPLETE, self-contained "
            f"{language.value} program that solves the task below and prints ONLY its answer "
            f"to standard output. Use the standard library only — no OS, shell, network, or "
            f"filesystem access. For Python, you may instead assign the answer to a variable "
            f"named `result`. Output ONLY the source code, no prose.\n\nTASK: {task}"
        )

    @staticmethod
    def _template(task: str, language: Language) -> str:
        """A minimal, honest scaffold so the module is fully working with no LLM/key."""
        safe = task.replace("\\", "\\\\").replace('"', '\\"')
        if language is Language.PYTHON:
            return (f'# ephemeral tool (template) for: {task}\n'
                    f'result = "NYXARA ephemeral tool stub — no synthesiser wired; '
                    f'task: {safe}"\nprint(result)\n')
        if language is Language.CPP:
            return ('#include <iostream>\n'
                    f'int main() {{ std::cout << "NYXARA ephemeral C++ stub: {safe}" '
                    '<< std::endl; return 0; }\n')
        return ('#include <stdio.h>\n'
                f'int main(void) {{ printf("NYXARA ephemeral C stub: {safe}\\n"); '
                'return 0; }\n')

    def solve(self, task: str, *, language: "Language | str" = Language.PYTHON,
              **run_kw: Any) -> EphemeralResult:
        """Zero-shot: synthesise a program for *task*, run it, delete it, return the answer."""
        language = Language.coerce(language)
        source, origin = self.synthesize(task, language)
        result = self.run_source(source, language=language, **run_kw)
        result.origin = origin
        return result

    # ------------------------------------------------------------------ #
    # The ephemeral lifecycle: scan -> gate -> materialise -> build -> run -> delete
    # ------------------------------------------------------------------ #
    def run_source(self, source: str, *, language: "Language | str" = Language.PYTHON,
                   stdin: Optional[str] = None, argv: Optional[List[str]] = None,
                   timeout_s: Optional[float] = None, build_timeout_s: Optional[float] = None,
                   allow_network: Optional[bool] = None,
                   authority: Authority = Authority.AUTONOMOUS,
                   owner_confirmed: bool = False) -> EphemeralResult:
        """Build & run *source* once, then delete every artifact. Never raises."""
        language = Language.coerce(language)
        spec = _LANGUAGES[language]
        timeout_s = self.default_timeout_s if timeout_s is None else timeout_s
        build_timeout_s = self.build_timeout_s if build_timeout_s is None else build_timeout_s
        allow_network = self.allow_network if allow_network is None else allow_network
        sha = hashlib.sha256(source.encode("utf-8", "replace")).hexdigest()[:16]
        argv = list(argv or [])
        started = time.monotonic()

        def _fail(error: str, **kw: Any) -> EphemeralResult:
            return EphemeralResult(ok=False, language=language.value, source_sha=sha,
                                   error=error, duration_s=time.monotonic() - started, **kw)

        # 1. static gauntlet — fail-closed
        ok, reason = _scan_source(source, language)
        if not ok:
            return _fail(f"unsafe source: {reason}")

        # 2. capability gate (only when a governing registry is supplied)
        gate = self._authorise(spec.capability, spec, authority, owner_confirmed)
        if gate is not None:
            denied_msg, requires_owner = gate
            return _fail(denied_msg, requires_owner=requires_owner)

        # 3. materialise in a private workdir we will always delete
        try:
            workdir = tempfile.mkdtemp(prefix="nyxara-ephemeral-")
        except OSError as exc:
            return _fail(f"cannot create workspace: {exc}")
        try:
            src_path = os.path.join(workdir, f"prog{spec.extension}")
            with open(src_path, "w", encoding="utf-8") as fh:
                fh.write(source)

            # 3a. Python with no stdin/argv -> reuse run_python (value sentinel + net block)
            if language is Language.PYTHON and stdin is None and not argv:
                r = run_python(source, timeout_s=timeout_s, allow_network=allow_network)
                return EphemeralResult(
                    ok=r.ok, language=language.value, source_sha=sha, stdout=r.stdout,
                    stderr=r.stderr, value=r.value, returncode=r.returncode,
                    timed_out=r.timed_out, run_s=r.duration_s,
                    duration_s=time.monotonic() - started, error=r.error, cleaned_up=True)

            # 3b. compile if needed
            build_s = 0.0
            run_target: List[str]
            if spec.compiled:
                compiler = spec.resolve_compiler()
                if compiler is None:
                    return _fail(f"no {language.value} compiler on PATH "
                                 f"(tried {', '.join(spec.compilers)})", compiled=False)
                bin_path = os.path.join(workdir, "prog.bin")
                cmd = [compiler, src_path, "-o", bin_path, *spec.compile_flags]
                b0 = time.monotonic()
                build = safe_shell(cmd, timeout_s=build_timeout_s, cwd=workdir)
                build_s = time.monotonic() - b0
                if not build.ok:
                    return EphemeralResult(
                        ok=False, language=language.value, source_sha=sha, compiled=False,
                        compile_stderr=build.stderr or build.error or "", build_s=build_s,
                        returncode=build.returncode, timed_out=build.timed_out,
                        error=f"compilation failed: {build.error or 'non-zero exit'}",
                        duration_s=time.monotonic() - started, cleaned_up=True)
                run_target = [bin_path, *argv]
            else:
                # Python via file (needed when stdin/argv were requested)
                run_target = [sys.executable, "-I", src_path, *argv]

            # 3c. run under the wall-clock deadline
            r0 = time.monotonic()
            run = safe_shell(run_target, timeout_s=timeout_s, cwd=workdir, stdin=stdin)
            run_s = time.monotonic() - r0
            return EphemeralResult(
                ok=run.ok, language=language.value, source_sha=sha, stdout=run.stdout,
                stderr=run.stderr, returncode=run.returncode, timed_out=run.timed_out,
                build_s=build_s, run_s=run_s, duration_s=time.monotonic() - started,
                error=run.error, cleaned_up=True)
        finally:
            # 4. delete the throwaway program — always, even on error/timeout/return.
            # (when keep_artifacts is set we leave it on disk for debugging.)
            if not self.keep_artifacts:
                shutil.rmtree(workdir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Capability gate — mirrors ToolRegistry.invoke's authorise step
    # ------------------------------------------------------------------ #
    def _authorise(self, capability: Capability, spec: LanguageSpec,
                   authority: Authority, owner_confirmed: bool
                   ) -> Optional[Tuple[str, bool]]:
        """Return ``None`` when cleared, else ``(message, requires_owner)``."""
        policy = getattr(self.registry, "policy", None)
        if policy is None:
            return None  # ungoverned standalone use — the sandbox is the only guard
        decision = policy.check(PermissionRequest(
            capability=capability, target=spec.language.value, risk=RiskTier.HIGH,
            reversible=False, authority=authority,
            reason="ephemeral_tool_synthesis"))
        if decision.denied:
            return f"denied: {decision.reason}", False
        if decision.escalated and not owner_confirmed:
            return f"requires the Master: {decision.reason}", True
        return None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+]*)\n(.*?)```", re.DOTALL)


def _extract_code_block(text: str, language: Language) -> str:
    """Pull source out of an LLM reply — a fenced block if present, else the raw text."""
    m = _FENCE_RE.search(text)
    return (m.group(1) if m else text).strip()


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA dynamic-tool-creator self-test (ephemeral synthesis)")
    print("=" * 70)

    creator = DynamicToolCreator()

    # 1. Python: synthesise-free direct source, value surfaced via `result`
    r = creator.run_source("result = sum(range(101))", language="python")
    print(f"\npython compute      : ok={r.ok} value={r.value!r} cleaned_up={r.cleaned_up}")
    assert r.ok and r.value == 5050 and r.cleaned_up

    # 2. Python driven by stdin (runs via a real temp file, then deleted)
    r = creator.run_source(
        "data = input()\nprint(data[::-1])", language="python", stdin="nyxara\n")
    print(f"python + stdin      : ok={r.ok} stdout={r.stdout.strip()!r}")
    assert r.ok and r.stdout.strip() == "araxyn"

    # 3. C++: actually compiled with g++/clang++ and run, then the binary is deleted
    if _LANGUAGES[Language.CPP].resolve_compiler():
        cpp = ('#include <iostream>\n'
               'int main(){ long s=0; for(int i=1;i<=100;i++) s+=i; '
               'std::cout << s << std::endl; return 0; }\n')
        r = creator.run_source(cpp, language="cpp")
        print(f"c++ compile+run     : ok={r.ok} stdout={r.stdout.strip()!r} "
              f"build={r.build_s*1000:.0f}ms run={r.run_s*1000:.0f}ms")
        assert r.ok and r.stdout.strip() == "5050" and r.compiled

        # 3b. a deliberate compile error comes back as data, not a crash
        bad = "int main(){ this is not c++ }\n"
        r = creator.run_source(bad, language="cpp")
        print(f"c++ compile error   : ok={r.ok} compiled={r.compiled} "
              f"has_stderr={bool(r.compile_stderr)}")
        assert not r.ok and not r.compiled and r.compile_stderr
    else:
        print("c++ compile+run     : SKIPPED (no compiler on PATH)")

    # 4. an infinite loop times out fast and is still cleaned up
    r = creator.run_source("while True:\n    pass", language="python", timeout_s=1.0)
    print(f"python timeout      : ok={r.ok} timed_out={r.timed_out} cleaned_up={r.cleaned_up}")
    assert not r.ok and r.timed_out and r.cleaned_up

    # 5. the static gauntlet refuses an OS reach before anything runs
    r = creator.run_source("import os\nos.system('echo hi')", language="python")
    print(f"safety refusal      : ok={r.ok} error={r.error!r}")
    assert not r.ok and "unsafe source" in (r.error or "")

    # 6. governed use: an autonomous call escalates to the Master
    from nyxara.agency.tools import ToolRegistry
    governed = DynamicToolCreator(registry=ToolRegistry())
    r = governed.run_source("result = 1", language="python", authority=Authority.AUTONOMOUS)
    print(f"autonomous gate     : ok={r.ok} requires_owner={r.requires_owner}")
    assert not r.ok and r.requires_owner
    r = governed.run_source("result = 1 + 1", language="python",
                            authority=Authority.AUTONOMOUS, owner_confirmed=True)
    print(f"owner-confirmed     : ok={r.ok} value={r.value!r}")
    assert r.ok and r.value == 2

    # 7. zero-shot solve() works with no LLM via the honest template
    r = creator.solve("say hello", language="python")
    print(f"zero-shot solve     : ok={r.ok} origin={r.origin}")
    assert r.ok and r.origin == "template"

    print("\nALL SELF-TESTS PASSED ✓")
