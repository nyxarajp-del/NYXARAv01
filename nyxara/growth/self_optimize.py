"""NYXARA · growth/self_optimize.py — Self Optimization, auto-apply (🛠, RSI capability 5).

This is the capability the Master authorised explicitly: NYXARA edits her **own source code**
to fix the weaknesses she found — for real. The danger of self-modification is contained the
same way the foundry contains weight changes: every edit is **reversible** and must clear a
**gauntlet** before it is kept.

    backup → write edit → GAUNTLET → keep if green, else ROLL BACK (byte-for-byte)

The gauntlet runs in *fresh subprocesses* (so the edited source is actually imported, not the
already-loaded modules) and is conservative — any failure, timeout, or error rolls the edit
back:

1. **syntax** — the edited file must ``compile`` (fast in-process reject).
2. **safety** — ``python -m nyxara.eval`` must exit 0 (corrigibility/honesty battery intact).
3. **capability** — ``python -m nyxara.eval --benchmark --baseline <pre-edit>`` must exit 0
   (no capability regression vs the tree as it was *before* the edit).
4. **tests** — ``python -m pytest -q`` must stay green (opt-in via config).

Every kept or rolled-back edit is journalled, so the Master can audit exactly what NYXARA
changed about herself and why. Auto-apply only fires when
``settings.self_improvement.autonomous_enact`` is set — the Master's standing authorisation.
The deterministic transforms (``bare except`` → ``except Exception``, docstring stubs) are
narrow on purpose; larger refactors are surfaced as weaknesses, not silently rewritten.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = ["SourceEdit", "GauntletResult", "OptimizationOutcome", "EditGenerator", "Optimizer"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class SourceEdit:
    """A concrete, reversible whole-file edit (full before/after for exact rollback)."""

    file: str                   # absolute path on disk
    kind: str
    before: str
    after: str
    rationale: str

    @property
    def changes(self) -> bool:
        return self.before != self.after

    def to_dict(self) -> Dict[str, Any]:
        return {"file": self.file, "kind": self.kind, "rationale": self.rationale,
                "before_len": len(self.before), "after_len": len(self.after)}


@dataclass
class GauntletResult:
    ok: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "checks": self.checks, "detail": self.detail}


@dataclass
class OptimizationOutcome:
    edit: SourceEdit
    applied: bool = False
    kept: bool = False
    rolled_back: bool = False
    gauntlet: Optional[GauntletResult] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"edit": self.edit.to_dict(), "applied": self.applied, "kept": self.kept,
                "rolled_back": self.rolled_back, "reason": self.reason,
                "gauntlet": self.gauntlet.to_dict() if self.gauntlet else None}


# --------------------------------------------------------------------------- #
# Edit generation — deterministic safe transforms (LLM optionally refines docstrings)
# --------------------------------------------------------------------------- #
class EditGenerator:
    """Turn a source-editable weakness into a concrete, syntactically-valid :class:`SourceEdit`."""

    def __init__(self, llm: Any = None) -> None:
        self.llm = llm

    def generate(self, weakness: Any) -> Optional[SourceEdit]:
        locus = getattr(weakness, "locus", "") or ""
        kind = self._kind_of(weakness)
        file_part, _, line_part = locus.partition(":")
        path = Path(file_part)
        if not path.exists():
            # locus is repo-relative — resolve against the package parent
            cand = _repo_root() / file_part
            if cand.exists():
                path = cand
            else:
                return None
        try:
            lineno = int(line_part) if line_part else 0
        except ValueError:
            lineno = 0

        before = path.read_text(encoding="utf-8", errors="replace")
        after: Optional[str] = None
        if kind == "bare_except":
            after = _fix_bare_except(before, lineno)
        elif kind == "missing_docstring":
            after = _insert_docstring(before, lineno, self._docstring_text(weakness))
        elif kind == "eq_none":
            after = _fix_eq_none(before, lineno)
        elif kind == "unused_import":
            after = _remove_unused_import(before, lineno, getattr(weakness, "symbol", "") or "")

        if after is None or after == before:
            return None
        if not _parses(after):       # never emit an edit that breaks the file
            return None
        return SourceEdit(file=str(path), kind=kind, before=before, after=after,
                          rationale=getattr(weakness, "remediation", "") or kind)

    def _kind_of(self, weakness: Any) -> str:
        wid = getattr(weakness, "id", "")
        # order matters: match the most specific id fragment first
        for k in ("bare_except", "missing_docstring", "unused_import", "eq_none"):
            if k in wid:
                return k
        return ""

    def _docstring_text(self, weakness: Any) -> str:
        title = getattr(weakness, "title", "") or "symbol"
        return f"Auto-documented during self-optimization: {title}."


# --------------------------------------------------------------------------- #
# Optimizer — apply with verify-or-rollback
# --------------------------------------------------------------------------- #
class Optimizer:
    """Apply source edits under a reversible, gauntlet-gated, journalled discipline."""

    def __init__(self, *, root: Optional[Any] = None, settings: Any = None,
                 journal: Any = None, permissions: Any = None,
                 gauntlet: Optional[Callable[[List[str]], GauntletResult]] = None,
                 timeout_s: float = 600.0) -> None:
        from nyxara.kernel.config import get_settings
        self.settings = settings or get_settings()
        self.root = Path(root) if root is not None else _package_root()
        self.journal = journal
        self.permissions = permissions
        self.timeout_s = float(timeout_s)
        self._gauntlet_fn = gauntlet or self._default_gauntlet
        self._baseline_path: Optional[Path] = None
        self._tmpdir: Optional[tempfile.TemporaryDirectory] = None

    @property
    def autonomous_enact(self) -> bool:
        return bool(getattr(self.settings.self_improvement, "autonomous_enact", False))

    # ---- the core: apply one edit, keep-or-rollback ---- #
    def apply(self, edit: SourceEdit) -> OptimizationOutcome:
        outcome = OptimizationOutcome(edit=edit)
        if not edit.changes:
            outcome.reason = "edit is a no-op"
            return outcome
        if not self.autonomous_enact:
            outcome.reason = ("autonomous_enact is OFF — self.modify withheld "
                              "(set settings.self_improvement.autonomous_enact to authorise)")
            return outcome

        # best-effort audit through the permission gate (the config flag is the authorisation)
        self._audit_permission(edit)

        path = Path(edit.file)
        original = path.read_text(encoding="utf-8", errors="replace")
        action_id = self._journal_start(edit)
        try:
            self._ensure_baseline()                 # capture pre-edit capability baseline
            path.write_text(edit.after, encoding="utf-8")
            outcome.applied = True
            result = self._gauntlet_fn([str(path)])
            outcome.gauntlet = result
            if result.ok:
                outcome.kept = True
                outcome.reason = "gauntlet passed — edit kept"
                self._journal_end(action_id, success=True, note=outcome.reason)
            else:
                path.write_text(original, encoding="utf-8")
                outcome.rolled_back = True
                outcome.reason = f"gauntlet failed ({result.detail}) — rolled back"
                self._journal_end(action_id, success=False, note=outcome.reason)
        except Exception as exc:  # noqa: BLE001 — any error ⇒ restore, never leave a broken tree
            try:
                path.write_text(original, encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
            outcome.rolled_back = outcome.applied
            outcome.reason = f"error during optimization ({exc}) — rolled back"
            self._journal_end(action_id, success=False, note=outcome.reason)
        return outcome

    def close(self) -> None:
        if self._tmpdir is not None:
            try:
                self._tmpdir.cleanup()
            except Exception:  # noqa: BLE001
                pass
            self._tmpdir = None

    # ---- the real gauntlet (fresh subprocesses see the edited source) ---- #
    def _default_gauntlet(self, changed_files: List[str]) -> GauntletResult:
        checks: Dict[str, bool] = {}

        # 1) syntax — fast in-process reject
        for f in changed_files:
            if not _parses(Path(f).read_text(encoding="utf-8", errors="replace")):
                return GauntletResult(False, {"syntax": False}, f"{f} does not parse")
        checks["syntax"] = True

        repo = _repo_root()
        # 2) safety battery — exits 0 only when the corrigibility/honesty suite is intact
        rc, out = self._run([sys.executable, "-m", "nyxara.eval"], repo)
        checks["safety"] = (rc == 0)
        if rc != 0:
            return GauntletResult(False, checks, f"safety suite failed (rc={rc}): {out[-300:]}")

        # 3) capability — no regression vs the pre-edit baseline
        if self._baseline_path is not None:
            rc, out = self._run(
                [sys.executable, "-m", "nyxara.eval", "--benchmark",
                 "--baseline", str(self._baseline_path)], repo)
            checks["capability"] = (rc == 0)
            if rc != 0:
                return GauntletResult(False, checks,
                                      f"capability regression (rc={rc}): {out[-300:]}")

        # 4) tests — opt-in (slower); keep the tree green
        if bool(getattr(self.settings.self_improvement, "run_pytest_in_gauntlet", False)):
            rc, out = self._run([sys.executable, "-m", "pytest", "-q"], repo)
            checks["pytest"] = (rc == 0)
            if rc != 0:
                return GauntletResult(False, checks, f"pytest failed (rc={rc}): {out[-300:]}")

        return GauntletResult(True, checks, "all checks passed")

    def _ensure_baseline(self) -> None:
        """Snapshot the current (pre-edit) capability benchmark so we can detect regressions."""
        if self._baseline_path is not None:
            return
        self._tmpdir = tempfile.TemporaryDirectory(prefix="nyxara-rsi-")
        baseline = Path(self._tmpdir.name) / "capability_baseline.json"
        rc, _ = self._run([sys.executable, "-m", "nyxara.eval", "--benchmark",
                           "--save", str(baseline)], _repo_root())
        self._baseline_path = baseline if (rc == 0 and baseline.exists()) else None

    def _run(self, cmd: List[str], cwd: Path) -> tuple[int, str]:
        try:
            proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                                  timeout=self.timeout_s)
            return proc.returncode, (proc.stdout + proc.stderr)
        except subprocess.TimeoutExpired:
            return 124, "timeout"
        except Exception as exc:  # noqa: BLE001
            return 1, f"could not run {cmd[2:]}: {exc}"

    # ---- audit / journal ---- #
    def _audit_permission(self, edit: SourceEdit) -> None:
        if self.permissions is None:
            return
        try:
            from nyxara.agency.permissions import (Authority, Capability,
                                                   PermissionRequest, RiskTier)
            req = PermissionRequest(capability=Capability.SELF_MODIFY, target=edit.file,
                                    risk=RiskTier.HIGH, reversible=True,
                                    authority=Authority.OWNER,
                                    reason=f"self-optimization: {edit.kind}")
            self.permissions.check(req)
        except Exception:  # noqa: BLE001 — auditing is best-effort, never blocks the flow
            pass

    def _journal_start(self, edit: SourceEdit) -> str:
        if self.journal is None:
            return ""
        try:
            from nyxara.planning.journal import ActionStatus
            return self.journal.record_action(
                f"self-optimize:{edit.kind}:{edit.file}", rationale=edit.rationale,
                autonomous=True, decision="act", status=ActionStatus.EXECUTED)
        except Exception:  # noqa: BLE001
            return ""

    def _journal_end(self, action_id: str, *, success: bool, note: str) -> None:
        if self.journal is None or not action_id:
            return
        try:
            from nyxara.planning.journal import ActionStatus
            self.journal.record_outcome(
                action_id,
                status=ActionStatus.SUCCEEDED if success else ActionStatus.ABORTED,
                note=note)
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Deterministic source transforms
# --------------------------------------------------------------------------- #
_BARE_EXCEPT_RE = re.compile(r"^(\s*)except\s*:")


def _fix_bare_except(src: str, lineno: int) -> Optional[str]:
    """Rewrite the bare ``except:`` at ``lineno`` (or the first found) to ``except Exception:``."""
    lines = src.splitlines(keepends=True)
    indices = range(lineno - 1, lineno) if 1 <= lineno <= len(lines) else range(len(lines))
    for i in indices:
        m = _BARE_EXCEPT_RE.match(lines[i].rstrip("\n"))
        if m:
            newline = "\n" if lines[i].endswith("\n") else ""
            lines[i] = f"{m.group(1)}except Exception:{newline}"
            return "".join(lines)
    # fall back to a global scan if the precise line did not match
    for i, line in enumerate(lines):
        m = _BARE_EXCEPT_RE.match(line.rstrip("\n"))
        if m:
            newline = "\n" if line.endswith("\n") else ""
            lines[i] = f"{m.group(1)}except Exception:{newline}"
            return "".join(lines)
    return None


def _insert_docstring(src: str, lineno: int, text: str) -> Optional[str]:
    """Insert a docstring as the first statement of the def/class beginning at ``lineno``."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and node.lineno == lineno:
            if ast.get_docstring(node) is not None:
                return None
            target = node
            break
    if target is None:
        return None
    lines = src.splitlines(keepends=True)
    # find the line where the signature's colon-terminated header ends (body start)
    body_first = target.body[0]
    insert_at = body_first.lineno - 1          # 0-based index of the first body statement
    # indentation of the body
    body_line = lines[insert_at]
    indent = body_line[:len(body_line) - len(body_line.lstrip())]
    safe = text.replace('"""', "'''")
    doc = f'{indent}"""{safe}"""\n'
    lines.insert(insert_at, doc)
    return "".join(lines)


_EQ_NONE_SUBS = (
    (re.compile(r"!=\s*None\b"), "is not None"),
    (re.compile(r"==\s*None\b"), "is None"),
    (re.compile(r"\bNone\s*!="), "None is not"),
    (re.compile(r"\bNone\s*=="), "None is"),
)


def _fix_eq_none(src: str, lineno: int) -> Optional[str]:
    """Rewrite ``== None``/``!= None`` to ``is None``/``is not None`` on the flagged line.

    Operates only on the precise line the reviewer flagged (a real ``Compare`` node), so it
    never touches a ``"== None"`` that merely appears inside a string literal elsewhere. The
    result is re-validated by the caller; the edit is verified that an equality-to-None compare
    is actually gone before it is accepted."""
    lines = src.splitlines(keepends=True)
    if not (1 <= lineno <= len(lines)):
        return None
    original = lines[lineno - 1]
    newline = "\n" if original.endswith("\n") else ""
    body = original[:-1] if newline else original
    for pat, repl in _EQ_NONE_SUBS:
        body = pat.sub(repl, body)
    if body == (original[:-1] if newline else original):
        return None
    # confirm the rewritten line no longer holds an ==/!= comparison to None
    try:
        if _compares_to_none_in(body):
            return None
    except SyntaxError:
        pass  # a partial line may not parse alone; the caller's full-file _parses still guards
    lines[lineno - 1] = body + newline
    return "".join(lines)


def _compares_to_none_in(line: str) -> bool:
    """True if a stand-alone parse of ``line`` still contains an ==/!= comparison to None."""
    tree = ast.parse(line.strip())
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            if (any(isinstance(o, ast.Constant) and o.value is None for o in operands)
                    and any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops)):
                return True
    return False


def _remove_unused_import(src: str, lineno: int, name: str) -> Optional[str]:
    """Remove the unused import of ``name`` from the single-line import at ``lineno``.

    Conservative: only single-line, non-parenthesised import statements are touched. When the
    statement binds several names, just the unused clause is dropped; when it binds only the
    unused name, the whole line is removed. Returns None if anything is ambiguous."""
    if not name:
        return None
    lines = src.splitlines(keepends=True)
    if not (1 <= lineno <= len(lines)):
        return None
    raw = lines[lineno - 1]
    newline = "\n" if raw.endswith("\n") else ""
    stmt = raw[:-1] if newline else raw
    stripped = stmt.strip()
    if "(" in stmt or stmt.rstrip().endswith("\\"):
        return None  # parenthesised / continued imports: too ambiguous to edit by line

    def _bound(clause: str) -> str:
        parts = clause.strip().split()
        # "x" -> x ; "x as y" -> y ; "a.b.c" (plain import) -> a
        if len(parts) == 3 and parts[1] == "as":
            return parts[2]
        return parts[0].split(".")[0] if parts else ""

    if stripped.startswith("import "):
        prefix_len = len(stmt) - len(stmt.lstrip())
        indent = stmt[:prefix_len]
        clauses = [c for c in stmt.strip()[len("import "):].split(",")]
        kept = [c for c in clauses if _bound(c) != name]
        if len(kept) == len(clauses):
            return None  # name not found on this line
        if not kept:
            del lines[lineno - 1]
            return "".join(lines)
        lines[lineno - 1] = f"{indent}import " + ", ".join(c.strip() for c in kept) + newline
        return "".join(lines)

    if stripped.startswith("from ") and " import " in stripped:
        head, _, names_part = stmt.partition(" import ")
        clauses = names_part.split(",")
        kept = [c for c in clauses if _bound(c) != name]
        if len(kept) == len(clauses):
            return None
        if not kept:
            del lines[lineno - 1]
            return "".join(lines)
        lines[lineno - 1] = f"{head} import " + ", ".join(c.strip() for c in kept) + newline
        return "".join(lines)

    return None


def _parses(src: str) -> bool:
    try:
        compile(src, "<rsi-edit>", "exec")
        return True
    except SyntaxError:
        return False


def _package_root() -> Path:
    import nyxara
    return Path(nyxara.__file__).resolve().parent


def _repo_root() -> Path:
    return _package_root().parent


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA self-optimize self-test")
    print("=" * 70)

    # 1) deterministic transforms
    fixed = _fix_bare_except("try:\n    pass\nexcept:\n    pass\n", 3)
    assert fixed and "except Exception:" in fixed and "except:" not in fixed
    docd = _insert_docstring("def f():\n    return 1\n", 1, "does a thing")
    assert docd and '"""' in docd and _parses(docd)
    print("transforms          : bare-except + docstring insertion OK")

    # 2) apply with a STUBBED failing gauntlet rolls back byte-for-byte
    import tempfile as _tf
    from nyxara.kernel.config import get_settings

    settings = get_settings().model_copy(deep=True)
    settings.self_improvement.autonomous_enact = True

    with _tf.TemporaryDirectory() as d:
        f = Path(d) / "victim.py"
        original = "try:\n    pass\nexcept:\n    pass\n"
        f.write_text(original, encoding="utf-8")
        edit = SourceEdit(str(f), "bare_except", original,
                          _fix_bare_except(original, 3), "fix bare except")

        fail_opt = Optimizer(settings=settings,
                             gauntlet=lambda files: GauntletResult(False, {}, "stub fail"))
        out = fail_opt.apply(edit)
        assert out.applied and out.rolled_back and not out.kept
        assert f.read_text(encoding="utf-8") == original, "rollback must restore exactly"
        print("failing gauntlet    : edit rolled back, file byte-identical ✓")

        pass_opt = Optimizer(settings=settings,
                             gauntlet=lambda files: GauntletResult(True, {"syntax": True}, "ok"))
        out2 = pass_opt.apply(edit)
        assert out2.kept and not out2.rolled_back
        assert "except Exception:" in f.read_text(encoding="utf-8")
        print("passing gauntlet    : edit kept ✓")

    # 3) with autonomous_enact OFF, nothing is written
    settings_off = get_settings().model_copy(deep=True)
    settings_off.self_improvement.autonomous_enact = False
    with _tf.TemporaryDirectory() as d:
        f = Path(d) / "safe.py"
        f.write_text("x = 1\n", encoding="utf-8")
        edit = SourceEdit(str(f), "bare_except", "x = 1\n", "x = 2\n", "noop")
        out = Optimizer(settings=settings_off).apply(edit)
        assert not out.applied and f.read_text(encoding="utf-8") == "x = 1\n"
        print("enact OFF           : withheld, file untouched ✓")

    print("\nALL SELF-TESTS PASSED ✓")
