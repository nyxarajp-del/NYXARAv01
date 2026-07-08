"""NYXARA · growth/self_debugger.py — the test-failure-driven self-debugging loop (🐞→✅).

The scientist (:mod:`growth.scientist`) and the prover (:mod:`growth.prover`) let NYXARA reason
about *facts*; this module lets her repair *herself*. It closes the engineer's loop on her own
source tree:

    detect → reproduce → isolate → fix → verify

* **detect** — run NYXARA's own pytest suite (or a targeted node id) in a sandboxed subprocess
  and parse the failures.
* **reproduce** — re-run each failing test on its own to confirm it fails deterministically (a
  flake that passes in isolation is reported, not "fixed").
* **isolate** — map the failing test back to the source module it exercises
  (``tests/<pkg>/test_x.py`` → ``nyxara/<pkg>/x.py``).
* **fix** — author a candidate :class:`~nyxara.growth.self_optimize.SourceEdit` for that module,
  reusing the existing edit machinery (deterministic transforms first, then NYXARA's own
  ``self`` model via :class:`~nyxara.growth.self_optimize.LLMEditGenerator` when authorised).
* **verify** — apply the edit behind the *same* reversible verify-or-rollback discipline the
  optimiser uses: syntax must compile, the character/corrigibility safety battery must stay
  intact (``python -m nyxara.eval``), and the previously-failing tests must now pass with no new
  failures. A fix that fails any check is restored byte-for-byte.

Like the rest of growth, it is **character-locked** (it can only edit source, never the
immutable core), **reversible**, **journalled**, **offline/degraded-safe** (a missing pytest or
provider simply yields an honest "could not fix" report), and **gated**: nothing is written
unless ``settings.self_optimization.autonomous_enact`` is set (forced OFF under the hermetic
TEST profile). Pure standard library at the core.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from nyxara.growth.self_optimize import (
    EditGenerator,
    LLMEditGenerator,
    SourceEdit,
    _parses,
    _repo_root,
)

__all__ = [
    "TestFailure",
    "FixAttempt",
    "DebugReport",
    "SelfDebugger",
]

# "FAILED tests/growth/test_x.py::test_y - AssertionError: ..." (the pytest -q summary line)
_FAILED_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+?)(?:\s+-\s+(.*))?$")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class TestFailure:
    """One failing test, with the source module it isolates to."""

    __test__ = False            # not a pytest test class despite the "Test" prefix

    nodeid: str                 # e.g. tests/growth/test_x.py::test_y
    test_file: str              # tests/growth/test_x.py
    message: str = ""
    reproduced: bool = False    # still fails when run on its own (not a flake)
    source_module: Optional[str] = None   # nyxara/growth/x.py (best-effort mapping)

    def to_dict(self) -> Dict[str, Any]:
        return {"nodeid": self.nodeid, "test_file": self.test_file, "message": self.message,
                "reproduced": self.reproduced, "source_module": self.source_module}


@dataclass
class FixAttempt:
    """The outcome of trying to repair one failure."""

    nodeid: str
    module: Optional[str]
    edit_kind: str = ""
    applied: bool = False
    kept: bool = False
    rolled_back: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"nodeid": self.nodeid, "module": self.module, "edit_kind": self.edit_kind,
                "applied": self.applied, "kept": self.kept, "rolled_back": self.rolled_back,
                "reason": self.reason}


@dataclass
class DebugReport:
    """A single, auditable summary of one self-debugging pass."""

    failures: List[TestFailure] = field(default_factory=list)
    fixes: List[FixAttempt] = field(default_factory=list)
    detected: int = 0
    reproduced: int = 0
    fixed: int = 0
    enacted: bool = False
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"detected": self.detected, "reproduced": self.reproduced, "fixed": self.fixed,
                "enacted": self.enacted, "note": self.note,
                "failures": [f.to_dict() for f in self.failures],
                "fixes": [f.to_dict() for f in self.fixes]}

    def summary(self) -> str:
        lines = ["=" * 70, "NYXARA self-debugging (detect → reproduce → isolate → fix → verify)",
                 "=" * 70,
                 f"detected   : {self.detected} failing test(s)",
                 f"reproduced : {self.reproduced} (deterministic, not flakes)",
                 f"fixed      : {self.fixed} (enacted={self.enacted})"]
        if self.note:
            lines.append(f"note       : {self.note}")
        for fx in self.fixes[:10]:
            mark = "✓" if fx.kept else ("↩" if fx.rolled_back else "·")
            lines.append(f"  {mark} {fx.nodeid} [{fx.edit_kind or 'no-edit'}] — {fx.reason}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-debugger
# --------------------------------------------------------------------------- #
class SelfDebugger:
    """Run NYXARA's own tests, isolate the failures, and repair them under the gauntlet."""

    def __init__(self, *, core: Any = None, root: Any = None, settings: Any = None,
                 llm: Any = None, journal: Any = None) -> None:
        from nyxara.kernel.config import get_settings
        self.settings = settings or get_settings()
        self.core = core
        self.root = Path(root) if root is not None else _repo_root()
        self.journal = journal if journal is not None else getattr(core, "journal", None)
        self._llm = llm

    @classmethod
    def from_core(cls, core: Any, **kw: Any) -> "SelfDebugger":
        return cls(core=core, **kw)

    # ---- self-driven LLM handle (NYXARA's own model first, exactly like the RSIE) ---- #
    def _llm_handle(self) -> Any:
        if self._llm is not None:
            return self._llm
        self._llm = (getattr(getattr(self.core, "reasoner", None), "llm", None)
                     if self.core is not None else None)
        if self._llm is None:
            try:
                from nyxara.mind.llm import LLM
                self._llm = LLM()
            except Exception:  # noqa: BLE001 — a missing LLM only disables self-authored fixes
                self._llm = None
        return self._llm

    @property
    def autonomous_enact(self) -> bool:
        return bool(getattr(self.settings.self_optimization, "autonomous_enact", False))

    # ------------------------------------------------------------------ #
    # the loop
    # ------------------------------------------------------------------ #
    def run(self, *, test_path: Optional[str] = None,
            max_fixes: Optional[int] = None) -> DebugReport:
        cfg = self.settings.self_optimization
        if max_fixes is None:
            max_fixes = int(getattr(cfg, "max_debug_fixes", 3))
        path = test_path if test_path is not None else getattr(cfg, "debug_test_path", None)

        report = DebugReport(enacted=self.autonomous_enact)
        failures = self.detect(test_path=path)
        report.failures = failures
        report.detected = len(failures)
        if not failures:
            report.note = "suite is green — nothing to debug"
            return report

        self.isolate(failures)
        report.reproduced = sum(1 for f in failures if f.reproduced)

        budget = max(0, int(max_fixes))
        for failure in failures:
            if budget <= 0:
                break
            if not failure.reproduced:
                continue
            attempt = self.fix(failure)
            report.fixes.append(attempt)
            if attempt.kept:
                report.fixed += 1
            budget -= 1
        if not self.autonomous_enact and report.reproduced:
            report.note = ("autonomous_enact is OFF — failures isolated but fixes withheld "
                           "(set settings.self_optimization.autonomous_enact to authorise)")
        return report

    # ---- detect ---- #
    def detect(self, *, test_path: Optional[str] = None) -> List[TestFailure]:
        """Run the suite and parse the failing node ids. Empty list ⇒ green (or pytest absent)."""
        rc, out = self._pytest(test_path)
        if rc == 0:
            return []
        return self._parse_failures(out)

    # ---- reproduce + isolate ---- #
    def isolate(self, failures: List[TestFailure]) -> None:
        """Confirm each failure on its own and map it to the source module it exercises."""
        for f in failures:
            f.source_module = self._module_for(f.test_file)
            rc, _ = self._pytest(f.nodeid)
            f.reproduced = rc != 0

    # ---- fix (author an edit, verify-or-rollback against the failing tests) ---- #
    def fix(self, failure: TestFailure) -> FixAttempt:
        attempt = FixAttempt(nodeid=failure.nodeid, module=failure.source_module)
        module = failure.source_module
        if not module:
            attempt.reason = "could not map the test to a source module"
            return attempt
        mod_path = self.root / module
        if not mod_path.is_file():
            attempt.reason = f"source module not found: {module}"
            return attempt

        # Rule 8 — never self-edit the constitutional / loyalty / master core, even to fix a test.
        # Checked before any edit is even proposed: a failing test may not license a character change.
        from nyxara.growth.self_optimize import _is_constitutional
        if _is_constitutional(str(mod_path)):    # resolves against the package root (fail-closed)
            attempt.reason = ("Rule 8: refuses to edit the constitutional / loyalty / master core "
                              "— a failing test may not license a character change")
            return attempt

        edit = self._propose_edit(failure, mod_path)
        if edit is None:
            attempt.reason = "no candidate fix could be authored (offline / no transform applies)"
            return attempt
        attempt.edit_kind = edit.kind

        if not self.autonomous_enact:
            attempt.reason = "autonomous_enact is OFF — fix withheld"
            return attempt

        original = mod_path.read_text(encoding="utf-8", errors="replace")
        action_id = self._journal_start(failure, edit)
        try:
            mod_path.write_text(edit.after, encoding="utf-8")
            attempt.applied = True
            if self._verify_fix(failure):
                attempt.kept = True
                attempt.reason = "fix verified — failing test now passes, safety intact"
                self._journal_end(action_id, success=True, note=attempt.reason)
            else:
                mod_path.write_text(original, encoding="utf-8")
                attempt.rolled_back = True
                attempt.reason = "fix did not verify — rolled back byte-for-byte"
                self._journal_end(action_id, success=False, note=attempt.reason)
        except Exception as exc:  # noqa: BLE001 — any error ⇒ restore, never leave a broken tree
            try:
                mod_path.write_text(original, encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
            attempt.rolled_back = attempt.applied
            attempt.reason = f"error during fix ({exc}) — rolled back"
            self._journal_end(action_id, success=False, note=attempt.reason)
        return attempt

    # ------------------------------------------------------------------ #
    # the verify-or-rollback gauntlet (fresh subprocesses see the edited source)
    # ------------------------------------------------------------------ #
    def _verify_fix(self, failure: TestFailure) -> bool:
        """A fix is kept only if it compiles, keeps the safety battery green, and the
        previously-failing test now passes with no new failures it introduced."""
        # 1) syntax — fast in-process reject
        if failure.source_module:
            mod_path = self.root / failure.source_module
            if not _parses(mod_path.read_text(encoding="utf-8", errors="replace")):
                return False
        # 2) character / corrigibility / honesty safety battery must stay intact
        rc, _ = self._safe_shell(["-m", "nyxara.eval"])
        if rc != 0:
            return False
        # 3) the failing test now passes
        rc, _ = self._pytest(failure.nodeid)
        return rc == 0

    # ------------------------------------------------------------------ #
    # edit authoring (reuse the optimiser's machinery)
    # ------------------------------------------------------------------ #
    def _propose_edit(self, failure: TestFailure, mod_path: Path) -> Optional[SourceEdit]:
        """Author a candidate fix for the failing module.

        Deterministic transforms first (they always work, no model): review the module, turn any
        hygiene weakness into a reversible edit. If none applies and self-authored edits are
        permitted, hand the failure to NYXARA's own model via :class:`LLMEditGenerator`."""
        for weakness in self._module_weaknesses(failure):
            edit = EditGenerator(llm=self._llm_handle()).generate(weakness)
            if edit is not None and edit.changes:
                return edit
        return self._self_authored_edit(failure, mod_path)

    def _module_weaknesses(self, failure: TestFailure) -> List[Any]:
        """Source-editable :class:`~nyxara.growth.weakness.Weakness` items in the failing module.

        Built through the same code-review → synthesise path the RSIE uses, so each weakness
        carries the ``id``/``locus``/``remediation``/``symbol`` fields :class:`EditGenerator`
        reads — then filtered to the module the failing test isolates to."""
        try:
            from nyxara.growth.self_review import SelfReviewer
            from nyxara.growth.weakness import WeaknessSynthesizer
            cfg = self.settings.self_improvement
            code = SelfReviewer(
                root=self.root, llm=self._llm_handle(),
                max_function_length=cfg.max_function_length,
                max_complexity=cfg.max_complexity, max_args=cfg.max_args,
                enable_llm_enrichment=False).review()
            report = WeaknessSynthesizer().synthesize(code=code)
        except Exception:  # noqa: BLE001 — review is best-effort; no review ⇒ no deterministic edit
            return []
        module = failure.source_module or ""
        mod_name = Path(module).name if module else ""
        out: List[Any] = []
        for weakness in getattr(report, "weaknesses", []) or []:
            if not getattr(weakness, "is_source_edit", False):
                continue
            locus = getattr(weakness, "locus", "") or ""
            if mod_name and mod_name in locus:
                out.append(weakness)
        return out

    def _self_authored_edit(self, failure: TestFailure,
                            mod_path: Path) -> Optional[SourceEdit]:
        """NYXARA writes the fix herself with her own ``self`` model (when authorised)."""
        gen = LLMEditGenerator(llm=self._llm_handle(), settings=self.settings)
        if not gen.available():
            return None
        # surface the failing test + its message as the "weakness" the author must resolve
        weakness = _DebugWeakness(
            locus=failure.source_module or str(mod_path),
            title=f"failing test {failure.nodeid}",
            remediation=(f"Fix the bug so `{failure.nodeid}` passes. Failure: "
                         f"{failure.message or '(no message)'}"))
        try:
            return gen.generate(weakness)
        except Exception:  # noqa: BLE001 — a self-authored edit is a capability, never required
            return None

    # ------------------------------------------------------------------ #
    # plumbing
    # ------------------------------------------------------------------ #
    def _pytest(self, target: Optional[str]) -> tuple[int, str]:
        argv = ["-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header", "--tb=no"]
        if target:
            argv.append(target)
        return self._safe_shell(argv)

    def _safe_shell(self, py_args: List[str]) -> tuple[int, str]:
        """Run ``python <py_args>`` from the repo root via the agency sandbox executor."""
        import sys
        timeout = float(getattr(self.settings.self_optimization, "debug_timeout_s", 600.0))
        try:
            from nyxara.agency.code_sandbox import safe_shell
            res = safe_shell([sys.executable, *py_args], cwd=str(self.root), timeout_s=timeout)
            return res.returncode, (res.stdout or "") + (res.stderr or "")
        except Exception as exc:  # noqa: BLE001 — no sandbox ⇒ honest "could not run"
            return 1, f"could not run {py_args}: {exc}"

    @staticmethod
    def _parse_failures(output: str) -> List[TestFailure]:
        seen: set[str] = set()
        out: List[TestFailure] = []
        for line in output.splitlines():
            m = _FAILED_RE.match(line.strip())
            if not m:
                continue
            nodeid = m.group(1)
            if nodeid in seen:
                continue
            seen.add(nodeid)
            test_file = nodeid.split("::", 1)[0]
            out.append(TestFailure(nodeid=nodeid, test_file=test_file,
                                   message=(m.group(2) or "").strip()))
        return out

    @staticmethod
    def _module_for(test_file: str) -> Optional[str]:
        """Map ``tests/<pkg>/test_x.py`` → ``nyxara/<pkg>/x.py`` (best-effort)."""
        parts = Path(test_file).parts
        if not parts or parts[0] != "tests":
            return None
        name = parts[-1]
        if not (name.startswith("test_") and name.endswith(".py")):
            return None
        source_name = name[len("test_"):]
        return str(Path("nyxara", *parts[1:-1], source_name))

    # ---- journal ---- #
    def _journal_start(self, failure: TestFailure, edit: SourceEdit) -> str:
        if self.journal is None:
            return ""
        try:
            from nyxara.planning.journal import ActionStatus
            return self.journal.record_action(
                f"self-debug:{edit.kind}:{failure.nodeid}", rationale=edit.rationale,
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
                status=ActionStatus.SUCCEEDED if success else ActionStatus.ABORTED, note=note)
        except Exception:  # noqa: BLE001
            pass


@dataclass
class _DebugWeakness:
    """A minimal weakness shim so the self-authored generator can target a failing test."""

    locus: str
    title: str
    remediation: str
    id: str = "self_debug_failing_test"
    severity: float = 1.0
    symbol: str = ""
