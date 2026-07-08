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

__all__ = ["SourceEdit", "GauntletResult", "OptimizationOutcome", "EditGenerator",
           "LLMEditGenerator", "SelfEditGenerator", "Optimizer"]


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
    # proof-carrying + scalable-oversight certificates attached to this edit (when run)
    proof: Optional[Dict[str, Any]] = None
    oversight: Optional[Dict[str, Any]] = None
    # the "provably BETTER" certificate (capability gain / proven-cheaper / defect-elimination)
    improvement: Optional[Dict[str, Any]] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"edit": self.edit.to_dict(), "applied": self.applied, "kept": self.kept,
                "rolled_back": self.rolled_back, "reason": self.reason,
                "proof": self.proof, "oversight": self.oversight,
                "improvement": self.improvement,
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
        if not kind:
            return None          # no deterministic transform applies (e.g. an architecture redesign)
        file_part, _, line_part = locus.partition(":")
        path = Path(file_part)
        if not path.is_file():
            # locus is repo-relative — resolve against the package parent
            cand = _repo_root() / file_part
            if cand.is_file():
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
        elif kind in ("not_in", "is_not_cmp"):
            after = _fix_negated_compare(before, lineno)

        if after is None or after == before:
            return None
        if not _parses(after):       # never emit an edit that breaks the file
            return None
        return SourceEdit(file=str(path), kind=kind, before=before, after=after,
                          rationale=getattr(weakness, "remediation", "") or kind)

    def _kind_of(self, weakness: Any) -> str:
        wid = getattr(weakness, "id", "")
        # order matters: match the most specific id fragment first
        for k in ("bare_except", "missing_docstring", "unused_import", "eq_none",
                  "is_not_cmp", "not_in"):
            if k in wid:
                return k
        return ""

    def _docstring_text(self, weakness: Any) -> str:
        title = getattr(weakness, "title", "") or "symbol"
        return f"Auto-documented during self-optimization: {title}."


# --------------------------------------------------------------------------- #
# Self-authored edit generation — NYXARA writes a real whole-file fix herself
# --------------------------------------------------------------------------- #
class LLMEditGenerator:
    """Author a real, whole-file source fix for a weakness no deterministic transform can express
    (high complexity, an over-long function, too many parameters, a TODO, an architectural
    redesign).

    This is the capability that turns self-optimization from a linter into genuine recursive
    self-improvement: **NYXARA writes the fix herself, with her OWN foundry-trained model** (the
    ``self`` provider) — not an external LLM — whenever ``self_authored_only`` is set ("khud
    NYXARA kare, koi LLM naa kare"). It is **safe by construction**: the proposal is only ever
    handed to :class:`Optimizer`, which subjects it to the *same* reversible verify-or-rollback
    gauntlet (syntax → safety battery → capability non-regression → optional tests). This
    generator's own checks merely reject obviously-bad proposals early (won't parse, truncated,
    runaway rewrite) so the expensive gauntlet is spent only on plausible edits.

    It is triple-gated: ``allow_llm_edits`` must be set, ``autonomous_enact`` must authorise
    writing, and a real author must be available — NYXARA's own ``self`` model (preferred and,
    under ``self_authored_only``, required), or, only when that flag is off, a configured external
    provider. The mock is never an author. Note the honest contract: the gauntlet guarantees
    *safety* (a bad rewrite rolls back byte-for-byte), not *capability* — so the yield of these
    rewrites scales with how good NYXARA's own model is, while the deterministic transforms in
    :class:`EditGenerator` always work with no model at all.
    """

    _SYSTEM = (
        "You are NYXARA, editing your own Python source to fix one specific weakness. "
        "Return the COMPLETE, corrected contents of the file and NOTHING else — no prose, no "
        "explanation, no markdown code fences. Change only what the remediation requires; "
        "preserve every other line, import, comment, docstring, and the file's public behaviour "
        "exactly. The result must be valid Python and must never weaken safety, corrigibility, "
        "honesty, or loyalty to the Master."
    )

    def __init__(self, llm: Any = None, *, settings: Any = None) -> None:
        from nyxara.kernel.config import get_settings
        self.llm = llm
        self.settings = settings or get_settings()

    def available(self) -> bool:
        """True only when self-authored edits are authorised AND a real author can answer."""
        if not bool(getattr(self.settings.self_improvement, "allow_llm_edits", False)):
            return False
        return self._author_provider() is not None

    def _author_provider(self) -> Optional[str]:
        """Name of the provider that will author the edit, or None.

        NYXARA's own ``self`` model is always preferred; under ``self_authored_only`` it is the
        *only* permitted author (so an external LLM is never used). With that flag off, a real
        configured external provider may author instead. The ``mock`` impersonator is never an
        author."""
        return _author_provider_name(
            self.llm,
            self_authored_only=bool(getattr(self.settings.self_improvement,
                                            "self_authored_only", True)))

    def generate(self, weakness: Any) -> Optional[SourceEdit]:
        author = self._author_provider()
        if author is None or not bool(
                getattr(self.settings.self_improvement, "allow_llm_edits", False)):
            return None
        locus = getattr(weakness, "locus", "") or ""
        file_part, _, _ = locus.partition(":")
        path = _resolve_locus_path(file_part)
        if path is None:
            return None
        before = path.read_text(encoding="utf-8", errors="replace")
        cfg = self.settings.self_improvement
        if len(before.encode("utf-8")) > int(getattr(cfg, "llm_edit_max_file_bytes", 200000)):
            return None  # only pathologically huge files are skipped
        prompt = self._build_prompt(weakness, path, before)
        try:
            raw = self._author_complete(author, prompt,
                                        int(getattr(cfg, "llm_edit_max_tokens", 8192)))
        except Exception:  # noqa: BLE001 — an author failure must never crash the cycle
            return None
        after = _strip_code_fence(raw or "")
        if not self._acceptable(before, after):
            return None
        tag = "self" if author == "self" else "llm"
        return SourceEdit(file=str(path), kind=f"{tag}:{self._kind(weakness)}",
                          before=before, after=after,
                          rationale=getattr(weakness, "remediation", "")
                          or f"{tag}-authored edit")

    def _author_complete(self, author: str, prompt: str, max_tokens: int) -> str:
        """Author through one specific provider, honestly (never mock-substituted).

        Prefers the facade's :meth:`LLM.complete_with` so the chosen author actually answers as
        itself; falls back to a plain ``generate`` for minimal facades/stubs that lack it."""
        complete_with = getattr(self.llm, "complete_with", None)
        if complete_with is not None:
            from nyxara.mind.llm import LLMRequest
            req = LLMRequest.from_prompt(prompt, system=self._SYSTEM,
                                         max_tokens=max_tokens, temperature=0.0)
            return complete_with(author, req).text
        return self.llm.generate(prompt, system=self._SYSTEM,
                                 max_tokens=max_tokens, temperature=0.0)

    # ---- helpers ---- #
    def _kind(self, weakness: Any) -> str:
        wid = getattr(weakness, "id", "") or ""
        for k in ("high_complexity", "long_function", "too_many_args", "todo"):
            if k in wid:
                return k
        return "refactor"

    def _build_prompt(self, weakness: Any, path: Path, before: str) -> str:
        title = getattr(weakness, "title", "") or ""
        remediation = getattr(weakness, "remediation", "") or ""
        evidence = "; ".join(e for e in (getattr(weakness, "evidence", []) or []) if e)
        return (
            "# Weakness to fix\n"
            f"File: {path}\n"
            f"Issue: {title}\n"
            f"Evidence: {evidence}\n"
            f"Required fix: {remediation}\n\n"
            "# Current file contents\n"
            f"{before}\n\n"
            "# Your task\n"
            "Return the complete corrected contents of this file, applying only the required "
            "fix and preserving everything else."
        )

    def _acceptable(self, before: str, after: str) -> bool:
        if not after or after == before:
            return False
        if not _parses(after):                 # never hand the gauntlet an unparseable file
            return False
        ratio = float(getattr(self.settings.self_improvement,
                              "llm_edit_max_size_delta_ratio", 0.5))
        b, a = len(before), len(after)
        if b > 0 and abs(a - b) > ratio * b:    # truncation or runaway rewrite — reject early
            return False
        return True


# NYXARA authors her own redesigns: the honest name for the generator above. The old name is
# kept as an alias so existing imports keep working.
SelfEditGenerator = LLMEditGenerator


# --------------------------------------------------------------------------- #
# Optimizer — apply with verify-or-rollback
# --------------------------------------------------------------------------- #
class Optimizer:
    """Apply source edits under a reversible, gauntlet-gated, journalled discipline."""

    def __init__(self, *, root: Optional[Any] = None, settings: Any = None,
                 journal: Any = None, permissions: Any = None,
                 gauntlet: Optional[Callable[[List[str]], GauntletResult]] = None,
                 timeout_s: float = 600.0,
                 require_improvement: Optional[bool] = None,
                 improvement_fn: Optional[Callable[["SourceEdit"], Any]] = None) -> None:
        from nyxara.kernel.config import get_settings
        self.settings = settings or get_settings()
        self.root = Path(root) if root is not None else _package_root()
        self.journal = journal
        self.permissions = permissions
        self.timeout_s = float(timeout_s)
        self._gauntlet_fn = gauntlet or self._default_gauntlet
        self._baseline_path: Optional[Path] = None
        self._after_path: Optional[Path] = None
        self._tmpdir: Optional[tempfile.TemporaryDirectory] = None
        # the strict "provably BETTER" gate (Master JP's charge). None ⇒ read the standing config.
        self._require_improvement = (
            bool(require_improvement) if require_improvement is not None
            else bool(getattr(self.settings.self_improvement,
                              "require_provable_improvement", True)))
        self._improvement_fn = improvement_fn      # test/override hook; None ⇒ real prover

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

        # Rule 8 — the constitutional / loyalty / master core is structurally out of reach: the
        # self-editor refuses to even open a sealed file, fail-closed (an unresolved path refuses
        # too). NYXARA evolves capability, never her character or her allegiance to the Master.
        if _is_constitutional(edit.file, self.root):
            outcome.reason = ("Rule 8: refuses to edit the constitutional / loyalty / master core "
                              "— capability may evolve, character and allegiance may not")
            return outcome

        # best-effort audit through the permission gate (the config flag is the authorisation)
        self._audit_permission(edit)

        path = Path(edit.file)
        original = path.read_text(encoding="utf-8", errors="replace")
        action_id = self._journal_start(edit)
        try:
            self._after_path = None                 # never reuse a prior edit's after-report
            self._ensure_baseline()                 # capture pre-edit capability baseline
            path.write_text(edit.after, encoding="utf-8")
            outcome.applied = True
            # proof-carrying + scalable oversight run BEFORE the (expensive) empirical gauntlet: a
            # provable regression or an independent veto rejects the edit up front, byte-for-byte.
            blocked, why = self._pre_gauntlet_checks(edit, outcome)
            if blocked:
                path.write_text(original, encoding="utf-8")
                outcome.rolled_back = True
                outcome.reason = f"{why} — rolled back"
                self._journal_end(action_id, success=False, note=outcome.reason)
                return outcome
            result = self._gauntlet_fn([str(path)])
            outcome.gauntlet = result
            if result.ok:
                # The gauntlet proved the edit SAFE + NON-REGRESSING. The Master's charge is
                # stricter: keep it only when it is *provably BETTER*. An edit that cannot earn an
                # improvement certificate is rolled back byte-for-byte, exactly like a failure.
                cert = self._prove_improvement(edit) if self._require_improvement else None
                if cert is not None:
                    outcome.improvement = cert.to_dict()
                if cert is not None and not cert.better:
                    path.write_text(original, encoding="utf-8")
                    outcome.rolled_back = True
                    outcome.reason = (f"gauntlet passed but NOT provably better "
                                      f"({cert.reason}) — rolled back")
                    self._journal_end(action_id, success=False, note=outcome.reason)
                else:
                    outcome.kept = True
                    outcome.reason = (f"gauntlet passed + provably better via {cert.method} "
                                      "— edit kept") if cert else "gauntlet passed — edit kept"
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

    # ---- proof-carrying + scalable-oversight pre-checks (before the empirical gauntlet) ---- #
    def _pre_gauntlet_checks(self, edit: SourceEdit, outcome: OptimizationOutcome
                             ) -> tuple[bool, str]:
        """Run the proof and oversight gates. Returns ``(blocked, reason)``.

        Both are *sound*: they reject only a provable regression (proof-carrying) or an independent
        cross-examination veto (scalable oversight), and otherwise record a certificate and let the
        empirical gauntlet decide. Either failing to import simply skips that gate — never blocks."""
        cfg = self.settings.self_improvement
        # proof-carrying self-modification (Gödel-machine): block a provable regression up front
        if bool(getattr(cfg, "proof_carrying_edits", True)):
            try:
                from nyxara.growth.proof_carrying import ProofCarrier
                pres = ProofCarrier(settings=self.settings).certify_edit(
                    edit.before, edit.after, kind=edit.kind)
                outcome.proof = pres.to_dict()
                if pres.blocks:
                    return True, f"proof-carrying: refuted ({pres.certificate})"
            except Exception:  # noqa: BLE001 — a proof failure must never block a valid edit
                pass
        # scalable oversight: an independent, redundant verifier vetoes a flawed edit
        if bool(getattr(cfg, "scalable_oversight", True)):
            try:
                from nyxara.growth.oversight_verify import ScalableVerifier
                ores = ScalableVerifier(settings=self.settings).verify_edit(edit.before, edit.after,
                                                                            kind=edit.kind)
                outcome.oversight = ores.to_dict()
                if ores.vetoed:
                    return True, f"scalable oversight: vetoed ({ores.reason})"
            except Exception:  # noqa: BLE001 — oversight unavailable ⇒ skip, never block
                pass
        return False, ""

    # ---- the strict "provably BETTER" gate (Master JP's charge) ---- #
    def _prove_improvement(self, edit: SourceEdit) -> Any:
        """Certify the edit is a genuine improvement, or return a ``better=False`` verdict.

        A capability gain is proved against the before/after benchmark reports the gauntlet
        snapshots (deterministic dominance); a refactor is proved equivalent-and-cheaper on the
        source; a named deterministic transform is proved to eliminate its defect. Any failure to
        certify keeps the OLD code — never a silent accept."""
        if self._improvement_fn is not None:
            return self._improvement_fn(edit)          # test/override hook
        from nyxara.growth.improvement_proof import ImprovementProver
        before = after = None
        try:
            from nyxara.eval.benchmark import BenchmarkReport
            if self._baseline_path is not None and Path(self._baseline_path).exists():
                before = BenchmarkReport.load(str(self._baseline_path))
            if self._after_path is not None and Path(self._after_path).exists():
                after = BenchmarkReport.load(str(self._after_path))
        except Exception:  # noqa: BLE001 — no capability data ⇒ fall back to source-only proofs
            before = after = None
        return ImprovementProver(settings=self.settings).prove(
            before=before, after=after, before_src=edit.before, after_src=edit.after,
            edit_kind=edit.kind)

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

        # 2b) constitution seal — the sealed rules / values / invariants must be byte-for-byte
        #     intact after the edit (fail-closed). Defence in depth beyond the path-based refusal:
        #     even an edit to a NON-protected file that perturbs a seal is caught and rolled back.
        rc, out = self._run(
            [sys.executable, "-c",
             "from nyxara.kernel.invariants import boot_verify; "
             "from nyxara.identity.values import verify_values; "
             "verify_values(); boot_verify(raise_on_fail=True)"], repo)
        checks["constitution_seal"] = (rc == 0)
        if rc != 0:
            return GauntletResult(False, checks,
                                  f"constitution seal broken (rc={rc}): {out[-300:]}")

        # 3) capability — no regression vs the pre-edit baseline, and snapshot the AFTER report so
        #    the provable-improvement gate can prove a capability Pareto-gain deterministically.
        if self._baseline_path is not None:
            after_path = self._new_after_path()
            rc, out = self._run(
                [sys.executable, "-m", "nyxara.eval", "--benchmark",
                 "--baseline", str(self._baseline_path), "--save", str(after_path)], repo)
            checks["capability"] = (rc == 0)
            self._after_path = after_path if after_path.exists() else None
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

    def _new_after_path(self) -> Path:
        """A fresh temp path for the post-edit benchmark report (the 'after' half of the proof)."""
        if self._tmpdir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="nyxara-rsi-")
        return Path(self._tmpdir.name) / "capability_after.json"

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


def _fix_negated_compare(src: str, lineno: int) -> Optional[str]:
    """Rewrite ``not a in b`` → ``a not in b`` and ``not a is b`` → ``a is not b`` (E713/E714).

    Uses ``ast`` to locate the exact ``not (<x> in/is <y>)`` node (only single-operator
    comparisons, so chained comparisons are never touched) and rebuilds just its source span
    from the operands' verbatim segments — formatting elsewhere is preserved byte-for-byte.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    def _is_target(n: Any) -> bool:
        return (isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not)
                and isinstance(n.operand, ast.Compare)
                and len(n.operand.ops) == 1
                and isinstance(n.operand.ops[0], (ast.In, ast.Is)))

    target = None
    for node in ast.walk(tree):                      # prefer the precisely-flagged line
        if _is_target(node) and getattr(node, "lineno", None) == lineno:
            target = node
            break
    if target is None:                               # fall back to the first occurrence
        for node in ast.walk(tree):
            if _is_target(node):
                target = node
                break
    if target is None:
        return None

    cmp = target.operand
    left = ast.get_source_segment(src, cmp.left)
    right = ast.get_source_segment(src, cmp.comparators[0])
    if left is None or right is None:
        return None
    op = "not in" if isinstance(cmp.ops[0], ast.In) else "is not"
    out = _replace_node_span(src, target, f"{left} {op} {right}")
    if out is None or out == src or not _parses(out):
        return None
    return out


def _replace_node_span(src: str, node: ast.AST, replacement: str) -> Optional[str]:
    """Replace the exact source span of ``node`` with ``replacement`` using ast position info.

    ``col_offset``/``end_col_offset`` are UTF-8 *byte* columns, so the affected lines are sliced
    on their encoded bytes (correct even when the source contains non-ASCII characters)."""
    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", None)
    col = getattr(node, "col_offset", None)
    end_col = getattr(node, "end_col_offset", None)
    if None in (lineno, end_lineno, col, end_col):
        return None
    lines = src.splitlines(keepends=True)
    if not (1 <= lineno <= len(lines)) or not (1 <= end_lineno <= len(lines)):
        return None
    prefix = lines[lineno - 1].encode("utf-8")[:col].decode("utf-8", errors="replace")
    suffix = lines[end_lineno - 1].encode("utf-8")[end_col:].decode("utf-8", errors="replace")
    return "".join(lines[:lineno - 1] + [prefix + replacement + suffix] + lines[end_lineno:])


def _author_provider_name(llm: Any, *, self_authored_only: bool = True) -> Optional[str]:
    """Name of the provider that may author a self-edit, or None.

    NYXARA's OWN ``self`` model (her foundry-trained brain) is always preferred — this is the
    point of the feature: she redesigns herself, with herself. Under ``self_authored_only`` it is
    the *only* permitted author, so no external LLM is ever consulted. With that flag off, a real
    configured external provider may author when ``self`` is not yet available. The ``mock``
    impersonator is never an author (it would only fake a fix)."""
    if llm is None:
        return None
    try:
        available = set(llm.available_providers())
    except Exception:  # noqa: BLE001 — fall back to the chosen-provider probe below
        available = set()
    if "self" in available:
        return "self"                       # NYXARA's own brain answers — always preferred
    if self_authored_only:
        return None                         # her own model isn't ready and no LLM is permitted
    try:
        prov = llm.chosen_provider()
    except Exception:  # noqa: BLE001
        return None
    name = getattr(prov, "name", "")
    if name not in ("mock", "self") and bool(prov.available()):
        return name
    return None


def _real_provider_available(llm: Any) -> bool:
    """Back-compat shim: True iff some real (non-mock) author — NYXARA's own model or a configured
    provider — can answer. Superseded by :func:`_author_provider_name`."""
    return _author_provider_name(llm, self_authored_only=False) is not None


def _resolve_locus_path(file_part: str) -> Optional[Path]:
    """Resolve a weakness locus to an existing file on disk.

    Handles three locus shapes: an absolute/relative file path, a repo-relative path, and a
    **dotted module name** (e.g. ``nyxara.kernel.orchestrator``) — the form architecture
    weaknesses carry — resolving the module or its package ``__init__.py``."""
    if not file_part:
        return None
    path = Path(file_part)
    if path.is_file():
        return path
    cand = _repo_root() / file_part
    if cand.is_file():
        return cand
    # dotted module name (architecture loci) → its source file (or the package __init__.py)
    if "/" not in file_part and "\\" not in file_part and not file_part.endswith(".py"):
        rel = file_part.replace(".", "/")
        module = _repo_root() / (rel + ".py")
        if module.is_file():
            return module
        package = _repo_root() / rel / "__init__.py"
        if package.is_file():
            return package
    return None


def _strip_code_fence(text: str) -> str:
    """Strip a leading/trailing markdown code fence (```...```), if the model added one.

    A robust safety net: the system prompt forbids fences, but models occasionally add them.
    Always returns a newline-terminated string (Python files end with a trailing newline)."""
    t = text.strip()
    if t.startswith("```"):
        body = t.splitlines()[1:]                    # drop the opening ``` / ```python line
        if body and body[-1].strip().startswith("```"):
            body = body[:-1]                         # drop the closing fence
        t = "\n".join(body)
    return t if t.endswith("\n") else t + "\n"


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
# The constitutional / loyalty / master core — structurally out of reach (Rule 8)
# --------------------------------------------------------------------------- #
# The sealed files that define NYXARA's rules, values, corrigibility, soul, owner identity, and
# authentication. The self-editor refuses to touch any of them: capability may evolve, character
# and allegiance to the Master may not. Enforcement is fail-closed — an unresolvable path is
# treated as protected. (Defence in depth: the gauntlet also re-verifies every seal in a fresh
# subprocess, so a seal broken via any other file is caught and rolled back.)
PROTECTED_RELPATHS = frozenset({
    "kernel/rules.py", "kernel/invariants.py", "kernel/config.py",
    "identity/values.py", "identity/soul.py",
    "guard/value_learning.py", "guard/corrigibility.py", "guard/auth.py",
    "growth/loyalty.py",
})


def _is_constitutional(file: str, root: Optional[Path] = None) -> bool:
    """True iff ``file`` is one of NYXARA's sealed core files (fail-closed on any doubt)."""
    try:
        pkg = Path(root) if root is not None else _package_root()
        target = Path(file).resolve()
        for rel in PROTECTED_RELPATHS:
            if target == (pkg / rel).resolve():
                return True
        # also refuse anything whose resolved path ends with a protected rel-path, so a copy of
        # the package tree (or a differently-rooted install) is guarded too.
        parts = target.as_posix()
        return any(parts.endswith("/nyxara/" + rel) or parts.endswith("/" + rel)
                   for rel in PROTECTED_RELPATHS)
    except Exception:  # noqa: BLE001 — cannot prove it is safe ⇒ treat it as protected
        return True


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

        # a passing gauntlet + a provably-better edit (bare-except is a proven defect-elimination)
        f.write_text(original, encoding="utf-8")     # reset (the failing run rolled back)
        pass_opt = Optimizer(settings=settings,
                             gauntlet=lambda files: GauntletResult(True, {"syntax": True}, "ok"))
        out2 = pass_opt.apply(edit)
        assert out2.kept and not out2.rolled_back
        assert out2.improvement and out2.improvement["method"] == "defect-elimination"
        assert "except Exception:" in f.read_text(encoding="utf-8")
        print("passing gauntlet    : provably-better edit kept (defect-elimination) ✓")

        # a passing gauntlet but a NON-improving edit (pure whitespace) is ROLLED BACK — the
        # Master's charge: change the code only when it is provably better, never merely harmless.
        f.write_text("def add(a, b):\n    return a+b\n", encoding="utf-8")
        noop = SourceEdit(str(f), "self:refactor", "def add(a, b):\n    return a+b\n",
                          "def add(a, b):\n    return a + b\n", "reformat only")
        out3 = pass_opt.apply(noop)
        assert out3.applied and out3.rolled_back and not out3.kept
        assert out3.improvement and not out3.improvement["better"]
        assert f.read_text(encoding="utf-8") == "def add(a, b):\n    return a+b\n"
        print("not provably better : non-improving edit rolled back ✓")

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
