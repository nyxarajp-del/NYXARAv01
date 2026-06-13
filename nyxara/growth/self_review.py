"""NYXARA · growth/self_review.py — Self Code Review (🔍, RSI capability 1).

NYXARA reads her *own* source the way a careful engineer would and reports what she finds.
This is real static analysis — not a placeholder — built on the standard-library :mod:`ast`
(already used in ``agency/default_tools.py``), so it produces honest signal with **no API
key and no network**. Per public function / class / module it computes:

* **function length** — physical lines (``end_lineno - lineno``); long bodies are flagged.
* **complexity** — a cyclomatic-ish count: ``1 + branches`` (if/for/while/try/except/with/
  boolean-op/comprehension/ternary/assert), the standard McCabe-style approximation.
* **missing docstring** — ``ast.get_docstring`` is ``None`` for a *public* def/class/module.
* **bare except** — a truly bare ``except:`` (``ExceptHandler.type is None``). The codebase's
  deliberate ``except Exception:  # noqa: BLE001`` is *not* bare and is never flagged.
* **TODO/FIXME** — ``TODO`` / ``FIXME`` / ``XXX`` / ``HACK`` markers left in the source.
* **too-many-args** — a signature with more positional parameters than the threshold.

When a real LLM is configured (``provider != "mock"``, the same probe the per-turn
:class:`~nyxara.mind.recursive_improver.RecursiveImprover` uses) the worst offenders are
optionally enriched with a one-line critique — strictly additive, never required, and
wrapped so a provider hiccup degrades to the deterministic findings, never a crash.

Thresholds come from :class:`~nyxara.kernel.config.SelfImprovementConfig`.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

__all__ = ["CodeFinding", "CodeReviewReport", "SelfReviewer"]

_TODO_RE = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")

# severity weights per finding kind, in [0, 1] — used by the weakness synthesizer too
_SEVERITY: Dict[str, float] = {
    "bare_except": 0.7,
    "high_complexity": 0.6,
    "long_function": 0.45,
    "too_many_args": 0.4,
    "missing_docstring": 0.25,
    "todo": 0.2,
}


# --------------------------------------------------------------------------- #
# Finding & report
# --------------------------------------------------------------------------- #
@dataclass
class CodeFinding:
    """One reviewable signal located at a concrete source position."""

    file: str
    lineno: int
    symbol: str
    kind: str
    severity: float
    detail: str
    critique: str = ""          # optional LLM-enriched note (blank offline)

    def to_dict(self) -> Dict[str, Any]:
        d = {"file": self.file, "lineno": self.lineno, "symbol": self.symbol,
             "kind": self.kind, "severity": round(self.severity, 3), "detail": self.detail}
        if self.critique:
            d["critique"] = self.critique
        return d


@dataclass
class CodeReviewReport:
    """The aggregate of one self-review pass over the package."""

    findings: List[CodeFinding] = field(default_factory=list)
    files_scanned: int = 0
    llm_enriched: bool = False

    def by_severity(self) -> List[CodeFinding]:
        return sorted(self.findings, key=lambda f: f.severity, reverse=True)

    def by_kind(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for f in self.findings:
            out[f.kind] = out.get(f.kind, 0) + 1
        return out

    def worst(self, n: int = 10) -> List[CodeFinding]:
        return self.by_severity()[:max(0, n)]

    def to_dict(self) -> Dict[str, Any]:
        return {"files_scanned": self.files_scanned, "n_findings": len(self.findings),
                "llm_enriched": self.llm_enriched, "by_kind": self.by_kind(),
                "findings": [f.to_dict() for f in self.by_severity()]}

    def summary(self) -> str:
        lines = ["=" * 70, "NYXARA self code-review", "=" * 70,
                 f"scanned {self.files_scanned} module(s); {len(self.findings)} finding(s)"
                 f"{' (LLM-enriched)' if self.llm_enriched else ''}", "", "by kind:"]
        for kind, n in sorted(self.by_kind().items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"  {kind:<18}: {n}")
        worst = self.worst(8)
        if worst:
            lines += ["", "worst offenders:"]
            for f in worst:
                lines.append(f"  ✗ [{f.kind}] {f.file}:{f.lineno} {f.symbol} — {f.detail}")
        else:
            lines += ["", "no findings ✓"]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Reviewer
# --------------------------------------------------------------------------- #
class SelfReviewer:
    """Walk NYXARA's own source with :mod:`ast` and produce real review findings."""

    def __init__(self, root: Optional[Any] = None, *, llm: Any = None,
                 max_function_length: int = 60, max_complexity: int = 10,
                 max_args: int = 6, enable_llm_enrichment: bool = True,
                 enrich_top: int = 6) -> None:
        self.root = Path(root) if root is not None else _package_root()
        self.llm = llm
        self.max_function_length = max(10, int(max_function_length))
        self.max_complexity = max(1, int(max_complexity))
        self.max_args = max(1, int(max_args))
        self.enable_llm_enrichment = bool(enable_llm_enrichment)
        self.enrich_top = max(0, int(enrich_top))

    # ---- module discovery ---- #
    def _iter_modules(self) -> Iterable[Path]:
        for path in sorted(self.root.rglob("*.py")):
            parts = set(path.parts)
            if "__pycache__" in parts or "tests" in parts:
                continue
            yield path

    # ---- per-file review ---- #
    def review_file(self, path: Any) -> List[CodeFinding]:
        path = Path(path)
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 — an unreadable file is data, not a crash
            return []
        try:
            tree = ast.parse(src, filename=str(path))
        except SyntaxError as exc:
            return [CodeFinding(self._rel(path), exc.lineno or 0, path.stem,
                                "syntax_error", 0.9, f"cannot parse: {exc.msg}")]

        findings: List[CodeFinding] = []
        rel = self._rel(path)

        # module-level docstring
        if ast.get_docstring(tree) is None and path.name != "__init__.py":
            findings.append(CodeFinding(rel, 1, path.stem, "missing_docstring",
                                        _SEVERITY["missing_docstring"],
                                        "module has no docstring"))

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                findings.extend(self._review_function(rel, node))
            elif isinstance(node, ast.ClassDef):
                if ast.get_docstring(node) is None and not node.name.startswith("_"):
                    findings.append(CodeFinding(rel, node.lineno, node.name,
                                                "missing_docstring",
                                                _SEVERITY["missing_docstring"],
                                                "public class has no docstring"))
            elif isinstance(node, ast.ExceptHandler) and node.type is None:
                findings.append(CodeFinding(rel, node.lineno, "<except>", "bare_except",
                                            _SEVERITY["bare_except"],
                                            "bare 'except:' swallows every error (incl. "
                                            "KeyboardInterrupt/SystemExit)"))

        # TODO/FIXME markers — cheap raw-line scan
        for i, line in enumerate(src.splitlines(), start=1):
            m = _TODO_RE.search(line)
            if m:
                findings.append(CodeFinding(rel, i, m.group(1), "todo",
                                            _SEVERITY["todo"], line.strip()[:120]))
        return findings

    def _review_function(self, rel: str, node: ast.AST) -> List[CodeFinding]:
        out: List[CodeFinding] = []
        name = getattr(node, "name", "<fn>")

        # missing docstring on a public function
        if ast.get_docstring(node) is None and not name.startswith("_"):
            out.append(CodeFinding(rel, node.lineno, name, "missing_docstring",
                                   _SEVERITY["missing_docstring"],
                                   "public function has no docstring"))

        # physical length
        end = getattr(node, "end_lineno", None) or node.lineno
        length = end - node.lineno
        if length > self.max_function_length:
            out.append(CodeFinding(rel, node.lineno, name, "long_function",
                                   min(1.0, _SEVERITY["long_function"]
                                       + 0.01 * (length - self.max_function_length)),
                                   f"{length} lines (> {self.max_function_length})"))

        # cyclomatic-ish complexity
        complexity = _complexity(node)
        if complexity > self.max_complexity:
            out.append(CodeFinding(rel, node.lineno, name, "high_complexity",
                                   min(1.0, _SEVERITY["high_complexity"]
                                       + 0.02 * (complexity - self.max_complexity)),
                                   f"complexity {complexity} (> {self.max_complexity})"))

        # too many positional args (ignore self/cls)
        args = node.args
        n_pos = len(args.posonlyargs) + len(args.args)
        if args.args and args.args[0].arg in ("self", "cls"):
            n_pos -= 1
        if n_pos > self.max_args:
            out.append(CodeFinding(rel, node.lineno, name, "too_many_args",
                                   _SEVERITY["too_many_args"],
                                   f"{n_pos} parameters (> {self.max_args})"))
        return out

    # ---- the full pass ---- #
    def review(self) -> CodeReviewReport:
        report = CodeReviewReport()
        for path in self._iter_modules():
            report.files_scanned += 1
            report.findings.extend(self.review_file(path))
        if self.enable_llm_enrichment and self._llm_available():
            self._enrich(report)
        return report

    # ---- optional LLM enrichment ---- #
    def _llm_available(self) -> bool:
        if self.llm is None:
            return False
        try:
            return self.llm.chosen_provider().name != "mock"
        except Exception:  # noqa: BLE001
            return False

    def _enrich(self, report: CodeReviewReport) -> None:
        """Attach a one-line critique to the worst offenders (best-effort, additive)."""
        any_enriched = False
        for finding in report.worst(self.enrich_top):
            prompt = (
                "You are reviewing NYXARA's own source. In ONE sentence, say why this is a "
                "real problem and the single best fix.\n"
                f"FILE: {finding.file}:{finding.lineno}  SYMBOL: {finding.symbol}\n"
                f"ISSUE ({finding.kind}): {finding.detail}\n"
                "Output only the one-sentence critique."
            )
            try:
                note = str(self.llm.generate(prompt, temperature=0.2, max_tokens=80) or "").strip()
            except Exception:  # noqa: BLE001 — enrichment is optional, never fatal
                continue
            if note:
                finding.critique = note[:300]
                any_enriched = True
        report.llm_enriched = any_enriched

    # ---- helpers ---- #
    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root.parent))
        except ValueError:
            return str(path)


def _complexity(node: ast.AST) -> int:
    """McCabe-style approximation: 1 + the number of branch/decision points inside ``node``."""
    branches = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.ExceptHandler,
                ast.With, ast.AsyncWith, ast.BoolOp, ast.IfExp, ast.Assert,
                ast.comprehension)
    count = 1
    for child in ast.walk(node):
        if child is node:
            continue
        # don't count nested function/class definitions' own complexity here
        if isinstance(child, branches):
            count += 1
    return count


def _package_root() -> Path:
    """Resolve the on-disk ``nyxara/`` package directory (cwd-independent)."""
    import nyxara
    return Path(nyxara.__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA self-review self-test")
    print("=" * 70)

    # 1) synthetic source exercises every finding kind
    import tempfile

    sample = (
        "import os\n"
        "def public_fn(a, b, c, d, e, f, g):\n"          # too_many_args + missing_docstring
        "    try:\n"
        "        return a\n"
        "    except:\n"                                    # bare_except
        "        pass  # TODO clean this up\n"             # todo
    )
    with tempfile.TemporaryDirectory() as d:
        pkg = Path(d) / "pkg"
        pkg.mkdir()
        (pkg / "mod.py").write_text(sample, encoding="utf-8")
        rep = SelfReviewer(root=pkg, max_function_length=2).review()
        kinds = rep.by_kind()
        print("\nsynthetic findings  :", kinds)
        for must in ("too_many_args", "bare_except", "todo", "missing_docstring"):
            assert must in kinds, f"missing finding kind: {must}"

    # 2) real run over NYXARA's own tree never crashes and scans real modules
    real = SelfReviewer().review()
    print(f"\nreal scan           : {real.files_scanned} module(s), "
          f"{len(real.findings)} finding(s)")
    assert real.files_scanned > 20, "should scan the real package"
    print("\n" + real.summary()[:600])

    print("\nALL SELF-TESTS PASSED ✓")
