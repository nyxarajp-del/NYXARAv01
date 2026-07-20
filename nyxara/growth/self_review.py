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

When a real LLM is configured (``provider != "native"``, the same probe the per-turn
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
    "mutable_default_arg": 0.6,     # a real bug: the default is shared across every call
    "long_function": 0.45,
    "too_many_args": 0.4,
    "eq_none": 0.35,
    "redundant_else_return": 0.33,
    "not_in": 0.32,
    "is_not_cmp": 0.32,
    "not_eq_cmp": 0.32,
    "unused_import": 0.3,
    "double_negation": 0.3,
    "empty_collection_call": 0.28,
    "redundant_pass": 0.28,
    "missing_docstring": 0.25,
    "todo": 0.2,
}

# the literal each empty builtin-collection call collapses to (C408)
_EMPTY_COLLECTION_LITERAL = {"list": "[]", "dict": "{}", "tuple": "()"}


def _ends_in_terminator(body: List["ast.stmt"]) -> bool:
    """True if a statement block's last statement unconditionally leaves the block.

    ``return``/``raise``/``break``/``continue`` all mean the following ``else:`` can never be
    reached via fall-through, so it is redundant and may be de-indented (RET505)."""
    return bool(body) and isinstance(body[-1], (ast.Return, ast.Raise, ast.Break, ast.Continue))


def _has_mutable_default(node: "ast.AST") -> bool:
    """True if a def has a mutable literal (or ``list()``/``dict()``/``set()``) as a default arg.

    A mutable default is created once and shared across every call — a classic latent bug (B006).
    Immutable defaults (``()``, ``tuple()``, constants) are never flagged."""
    args = getattr(node, "args", None)
    if args is None:
        return False
    candidates = list(getattr(args, "defaults", []) or []) + [
        d for d in (getattr(args, "kw_defaults", []) or []) if d is not None]
    return any(_is_mutable_default(d) for d in candidates)


def _is_mutable_default(default: "ast.AST") -> bool:
    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
        return True
    if (isinstance(default, ast.Call) and isinstance(default.func, ast.Name)
            and default.func.id in ("list", "dict", "set")
            and not default.args and not default.keywords):
        return True
    return False


def _compares_to_none(node: "ast.Compare") -> bool:
    """True if ``node`` uses ``==``/``!=`` against the ``None`` literal (PEP 8 / E711)."""
    operands = [node.left, *node.comparators]
    has_none = any(isinstance(o, ast.Constant) and o.value is None for o in operands)
    has_eq = any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops)
    return has_none and has_eq


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
            elif isinstance(node, ast.Compare) and _compares_to_none(node):
                findings.append(CodeFinding(rel, node.lineno, "<compare>", "eq_none",
                                            _SEVERITY["eq_none"],
                                            "comparison to None should use 'is'/'is not', "
                                            "not '=='/'!=' (PEP 8 / E711)"))
            elif (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
                  and isinstance(node.operand, ast.Compare)
                  and len(node.operand.ops) == 1
                  and isinstance(node.operand.ops[0], (ast.In, ast.Is))):
                is_in = isinstance(node.operand.ops[0], ast.In)
                kind = "not_in" if is_in else "is_not_cmp"
                op = "not in" if is_in else "is not"
                findings.append(CodeFinding(
                    rel, node.lineno, "<compare>", kind, _SEVERITY[kind],
                    f"negated membership/identity test should use the '{op}' operator "
                    f"(PEP 8 / {'E713' if is_in else 'E714'})"))
            elif (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
                  and isinstance(node.operand, ast.Compare)
                  and len(node.operand.ops) == 1
                  and isinstance(node.operand.ops[0], (ast.Eq, ast.NotEq))
                  and not _compares_to_none(node.operand)):
                findings.append(CodeFinding(
                    rel, node.lineno, "<compare>", "not_eq_cmp", _SEVERITY["not_eq_cmp"],
                    "negated equality should use the '!='/'==' operator directly "
                    "(SIM201/SIM202)"))
            elif (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
                  and isinstance(node.operand, ast.UnaryOp)
                  and isinstance(node.operand.op, ast.Not)):
                findings.append(CodeFinding(
                    rel, node.lineno, "<not>", "double_negation",
                    _SEVERITY["double_negation"],
                    "double negation 'not not x' should be 'bool(x)' (SIM208)"))
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id in _EMPTY_COLLECTION_LITERAL
                  and not node.args and not node.keywords):
                lit = _EMPTY_COLLECTION_LITERAL[node.func.id]
                findings.append(CodeFinding(
                    rel, node.lineno, node.func.id, "empty_collection_call",
                    _SEVERITY["empty_collection_call"],
                    f"empty '{node.func.id}()' call should be the literal '{lit}' (C408)"))

        # block-structure findings that need the enclosing statement list (not a bare node walk)
        findings.extend(self._block_findings(rel, tree))

        # unused imports — module-level imports whose bound name is never referenced
        findings.extend(self._unused_imports(rel, tree, src, path))

        # TODO/FIXME markers — cheap raw-line scan
        for i, line in enumerate(src.splitlines(), start=1):
            m = _TODO_RE.search(line)
            if m:
                findings.append(CodeFinding(rel, i, m.group(1), "todo",
                                            _SEVERITY["todo"], line.strip()[:120]))
        return findings

    def _unused_imports(self, rel: str, tree: ast.AST, src: str, path: Path
                        ) -> List[CodeFinding]:
        """Top-level imports whose bound name is never referenced anywhere in the module.

        Conservative on purpose: skips ``__init__.py`` (re-export surface), ``__future__``,
        star imports, names listed in ``__all__``, and any import line carrying a ``# noqa``.
        """
        if path.name == "__init__.py":
            return []
        # every name read anywhere in the module (the root of any attribute chain is a Name)
        used: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used.add(node.id)
        # names a module deliberately exports count as "used"
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)):
                for elt in getattr(node.value, "elts", []):
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        used.add(elt.value)

        lines = src.splitlines()
        out: List[CodeFinding] = []
        for node in tree.body:                       # top-level imports only
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                continue
            line_txt = lines[node.lineno - 1] if 0 < node.lineno <= len(lines) else ""
            if "noqa" in line_txt:
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if any(a.name == "*" for a in node.names):    # star import — leave alone
                    continue
                for alias in node.names:
                    bound = (alias.asname or alias.name).split(".")[0]
                    if bound not in used:
                        out.append(CodeFinding(
                            rel, node.lineno, bound, "unused_import",
                            _SEVERITY["unused_import"],
                            f"import '{bound}' is never used"))
        return out

    def _block_findings(self, rel: str, tree: ast.AST) -> List[CodeFinding]:
        """Findings that depend on a statement's *block* context (redundant pass / else-return).

        A bare :func:`ast.walk` sees nodes but not which block they belong to, so these two checks
        inspect each node's ``body``/``orelse``/``finalbody`` statement lists directly."""
        out: List[CodeFinding] = []
        for node in ast.walk(tree):
            # a `pass` that shares a block with other statements is dead weight (PIE790)
            for block_field in ("body", "orelse", "finalbody"):
                block = getattr(node, block_field, None)
                if isinstance(block, list) and len(block) > 1:
                    for stmt in block:
                        if isinstance(stmt, ast.Pass):
                            out.append(CodeFinding(
                                rel, stmt.lineno, "<pass>", "redundant_pass",
                                _SEVERITY["redundant_pass"],
                                "redundant 'pass' — the block has other statements (PIE790)"))
            # an `else:` after an `if` branch that always returns/raises/breaks/continues (RET505)
            if isinstance(node, ast.If) and node.orelse and _ends_in_terminator(node.body):
                # an `elif` is an If sitting in `orelse` at the SAME column — never flag it
                if (len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)
                        and node.orelse[0].col_offset == node.col_offset):
                    continue
                out.append(CodeFinding(
                    rel, node.orelse[0].lineno, "<else>", "redundant_else_return",
                    _SEVERITY["redundant_else_return"],
                    "the 'if' branch ends in return/raise/break/continue — the 'else' is "
                    "redundant and can be de-indented (RET505)"))
        return out

    def _review_function(self, rel: str, node: ast.AST) -> List[CodeFinding]:
        out: List[CodeFinding] = []
        name = getattr(node, "name", "<fn>")

        # missing docstring on a public function
        if ast.get_docstring(node) is None and not name.startswith("_"):
            out.append(CodeFinding(rel, node.lineno, name, "missing_docstring",
                                   _SEVERITY["missing_docstring"],
                                   "public function has no docstring"))

        # mutable default argument — created once, shared across every call (a real bug, B006)
        if _has_mutable_default(node):
            out.append(CodeFinding(rel, node.lineno, name, "mutable_default_arg",
                                   _SEVERITY["mutable_default_arg"],
                                   "mutable default argument is shared across calls — use a "
                                   "None sentinel (B006)"))

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
            return self.llm.chosen_provider().name != "native"
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
