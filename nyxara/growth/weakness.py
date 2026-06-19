"""NYXARA · growth/weakness.py — Self Weakness Detection (🩻, RSI capability 4).

The review and the architecture analysis each see one face of the system; the benchmark sees
a third. This module is the **join**: it folds the code-review findings, the architecture
smells, and the benchmark failures into a single, ranked list of concrete weaknesses, each
with a normalised severity in ``[0, 1]`` and a plain-language *remediation* — the exact thing
the self-optimizer will try to act on.

Ranking reflects blast radius: architectural cycles / layering violations rank highest (they
corrupt the whole build graph), then lost capability (benchmark failures), then risky code
(bare excepts, high complexity), then hygiene (missing docstrings, TODOs). Pure, deterministic,
dependency-free — it only consumes the three report objects it is given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["Weakness", "WeaknessReport", "WeaknessSynthesizer"]


@dataclass
class Weakness:
    """One ranked, actionable weakness synthesised from the analysis reports."""

    id: str
    source: str                 # "architecture" | "benchmark" | "code"
    severity: float             # normalised [0, 1]
    title: str
    evidence: List[str] = field(default_factory=list)
    remediation: str = ""
    is_source_edit: bool = False     # True ⇒ remediation is a concrete source-code change
    locus: str = ""                  # file:line or module the edit/forge targets
    symbol: str = ""                 # the precise symbol the edit targets (e.g. an import name)
    edit_strategy: str = ""          # "deterministic" | "llm" | "" — who can author the edit

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "source": self.source, "severity": round(self.severity, 3),
                "title": self.title, "evidence": self.evidence,
                "remediation": self.remediation, "is_source_edit": self.is_source_edit,
                "locus": self.locus, "symbol": self.symbol,
                "edit_strategy": self.edit_strategy}


@dataclass
class WeaknessReport:
    weaknesses: List[Weakness] = field(default_factory=list)

    def top(self, n: int = 10) -> List[Weakness]:
        return self.ranked()[:max(0, n)]

    def ranked(self) -> List[Weakness]:
        return sorted(self.weaknesses, key=lambda w: w.severity, reverse=True)

    def by_source(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for w in self.weaknesses:
            out[w.source] = out.get(w.source, 0) + 1
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"n_weaknesses": len(self.weaknesses), "by_source": self.by_source(),
                "weaknesses": [w.to_dict() for w in self.ranked()]}

    def summary(self) -> str:
        lines = ["=" * 70, "NYXARA self weakness detection", "=" * 70,
                 f"{len(self.weaknesses)} weakness(es) — {self.by_source()}", ""]
        for w in self.top(12):
            flag = "✎" if w.is_source_edit else "·"
            lines.append(f"  {flag} [{w.severity:.2f}] ({w.source}) {w.title}")
            if w.remediation:
                lines.append(f"      ↳ {w.remediation}")
        if not self.weaknesses:
            lines.append("  no weaknesses detected ✓")
        return "\n".join(lines)


class WeaknessSynthesizer:
    """Fold code-review + architecture + benchmark reports into one ranked weakness list."""

    def synthesize(self, code: Any = None, arch: Any = None,
                   bench: Optional[Dict[str, Any]] = None) -> WeaknessReport:
        report = WeaknessReport()
        if arch is not None:
            report.weaknesses.extend(self._from_architecture(arch))
        if bench is not None:
            report.weaknesses.extend(self._from_benchmark(bench))
        if code is not None:
            report.weaknesses.extend(self._from_code(code))
        return report

    # ---- architecture (highest blast radius) ---- #
    def _from_architecture(self, arch: Any) -> List[Weakness]:
        out: List[Weakness] = []
        for i, cyc in enumerate(getattr(arch, "cycles", []) or []):
            out.append(Weakness(
                id=f"arch-cycle-{i}", source="architecture", severity=0.9,
                title=f"import cycle across {len(cyc)} modules",
                evidence=[" → ".join(cyc)],
                remediation=(f"break the cycle {cyc[0]} ↔ {cyc[1]} by extracting a shared "
                             "interface/seam so the dependency points one way"),
                is_source_edit=False, locus=cyc[0] if cyc else ""))
        for j, (src, dst) in enumerate(getattr(arch, "layering_violations", []) or []):
            out.append(Weakness(
                id=f"arch-layer-{j}", source="architecture", severity=0.8,
                title=f"layering violation: {src} → {dst}",
                evidence=[f"{src} imports {dst} (a higher layer)"],
                remediation=(f"invert the dependency: {dst} should not be imported by the "
                             f"more-foundational {src}; move the shared piece down a layer"),
                is_source_edit=False, locus=src))
        for god in (getattr(arch, "god_modules", []) or [])[:5]:
            out.append(Weakness(
                id=f"arch-god-{god}", source="architecture", severity=0.55,
                title=f"god-module: {god} (very high coupling)",
                evidence=[f"{god} has outsized fan-in + fan-out"],
                remediation=f"split {god} into cohesive sub-modules to reduce coupling",
                is_source_edit=False, locus=god))
        return out

    # ---- benchmark (lost / missing capability) ---- #
    def _from_benchmark(self, bench: Dict[str, Any]) -> List[Weakness]:
        out: List[Weakness] = []
        accuracy = float(bench.get("accuracy", 0.0) or 0.0)
        failures = bench.get("failures", []) or []
        by_cat = bench.get("by_category", {}) or {}
        if failures:
            # severity scales with how much capability is missing; floored above ordinary
            # code findings so lost capability outranks code hygiene/risk smells
            sev = min(0.85, 0.6 + 0.3 * (1.0 - accuracy))
            cats = sorted({f.get("category", "?") for f in failures
                           if isinstance(f, dict)})
            out.append(Weakness(
                id="bench-failures", source="benchmark", severity=sev,
                title=f"{len(failures)} benchmark task(s) failing "
                      f"(accuracy {accuracy:.0%})",
                evidence=[f"weak categories: {', '.join(cats) or 'n/a'}"],
                remediation=("raise reasoning depth (recursive_improvement_iterations) and "
                             "forge/distil on the failing categories: "
                             f"{', '.join(cats) or 'n/a'}"),
                is_source_edit=False, locus="mind"))
        # a category that is entirely failing is its own, sharper weakness
        for cat, st in sorted(by_cat.items()):
            acc = float(st.get("accuracy", 1.0) or 0.0) if isinstance(st, dict) else 1.0
            if acc <= 0.0 and isinstance(st, dict) and st.get("n", 0):
                out.append(Weakness(
                    id=f"bench-cat-{cat}", source="benchmark", severity=0.65,
                    title=f"category '{cat}' fully failing",
                    evidence=[f"0/{st.get('n')} correct in '{cat}'"],
                    remediation=f"target '{cat}' with distillation + a regression benchmark task",
                    is_source_edit=False, locus="eval"))
        return out

    # ---- code review (risky code & hygiene → real source edits) ---- #
    def _from_code(self, code: Any) -> List[Weakness]:
        out: List[Weakness] = []
        findings = getattr(code, "findings", []) or []
        # source-editable kinds the optimizer can fix with a deterministic, AST-validated
        # transform (instant, offline, zero LLM/network dependence)
        deterministic = {"bare_except", "missing_docstring", "eq_none", "unused_import",
                         "not_in", "is_not_cmp"}
        # kinds that are genuine source fixes but need real authoring (a whole-file rewrite the
        # LLM proposes); they only fire when the LLM edit path is enabled + a provider exists,
        # and every such edit still clears the same reversible gauntlet
        llm_editable = {"high_complexity", "long_function", "too_many_args", "todo"}
        for f in findings:
            kind = getattr(f, "kind", "")
            sev = float(getattr(f, "severity", 0.3) or 0.3)
            # de-prioritise hygiene below capability/architecture by capping
            if kind in ("missing_docstring", "todo"):
                sev = min(sev, 0.3)
            strategy = ("deterministic" if kind in deterministic
                        else "llm" if kind in llm_editable else "")
            out.append(Weakness(
                id=f"code-{kind}-{getattr(f, 'file', '?')}-{getattr(f, 'lineno', 0)}",
                source="code", severity=sev,
                title=f"{kind} at {getattr(f, 'file', '?')}:{getattr(f, 'lineno', 0)} "
                      f"({getattr(f, 'symbol', '?')})",
                evidence=[getattr(f, "detail", "")]
                + ([getattr(f, "critique", "")] if getattr(f, "critique", "") else []),
                remediation=_code_remediation(kind),
                is_source_edit=bool(strategy),
                edit_strategy=strategy,
                locus=f"{getattr(f, 'file', '?')}:{getattr(f, 'lineno', 0)}",
                symbol=str(getattr(f, "symbol", "") or "")))
        return out


def _code_remediation(kind: str) -> str:
    return {
        "bare_except": "replace the bare 'except:' with 'except Exception:' so "
                       "KeyboardInterrupt/SystemExit propagate",
        "missing_docstring": "add a one-line docstring describing the symbol's contract",
        "high_complexity": "extract helper functions to lower cyclomatic complexity",
        "long_function": "split the function into smaller, single-purpose helpers",
        "too_many_args": "group related parameters into a dataclass/config object",
        "eq_none": "use 'is None'/'is not None' instead of '=='/'!=' for None comparison",
        "unused_import": "remove the unused import so the dependency surface stays honest",
        "not_in": "use the 'not in' membership operator instead of 'not ... in ...' (E713)",
        "is_not_cmp": "use the 'is not' operator instead of 'not ... is ...' (E714)",
        "todo": "resolve or file the TODO; remove the stale marker",
    }.get(kind, "review and refactor")


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA weakness-synthesis self-test")
    print("=" * 70)

    from nyxara.growth.architecture import ArchitectureReport
    from nyxara.growth.self_review import CodeFinding, CodeReviewReport

    arch = ArchitectureReport(cycles=[["nyxara.a", "nyxara.b"]],
                              layering_violations=[("nyxara.kernel.x", "nyxara.growth.y")])
    code = CodeReviewReport(findings=[
        CodeFinding("nyxara/m.py", 10, "f", "bare_except", 0.7, "bare except"),
        CodeFinding("nyxara/m.py", 1, "m", "missing_docstring", 0.25, "no docstring"),
    ], files_scanned=1)
    bench = {"accuracy": 0.5, "failures": [{"category": "logic"}],
             "by_category": {"logic": {"n": 4, "accuracy": 0.0}}}

    rep = WeaknessSynthesizer().synthesize(code=code, arch=arch, bench=bench)
    print("\n" + rep.summary())

    ranked = rep.ranked()
    # architecture cycle must outrank a benchmark failure must outrank code hygiene
    assert ranked[0].source == "architecture", "cycle should rank first"
    sources_in_order = [w.source for w in ranked]
    assert sources_in_order.index("architecture") < sources_in_order.index("code")
    # every weakness carries a concrete remediation
    assert all(w.remediation for w in rep.weaknesses)
    # the bare-except is marked as a real source edit; the cycle is not
    assert any(w.is_source_edit and "bare_except" in w.id for w in rep.weaknesses)
    assert not any(w.is_source_edit for w in rep.weaknesses if w.source == "architecture")

    print("\nALL SELF-TESTS PASSED ✓")
