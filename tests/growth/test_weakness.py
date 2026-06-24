"""Tests for nyxara.growth.weakness — synthesising ranked weaknesses."""

from __future__ import annotations

from nyxara.growth.architecture import ArchitectureReport
from nyxara.growth.self_review import CodeFinding, CodeReviewReport
from nyxara.growth.weakness import WeaknessReport, WeaknessSynthesizer


def _inputs():
    arch = ArchitectureReport(
        cycles=[["nyxara.a", "nyxara.b"]],
        layering_violations=[("nyxara.kernel.x", "nyxara.growth.y")])
    code = CodeReviewReport(findings=[
        CodeFinding("nyxara/m.py", 10, "f", "bare_except", 0.7, "bare except"),
        CodeFinding("nyxara/m.py", 1, "m", "missing_docstring", 0.25, "no docstring"),
    ], files_scanned=1)
    bench = {"accuracy": 0.5, "failures": [{"category": "logic"}],
             "by_category": {"logic": {"n": 4, "accuracy": 0.0}}}
    return code, arch, bench


def test_synthesizes_from_all_three_sources():
    code, arch, bench = _inputs()
    rep = WeaknessSynthesizer().synthesize(code=code, arch=arch, bench=bench)
    assert isinstance(rep, WeaknessReport)
    sources = rep.by_source()
    assert sources.get("architecture") and sources.get("benchmark") and sources.get("code")


def test_ranking_architecture_above_benchmark_above_code():
    code, arch, bench = _inputs()
    rep = WeaknessSynthesizer().synthesize(code=code, arch=arch, bench=bench)
    order = [w.source for w in rep.ranked()]
    assert order.index("architecture") < order.index("benchmark")
    assert order.index("benchmark") < order.index("code")


def test_every_weakness_has_remediation():
    code, arch, bench = _inputs()
    rep = WeaknessSynthesizer().synthesize(code=code, arch=arch, bench=bench)
    assert all(w.remediation for w in rep.weaknesses)


def test_code_and_architecture_marked_as_source_edit():
    code, arch, bench = _inputs()
    rep = WeaknessSynthesizer().synthesize(code=code, arch=arch, bench=bench)
    editable = [w for w in rep.weaknesses if w.is_source_edit]
    assert any("bare_except" in w.id for w in editable)
    # architectural weaknesses are now self-authored ("llm" strategy) source edits — NYXARA may
    # attempt the redesign herself, gauntlet-gated
    arch_w = [w for w in rep.weaknesses if w.source == "architecture"]
    assert arch_w and all(w.is_source_edit and w.edit_strategy == "llm" for w in arch_w)
    # benchmark weaknesses remain directives, not direct source edits
    assert not any(w.is_source_edit for w in rep.weaknesses if w.source == "benchmark")


def test_empty_inputs_are_safe():
    rep = WeaknessSynthesizer().synthesize()
    assert rep.weaknesses == []
    assert rep.to_dict()["n_weaknesses"] == 0
