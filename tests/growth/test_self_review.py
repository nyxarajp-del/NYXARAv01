"""Tests for nyxara.growth.self_review — real AST self code review."""

from __future__ import annotations

from pathlib import Path

from nyxara.growth.self_review import CodeReviewReport, SelfReviewer

_SAMPLE = (
    "import os\n"
    "def public_fn(a, b, c, d, e, f, g):\n"      # too_many_args + missing_docstring
    "    try:\n"
    "        return a\n"
    "    except:\n"                               # bare_except
    "        pass  # TODO clean this up\n"         # todo
    "def long_public():\n"                         # long_function + missing_docstring
    + "".join(f"    x{i} = {i}\n" for i in range(14))
)


def _toy_pkg(tmp_path: Path) -> Path:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "mod.py").write_text(_SAMPLE, encoding="utf-8")
    return pkg


def test_surfaces_each_finding_kind(tmp_path: Path):
    rep = SelfReviewer(root=_toy_pkg(tmp_path), max_function_length=2).review()
    kinds = rep.by_kind()
    for must in ("too_many_args", "bare_except", "todo", "missing_docstring",
                 "long_function"):
        assert must in kinds, f"missing finding kind: {must}"


def _one_mod(tmp_path: Path, body: str) -> Path:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "m.py").write_text('"""m."""\n' + body, encoding="utf-8")
    return pkg


def test_detects_eq_none(tmp_path: Path):
    rep = SelfReviewer(root=_one_mod(tmp_path, "def _f(x):\n    return x == None\n")).review()
    assert "eq_none" in rep.by_kind()


def test_is_none_not_flagged(tmp_path: Path):
    rep = SelfReviewer(root=_one_mod(tmp_path, "def _f(x):\n    return x is None\n")).review()
    assert "eq_none" not in rep.by_kind()


def test_detects_unused_import(tmp_path: Path):
    rep = SelfReviewer(root=_one_mod(tmp_path, "import os\ndef _f():\n    return 1\n")).review()
    assert "unused_import" in rep.by_kind()


def test_used_import_not_flagged(tmp_path: Path):
    rep = SelfReviewer(root=_one_mod(
        tmp_path, "import os\ndef _f():\n    return os.getpid()\n")).review()
    assert "unused_import" not in rep.by_kind()


def test_noqa_import_not_flagged(tmp_path: Path):
    rep = SelfReviewer(root=_one_mod(
        tmp_path, "import os  # noqa: F401\ndef _f():\n    return 1\n")).review()
    assert "unused_import" not in rep.by_kind()


def test_all_exported_import_not_flagged(tmp_path: Path):
    rep = SelfReviewer(root=_one_mod(
        tmp_path, "from x import thing\n__all__ = ['thing']\n")).review()
    assert "unused_import" not in rep.by_kind()


def test_deliberate_except_exception_is_not_flagged_as_bare(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "m.py").write_text(
        '"""m."""\n'
        "def _f():\n"
        "    try:\n"
        "        pass\n"
        "    except Exception:  # noqa: BLE001\n"
        "        pass\n", encoding="utf-8")
    rep = SelfReviewer(root=pkg).review()
    assert "bare_except" not in rep.by_kind()


def test_real_package_scans_without_crashing():
    rep = SelfReviewer().review()
    assert isinstance(rep, CodeReviewReport)
    assert rep.files_scanned > 20
    assert "by_kind" in rep.to_dict()


def test_offline_does_not_enrich():
    # under the TEST profile the provider is mock → no LLM enrichment
    rep = SelfReviewer().review()
    assert rep.llm_enriched is False


def test_worst_is_sorted_by_severity(tmp_path: Path):
    rep = SelfReviewer(root=_toy_pkg(tmp_path), max_function_length=2).review()
    worst = rep.worst(5)
    sev = [f.severity for f in worst]
    assert sev == sorted(sev, reverse=True)
