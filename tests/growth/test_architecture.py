"""Tests for nyxara.growth.architecture — real import-graph analysis."""

from __future__ import annotations

from pathlib import Path

import pytest

from nyxara.growth.architecture import (ArchitectureAnalyzer, ArchitectureReport,
                                        _find_cycles, _tarjan_scc)


def _cycle_pkg(tmp_path: Path) -> Path:
    pkg = tmp_path / "toy"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "a.py").write_text("from toy import b\n", encoding="utf-8")
    (pkg / "b.py").write_text("from toy import a\n", encoding="utf-8")
    return pkg


def test_detects_cycle(tmp_path: Path):
    rep = ArchitectureAnalyzer(root=_cycle_pkg(tmp_path), pkg="toy").analyze()
    assert any(set(c) == {"toy.a", "toy.b"} for c in rep.cycles)


def test_cycle_detection_without_networkx(tmp_path: Path, monkeypatch):
    # force the pure-dict Tarjan fallback by hiding networkx
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "networkx":
            raise ImportError("hidden for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cycles, used_nx = _find_cycles([("x", "y"), ("y", "x")])
    assert used_nx is False
    assert any(set(c) == {"x", "y"} for c in cycles)


def test_tarjan_finds_three_cycle():
    sccs = _tarjan_scc([("a", "b"), ("b", "c"), ("c", "a"), ("d", "e")])
    assert any(set(c) == {"a", "b", "c"} for c in sccs)
    assert all(len(c) > 1 for c in sccs)       # singletons are never cycles


def test_flags_layering_violation(tmp_path: Path):
    # a lower-layer module importing a higher-layer one (kernel → growth)
    root = tmp_path / "nyxara"
    (root / "kernel").mkdir(parents=True)
    (root / "growth").mkdir(parents=True)
    (root / "__init__.py").write_text("", encoding="utf-8")
    (root / "kernel" / "__init__.py").write_text("", encoding="utf-8")
    (root / "growth" / "__init__.py").write_text("", encoding="utf-8")
    (root / "kernel" / "k.py").write_text("from nyxara.growth import g\n", encoding="utf-8")
    (root / "growth" / "g.py").write_text("x = 1\n", encoding="utf-8")
    rep = ArchitectureAnalyzer(root=root, pkg="nyxara").analyze()
    assert any(src.startswith("nyxara.kernel") and "growth" in dst
               for src, dst in rep.layering_violations)


def test_lazy_imports_are_not_counted(tmp_path: Path):
    # a function-local import must NOT create an edge (it is a deliberate cycle-breaker)
    pkg = tmp_path / "toy"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "a.py").write_text("def f():\n    from toy import b\n    return b\n", encoding="utf-8")
    (pkg / "b.py").write_text("x = 1\n", encoding="utf-8")
    rep = ArchitectureAnalyzer(root=pkg, pkg="toy").analyze()
    assert ("toy.a", "toy.b") not in rep.edges


def test_real_package_has_no_load_time_cycles():
    rep = ArchitectureAnalyzer().analyze()
    assert isinstance(rep, ArchitectureReport)
    assert len(rep.modules) > 50
    # the codebase uses lazy imports to stay acyclic at load time
    assert rep.cycles == []
