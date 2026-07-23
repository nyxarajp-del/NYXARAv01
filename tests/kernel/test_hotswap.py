"""Tests for kernel/hotswap.py — bounded live module reload with safe fallback (Change set C).

Fast + standalone: builds throwaway temp modules and reloads them; also pins the constitutional guard.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from nyxara.kernel.hotswap import HotSwapper


def _write(pkgdir: Path, name: str, body: str) -> str:
    p = pkgdir / f"{name}.py"
    p.write_text(body, encoding="utf-8")
    return str(p)


def test_hot_swap_reloads_a_safe_leaf_module(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    path = _write(tmp_path, "hs_leaf", "VALUE = 1\n")
    mod = importlib.import_module("hs_leaf")
    assert mod.VALUE == 1
    try:
        Path(path).write_text("VALUE = 2\n", encoding="utf-8")
        res = HotSwapper().swap(path)
        assert res.hot_swapped is True
        assert sys.modules["hs_leaf"].VALUE == 2      # live in the running process, no restart
    finally:
        sys.modules.pop("hs_leaf", None)


def test_unimported_path_is_not_swapped(tmp_path):
    path = _write(tmp_path, "hs_never_imported", "X = 0\n")
    res = HotSwapper().swap(path)
    assert res.hot_swapped is False
    assert "not currently imported" in res.reason


def test_referenced_module_is_disk_only(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    dep_path = _write(tmp_path, "hs_dep", "def helper():\n    return 41\n")
    _write(tmp_path, "hs_user", "from hs_dep import helper\n")
    importlib.import_module("hs_dep")
    importlib.import_module("hs_user")   # holds a reference to hs_dep.helper
    try:
        res = HotSwapper().swap(dep_path)
        assert res.hot_swapped is False
        assert "stale bindings" in res.reason or "referenced by" in res.reason
    finally:
        sys.modules.pop("hs_user", None)
        sys.modules.pop("hs_dep", None)


def test_constitutional_core_is_never_swapped():
    # identity/soul.py is a sealed core file — must be refused regardless of import state
    soul = Path("nyxara/identity/soul.py").resolve()
    res = HotSwapper().is_safe(str(soul))
    assert res.hot_swapped is False
    assert "constitutional" in res.reason or "master core" in res.reason
