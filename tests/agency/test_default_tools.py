"""Tests for nyxara.agency.default_tools — the default executable toolset."""

from __future__ import annotations

import os

import pytest

from nyxara.agency.default_tools import build_default_tools, safe_calculate
from nyxara.agency.permissions import Authority
from nyxara.agency.tools import ToolRegistry


def _reg(**kw):
    return build_default_tools(ToolRegistry(), **kw)


def test_registers_expected_tools():
    names = set(_reg().names())
    assert {"now", "calculate", "read_file", "list_dir", "write_file",
            "web_fetch", "web_search"} <= names


def test_calculate_runs():
    r = _reg().invoke("calculate", {"expression": "2*(3+4)"}, authority=Authority.OWNER)
    assert r.ok and r.value == 14.0


def test_safe_calculate_rejects_non_arithmetic():
    with pytest.raises(Exception):
        safe_calculate("__import__('os').system('echo hi')")
    with pytest.raises(Exception):
        safe_calculate("open('x')")


def test_now_is_iso():
    r = _reg().invoke("now", {}, authority=Authority.OWNER)
    assert r.ok and "T" in r.value


def test_read_and_write_file(tmp_path):
    reg = _reg()
    p = tmp_path / "note.txt"
    w = reg.invoke("write_file", {"path": str(p), "content": "hello"},
                   authority=Authority.OWNER)
    assert w.ok and os.path.exists(p)
    r = reg.invoke("read_file", {"path": str(p)}, authority=Authority.OWNER)
    assert r.ok and r.value == "hello"


def test_list_dir(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    r = _reg().invoke("list_dir", {"path": str(tmp_path)}, authority=Authority.OWNER)
    assert r.ok and "a.txt" in r.value


def test_memory_tools_present_only_with_store():
    assert "remember_fact" not in _reg().names()
    from nyxara.memory.store import MemoryStore
    reg = _reg(memory=MemoryStore())
    assert "remember_fact" in reg.names() and "recall_memory" in reg.names()


def test_memory_roundtrip_through_tools():
    from nyxara.memory.store import MemoryStore
    mem = MemoryStore()
    reg = _reg(memory=mem)
    w = reg.invoke("remember_fact", {"text": "the sky is teal today"},
                   authority=Authority.OWNER)
    assert w.ok
    r = reg.invoke("recall_memory", {"query": "what colour is the sky"},
                   authority=Authority.OWNER)
    assert r.ok and any("teal" in s for s in r.value)
