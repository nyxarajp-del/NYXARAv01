"""Tests for the extended default toolset: knowledge, code, shell, http (capability-gated)."""

from __future__ import annotations

from nyxara.agency.default_tools import build_default_tools
from nyxara.agency.permissions import Authority
from nyxara.agency.tools import ToolRegistry
from nyxara.memory.store import MemoryStore


def _reg(**kw):
    return build_default_tools(ToolRegistry(), **kw)


def test_code_tools_registered():
    names = set(_reg().names())
    assert {"run_python", "run_shell", "http_request"} <= names


def test_knowledge_tools_registered_with_memory():
    names = set(_reg(memory=MemoryStore()).names())
    assert {"knowledge_ingest", "knowledge_search"} <= names


def test_knowledge_ingest_then_search():
    reg = _reg(memory=MemoryStore())
    r = reg.invoke("knowledge_ingest",
                   {"text": "NYXARA is owned by Jaypal Khoja, called JP.", "source": "profile"},
                   authority=Authority.OWNER, owner_confirmed=True)
    assert r.ok and r.value["chunks"] >= 1
    r = reg.invoke("knowledge_search", {"query": "who owns NYXARA"},
                   authority=Authority.OWNER, owner_confirmed=True)
    assert r.ok and any("Jaypal" in c["text"] for c in r.value)


def test_run_python_owner_executes():
    reg = _reg()
    r = reg.invoke("run_python", {"code": "result = sum(range(10))"},
                   authority=Authority.OWNER, owner_confirmed=True)
    assert r.ok and r.value["value"] == 45


def test_code_exec_autonomous_requires_owner():
    reg = _reg()
    r = reg.invoke("run_python", {"code": "print(1)"}, authority=Authority.AUTONOMOUS)
    assert r.requires_owner and not r.ok


def test_http_request_ssrf_guard():
    reg = _reg()
    r = reg.invoke("http_request", {"url": "http://127.0.0.1/"},
                   authority=Authority.OWNER, owner_confirmed=True)
    assert r.ok  # the tool ran; the request itself was refused as data
    assert r.value["ok"] is False
