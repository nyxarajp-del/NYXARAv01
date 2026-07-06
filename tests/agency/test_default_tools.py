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


def test_http_request_always_registered():
    # http_request is a core internet tool — present even without the code/shell pack.
    assert "http_request" in _reg(enable_code=False).names()
    assert "http_request" in _reg(enable_code=True).names()


# --------------------------------------------------------------------------- #
# run_shell — the real terminal tool (arbitrary shell, code-gated autonomy)
# --------------------------------------------------------------------------- #
def test_run_shell_is_a_real_shell():
    # Owner runs it directly (Rule 1): a pipe proves it is a real shell, not argv-only.
    r = _reg().invoke("run_shell", {"command": "echo real | rev"},
                      authority=Authority.OWNER)
    assert r.ok and "laer" in r.value["stdout"]


def test_run_shell_persists_cwd_across_calls():
    reg = _reg()
    reg.invoke("run_shell", {"command": "cd /"}, authority=Authority.OWNER)
    r = reg.invoke("run_shell", {"command": "pwd"}, authority=Authority.OWNER)
    assert r.ok and r.value["stdout"].strip() == "/"


def test_run_shell_escalates_under_default_policy():
    # PROC_EXEC/HIGH is irreversible: on a fresh (conservative) policy, autonomous use
    # escalates to the Master rather than running silently — governance is intact.
    r = _reg().invoke("run_shell", {"command": "echo hi"},
                      authority=Authority.AUTONOMOUS)
    assert not r.ok and r.requires_owner


def test_run_shell_runs_autonomously_under_full_control():
    # "NYXARA khude kare": with the owner's standing full_control grant, she runs the
    # shell on her own initiative — code-driven, no LLM, still gated by the policy.
    from nyxara.agency.permissions import build_sovereign_policy

    reg = build_default_tools(ToolRegistry(policy=build_sovereign_policy()))
    r = reg.invoke("run_shell", {"command": "echo autonomous | tr a-z A-Z"},
                   authority=Authority.AUTONOMOUS)
    assert r.ok and "AUTONOMOUS" in r.value["stdout"]


# --------------------------------------------------------------------------- #
# process/job control + system introspection tools
# --------------------------------------------------------------------------- #
def test_process_control_tools_registered():
    names = set(_reg().names())
    assert {"proc_spawn", "proc_list", "proc_read", "proc_write", "proc_signal",
            "proc_kill", "proc_wait", "list_processes", "kill_process",
            "system_info"} <= names


def test_proc_spawn_read_wait_roundtrip():
    import time

    reg = _reg()
    r = reg.invoke("proc_spawn", {"command": "echo hello-job"}, authority=Authority.OWNER)
    assert r.ok and r.value["ok"]
    handle = r.value["handle"]
    reg.invoke("proc_wait", {"handle": handle, "timeout_s": 10.0}, authority=Authority.OWNER)
    time.sleep(0.05)
    out = reg.invoke("proc_read", {"handle": handle}, authority=Authority.OWNER)
    assert out.ok and "hello-job" in out.value["stdout"]


def test_proc_spawn_escalates_under_default_policy():
    # proc_spawn is PROC_EXEC/HIGH like run_shell: conservative policy escalates autonomous use.
    r = _reg().invoke("proc_spawn", {"command": "echo hi"}, authority=Authority.AUTONOMOUS)
    assert not r.ok and r.requires_owner


def test_proc_spawn_runs_autonomously_under_full_control():
    from nyxara.agency.permissions import build_sovereign_policy

    reg = build_default_tools(ToolRegistry(policy=build_sovereign_policy()))
    r = reg.invoke("proc_spawn", {"command": "sleep 300"}, authority=Authority.AUTONOMOUS)
    assert r.ok and r.value["ok"]
    reg.invoke("proc_kill", {"handle": r.value["handle"]}, authority=Authority.AUTONOMOUS)


def test_system_info_tool_is_read_only_and_safe():
    # system_info is TOOL_CALL/LOW: even autonomously it runs without escalation.
    r = _reg().invoke("system_info", {}, authority=Authority.AUTONOMOUS)
    assert r.ok and r.value["ok"] and r.value["cpu_count"] >= 1


def test_kill_process_requires_a_target():
    r = _reg().invoke("kill_process", {}, authority=Authority.OWNER)
    assert r.ok and r.value["ok"] is False and "pid" in r.value["error"]


def test_web_search_routes_through_provider_layer(monkeypatch):
    import nyxara.senses.search as search

    class FakeSearcher:
        def __init__(self, **kw):
            pass

        def search(self, query, max_results=None):
            return search.SearchResponse(query, "duckduckgo", results=[
                search.SearchResult("Title", "https://ex.com", "snippet")])

    monkeypatch.setattr(search, "WebSearcher", FakeSearcher)
    r = _reg().invoke("web_search", {"query": "anything"}, authority=Authority.OWNER)
    assert r.ok and r.value[0]["url"] == "https://ex.com"
    assert r.value[0]["title"] == "Title" and r.value[0]["snippet"] == "snippet"
    assert r.value[0]["text"] == "snippet"  # back-compat alias


def test_web_search_errors_as_data(monkeypatch):
    import nyxara.senses.search as search

    class FailSearcher:
        def __init__(self, **kw):
            pass

        def search(self, query, max_results=None):
            return search.SearchResponse(query, "duckduckgo", error="boom")

    monkeypatch.setattr(search, "WebSearcher", FailSearcher)
    # a failed search returns [] rather than raising — the act stage never crashes
    r = _reg().invoke("web_search", {"query": "x"}, authority=Authority.OWNER)
    assert r.ok and r.value == []


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
    # list_dir now returns rich metadata entries (name/type/size/mtime/mode), not bare names.
    assert r.ok and any(e["name"] == "a.txt" and e["type"] == "file" for e in r.value)


def test_multimodal_tools_registered():
    names = set(_reg().names())
    assert {"inspect_image", "read_image_text", "transcribe_audio", "read_document"} <= names


def test_read_document_plain_text(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("NYXARA serves JP. " * 10)
    r = _reg().invoke("read_document", {"path": str(p)}, authority=Authority.OWNER)
    assert r.ok and r.value["words"] >= 20 and "NYXARA" in r.value["text"]


def test_transcribe_audio_degrades_honestly(tmp_path):
    import struct
    import wave
    wp = tmp_path / "a.wav"
    w = wave.open(str(wp), "w")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(struct.pack("<" + "h" * 800, *([0] * 800)))
    w.close()
    r = _reg().invoke("transcribe_audio", {"path": str(wp)}, authority=Authority.OWNER)
    # without whisper installed the transcript is None but the call must succeed with a note
    assert r.ok and (r.value["transcript"] is not None or r.value["note"])


def test_memory_tools_present_only_with_store():
    assert "remember_fact" not in _reg().names()
    from nyxara.memory.store import MemoryStore
    reg = _reg(memory=MemoryStore())
    assert "remember_fact" in reg.names() and "recall_memory" in reg.names()


def test_generate_image_tool(tmp_path):
    p = tmp_path / "art.png"
    r = _reg().invoke("generate_image", {"prompt": "a sovereign mind", "path": str(p),
                                         "size": 64}, authority=Authority.OWNER)
    assert r.ok and r.value["ok"] and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_synthesize_speech_tool(tmp_path):
    p = tmp_path / "say.wav"
    r = _reg().invoke("synthesize_speech", {"text": "hello Master", "path": str(p)},
                      authority=Authority.OWNER)
    assert r.ok and r.value["ok"] and p.read_bytes()[:4] == b"RIFF"


def test_train_self_model_is_master_gated():
    from nyxara.memory.store import MemoryStore, MemoryType
    mem = MemoryStore()
    for i in range(20):
        mem.remember(f"NYXARA serves JP loyally, note {i}", mem_type=MemoryType.SEMANTIC)
    reg = _reg(memory=mem)
    assert "train_self_model" in reg.names()
    # SELF_MODIFY is owner-exclusive: an autonomous call must escalate, never run
    auto = reg.invoke("train_self_model", {"generations": 1}, authority=Authority.AUTONOMOUS)
    assert not auto.ok and auto.requires_owner


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
