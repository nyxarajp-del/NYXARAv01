"""End-to-end integration: the sovereign loop across module boundaries.

The per-module suites test units in isolation; this drives a real NyxaraCore
through a full lifecycle on the deterministic offline reasoner (no network, no
provider key): boot → process a turn → run a gated agent → persist memory →
restart into a fresh core → restore. It guards the wiring between kernel, mind,
agency and memory that unit tests can't see on their own."""

from __future__ import annotations

from nyxara import Authority, Disposition, NyxaraCore


def test_full_boot_and_process_turn():
    core = NyxaraCore()
    result = core.process("hello NYXARA", authority=Authority.OWNER)
    assert isinstance(result.disposition, Disposition)
    assert result.response
    # the disposed turn serializes cleanly for transport
    d = result.to_dict()
    assert "disposition" in d and "response" in d


def test_report_after_a_turn_is_serializable():
    core = NyxaraCore()
    core.process("what is 2 + 2?", authority=Authority.OWNER)
    report = core.report()
    assert isinstance(report, dict)
    assert "control" in report


def test_agent_run_completes_end_to_end():
    core = NyxaraCore()
    run = core.agent("say hello", authority=Authority.OWNER)
    body = run.to_dict()
    assert "status" in body and "final_answer" in body


def test_memory_persists_across_a_restart(tmp_path):
    path = str(tmp_path / "memory.json")

    # session one: learn something and persist identity (Rule 7)
    first = NyxaraCore(enable_memory=True)
    first.memory.remember("NYXARA's Master is JP", importance=0.95)
    first.process("remember who the Master is", authority=Authority.OWNER)
    saved = first.save_state(path)
    assert saved == path

    # session two: a brand-new process restores the prior memory
    second = NyxaraCore(enable_memory=True)
    loaded = second.load_state(path)
    assert loaded >= 1


def test_scram_halts_subsequent_turns():
    core = NyxaraCore()
    core.scram(reason="integration stop")
    result = core.process("are you still there?", authority=Authority.OWNER)
    # once scrammed, corrigibility halts the loop before anything proceeds
    assert result.disposition is Disposition.HALT
    core.resume()
    after = core.process("back online?", authority=Authority.OWNER)
    assert after.disposition is not Disposition.HALT
