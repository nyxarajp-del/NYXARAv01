"""Tests that the Intuition Core is genuinely WIRED into the live sovereign cycle:

* built on the core (``core.intuition``),
* feeding System 1 in dual-process (a real hunch, not the old echo stub),
* load-bearing in ``_arbitrate`` (a machine-verified leap on a reversible, low-stakes turn
  raises the candidate's confidence and is recorded as the last intuition),
* seeding the Eureka prove-loop,
* exposed as ``core.intuit(...)``.

All offline — no LLM, no API key — proving NYXARA does it herself.
"""

from __future__ import annotations

from nyxara.kernel.orchestrator import Candidate, NyxaraCore


def test_intuition_is_built_on_core():
    core = NyxaraCore()
    assert core.intuition is not None
    assert type(core.intuition).__name__ == "IntuitionCore"


def test_dual_process_uses_real_intuition_not_echo():
    core = NyxaraCore()
    assert core.dual_process is not None
    from nyxara.mind.faculties import Task, TaskType
    task = Task(TaskType.SEQUENCE, "1, 4, 9, 16, 25", payload=[1, 4, 9, 16, 25])
    fast = core.dual_process.system1.respond(task)
    # the old stub echoed task.description; the real Core returns the leapt answer 36
    assert fast.answer == 36


def test_arbitrate_is_load_bearing_on_safe_turn():
    core = NyxaraCore()
    cand = Candidate(text="", kind="respond", confidence=0.30, reversible=True)
    before = cand.confidence
    out = core._arbitrate("what comes next: 1, 4, 9, 16, 25 ?", cand, memories=[])
    # a self-verified leap raises confidence (never lowers) and is captured
    assert out.confidence >= before
    assert core._last_intuition is not None
    assert core._last_intuition.answer == 36
    assert "intuition" in (out.rationale or "")


def test_arbitrate_colour_only_on_irreversible_turn():
    core = NyxaraCore()
    cand = Candidate(text="do irreversible thing", kind="act", confidence=0.40,
                     reversible=False)
    out = core._arbitrate("what comes next: 1, 4, 9, 16, 25 ?", cand, memories=[])
    # high-stakes / irreversible -> intuition must NOT alter the candidate
    assert out.confidence == 0.40
    assert core._last_intuition is None


def test_eureka_seed_source_is_wired():
    core = NyxaraCore()
    if core.eureka is not None:
        assert core.eureka._seed_source is not None


def test_public_intuit_entrypoint():
    core = NyxaraCore()
    out = core.intuit([2, 4, 6, 8, 10])
    assert out["leap"] is not None
    assert out["leap"]["answer"] == 12
    assert out["leap"]["self_verified"] is True


def test_process_a_puzzle_end_to_end():
    from nyxara.kernel.orchestrator import Authority
    core = NyxaraCore()
    res = core.process("what comes next: 3, 6, 9, 12, 15 ?", authority=Authority.OWNER)
    # the turn completes and the verified leap (18) is captured as the last intuition
    assert res is not None
    assert core._last_intuition is not None and core._last_intuition.answer == 18
