"""End-to-end tests for true generalization in a live NyxaraCore turn (offline, no LLM).

These drive a real :class:`~nyxara.kernel.orchestrator.NyxaraCore` with the deterministic
MockProvider (no network, no key) and assert behavioural realness:

* a genuinely NEW task, shown by a few examples in the prompt, is solved on a held-out input in
  the live turn by her own skill-induction faculty — not the mock LLM;
* ordinary conversation is not hijacked (the cascade declines on unstructured chat);
* on the action side, a task that needs a tool she has no code for is forged and re-dispatched
  in the same turn, producing a real result, with every gate still running.
"""

from __future__ import annotations

from nyxara.agency.permissions import Authority
from nyxara.kernel.orchestrator import Candidate, NyxaraCore


def test_from_examples_task_answered_in_live_turn_by_own_faculty():
    core = NyxaraCore()
    r = core.process("apply the rule: ab -> ba, cd -> dc. now: xy", authority=Authority.OWNER)
    cand = r.candidate
    # the reversal of the unseen input can only come from her induced program (no LLM here)
    assert "yx" in cand.text
    assert "skill_induction" in cand.rationale
    # it is a normal reply candidate that cleared the gates — nothing was bypassed
    assert cand.kind == "respond"
    assert r.gates.get("oversight") == "allowed"


def test_from_examples_reversal_on_a_longer_word():
    core = NyxaraCore()
    r = core.process("pattern: hello -> olleh, world -> dlrow. now: nyxara",
                     authority=Authority.OWNER)
    assert "araxyn" in r.candidate.text


def test_ordinary_chat_is_not_hijacked():
    core = NyxaraCore()
    r = core.process("hello, how are you today?", authority=Authority.OWNER)
    # a greeting must not be treated as a from-examples task
    assert "skill_induction" not in r.candidate.rationale
    assert "araxyn" not in r.candidate.text


def test_candidate_from_preserves_missing_tool_intent():
    from nyxara.mind.llm_reasoner import LLMReasoner

    class _EmptyRegistry:
        def get(self, name):  # every tool is missing
            return None

    lr = LLMReasoner(tools=_EmptyRegistry())
    cand = lr._candidate_from(
        {"kind": "act", "tool": "reverse_text", "tool_args": {"text": "hi"}, "text": "do it"},
        "reverse the text")
    # degraded to a reply so no gate acts on a nonexistent tool …
    assert cand.kind == "respond"
    # … but the intent is preserved so the kernel can forge and re-dispatch
    assert cand.wanted_tool == "reverse_text"
    assert cand.wanted_tool_args == {"text": "hi"}


def test_action_side_forge_and_redispatch_produces_real_result():
    core = NyxaraCore()
    assert core.tool_forge is not None, "tool forging must be available"
    gap = Candidate(text="", kind="respond", wanted_tool="reverse_text",
                    wanted_tool_args={"text": "hello"})
    rebuilt = core._maybe_forge_and_redispatch("reverse the text hello", gap, Authority.OWNER)
    # a novel action is no longer degraded to talk: a real tool was forged and bound to the action
    assert rebuilt.kind == "act"
    assert rebuilt.tool and core.tools.get(rebuilt.tool) is not None
    # and it actually runs, producing the real reversed output
    res = core._dispatch_tool(rebuilt, Authority.OWNER)
    assert getattr(res, "ok", False) is True
    assert res.value == "olleh"


def test_forge_gap_tried_once_then_left_as_reply():
    core = NyxaraCore()
    gap = Candidate(text="fallback reply", kind="respond", wanted_tool="reverse_text",
                    wanted_tool_args={"text": "hi"})
    core._maybe_forge_and_redispatch("x", gap, Authority.OWNER)
    # a second identical gap in the same session is not re-forged — it stays a reply
    again = Candidate(text="fallback reply", kind="respond", wanted_tool="reverse_text",
                      wanted_tool_args={"text": "yo"})
    out = core._maybe_forge_and_redispatch("x", again, Authority.OWNER)
    assert out.kind == "respond" and out.text == "fallback reply"
