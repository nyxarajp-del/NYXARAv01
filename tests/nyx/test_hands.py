"""NYX V.02 · tool agency — sight of the toolset, not extra permission.

The measured gap was never permission: full operational control ships on by default. It was
that `NyxBrain` was never handed `core.tools`, and `NyxReasoner` deliberately never set
`tool`/`tool_args` (`reasoner.py:14`). Both were pure wiring, and both are asserted here —
along with the restraints that must survive the wiring.
"""

from __future__ import annotations

from nyxara.agency.permissions import Authority, Capability, RiskTier
from nyxara.agency.tools import ToolParam, ToolRegistry, ToolResult, ToolSpec
from nyxara.kernel.config import NyxConfig
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.hands import Hands


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSpec("echo_tool", handler=lambda text: text,
                          description="repeat the given text back",
                          params=[ToolParam("text", "str")],
                          capability=Capability.TOOL_CALL, risk=RiskTier.TRIVIAL))
    reg.register(ToolSpec("boom_tool", handler=lambda: 1 / 0,
                          description="a tool that always fails",
                          capability=Capability.TOOL_CALL, risk=RiskTier.TRIVIAL))
    return reg


def _brain(**kw) -> NyxBrain:
    brain = NyxBrain(NyxConfig(**kw))
    brain.attach_kernel(tools=_registry())
    return brain


# --------------------------------------------------------------------------- #
# Sight — the wiring gap
# --------------------------------------------------------------------------- #
def test_a_brain_with_no_registry_is_blind_and_says_so():
    """This is exactly the V.01 state, and it is a wiring problem, not a permission one."""
    hands = Hands(NyxBrain(NyxConfig()))
    assert hands.available() is False
    assert "blind to tools" in hands.describe()
    assert "wiring problem, not a permission one" in hands.describe()


def test_attach_kernel_gives_her_sight():
    brain = _brain()
    assert brain.hands.available()
    assert {"echo_tool", "boom_tool"} <= {s["name"] for s in brain.hands.catalog()}


def test_the_catalogue_shows_risk_and_reversibility():
    spec = next(s for s in _brain().hands.catalog() if s["name"] == "echo_tool")
    assert spec["risk"] == "trivial" and spec["reversible"] is True


# --------------------------------------------------------------------------- #
# Judgement — not every turn wants a tool
# --------------------------------------------------------------------------- #
def test_an_instruction_wants_a_tool():
    wanted, why = _brain().hands.wants_tool("run the tests")
    assert wanted and "instruction" in why


def test_a_question_about_the_world_does_not():
    wanted, why = _brain().hands.wants_tool("what is gravity")
    assert not wanted and "reasoning, not a tool" in why


def test_a_question_about_this_machine_does():
    wanted, why = _brain().hands.wants_tool("how many lines are in this repo?")
    assert wanted and "about this machine" in why


def test_a_hinglish_question_about_the_machine_is_understood_too():
    wanted, _why = _brain().hands.wants_tool("is repo me kitni lines hain?")
    assert wanted


def test_a_statement_never_wants_a_tool():
    wanted, why = _brain().hands.wants_tool("gravity pulls the apple down")
    assert not wanted and "does not ask for anything to be done" in why


def test_an_empty_turn_wants_nothing():
    assert _brain().hands.wants_tool("")[0] is False


# --------------------------------------------------------------------------- #
# Reaching — through the pipeline, never around it
# --------------------------------------------------------------------------- #
def test_a_named_tool_runs_and_returns_its_value():
    reach = _brain().act("say hello", tool="echo_tool", args={"text": "hello"})
    assert reach.ran and reach.ok and reach.value == "hello"


def test_a_failing_tool_is_recorded_as_a_failure_not_a_crash():
    brain = _brain()
    reach = brain.act("break", tool="boom_tool", args={})
    assert reach.ran and reach.ok is False and reach.error
    assert brain.hands.record["boom_tool"].failed == 1


def test_a_dry_run_does_not_report_itself_as_having_run():
    """``dry_run`` answers *what would happen*. Nothing executed, so nothing may say it did."""
    brain = _brain()
    reach = brain.act("echo this back", tool="echo_tool", args={"text": "hi"}, dry_run=True)
    assert reach.ran is False and reach.ok is False
    assert "nothing was executed" in reach.why


def test_a_dry_run_is_distinguishable_from_a_refusal():
    """``ran: false`` alone cannot tell the two apart, so the reason has to say which it was."""
    brain = _brain()
    dry = brain.act("echo this back", tool="echo_tool", args={"text": "hi"}, dry_run=True)
    declined = brain.act("gravity pulls the apple down")
    assert "dry run" in dry.why and not dry.refused
    assert "dry run" not in declined.why


def test_a_dry_run_does_not_move_the_tools_track_record():
    """Crediting a use for something that never ran teaches her a reliability she never measured.

    The record is the input to :meth:`Hands._trust`, which decides what she reaches for next —
    so a phantom use does not merely look wrong, it changes later behaviour.
    """
    brain = _brain()
    for _ in range(5):
        brain.act("echo this back", tool="echo_tool", args={"text": "hi"}, dry_run=True)
    assert "echo_tool" not in brain.hands.record
    assert brain.hands.reaches == 0


def test_a_real_reach_after_a_dry_run_is_the_first_recorded_use():
    brain = _brain()
    brain.act("echo this back", tool="echo_tool", args={"text": "hi"}, dry_run=True)
    reach = brain.act("echo this back", tool="echo_tool", args={"text": "hi"})
    assert reach.ran is True
    assert brain.hands.record["echo_tool"].uses == 1
    assert brain.hands.reaches == 1


def test_a_turn_that_does_not_want_a_tool_is_declined_not_forced():
    brain = _brain()
    reach = brain.act("gravity pulls the apple down")
    assert reach.wanted is False and reach.ran is False
    assert brain.hands.declined == 1


def test_execution_goes_through_the_registry_invoke():
    """No path in this module may bypass the capability/risk pipeline."""
    seen = {}
    brain = _brain()
    original = brain.tools.invoke

    def spy(name, args=None, **kw):
        seen["name"] = name
        seen["authority"] = kw.get("authority")
        return original(name, args, **kw)

    brain.tools.invoke = spy
    brain.act("say hi", tool="echo_tool", args={"text": "hi"})
    assert seen["name"] == "echo_tool"
    assert seen["authority"] == Authority.AUTONOMOUS


def test_a_gate_refusal_is_not_counted_as_a_tool_failure():
    """Distrusting a tool because the gate declined would slowly poison a working tool."""
    hands = _brain().hands
    from nyxara.nyx.hands import Reach
    reach = Reach(tool="echo_tool")
    hands._absorb(reach, ToolResult(tool="echo_tool", ok=False, requires_owner=True,
                                    decision={"allowed": False, "reason": "owner required"}))
    assert reach.refused and reach.ran is False
    assert hands.record["echo_tool"].refused == 1
    assert hands.record["echo_tool"].failed == 0


# --------------------------------------------------------------------------- #
# Corrigibility — untouched, and asked before every reach
# --------------------------------------------------------------------------- #
class _Scrammed:
    def scrammed(self) -> bool:
        return True


class _Paused:
    paused = True


class _Broken:
    @property
    def scrammed(self):
        raise RuntimeError("unreadable")


def test_scram_stops_her_before_anything_runs():
    reach = _brain().act("say hi", tool="echo_tool", args={"text": "hi"},
                         oversight=_Scrammed())
    assert reach.ran is False and "scrammed" in reach.why


def test_a_pause_stops_her_too():
    reach = _brain().act("say hi", tool="echo_tool", args={"text": "hi"},
                         oversight=_Paused())
    assert reach.ran is False


def test_an_unreadable_oversight_is_treated_as_stop():
    assert Hands._stopped(_Broken()) is True


def test_autonomy_can_be_switched_off_for_a_deployment():
    brain = _brain(hands_autonomous=False)
    reach = brain.act("say hi", tool="echo_tool", args={"text": "hi"})
    assert reach.ran is False and "switched off" in reach.why


def test_disabling_hands_leaves_v01_behaviour():
    brain = NyxBrain(NyxConfig(hands_enabled=False))
    assert brain.hands is None and brain.act("anything") is None


# --------------------------------------------------------------------------- #
# The track record — measured, and thin records stay thin
# --------------------------------------------------------------------------- #
def test_a_thin_record_does_not_move_the_score():
    """The same discipline metacog uses: below a few samples the record means nothing."""
    brain = _brain()
    assert brain.hands._trust("boom_tool") == 1.0   # never used: no evidence either way
    for _ in range(2):
        brain.act("break", tool="boom_tool", args={})
    assert brain.hands.record["boom_tool"].samples == 2
    assert brain.hands._trust("boom_tool") == 1.0   # two failures is still too thin to matter


def test_a_measured_failure_lowers_a_tools_standing():
    brain = _brain()
    for _ in range(4):
        brain.act("break", tool="boom_tool", args={})
    assert brain.hands._trust("boom_tool") < 1.0
    assert brain.hands.record["boom_tool"].samples == 4


def test_the_record_survives_a_restart():
    brain = _brain()
    brain.act("say hi", tool="echo_tool", args={"text": "hi"})
    reborn = _brain()
    reborn.load_dict(brain.to_dict())
    assert reborn.hands.record["echo_tool"].ok == 1


def test_stats_say_this_is_sight_and_not_permission():
    stats = _brain().hands.stats()
    assert "not extra permission" in stats["note"]
    assert "never lowers a risk tier" in stats["note"]


# --------------------------------------------------------------------------- #
# The reason-seat may now name a tool — and only fill an empty field
# --------------------------------------------------------------------------- #
class _Candidate:
    def __init__(self, tool: str = "") -> None:
        self.kind = "respond"
        self.text = "base answer"
        self.tool = tool
        self.tool_args: dict = {}
        self.risk = "low"
        self.reversible = True
        self.capability = "tool.call"
        self.confidence = 0.5
        self.rationale = ""


class _Thought:
    def __init__(self, stimulus: str) -> None:
        self.stimulus = stimulus
        self.intent = None


def _seat(brain, **kw):
    from nyxara.nyx.reasoner import NyxReasoner
    return NyxReasoner(base=lambda *_a, **_k: _Candidate(), brain=brain, **kw)


def test_the_seat_can_fill_an_empty_tool_field():
    brain = _brain()
    candidate = _Candidate()
    _seat(brain)._propose_tool(candidate, _Thought("run the echo_tool please"))
    assert candidate.tool  # she named something rather than staying blind


def test_the_seat_never_replaces_a_tool_the_base_chose():
    brain = _brain()
    candidate = _Candidate(tool="already_chosen")
    _seat(brain)._propose_tool(candidate, _Thought("run the echo_tool please"))
    assert candidate.tool == "already_chosen"


def test_the_seat_proposes_nothing_when_the_turn_does_not_want_a_tool():
    brain = _brain()
    candidate = _Candidate()
    _seat(brain)._propose_tool(candidate, _Thought("gravity pulls the apple down"))
    assert candidate.tool == ""


def test_the_seat_never_touches_risk_reversibility_or_capability():
    brain = _brain()
    candidate = _Candidate()
    before = (candidate.risk, candidate.reversible, candidate.capability)
    _seat(brain)._propose_tool(candidate, _Thought("run the echo_tool please"))
    assert (candidate.risk, candidate.reversible, candidate.capability) == before


def test_tool_proposal_can_be_switched_off():
    brain = _brain()
    candidate = _Candidate()
    _seat(brain, may_propose_tools=False)._propose_tool(
        candidate, _Thought("run the echo_tool please"))
    assert candidate.tool == ""


def test_a_brain_without_hands_leaves_the_candidate_untouched():
    candidate = _Candidate()
    _seat(NyxBrain(NyxConfig(hands_enabled=False)))._propose_tool(
        candidate, _Thought("run the echo_tool please"))
    assert candidate.tool == ""


# --------------------------------------------------------------------------- #
# Fail-soft
# --------------------------------------------------------------------------- #
def test_everything_is_fail_soft_on_junk():
    hands = _brain().hands
    assert hands.reach(None).ran is False
    assert hands.choose("") == []
    assert hands.load_dict(None) is False
    assert hands.load_dict({"record": "junk"}) is False   # corrupt, and it says so


def test_an_unknown_tool_is_named_as_unknown_rather_than_refused_generically():
    """The registry already knows the tool does not exist; that beats the gate's generic no.

    Once an unmodelled action became UNKNOWN-reversibility, a bad tool name was refused with
    "I could not model this action" — true, but far less useful than the fact the registry was
    holding all along.
    """
    reach = _brain().act("x", tool="no_such_tool", args={})
    assert reach.ran is False and reach.error
    assert "no tool called" in reach.error
