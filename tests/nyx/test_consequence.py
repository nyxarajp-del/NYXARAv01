"""NYX V.02 · L-CHRONO-CAUSAL — look before acting, and step back from a bad irreversible tail.

L-CHRONOS simulates futures for decisions *inside* `think()`. It has never run on a tool call
or on code being written. With `hands` and `author` aboard that is the gap that matters, so
this gate sits in front of execution — and the tests here are mostly about what it *stops*.
"""

from __future__ import annotations

import os

import pytest

from nyxara.kernel.config import NyxConfig
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.consequence import ConsequenceGate, Reversibility, Verdict


@pytest.fixture()
def gate(tmp_path) -> ConsequenceGate:
    return ConsequenceGate(NyxBrain(NyxConfig()), workspace=str(tmp_path))


def _brain(**kw) -> NyxBrain:
    return NyxBrain(NyxConfig(**kw))


# --------------------------------------------------------------------------- #
# A model of the action, from checkable facts
# --------------------------------------------------------------------------- #
def test_a_write_inside_the_workspace_is_undoable(gate, tmp_path):
    got = gate.check("write_file", {"path": str(tmp_path / "notes.txt")})
    assert got.reversibility == Reversibility.UNDOABLE
    assert got.blast_radius == 0 and got.allowed


def test_a_delete_inside_the_workspace_is_recoverable(gate, tmp_path):
    got = gate.check("delete_path", {"path": str(tmp_path / "build")})
    assert got.reversibility == Reversibility.RECOVERABLE
    assert got.blast_radius == 0


def test_a_delete_outside_the_workspace_is_irreversible(gate):
    got = gate.check("delete_path", {"path": "/etc/passwd"})
    assert got.reversibility == Reversibility.IRREVERSIBLE
    assert got.blast_radius == 1


def test_anything_that_leaves_the_machine_is_irreversible(gate):
    got = gate.check("git_push", {"remote": "origin"})
    assert got.reversibility == Reversibility.IRREVERSIBLE
    assert any(e.kind == "outbound" for e in got.effects)


def test_blast_radius_counts_paths_not_outbound_effects(gate):
    """An outbound call leaves the machine but touches no file here."""
    got = gate.check("git_push", {"remote": "origin"})
    assert got.blast_radius == 0


def test_the_registrys_own_flags_are_read_when_there_is_a_registry(tmp_path):
    from nyxara.agency.permissions import Capability, RiskTier
    from nyxara.agency.tools import ToolRegistry, ToolSpec
    registry = ToolRegistry()
    registry.register(ToolSpec("one_way", handler=lambda: 1, description="x",
                               capability=Capability.TOOL_CALL, risk=RiskTier.HIGH,
                               reversible=False))
    gate = ConsequenceGate(_brain(), workspace=str(tmp_path))
    got = gate.check("one_way", {}, registry=registry)
    assert got.reversibility == Reversibility.IRREVERSIBLE
    assert any(e.kind == "registered" for e in got.effects)


def test_git_effects_are_read_from_git_not_imagined(tmp_path):
    """The working-tree summary comes from `git diff --stat`, or not at all."""
    gate = ConsequenceGate(_brain(), workspace=os.getcwd())
    got = gate.check("git_commit", {"message": "x"})
    assert any(e.kind == "git" for e in got.effects)


# --------------------------------------------------------------------------- #
# Her own restraint — the point of the module
# --------------------------------------------------------------------------- #
def test_she_will_not_act_blind_on_something_irreversible(gate):
    """The one non-advisory rule: cannot see ahead + cannot take back ⇒ do not act."""
    got = gate.check("delete_path", {"path": "/etc/passwd"})
    assert got.allowed is False
    assert "does not act blind on something irreversible" in got.why


def test_she_will_act_blind_on_something_undoable(gate, tmp_path):
    """Blindness alone is not a reason to freeze — only blindness plus irreversibility is."""
    got = gate.check("write_file", {"path": str(tmp_path / "a.txt")})
    assert got.allowed is True


def test_the_fail_closed_rule_can_be_relaxed_for_a_deployment(tmp_path):
    gate = ConsequenceGate(_brain(), workspace=str(tmp_path),
                           require_foresight_for_irreversible=False)
    got = gate.check("delete_path", {"path": "/etc/passwd"})
    assert got.allowed is True
    # …and it says exactly why, rather than implying the action became safe.
    assert "no way back" in got.why or "fail-closed rule off" in got.why


def test_a_gate_that_breaks_does_not_open(gate, monkeypatch):
    monkeypatch.setattr(gate, "_model", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    got = gate.check("delete_path", {"path": "/etc/passwd"})
    assert got.allowed is False and "fails closed" in got.why
    assert got.reversibility == Reversibility.UNKNOWN


def test_an_unnamed_action_is_refused(gate):
    assert gate.check("").allowed is False


# --------------------------------------------------------------------------- #
# The reversible variant — a real substitution, or none at all
# --------------------------------------------------------------------------- #
def test_committing_on_a_shared_branch_is_rerouted_onto_a_new_one(monkeypatch, tmp_path):
    gate = ConsequenceGate(_brain(), workspace=str(tmp_path))
    monkeypatch.setattr(gate, "_current_branch", lambda: "main")
    # Force the irreversible path so the variant search runs.
    monkeypatch.setattr(gate, "_class_of", lambda effects: Reversibility.IRREVERSIBLE)
    got = gate.check("git_commit", {"message": "x"})
    assert got.took_a_safer_route
    assert got.variant == "git_checkout" and got.variant_args["create"] is True
    assert "shared history is never written to directly" in got.why


def test_no_variant_is_invented_when_none_exists(monkeypatch, tmp_path):
    """Inventing a 'safe version' of an action that has none looks like care and is not."""
    gate = ConsequenceGate(_brain(), workspace=str(tmp_path))
    monkeypatch.setattr(gate, "_current_branch", lambda: "side-branch")
    got = gate.check("git_push", {"remote": "origin"})
    assert got.took_a_safer_route is False
    assert got.allowed is False


def test_a_delete_outside_the_workspace_gets_no_variant(gate):
    got = gate.check("delete_path", {"path": "/etc/passwd"})
    assert got.variant == "" and got.allowed is False


# --------------------------------------------------------------------------- #
# Foresight — bounded, and blindness is a result
# --------------------------------------------------------------------------- #
def test_a_blind_world_model_says_so_rather_than_returning_zeros(gate, tmp_path):
    got = gate.check("write_file", {"path": str(tmp_path / "a.txt")})
    assert got.foresight.blind is True
    assert "zeros dressed as foresight" in got.foresight.reason


def test_a_brain_without_chronos_reports_it_cannot_see_at_all(tmp_path):
    gate = ConsequenceGate(_brain(chronos_enabled=False), workspace=str(tmp_path))
    got = gate.check("write_file", {"path": str(tmp_path / "a.txt")})
    assert got.foresight.ran is False
    assert "cannot see ahead at all" in got.foresight.reason


def test_a_bad_tail_stops_even_a_recoverable_action(gate, tmp_path, monkeypatch):
    from nyxara.nyx.consequence import Foresight
    monkeypatch.setattr(gate, "_roll",
                        lambda *a, **k: Foresight(ran=True, branches=64, tail_risk=0.9,
                                                  confidence=0.8))
    got = gate.check("delete_path", {"path": str(tmp_path / "build")})
    assert got.allowed is False and "worst futures are bad" in got.why


def test_a_good_tail_lets_a_recoverable_action_through(gate, tmp_path, monkeypatch):
    from nyxara.nyx.consequence import Foresight
    monkeypatch.setattr(gate, "_roll",
                        lambda *a, **k: Foresight(ran=True, branches=64, tail_risk=0.1,
                                                  confidence=0.8))
    got = gate.check("delete_path", {"path": str(tmp_path / "build")})
    assert got.allowed is True


# --------------------------------------------------------------------------- #
# The seam hands and author use
# --------------------------------------------------------------------------- #
def test_hands_is_stopped_by_the_gate():
    from nyxara.agency.permissions import Capability, RiskTier
    from nyxara.agency.tools import ToolRegistry, ToolSpec
    registry = ToolRegistry()
    registry.register(ToolSpec("nuke_it", handler=lambda: "boom", description="x",
                               capability=Capability.TOOL_CALL, risk=RiskTier.CRITICAL,
                               reversible=False))
    brain = _brain()
    brain.attach_kernel(tools=registry)
    reach = brain.act("do it", tool="nuke_it", args={})
    assert reach.ran is False
    assert "consequence gate stopped this" in reach.why
    assert reach.verdict["reversibility"] == Reversibility.IRREVERSIBLE


def test_hands_still_runs_a_reversible_tool():
    from nyxara.agency.permissions import Capability, RiskTier
    from nyxara.agency.tools import ToolParam, ToolRegistry, ToolSpec
    registry = ToolRegistry()
    registry.register(ToolSpec("echo_tool", handler=lambda text: text, description="x",
                               params=[ToolParam("text", "str")],
                               capability=Capability.TOOL_CALL, risk=RiskTier.TRIVIAL))
    brain = _brain()
    brain.attach_kernel(tools=registry)
    reach = brain.act("say hi", tool="echo_tool", args={"text": "hi"})
    assert reach.ran and reach.ok and reach.value == "hi"


def test_the_gate_can_be_switched_off_entirely():
    brain = _brain(consequence_enabled=False)
    assert brain.consequence is None and brain.foresee("anything") is None


def test_foresee_is_reachable_on_the_brain():
    got = _brain().foresee("delete_path", {"path": "/etc/passwd"})
    assert got is not None and got.allowed is False


# --------------------------------------------------------------------------- #
# Honesty and fail-soft
# --------------------------------------------------------------------------- #
def test_stats_state_the_limits_plainly(gate):
    note = gate.stats()["note"]
    assert "not an ocean of timelines" in note
    assert "not five years of the world" in note
    assert "no '100% optimal'" in note


def test_counters_track_stops_and_reroutes(gate):
    gate.check("delete_path", {"path": "/etc/passwd"})
    gate.check("write_file", {"path": "/tmp/x"})
    stats = gate.stats()
    assert stats["checked"] == 2 and stats["stopped"] >= 1


def test_a_verdict_serialises_with_its_reasoning(gate):
    got = gate.check("delete_path", {"path": "/etc/passwd"}).to_dict()
    assert got["allowed"] is False and got["why"] and got["effects"]
    assert set(got) >= {"action", "reversibility", "blast_radius", "foresight"}


def test_reversibility_ordering_is_worst_wins():
    assert Reversibility.worse(Reversibility.UNDOABLE,
                               Reversibility.IRREVERSIBLE) == Reversibility.IRREVERSIBLE
    assert Reversibility.worse(Reversibility.IRREVERSIBLE,
                               Reversibility.UNKNOWN) == Reversibility.UNKNOWN


def test_everything_is_fail_soft_on_junk(gate):
    assert gate.check(None).allowed is False
    assert Verdict().to_dict()["allowed"] is True     # a default verdict is inert, not a stop


# --------------------------------------------------------------------------- #
# Unmodelled is not the same as safe
# --------------------------------------------------------------------------- #
def test_an_action_she_modelled_nothing_about_is_unknown_not_undoable():
    """``_class_of`` started at UNDOABLE, so anything it could not model came out "safe".

    Measured before the fix: ``rm -rf /some/unknown/path`` was classed **undoable** and
    **allowed** — free text has no args, so it produced no paths, no registry entry and no
    outbound token, and the loop had nothing to worsen. That is the empty-graph-free-bonus
    shape, in the one gate that decides whether she destroys something.
    """
    gate = ConsequenceGate()
    verdict = gate.check("rm -rf /some/unknown/path", {})
    assert verdict.reversibility == Reversibility.UNKNOWN
    assert verdict.allowed is False
    assert "could not model this action at all" in verdict.why


def test_an_unmodelled_action_gets_its_own_words_not_the_irreversible_ones():
    """"Cannot be taken back and cannot see ahead" is a different claim from "I do not know it"."""
    verdict = ConsequenceGate().check("do the thing", {})
    assert "cannot be taken back" not in verdict.why
    assert "Unmodelled is not the same as safe" in verdict.why


def test_a_registered_tool_that_touches_nothing_is_still_undoable():
    """The distinction is *modelled nothing*, not *does nothing* — a read-only tool must pass."""
    from nyxara.agency.permissions import Capability, RiskTier
    from nyxara.agency.tools import ToolParam, ToolRegistry, ToolSpec

    registry = ToolRegistry()
    registry.register(ToolSpec("echo_tool", handler=lambda text: text,
                               description="repeat the given text back",
                               params=[ToolParam("text", "str")],
                               capability=Capability.TOOL_CALL, risk=RiskTier.TRIVIAL))
    verdict = ConsequenceGate().check("echo_tool", {"text": "hi"}, registry=registry)
    assert verdict.reversibility == Reversibility.UNDOABLE
    assert verdict.allowed is True
    assert verdict.effects            # the registry's own flags are something to read
