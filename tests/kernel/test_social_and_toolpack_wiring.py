"""Wiring tests: the orchestrator now composes the social faculties and the tool packs.

Two dormant-but-real capabilities are made live here:
* the social-cognition faculties (empathy, persons, common ground, culture, repair) run on
  every turn as non-governing colour, surfaced in CycleResult.social; and
* the domain tool packs register their pure-stdlib tools (summarize_text, python_check) onto
  the same gated registry.
"""

from __future__ import annotations

from nyxara import Authority, NyxaraCore


def test_social_faculties_are_built():
    core = NyxaraCore()
    assert core.persons is not None
    assert core.empathy is not None
    assert core.common_ground is not None
    assert core.culture is not None
    assert core.repair is not None


def test_turn_produces_a_social_read():
    core = NyxaraCore()
    result = core.process(
        "I'm really happy, thank you so much for the help!", authority=Authority.OWNER)
    snap = result.social
    assert isinstance(snap, dict) and snap, "social read should be populated on an owner turn"
    assert snap.get("trust") == "owner"
    assert "empathy" in snap and snap["empathy"]["reads"]
    assert "style" in snap
    # the same read is reachable via introspection and serialises in to_dict
    assert core.social_snapshot() == snap
    assert "social" in result.to_dict()


def test_repair_triggers_on_a_correction():
    core = NyxaraCore()
    core.process("Restart the backup service.", authority=Authority.OWNER)
    result = core.process("No, that's not what I meant.", authority=Authority.OWNER)
    assert result.social.get("repair"), "a correction should surface a repair action"
    assert result.social["repair"]["type"] in {
        "apology_retry", "paraphrase", "confirmation", "specification", "clarification"}


def test_social_read_resets_between_turns():
    core = NyxaraCore()
    core.process("No, that's wrong.", authority=Authority.OWNER)
    # a plain, untroubled turn should not carry the previous turn's repair action
    result = core.process("Thanks, that is perfect.", authority=Authority.OWNER)
    assert "repair" not in result.social


def test_toolpacks_registered_in_default_toolset():
    core = NyxaraCore()
    assert core.tools is not None
    names = set(core.tools.names())
    assert "summarize_text" in names
    assert "python_check" in names
