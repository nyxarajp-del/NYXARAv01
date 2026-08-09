"""NYX V.02 · in-context learning — a procedure learned inside one turn, and kept.

Two measurements are pinned here. Before this phase: ``solve("7")`` returned ``None`` after a
×2 skill had been learned from ``3 → 6``, and ``john smith → Smith, John`` induced as ``None``
because field permutation was not in the operation set at all. Both now work, and the honesty
rule that comes with a finite operation set — refuse, do not guess — is pinned too.
"""

from __future__ import annotations

from nyxara.cognition.skill_induction import SkillInductionEngine
from nyxara.kernel.config import NyxConfig
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.icl import InContextLearner

SHOUT = "apple -> APPLE!\nmango -> MANGO!\ncar -> ?"
NAMES = "john smith -> Smith, John\nmary jones -> Jones, Mary\nada lovelace -> ?"


def _learner(**kw) -> InContextLearner:
    return InContextLearner(**kw)


# --------------------------------------------------------------------------- #
# Reading demonstrations out of a turn — in whatever shape they arrived
# --------------------------------------------------------------------------- #
def test_arrow_demonstrations_are_read():
    demos, probe = _learner().demonstrations(SHOUT)
    assert [(d.given, d.wanted) for d in demos] == [("apple", "APPLE!"), ("mango", "MANGO!")]
    assert probe == "car"


def test_every_common_separator_is_understood():
    for block in ("a -> A\nb -> B", "a → A\nb → B", "a: A\nb: B",
                  "a | A\nb | B", "a\tA\nb\tB", "a => A\nb => B"):
        demos, _probe = _learner().demonstrations(block)
        assert len(demos) == 2, block


def test_a_trailing_bare_line_is_the_probe():
    demos, probe = _learner().demonstrations("apple -> APPLE!\nmango -> MANGO!\ncar")
    assert len(demos) == 2 and probe == "car"


def test_ordinary_prose_is_not_a_demonstration_block():
    demos, probe = _learner().demonstrations(
        "gravity pulls the apple down, which is why it falls")
    assert demos == [] and probe == ""


def test_one_pair_is_not_a_demonstration_block():
    """A single example is consistent with infinitely many rules, so it teaches nothing."""
    assert _learner().demonstrations("apple -> APPLE!")[0] == []


# --------------------------------------------------------------------------- #
# The whole act — read, induce, answer
# --------------------------------------------------------------------------- #
def test_two_demonstrations_teach_a_new_procedure():
    got = _learner().learn(SHOUT)
    assert got.learned and got.answer == "CAR!"
    assert "upper" in got.program


def test_the_measured_reorder_gap_is_closed():
    """``john smith -> Smith, John`` induced as None before the reorder family existed."""
    got = _learner().learn(NAMES)
    assert got.learned and got.answer == "Lovelace, Ada"
    assert "reorder" in got.program


def test_arithmetic_transfers_to_an_unseen_number():
    got = _learner().learn("3 : 6\n5 : 10\n7 : ?")
    assert got.answer == "14"


def test_a_held_out_demonstration_is_what_earns_verified():
    """Fit is not transfer. Three demos let one be withheld and actually checked."""
    two = _learner().learn("apple -> APPLE!\nmango -> MANGO!\ncar -> ?")
    three = _learner().learn("apple -> APPLE!\nmango -> MANGO!\nplum -> PLUM!\ncar -> ?")
    assert two.generalized is False
    assert three.generalized is True


def test_outside_the_operation_set_she_refuses_rather_than_guesses():
    got = _learner().learn(
        "cat -> the moon is made of green cheese\ndog -> quantum chromodynamics\nfish -> ?")
    assert got.learned is False and got.answer == ""
    assert "could not reduce them to a rule" in got.refusal
    assert "rather say that than invent" in got.refusal


def test_a_refusal_still_names_the_operation_set():
    got = _learner().learn("a -> zzz\nb -> qqq\nc -> ?")
    assert "field reordering" in got.refusal and "slot templates" in got.refusal


def test_learning_never_raises_on_junk():
    learner = _learner()
    assert learner.learn(None).learned is False
    assert learner.learn("").demos == []
    assert learner.demonstrations(None) == ([], "")
    assert learner.apply("nope", "x") is None


# --------------------------------------------------------------------------- #
# The measured `solve("7")` bug — the match floor and the identified probe
# --------------------------------------------------------------------------- #
def test_the_anchorless_floor_still_protects_an_unrelated_turn():
    engine = SkillInductionEngine()
    engine.induce("double", [("3", "6"), ("5", "10"), ("8", "16")])
    assert engine.solve("what is the weather like today") is None
    assert engine.solve("7") is None                 # not identified as a probe → floor applies


def test_an_identified_probe_bypasses_the_lexical_floor():
    """The measured failure: a ×2 skill answered None to "7" because "7" is not like "3"."""
    engine = SkillInductionEngine()
    engine.induce("double", [("3", "6"), ("5", "10"), ("8", "16")])
    got = engine.solve("7", probe=True)
    assert got is not None and got[0] == "14"


def test_a_probe_still_has_to_change_the_input():
    engine = SkillInductionEngine()
    engine.induce("shout", [("apple", "APPLE!"), ("mango", "MANGO!")])
    assert engine.solve("", probe=True) is None


# --------------------------------------------------------------------------- #
# The reorder operation family, on its own
# --------------------------------------------------------------------------- #
def test_reorder_generalises_to_an_unseen_input():
    engine = SkillInductionEngine()
    skill = engine.induce("names", [("john smith", "Smith, John"),
                                    ("mary jones", "Jones, Mary"),
                                    ("alan turing", "Turing, Alan")])
    assert skill is not None and skill.generalized
    assert skill.apply("ada lovelace") == "Lovelace, Ada"


def test_reorder_needs_a_consistent_permutation_across_demos():
    """One pair can always be explained by *some* permutation; two disagreeing ones cannot."""
    from nyxara.cognition.skill_induction import _induce_reorder
    assert _induce_reorder(["a b", "c d"], ["b a", "c d"]) is None
    assert _induce_reorder(["a b"], ["a b"]) is None          # the identity is not a discovery


def test_reorder_is_inert_on_the_wrong_arity():
    engine = SkillInductionEngine()
    skill = engine.induce("names", [("john smith", "Smith, John"),
                                    ("mary jones", "Jones, Mary")])
    assert skill is not None
    assert skill.apply("madonna") == "madonna"       # one field: no fire, and no crash


# --------------------------------------------------------------------------- #
# Through the brain — and across a restart
# --------------------------------------------------------------------------- #
def _brain(**kw) -> NyxBrain:
    return NyxBrain(NyxConfig(**kw))


def test_a_demonstration_turn_is_answered_by_the_induced_procedure():
    thought = _brain().think(SHOUT)
    assert thought.answer == "CAR!"
    assert thought.winner is not None and thought.winner.source == "skill"
    assert thought.induced is not None and thought.induced.learned


def test_an_ordinary_turn_induces_nothing():
    thought = _brain().think("what is gravity")
    assert thought.induced is None or thought.induced.learned is False


def test_a_learned_skill_survives_a_restart():
    brain = _brain()
    brain.think(SHOUT)
    reborn = _brain()
    reborn.load_dict(brain.to_dict())
    assert len(reborn.icl.skills()) == 1
    got = reborn.icl.solve("banana", probe=True)
    assert got is not None and got[0] == "BANANA!"


def test_learn_from_is_reachable_on_the_brain():
    got = _brain().learn_from(NAMES)
    assert got is not None and got.answer == "Lovelace, Ada"


def test_stats_say_what_kind_of_learning_this_is():
    brain = _brain()
    brain.think(SHOUT)
    stats = brain.stats()["icl"]
    assert stats["skills_induced"] == 1 and stats["available"] is True
    assert "no weight is updated anywhere" in stats["note"]


def test_the_skill_specialist_is_seated_and_honest():
    brain = _brain()
    seated = [s.name for s in brain.workspace.specialists]
    assert "skill" in seated


def test_disabling_in_context_learning_leaves_v01_behaviour():
    brain = _brain(icl_enabled=False)
    assert brain.icl is None
    thought = brain.think(SHOUT)
    assert thought.induced is None


def test_a_corrupt_sidecar_never_raises():
    learner = _learner()
    assert learner.load_dict(None) is False
    assert learner.load_dict({"skills": ["junk", {"name": ""}]}) is True
    assert learner.skills() == []
