"""Phase 5: skills she invents, keeps, and can use somewhere they were not invented for.

The milestone is two claims and only one was answerable::

    skills_created > 0    and    the skills transfer to unseen tasks

**The first was true and unreportable.** `_compress_programs` fires on the dream cadence and
really does adopt abstractions — measured, 12 solved and 1 adopted on turn 31 — and neither number
was ever rolled into the session totals, so `stats()` had no key for them at all. Same shape as
`bits_gained` before it: carried on the report, absent from the sum.

**The second was unreachable.** `NoesisEngine` has always called
``self.transfer.transfer_all(library, adopted, tasks)`` on every SLEEP, and nothing in the
repository implemented that interface. The attribute defaulted to ``None``, the guard above the
call was always false, and ``transferred`` was structurally 0 on every cycle ever run.

Two further Phase 5 items were built and homeless: an adopted program and a promoted reasoning
shape are both *how to do something*, and the **procedural** memory level — the one whose policy
is stability by use over ninety days — was empty on every session while both were being produced.
"""

from __future__ import annotations


from nyxara.njp import NJPBrain
from nyxara.njp.levels import Level


def _worked(turns: int = 120) -> NJPBrain:
    brain = NJPBrain()
    for i in range(turns):
        brain.think(("birds need water", "a sparrow is a bird")[i % 2])
    return brain


# --------------------------------------------------------------------------- #
# skills_created > 0
# --------------------------------------------------------------------------- #

def test_the_session_can_say_how_many_skills_were_created():
    brain = _worked()
    totals = brain.loop.stats()["totals"]
    assert totals["programs_solved"] > 0, totals
    assert totals["programs_adopted"] > 0, totals


def test_an_adopted_abstraction_is_a_real_new_primitive():
    """Not a macro comment — an entry in the language with a typed body."""
    brain = _worked()
    library = brain.noesis.library
    assert library.abstractions, brain.noesis.report()
    name, abstraction = next(iter(library.abstractions.items()))
    assert abstraction.body is not None
    assert abstraction.arity >= 0
    assert library.entry(name) is not None


# --------------------------------------------------------------------------- #
# and they transfer to unseen tasks
# --------------------------------------------------------------------------- #

def test_the_transfer_verifier_is_attached():
    """It defaulted to None, so the milestone's second half could never be reached."""
    from nyxara.growth.zeroshot import ZeroShotTransfer
    assert isinstance(NJPBrain().noesis.transfer, ZeroShotTransfer)


def test_a_skill_is_confirmed_on_a_task_that_arrived_after_it():
    """The held-out discipline is in the timing.

    `transfer_all` is called with the abstractions just adopted and the tasks just solved — and
    the adoption was extracted from those solutions. Checking them against each other is the
    induction restated, so an adoption is held and verified on the *next* cycle's tasks.
    """
    from nyxara.growth.noesis import NoesisEngine
    from nyxara.growth.zeroshot import ZeroShotTransfer
    transfer = ZeroShotTransfer()
    engine = NoesisEngine(seed=42, tasks_per_cycle=12, transfer=transfer)
    for _ in range(6):
        engine.step()
    stats = transfer.stats()
    assert stats["transferred"] > 0, stats
    assert stats["abstractions_that_transferred"] > 0, stats
    # A claim of transfer that cannot show the program is one nobody can check.
    assert all(row["program"] for row in stats["examples"]), stats


def test_a_transfer_that_only_fits_the_examples_is_refused():
    """Fitting a handful of examples is exactly how a wrong program looks right.

    Measured before the falsifier was added: `abs0 := count_gt(x, $0)` was confirmed as
    transferring to `count_over_2` via `abs0(0)` — counting elements above *zero*, which agreed
    with that task's examples and is a different function.
    """
    from nyxara.growth.noesis import NoesisEngine
    from nyxara.growth.zeroshot import ZeroShotTransfer
    transfer = ZeroShotTransfer()
    engine = NoesisEngine(seed=42, tasks_per_cycle=12, transfer=transfer)
    for _ in range(6):
        engine.step()
    stats = transfer.stats()
    assert stats["refuted_by_red_team"] > 0, stats


def test_a_task_with_no_oracle_is_never_confirmed():
    """An unfalsifiable confirmation is what makes a metric drift."""
    from nyxara.growth.noesis import Prog
    from nyxara.growth.zeroshot import ZeroShotTransfer
    transfer = ZeroShotTransfer()

    class _Task:
        name, input_type, ret_type, oracle = "probe", "intlist", "int", None
        examples = ((( 1, 2), 3),)

    assert transfer._survives(Prog(kind="lit", rtype="int", value=1), _Task(), None) is False


def test_the_loop_reports_transfer_as_a_session_total():
    brain = _worked()
    assert "programs_transferred" in brain.loop.stats()["totals"]


# --------------------------------------------------------------------------- #
# procedural memory — where a skill lives
# --------------------------------------------------------------------------- #

def test_an_adopted_program_is_filed_in_procedural_memory():
    """A skill that lives only in the organ that produced it is one the rest of her cannot find."""
    brain = _worked()
    keys = [brain.levels.entries[k].key for k in brain.levels.levels[Level.PROCEDURAL]]
    assert any(k.startswith("program:") for k in keys), keys


def test_a_promoted_reasoning_shape_is_filed_there_too():
    brain = NJPBrain()
    session = ["birds need water", "a sparrow is a bird", "what does a sparrow need?",
               "sparrows need water", "a crow is a bird", "what does a crow need?",
               "crows need water", "a robin is a bird", "what does a robin need?",
               "robins need water"]
    for _ in range(6):
        for line in session:
            brain.think(line)
    keys = [brain.levels.entries[k].key for k in brain.levels.levels[Level.PROCEDURAL]]
    assert any(k.startswith("shape:") for k in keys), keys


# --------------------------------------------------------------------------- #
# strategy library — a bandit that never explores has not learned anything
# --------------------------------------------------------------------------- #

def test_the_least_tried_option_is_explored_not_the_first():
    """`untried[0]` is insertion order, so the option registered first monopolises the arm.

    A reward needs a *solved problem*, not merely a turn, so rewards are sparse — and the later
    options were never reached at all. Measured over 254 selections: derive at 3 trials, recall
    and a promoted shape at 0, and `switches` 0 because the choice never changed.
    """
    from nyxara.njp.selfmodel import MetaLearner
    learner = MetaLearner()
    learner.register("arm", "first", value="first")
    learner.register("arm", "second", value="second")
    picked = []
    for _ in range(4):
        chosen = learner.choose("arm")
        picked.append(chosen.name)
        learner.reward("arm", 1.0)
    assert set(picked) == {"first", "second"}, picked


def test_the_bandit_actually_switches_over_a_question_heavy_session():
    brain = NJPBrain()
    for line in ("birds need water", "a sparrow is a bird", "a crow is a bird",
                 "plants need water", "aag se garmi hoti hai", "garmi se pasina hota hai",
                 "a rose is a plant"):
        brain.think(line)
    for _ in range(6):
        for question in ("what does a sparrow need?", "what does a crow need?", "why pasina",
                         "what is a bird?", "what does a rose need?", "what causes pasina",
                         "what does a plant need?", "what is a rose?"):
            brain.think(question)
    assert brain.meta.stats()["switches"] > 0, brain.meta.stats()["arms"]
