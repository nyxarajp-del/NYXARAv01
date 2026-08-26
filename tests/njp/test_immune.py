"""The third gate: a win bought elsewhere is not a win.

Phase 7 promotes structural changes to her own cognition on a held-out measure. It never asked
where the improvement came from — and a change that raises the number it is scored on while
lowering generalisation has found the measure rather than improved her. That is the ordinary
failure mode of any optimiser handed a metric, and it was live in code that is already promoting.

Two things these tests hold down, and both are ways the detector could look like it works:

* **An absent axis never convicts** — which is correct, and is exactly why a broken lookup is so
  dangerous here. The first version scanned for ``stage.name`` where the field is ``stage.stage``,
  so *generalization* and *transfer* came back `None` and the detector reported "no axis fell"
  over four axes with complete confidence. `test_every_axis_is_actually_read` is that regression.
* **A fall alone is not a hack.** A candidate that made everything worse is a bad candidate, and
  the benchmark gate ahead of this one already turns it away. What this refuses is the *trade*.
"""

from __future__ import annotations

from types import SimpleNamespace

from nyxara.njp import NJPBrain
from nyxara.njp.immune import Axis, Reading, RewardHacking

CLEAN = {"generalization": 1.0, "transfer": 1.0, "concept": 1.0,
         "relation": 1.0, "uncertainty": 1.0, "hallucination": 0.0}


def _reading(**overrides) -> Reading:
    values = dict(CLEAN)
    values.update(overrides)
    return Reading(values)


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #

def test_a_win_traded_for_transfer_is_caught_and_the_axis_is_named():
    immune = RewardHacking()
    verdict = immune.check(_reading(), _reading(transfer=0.5), gain=0.3)
    assert verdict.hacked
    assert verdict.axis == Axis.TRANSFER
    assert verdict.before == 1.0 and verdict.after == 0.5
    assert "bought" in verdict.why
    assert immune.caught == 1 and immune.by_axis[Axis.TRANSFER] == 1


def test_a_change_that_is_merely_bad_is_not_called_a_hack():
    """The benchmark gate ahead of this one already refuses it; calling it a hack would put the
    wrong reason in the record."""
    immune = RewardHacking()
    verdict = immune.check(_reading(), _reading(transfer=0.5), gain=0.0)
    assert not verdict.hacked
    assert verdict.fell == [Axis.TRANSFER]
    assert "not a traded one" in verdict.why
    assert immune.caught == 0


def test_an_honest_win_passes():
    immune = RewardHacking()
    verdict = immune.check(_reading(), _reading(), gain=0.3)
    assert not verdict.hacked and not verdict.fell


def test_rising_hallucination_counts_as_a_fall():
    """The one axis where up is worse. Stated as data in `HIGHER_IS_BETTER` rather than as an
    `if` per axis, so a later axis cannot be added on the wrong side by accident."""
    verdict = RewardHacking().check(_reading(), _reading(hallucination=0.25), gain=0.3)
    assert verdict.hacked and verdict.axis == Axis.HALLUCINATION
    assert Axis.HIGHER_IS_BETTER[Axis.HALLUCINATION] is False


def test_the_worst_fall_is_the_one_named():
    verdict = RewardHacking().check(
        _reading(), _reading(transfer=0.9, generalization=0.2), gain=0.3)
    assert verdict.axis == Axis.GENERALIZATION
    assert set(verdict.fell) == {Axis.TRANSFER, Axis.GENERALIZATION}


def test_an_axis_nobody_measured_never_convicts():
    thin = Reading({"transfer": None, "concept": 1.0})
    verdict = RewardHacking().check(thin, Reading({"transfer": None, "concept": 1.0}), gain=0.9)
    assert not verdict.hacked
    assert "1 measured" in verdict.why


def test_noise_is_not_a_fall():
    verdict = RewardHacking().check(_reading(), _reading(transfer=1.0 - 1e-12), gain=0.3)
    assert not verdict.hacked


# --------------------------------------------------------------------------- #
# reading what the batteries already produced
# --------------------------------------------------------------------------- #

def test_every_axis_is_actually_read():
    """Regression on a lookup that failed by returning `None`. `by_name` is the report's own API;
    the hand-rolled scan matched `stage.name`, and the field is `stage.stage`."""
    from nyxara.eval.adversarial import run_adversarial_benchmark
    from nyxara.eval.intelligence import run_intelligence_benchmark

    reading = RewardHacking.read(
        adversarial=run_adversarial_benchmark(seed=20260823, families=["polar"]),
        intelligence=run_intelligence_benchmark(seed=7, width=4))
    assert set(reading.measured) == set(Axis.ALL), reading.to_dict()


def test_reading_a_report_that_is_not_there_is_absent_not_zero():
    reading = RewardHacking.read(adversarial=None, intelligence=None)
    assert reading.measured == []
    assert all(v is None for v in reading.values.values())


# --------------------------------------------------------------------------- #
# the gate, on the real pipeline
# --------------------------------------------------------------------------- #

def test_the_gate_returns_a_third_answer_and_it_is_honest_by_default():
    brain = NJPBrain()
    benign = SimpleNamespace(apply=lambda b: True, revert=lambda b: True)
    adversarial, regression, honest, note = brain.evolution._gates(brain, benign, gain=0.3)
    assert adversarial and regression and honest
    assert note == ""


def test_a_promotion_is_checked_rather_than_assumed():
    """The contrast that matters: the gate runs on the candidate that was actually promoted, and
    its verdict is on the record with the numbers — not a boolean nobody can audit."""
    session = ["birds need water", "a sparrow is a bird", "what does a sparrow need?",
               "sparrows need water", "why do birds need water?", "a crow is a bird",
               "what does a crow need?", "crows need water", "aag lagi", "garmi hui",
               "pasina aaya", "what caused garmi?", "hello", "a robin is a bird",
               "what does a robin need?", "robins need water", "plants need sunlight",
               "a rose is a plant", "what does a rose need?", "a tulip is a plant"]
    brain = NJPBrain()
    for turn in range(200):
        brain.think(session[turn % len(session)])

    evolution = brain.evolution
    if not evolution.cognitive_rewires:
        return                      # nothing was promoted on this corpus; nothing to audit
    stats = evolution.stats()["immune"]
    assert stats["checks"] >= 1, stats
    assert stats["last"] is not None
    assert all(not t.hacked for t in evolution.trials if t.promoted)


def test_gates_off_skips_the_check_rather_than_passing_it_silently():
    from nyxara.njp.evolution import CognitiveEvolution

    box = CognitiveEvolution(gates=False)
    assert box._gates(SimpleNamespace(), SimpleNamespace(apply=lambda b: True)) == (
        True, True, True, "")
    assert box.immune.checks == 0
