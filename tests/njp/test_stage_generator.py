"""Phase 5 ⑨ — proposing a rung the fixed ladder has no way to express.

:mod:`nyxara.njp.curriculum` walks nine rungs in order. Measured on a brain fresh off both
corpora: five rungs mastered (B, C, D, E, H), ``depth 0``, and ``next_stage`` returning **A** —
blocked not on failure but on *"only 3 of 20 required predictive.scored"*. It returns A on the
next assessment too, and the one after that.

The plan's rule for this phase is the design constraint: *generation and evaluation must stay
separate — a system that writes its own tests will optimise them.* Two tests here are that rule,
and they are the ones that would take the mechanism out:

* :func:`test_the_generator_reaches_past_the_ladder` — if every proposal is the rung
  ``next_stage`` already returns, this organ is an alias and buys nothing;
* :func:`test_the_generator_never_sees_the_held_out_split` — inspects the module's own source,
  because a rule enforced only by intention is not enforced.

The rest check the subtler half: that no source picks its own bar.
"""

from __future__ import annotations

import pytest

from nyxara.njp.curriculum import STAGES, Curriculum, Stage
from nyxara.njp.propose import Source, StageGenerator, _MIN_READINGS


class _Brain:
    """A brain that is only its stats block, which is all the generator is allowed to read."""

    def __init__(self, blocks, self_model=None):
        self._blocks = blocks
        self.self_model = self_model

    def stats(self):
        return dict(self._blocks)


def _plain(**over):
    blocks = {
        "predictive": {"accuracy": 0.9, "scored": 40},
        "concepts": {"concepts": 9, "compression": 2.5, "observations": 40},
        "world": {"causal_links": 9, "stated_laws": 9},
        "universe": {"usable_relations": 9, "observations": 40},
        "metareason": {"assertable_rate": 0.9, "solved": 9},
        "agency": {"success_rate": 0.9, "acted": 9},
        "beliefs": {"known": 9, "beliefs": 9},
        "field": {"meta_accepted": 2, "meta_trials": 2},
    }
    for organ, patch in over.items():
        blocks[organ] = {**blocks.get(organ, {}), **patch}
    return _Brain(blocks)


# --------------------------------------------------------------------------- #
# The two falsifiers
# --------------------------------------------------------------------------- #
def test_the_generator_reaches_past_the_ladder():
    """A proposal identical to `next_stage` every time means the generator is an alias."""
    import os

    os.environ.setdefault("NYXARA_NJP__EVOLVE_ENABLED", "false")
    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    for line in ("a sparrow is a bird", "birds can fly", "fire causes heat",
                 "a dog is a mammal", "what is a sparrow"):
        brain.think(line)

    fixed = Curriculum().next_stage(brain)
    assert fixed is not None
    generator = StageGenerator(brain)
    generator.observe(brain)
    proposals = generator.propose(brain)

    assert proposals, "a brain this early has starved rungs to name"
    keys = {(p.stage.organ, p.stage.metric, p.stage.threshold) for p in proposals}
    assert (fixed.organ, fixed.metric, fixed.threshold) not in keys or len(keys) > 1, \
        "every proposal was the rung the ladder already returns"


def test_the_generator_never_sees_the_held_out_split():
    """The separation rule, enforced structurally rather than by intention.

    Checked over the module's **imports and code**, parsed, rather than over its text — the
    docstring names ``nyxara.eval.capability`` in order to explain the rule, and a substring
    search cannot tell an explanation from a dependency.
    """
    import ast
    import inspect

    from nyxara.njp import propose

    tree = ast.parse(inspect.getsource(propose))
    imported: list = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("eval" in name for name in imported), imported

    # And nothing reaches it by attribute either, which an import check alone would miss.
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("capability", "_holdout", "split"), ast.dump(node)[:120]


# --------------------------------------------------------------------------- #
# No source picks its own bar
# --------------------------------------------------------------------------- #
def test_a_regression_is_measured_against_her_own_best():
    """The bar is a number she produced herself, so it cannot be argued down."""
    good = _plain(concepts={"compression": 2.5, "observations": 40})
    generator = StageGenerator(good)
    for _ in range(_MIN_READINGS):
        generator.observe(good)

    fallen = _plain(concepts={"compression": 1.6, "observations": 40})
    proposals = [p for p in generator.propose(fallen) if p.source == Source.REGRESSION]
    assert proposals, "compression fell from 2.5 to 1.6 and nothing noticed"
    assert proposals[0].stage.threshold == pytest.approx(2.5)
    assert "2.5" in proposals[0].why and "1.6" in proposals[0].why


def test_one_reading_is_not_a_record():
    """A best set from a single sighting is a bar chosen by accident."""
    good = _plain(concepts={"compression": 2.5, "observations": 40})
    generator = StageGenerator(good)
    generator.observe(good)                    # one reading only
    fallen = _plain(concepts={"compression": 1.0, "observations": 40})
    assert not [p for p in generator.propose(fallen) if p.source == Source.REGRESSION]


def test_ordinary_jitter_is_not_a_regression():
    good = _plain(concepts={"compression": 2.5, "observations": 40})
    generator = StageGenerator(good)
    for _ in range(_MIN_READINGS):
        generator.observe(good)
    nudged = _plain(concepts={"compression": 2.48, "observations": 40})
    assert not [p for p in generator.propose(nudged) if p.source == Source.REGRESSION]


def test_a_starved_rung_takes_the_ladders_own_sample_floor():
    """The amount of evidence required is the curriculum's number, not the generator's."""
    starved = _plain(predictive={"accuracy": 0.9, "scored": 3})
    proposals = [p for p in StageGenerator(starved).propose(starved)
                 if p.source == Source.STARVED and p.stage.letter == "A"]
    assert proposals
    stage_a = next(s for s in STAGES if s.letter == "A")
    assert proposals[0].stage.threshold == pytest.approx(stage_a.min_samples)
    # And it names the counter that must move, not the metric she cannot work on directly.
    assert proposals[0].stage.metric == stage_a.sample_metric
    assert proposals[0].stage.organ == stage_a.sample_organ


def test_a_starved_rung_that_is_also_clearly_failing_is_not_proposed_as_starved():
    """Nearly-evidenced and far under the bar is a failing rung, not a hungry one."""
    failing = _plain(metareason={"assertable_rate": 0.05, "solved": 3})
    proposals = [p for p in StageGenerator(failing).propose(failing)
                 if p.source == Source.STARVED and p.stage.letter == "F"]
    assert not proposals


def test_a_rung_with_enough_evidence_is_never_starved():
    plain = _plain()
    assert not [p for p in StageGenerator(plain).propose(plain)
                if p.source == Source.STARVED]


def test_a_weak_capability_takes_the_self_models_own_line():
    class _Cap:
        name, level, weak = "recall", 0.2, True

    class _Model:
        def weakest(self):
            return _Cap()

    brain = _Brain(_plain()._blocks, self_model=_Model())
    proposals = [p for p in StageGenerator(brain).propose(brain)
                 if p.source == Source.WEAKNESS]
    assert proposals
    assert proposals[0].stage.threshold == pytest.approx(0.5), "Capability.weak is level < 0.5"
    assert "recall" in proposals[0].why


def test_an_untested_capability_is_not_a_weak_one():
    class _Cap:
        name, level, weak = "recall", 0.2, False

    class _Model:
        def weakest(self):
            return _Cap()

    brain = _Brain(_plain()._blocks, self_model=_Model())
    assert not [p for p in StageGenerator(brain).propose(brain)
                if p.source == Source.WEAKNESS]


# --------------------------------------------------------------------------- #
# Every proposal must be evaluable by the code that already existed
# --------------------------------------------------------------------------- #
def test_every_proposed_stage_can_be_assessed():
    """A rung whose metric cannot be dug out of a stats block can never be scored.

    The generator produces a real `Stage`, so `Curriculum.assess` scores it with exactly the code
    that scores the nine — which is what keeps generation and evaluation apart in practice.
    """
    import os

    os.environ.setdefault("NYXARA_NJP__EVOLVE_ENABLED", "false")
    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    for line in ("a sparrow is a bird", "fire causes heat", "what is a sparrow"):
        brain.think(line)
    generator = StageGenerator(brain)
    generator.observe(brain)

    for proposal in generator.propose(brain):
        assert proposal.named
        report = Curriculum(stages=(proposal.stage,)).assess(brain)
        result = report.results[0]
        assert result.present, f"{proposal.stage.organ} is not a stats block"
        assert result.value is not None, \
            f"{proposal.stage.organ}.{proposal.stage.metric} reads nothing — unassessable"


def test_nothing_is_proposed_for_a_brain_with_nothing_wrong():
    plain = _plain()
    assert StageGenerator(plain).propose(plain) == []


def test_a_missing_organ_proposes_nothing_rather_than_raising():
    assert StageGenerator(_Brain({})).propose(_Brain({})) == []
    assert StageGenerator(None).propose(None) == []


def test_marks_survive_a_round_trip():
    good = _plain(concepts={"compression": 2.5, "observations": 40})
    generator = StageGenerator(good)
    for _ in range(_MIN_READINGS):
        generator.observe(good)
    revived = StageGenerator(good)
    revived.load_dict(generator.to_dict())
    assert revived.best == generator.best
